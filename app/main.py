"""Build the app. Run it with: uvicorn --factory app.main:create_app --no-access-log"""

import logging
import random
import sqlite3
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import routes_admin, routes_auth, routes_feed, routes_me, routes_social, ui, ui_admin
from app.web import PageRouting, Redirect, is_page, redirect, render
from app.actions import ActionError
from app.backups import start_backups
from app.config import Settings
from app.matchers import build_matchers
from app.ratelimit import RateLimiter
from app.security import install_security
from app.services import Services
from live_store import LiveStore
from manager import ProfileManager
import paths


def refuse_agent_databases(settings: Settings) -> None:
    """Stop at startup if the live app points at simulated-agent data.

    Real people must never be shown agents, and the agents' data must never get live
    tables or real people's rows mixed into it. This runs before any table is created.
    """
    agent_files = {paths.MAIN_DB, paths.PERSONA_DB, paths.SOURCE_MAIN_DB, paths.SOURCE_PERSONA_DB}
    for path in (settings.main_db, settings.persona_db):
        if path.resolve() in {p.resolve() for p in agent_files}:
            raise RuntimeError(f"{path} holds simulated-agent data. Point APP_MAIN_DB and APP_PERSONA_DB at the live database.")
    checks = ((settings.main_db, "SELECT 1 FROM profiles WHERE agent = 1 LIMIT 1"), (settings.persona_db, "SELECT 1 FROM personas LIMIT 1"))
    for path, sql in checks:
        if not path.exists():
            continue
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            try:
                found = conn.execute(sql).fetchone()
            except sqlite3.OperationalError:  # The table doesn't exist yet.
                found = None
        if found:
            raise RuntimeError(f"{path} contains simulated agents or personas. The live app needs its own database.")


def create_app(
    settings: Settings | None = None,
    rng: random.Random | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if settings.production and (problems := settings.production_problems()):
        raise RuntimeError("Unsafe settings for APP_ENV=production:\n- " + "\n- ".join(problems))
    refuse_agent_databases(settings)
    settings.main_db.parent.mkdir(parents=True, exist_ok=True)
    manager = ProfileManager(settings.main_db, settings.persona_db)  # Creates every table.
    rng = rng or random.Random(settings.matcher_seed)
    services = Services(
        settings=settings,
        manager=manager,
        live=LiveStore(settings.main_db),
        matchers=build_matchers(settings.enabled_matchers, settings.default_matcher, rng),
        rng=rng,
        login_per_ip=RateLimiter(settings.login_requests_per_ip, settings.login_window_seconds),
    )
    services.consent_text()  # Fail at startup, not on first use, if the text is missing.
    services.initialize_controls()
    services.give_display_names()
    services.live.adopt_admin_passwords()

    app = FastAPI(title="Matching study", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.services = services
    install_security(app, settings.site_url, settings.cookie_secure)

    error_pages = {
        400: ("Something's wrong with that request", "Please go back and try again."),
        403: ("You can't open this page", "You don't have access to this page."),
        404: ("Page not found", "We couldn't find that page."),
        405: ("Page not found", "We couldn't find that page."),
        422: ("Something in the form wasn't right", "Please go back and check what you entered."),
        429: ("Please slow down", "You've done that too many times. Wait a little and try again."),
    }

    def error_page(request: Request, status: int, message: str | None = None) -> Response:
        heading, default = error_pages.get(status, ("Something went wrong", "Please try again later."))
        # Only messages written for users are shown, never internal details.
        return render(request, "error.html", {"heading": heading, "message": message or default}, status)

    if settings.backup_every_hours > 0:
        app.state.backups = start_backups(settings.main_db, settings.backup_dir, settings.backup_every_hours, settings.backup_keep)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        """For the host's health checks: the app is up and can read the database."""
        services.live.get_setting("match_threshold")
        return {"ok": True}

    @app.exception_handler(Redirect)
    def go(request: Request, error: Redirect) -> Response:
        return redirect(request, error.url)

    @app.exception_handler(StarletteHTTPException)
    def http_error(request: Request, error: StarletteHTTPException) -> Response:
        if is_page(request):
            safe = error.detail if error.status_code in (403, 422) and isinstance(error.detail, str) else None
            return error_page(request, error.status_code, safe)
        return JSONResponse({"detail": error.detail}, status_code=error.status_code, headers=getattr(error, "headers", None))

    @app.exception_handler(Exception)
    def crash(request: Request, error: Exception) -> Response:
        logging.getLogger(__name__).exception("Unhandled error")
        services.live.record_event("server_error")
        if is_page(request):
            return error_page(request, 500)
        return JSONResponse({"detail": "Something went wrong"}, status_code=500)

    def invalid(error: ValidationError | RequestValidationError) -> JSONResponse:
        # Field names and messages only. Echoing the rejected values back would repeat
        # what the user sent, and some values, like infinity, can't be written as JSON.
        fields = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in error.errors()]
        return JSONResponse({"detail": fields}, status_code=422)

    @app.exception_handler(RequestValidationError)
    def invalid_request(request: Request, error: RequestValidationError) -> Response:
        return error_page(request, 422) if is_page(request) else invalid(error)

    @app.exception_handler(ValidationError)
    def invalid_data(request: Request, error: ValidationError) -> JSONResponse:
        return invalid(error)  # From the data models, for example an age under 18.

    @app.exception_handler(ActionError)
    def refused(request: Request, error: ActionError) -> Response:
        if is_page(request):
            return error_page(request, error.status, error.message if error.status < 500 else None)
        return JSONResponse({"detail": error.detail}, status_code=error.status)

    for module in (routes_auth, routes_me, routes_feed, routes_social, routes_admin, ui, ui_admin):
        app.include_router(module.router)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"), name="static")
    app.add_middleware(PageRouting)  # Added last, so it runs first and routes pages before anything else.
    return app
