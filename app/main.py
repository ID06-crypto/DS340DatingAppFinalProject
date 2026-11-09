"""Build the app. Run it with: uvicorn --factory app.main:create_app --no-access-log"""

import logging
import random
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
from app.config import Settings
from app.emails import EmailSender, email_sender_for
from app.matchers import build_matchers
from app.photos import LocalDiskPhotoStorage, PhotoStorage
from app.ratelimit import RateLimiter
from app.security import install_log_redaction, install_security
from app.services import Services
from live_store import LiveStore
from manager import ProfileManager


def create_app(
    settings: Settings | None = None,
    email_sender: EmailSender | None = None,
    photo_storage: PhotoStorage | None = None,
    rng: random.Random | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.main_db.parent.mkdir(parents=True, exist_ok=True)
    manager = ProfileManager(settings.main_db, settings.persona_db)  # Creates every table.
    rng = rng or random.Random(settings.matcher_seed)
    services = Services(
        settings=settings,
        manager=manager,
        live=LiveStore(settings.main_db),
        email_sender=email_sender or email_sender_for(settings),
        photo_storage=photo_storage or LocalDiskPhotoStorage(settings.photo_dir),
        matchers=build_matchers(settings.enabled_matchers, settings.default_matcher, rng),
        rng=rng,
        login_per_email=RateLimiter(settings.login_requests_per_email, settings.login_window_seconds),
        login_per_ip=RateLimiter(settings.login_requests_per_ip, settings.login_window_seconds),
        messages_per_user=RateLimiter(settings.messages_per_minute, 60),
    )
    services.consent_text()  # Fail at startup, not on first use, if the text is missing.
    services.initialize_controls()

    app = FastAPI(title="Matching study", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.services = services
    install_security(app, settings.site_url, settings.cookie_secure)
    install_log_redaction()

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
