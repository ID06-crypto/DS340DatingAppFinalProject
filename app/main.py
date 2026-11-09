"""Build the app. Run it with: uvicorn --factory app.main:create_app --no-access-log"""

import random

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app import routes_admin, routes_auth, routes_feed, routes_me, routes_social
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
        matchers=build_matchers(settings.enabled_matchers, rng),
        rng=rng,
        login_per_email=RateLimiter(settings.login_requests_per_email, settings.login_window_seconds),
        login_per_ip=RateLimiter(settings.login_requests_per_ip, settings.login_window_seconds),
        messages_per_user=RateLimiter(settings.messages_per_minute, 60),
    )
    services.consent_text()  # Fail at startup, not on first use, if the text is missing.

    app = FastAPI(title="Matching study", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.services = services
    install_security(app, settings.site_url, settings.cookie_secure)
    install_log_redaction()

    def invalid(error: ValidationError | RequestValidationError) -> JSONResponse:
        # Field names and messages only. Echoing the rejected values back would repeat
        # what the user sent, and some values, like infinity, can't be written as JSON.
        fields = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in error.errors()]
        return JSONResponse({"detail": fields}, status_code=422)

    @app.exception_handler(RequestValidationError)
    def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        return invalid(error)

    @app.exception_handler(ValidationError)
    def invalid_data(request: Request, error: ValidationError) -> JSONResponse:
        return invalid(error)  # From the data models, for example an age under 18.

    for module in (routes_auth, routes_me, routes_feed, routes_social, routes_admin):
        app.include_router(module.router)
    return app
