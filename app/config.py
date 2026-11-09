"""Settings, read from environment variables. Each one is documented in the README."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from paths import DATA_DIR, ROOT

LIVE_DIR = DATA_DIR / "live"


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in ("1", "true", "yes"):
        return True
    if value.strip().lower() in ("0", "false", "no"):
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _list(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.environ.get(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    site_url: str = "http://localhost:8000"
    allowed_email_domains: tuple[str, ...] = ("bu.edu",)
    admin_emails: tuple[str, ...] = ()
    main_db: Path = LIVE_DIR / "main.db"
    persona_db: Path = LIVE_DIR / "personas.db"
    photo_dir: Path = LIVE_DIR / "photos"
    consent_dir: Path = ROOT / "consent"
    consent_version: str = "1"
    require_photo: bool = True
    match_threshold: float = 7.0
    cookie_secure: bool = True
    session_days: int = 14
    login_token_minutes: int = 15
    enabled_matchers: tuple[str, ...] = ("random",)
    matcher_seed: int | None = None
    email_sender: str = "console"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = field(default="", repr=False)
    smtp_from: str = ""
    smtp_starttls: bool = True
    # Requests per window. They live in memory, so they assume one server process.
    login_requests_per_email: int = 3
    login_requests_per_ip: int = 20
    login_window_seconds: int = 900
    messages_per_minute: int = 30

    @classmethod
    def from_env(cls) -> "Settings":
        seed = os.environ.get("MATCHER_SEED")
        return cls(
            site_url=os.environ.get("SITE_URL", cls.site_url).rstrip("/"),
            allowed_email_domains=_list("ALLOWED_EMAIL_DOMAINS", "bu.edu"),
            admin_emails=_list("ADMIN_EMAILS", ""),
            main_db=Path(os.environ.get("APP_MAIN_DB", cls.main_db)),
            persona_db=Path(os.environ.get("APP_PERSONA_DB", cls.persona_db)),
            photo_dir=Path(os.environ.get("PHOTO_DIR", cls.photo_dir)),
            consent_dir=Path(os.environ.get("CONSENT_DIR", cls.consent_dir)),
            consent_version=os.environ.get("CONSENT_VERSION", cls.consent_version),
            require_photo=_bool("REQUIRE_PHOTO", True),
            match_threshold=float(os.environ.get("MATCH_THRESHOLD", cls.match_threshold)),
            cookie_secure=_bool("COOKIE_SECURE", True),
            session_days=int(os.environ.get("SESSION_DAYS", cls.session_days)),
            login_token_minutes=int(os.environ.get("LOGIN_TOKEN_MINUTES", cls.login_token_minutes)),
            enabled_matchers=_list("ENABLED_MATCHERS", "random"),
            matcher_seed=int(seed) if seed else None,
            email_sender=os.environ.get("EMAIL_SENDER", cls.email_sender),
            smtp_host=os.environ.get("SMTP_HOST", ""),
            smtp_port=int(os.environ.get("SMTP_PORT", cls.smtp_port)),
            smtp_username=os.environ.get("SMTP_USERNAME", ""),
            smtp_password=os.environ.get("SMTP_PASSWORD", ""),
            smtp_from=os.environ.get("SMTP_FROM", ""),
            smtp_starttls=_bool("SMTP_STARTTLS", True),
        )
