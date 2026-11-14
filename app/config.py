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
    # "production" turns on the startup safety checks in production_problems().
    app_env: str = "development"
    # Copies of the database made by the running app. 0 turns them off.
    backup_every_hours: int = 0
    backup_keep: int = 14
    allowed_email_domains: tuple[str, ...] = ()  # Empty means any domain.
    admin_emails: tuple[str, ...] = ()
    main_db: Path = LIVE_DIR / "main.db"
    persona_db: Path = LIVE_DIR / "personas.db"
    consent_dir: Path = ROOT / "consent"
    consent_version: str = "2"
    match_threshold: float = 7.0
    # Times admins enter, like when rating opens, are read and shown in this time zone.
    study_timezone: str = "America/New_York"
    cookie_secure: bool = True
    session_days: int = 14
    enabled_matchers: tuple[str, ...] = ("random",)
    default_matcher: str = "random"  # Serves users who opt out of the experiment.
    matcher_seed: int | None = None
    # Analytics. POSITIVE_THRESHOLD is on the 1 to 10 scale, PREDICTION_THRESHOLD on 0 to 1.
    positive_threshold: float = 7.0
    prediction_threshold: float = 0.5
    min_group_size: int = 10
    rapid_rating_ms: int = 800
    admin_step_up_minutes: int = 60
    export_salt: str = field(default="", repr=False)
    source_repo_url: str = ""
    # Requests per window. They live in memory, so they assume one server process.
    login_requests_per_ip: int = 30  # Failed sign-ins and sign-ups per address, per window.
    login_window_seconds: int = 900

    @classmethod
    def from_env(cls) -> "Settings":
        seed = os.environ.get("MATCHER_SEED")
        return cls(
            site_url=os.environ.get("SITE_URL", cls.site_url).rstrip("/"),
            allowed_email_domains=_list("ALLOWED_EMAIL_DOMAINS", ""),
            admin_emails=_list("ADMIN_EMAILS", ""),
            main_db=Path(os.environ.get("APP_MAIN_DB", cls.main_db)),
            persona_db=Path(os.environ.get("APP_PERSONA_DB", cls.persona_db)),
            consent_dir=Path(os.environ.get("CONSENT_DIR", cls.consent_dir)),
            consent_version=os.environ.get("CONSENT_VERSION", cls.consent_version),
            match_threshold=float(os.environ.get("MATCH_THRESHOLD", cls.match_threshold)),
            study_timezone=os.environ.get("STUDY_TIMEZONE", cls.study_timezone),
            cookie_secure=_bool("COOKIE_SECURE", True),
            login_requests_per_ip=int(os.environ.get("LOGIN_REQUESTS_PER_IP", cls.login_requests_per_ip)),
            session_days=int(os.environ.get("SESSION_DAYS", cls.session_days)),
            enabled_matchers=_list("ENABLED_MATCHERS", "random"),
            default_matcher=os.environ.get("DEFAULT_MATCHER", cls.default_matcher),
            matcher_seed=int(seed) if seed else None,
            positive_threshold=float(os.environ.get("POSITIVE_THRESHOLD", cls.positive_threshold)),
            prediction_threshold=float(os.environ.get("PREDICTION_THRESHOLD", cls.prediction_threshold)),
            min_group_size=int(os.environ.get("MIN_GROUP_SIZE", cls.min_group_size)),
            rapid_rating_ms=int(os.environ.get("RAPID_RATING_MS", cls.rapid_rating_ms)),
            admin_step_up_minutes=int(os.environ.get("ADMIN_STEP_UP_MINUTES", cls.admin_step_up_minutes)),
            export_salt=os.environ.get("EXPORT_SALT", ""),
            source_repo_url=os.environ.get("SOURCE_REPO_URL", ""),
            app_env=os.environ.get("APP_ENV", cls.app_env).strip().lower(),
            backup_every_hours=int(os.environ.get("BACKUP_EVERY_HOURS", cls.backup_every_hours)),
            backup_keep=int(os.environ.get("BACKUP_KEEP", cls.backup_keep)),
        )

    @property
    def production(self) -> bool:
        return self.app_env == "production"

    @property
    def backup_dir(self) -> Path:
        return self.main_db.parent / "backups"

    def production_problems(self) -> list[str]:
        """Settings that would be unsafe, or lose data, on the real site."""
        problems = []
        if not self.site_url.startswith("https://"):
            problems.append("SITE_URL must be the site's https:// address.")
        if not self.cookie_secure:
            problems.append("COOKIE_SECURE must be true.")
        if len(self.export_salt) < 16:
            problems.append("EXPORT_SALT must be set to a long random secret, and never changed.")
        if not self.admin_emails:
            problems.append("ADMIN_EMAILS must list at least one admin.")
        for path in (self.main_db, self.persona_db):
            if path.resolve().is_relative_to(ROOT.resolve()):
                problems.append(f"{path} is inside the code folder, which is wiped on every deploy. Put the databases on the persistent disk.")
        if self.backup_every_hours <= 0:
            problems.append("BACKUP_EVERY_HOURS must be set, for example to 24.")
        return problems
