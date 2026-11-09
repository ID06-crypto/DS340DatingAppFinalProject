"""Email normalization, tokens, CSRF checks, security headers, and log redaction."""

import logging
import re
import secrets
from collections.abc import Iterable
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse

SESSION_COOKIE = "session"
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_LOCAL_PART = re.compile(r"[a-z0-9._%-]+")
_TOKEN_IN_TEXT = re.compile(r"(token=)[^&\s\"']+")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def normalize_email(raw: str, allowed_domains: Iterable[str] | None = None) -> str | None:
    """Lowercase, strip, and drop any "+tag". Returns None for a bad or disallowed address."""
    email = raw.strip().lower()
    if email.count("@") != 1:
        return None
    local, domain = email.split("@")
    local = local.split("+", 1)[0]
    if not local or not _LOCAL_PART.fullmatch(local) or not domain:
        return None
    if allowed_domains is not None and domain not in set(allowed_domains):
        return None
    return f"{local}@{domain}"


def new_secret() -> str:
    """A random login token or session id. Only its hash is ever stored."""
    return secrets.token_urlsafe(32)


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def same_origin(request: Request, site_url: str) -> bool:
    """CSRF check: a state-changing request must come from the configured site.

    Browsers send Origin on cross-site POSTs, and Referer as a fallback. A request with
    neither is refused, so non-browser clients must send Origin themselves.
    """
    expected = origin_of(site_url)
    origin = request.headers.get("origin")
    if origin is not None:
        return origin == expected
    referer = request.headers.get("referer")
    return referer is not None and origin_of(referer) == expected


def install_security(app, site_url: str, secure: bool) -> None:
    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method in UNSAFE_METHODS and not same_origin(request, site_url):
            response = JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
        else:
            response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        response.headers.setdefault("Cache-Control", "no-store")
        if secure:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response


class RedactTokens(logging.Filter):
    """Removes login tokens from access-log lines, since /auth/verify carries one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                _TOKEN_IN_TEXT.sub(r"\1[redacted]", arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        record.msg = _TOKEN_IN_TEXT.sub(r"\1[redacted]", str(record.msg))
        return True


def install_log_redaction() -> None:
    for name in ("uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactTokens) for f in logger.filters):
            logger.addFilter(RedactTokens())
