"""Email normalization, tokens, CSRF checks, security headers, and log redaction."""

import hashlib
import hmac
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
_DOMAIN = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)+")

# Only this site's own files: no inline scripts or styles, and nothing from other origins.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
    "connect-src 'self'; font-src 'self'; form-action 'self'; frame-ancestors 'none'; "
    "base-uri 'none'; object-src 'none'"
)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def normalize_email(raw: str, allowed_domains: Iterable[str] | None = None) -> str | None:
    """Lowercase, strip, and drop any "+tag". Returns None for a bad or disallowed address.

    Any domain is accepted unless allowed_domains lists some.
    """
    email = raw.strip().lower()
    if email.count("@") != 1:
        return None
    local, domain = email.split("@")
    local = local.split("+", 1)[0]
    if not local or not _LOCAL_PART.fullmatch(local) or not _DOMAIN.fullmatch(domain):
        return None
    allowed = set(allowed_domains or ())
    if allowed and domain not in allowed:
        return None
    return f"{local}@{domain}"


# Admin passwords: scrypt from the standard library, with a random salt per password.
# Stored as "scrypt$n$r$p$salt$hash", so the cost can be raised later without breaking old hashes.

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 200


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    """Whether the password matches. With nothing stored, it still does the work of one
    check, so a wrong email takes as long as a wrong password."""
    try:
        _, n, r, p, salt, expected = (stored or "").split("$")
        n_, r_, p_, salt_bytes, expected_bytes = int(n), int(r), int(p), bytes.fromhex(salt), bytes.fromhex(expected)
    except ValueError:
        n_, r_, p_, salt_bytes, expected_bytes = SCRYPT_N, SCRYPT_R, SCRYPT_P, b"\0" * 16, b""
    digest = hashlib.scrypt(password.encode()[:1024], salt=salt_bytes, n=n_, r=r_, p=p_, dklen=32)
    return bool(expected_bytes) and hmac.compare_digest(digest, expected_bytes)


def temporary_password() -> str:
    """A temporary password an admin can read out or paste, like "kq7m-x2pd-9wfa"."""
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"  # No 0/o, 1/l/i, which are easy to misread.
    return "-".join("".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(3))


def password_problem(password: str) -> str | None:
    """Why a new password isn't allowed, or None."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Use at least {MIN_PASSWORD_LENGTH} characters."
    if len(password) > MAX_PASSWORD_LENGTH:
        return f"Use at most {MAX_PASSWORD_LENGTH} characters."
    return None


def new_secret() -> str:
    """A random login token or session id. Only its hash is ever stored."""
    return secrets.token_urlsafe(32)


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def same_origin(request: Request, site_url: str, page: bool = False) -> bool:
    """CSRF check: a state-changing request must come from the configured site.

    Browsers send Origin on cross-site POSTs, and Referer as a fallback. For the JSON API,
    a request with neither is refused, so non-browser clients must send Origin themselves.

    Pages use Referrer-Policy: no-referrer, so browsers send "Origin: null" on ordinary form
    posts. A page request with a null or missing Origin is let through here, and every page
    that changes anything then requires the session's CSRF token. A foreign Origin is
    always refused.
    """
    expected = origin_of(site_url)
    origin = request.headers.get("origin")
    if page and origin in (None, "null"):
        return True
    if origin is not None:
        return origin == expected
    referer = request.headers.get("referer")
    return referer is not None and origin_of(referer) == expected


def install_security(app, site_url: str, secure: bool) -> None:
    @app.middleware("http")
    async def guard(request: Request, call_next):
        page = request.url.path.startswith("/ui/")
        if request.method in UNSAFE_METHODS and not same_origin(request, site_url, page):
            response = JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
        else:
            response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=3600")
        else:
            response.headers.setdefault("Cache-Control", "no-store")  # Pages hold personal data.
        if secure:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response
