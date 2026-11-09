"""Shared plumbing for the web pages: routing, CSRF, page access, and rendering.

Browsers and the JSON API share URLs such as /me and /matches. A request that asks for
HTML (a browser navigation) or comes from HTMX is routed to the page version, which lives
under /ui internally. Everything else reaches the JSON API, unchanged.
"""

import hashlib
import hmac
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.deps import fresh_sign_in, services, signed_in
from app.services import Services, first_name
from live_store import Account

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
UI_PREFIX = "/ui"


def wants_page(scope: dict) -> bool:
    headers = dict(scope.get("headers") or [])
    accept = headers.get(b"accept", b"").decode("latin-1")
    return b"hx-request" in headers or "text/html" in accept


class PageRouting:
    """ASGI middleware that sends browser and HTMX requests to the page routes."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and not scope["path"].startswith(("/static/", UI_PREFIX + "/")) and wants_page(scope):
            scope = dict(scope)
            scope["path"] = UI_PREFIX + scope["path"]
            scope["raw_path"] = scope["path"].encode()
        await self.app(scope, receive, send)


def is_page(request: Request) -> bool:
    return request.url.path.startswith(UI_PREFIX + "/") or request.url.path == UI_PREFIX


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


class Redirect(Exception):
    """Raised by page dependencies to send the browser elsewhere."""

    def __init__(self, url: str) -> None:
        self.url = url


def redirect(request: Request, url: str) -> Response:
    """A 303 redirect, or for HTMX, an instruction to load the page."""
    if is_htmx(request):
        return Response(status_code=204, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=303)


# CSRF


def csrf_token(session_id: str | None) -> str | None:
    """A per-session token, derived from the secret session id, so nothing extra is stored."""
    if not session_id:
        return None
    return hmac.new(session_id.encode(), b"csrf", hashlib.sha256).hexdigest()


async def require_csrf(request: Request) -> None:
    """Every state-changing page request must carry the session's CSRF token.

    Forms send it as csrf_token, and HTMX sends it as the X-CSRF-Token header.
    """
    expected = csrf_token(getattr(request.state, "session_id", None) or request.cookies.get("session"))
    sent = request.headers.get("x-csrf-token")
    if sent is None:
        form = await request.form()
        sent = form.get("csrf_token")
    if not expected or not isinstance(sent, str) or not hmac.compare_digest(sent, expected):
        raise HTTPException(403, "This form expired. Go back, reload the page, and try again.")


# Page access


def next_onboarding_step(s: Services, account: Account) -> str | None:
    """Where an account must go before it can use the app, or None when it's ready."""
    if account.consent_version != s.settings.consent_version:
        return "/onboarding/consent"
    if account.user_id is None:
        return "/onboarding/profile"
    if s.manager.get_preferences(account.user_id) is None:
        return "/onboarding/preferences"
    if s.settings.require_photo and not s.live.photos_for(account.user_id):
        return "/onboarding/photos"
    if account.status != "approved":
        return "/pending"
    return None


def page_account(request: Request, s: Services = Depends(services)) -> Account:
    """Signed in, or sent to the landing page to sign in."""
    account = signed_in(request, s)
    if account is None:
        raise Redirect("/")
    request.state.account = account
    return account


def page_consented(account: Account = Depends(page_account), s: Services = Depends(services)) -> Account:
    if account.consent_version != s.settings.consent_version:
        raise Redirect("/onboarding/consent")
    return account


def page_member(account: Account = Depends(page_consented), s: Services = Depends(services)) -> Account:
    """Fully onboarded and approved. Anyone else is sent to their next onboarding step."""
    step = next_onboarding_step(s, account)
    if step is not None:
        raise Redirect(step)
    return account


def page_admin_any(account: Account = Depends(page_consented)) -> Account:
    """An approved admin, whether or not they signed in recently."""
    if account.role != "admin" or account.status != "approved":
        raise HTTPException(403, "Admins only")
    return account


def page_admin(request: Request, account: Account = Depends(page_admin_any), s: Services = Depends(services)) -> Account:
    """An admin who signed in recently. Otherwise they must sign in again (step-up)."""
    if not fresh_sign_in(request, s):
        raise Redirect("/admin/step-up")
    return account


# Rendering


def height_text(cm: int) -> str:
    inches = round(cm / 2.54)
    return f"{cm} cm ({inches // 12} ft {inches % 12} in)"


TEMPLATES.env.globals.update(first_name=first_name, height_text=height_text)


def render(request: Request, template: str, context: dict[str, Any] | None = None, status: int = 200) -> HTMLResponse:
    account: Account | None = getattr(request.state, "account", None)
    session_id = getattr(request.state, "session_id", None)
    base = {
        "account": account,
        "csrf": csrf_token(session_id),
        "is_member": False,
        "is_admin": bool(account and account.role == "admin" and account.status == "approved"),
    }
    s: Services = request.app.state.services
    if account is not None:
        base["is_member"] = next_onboarding_step(s, account) is None
    return TEMPLATES.TemplateResponse(request, template, {**base, **(context or {})}, status_code=status)
