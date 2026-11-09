"""Who may call what. Each dependency adds one requirement to the one before it."""

from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, Request

from app.security import SESSION_COOKIE
from app.services import Services
from live_store import Account, sha256, utc_now


def services(request: Request) -> Services:
    return request.app.state.services


def signed_in(request: Request, s: Services) -> Account | None:
    """The signed-in account, or None. Also notes when the session started, for step-up."""
    session_id = request.cookies.get(SESSION_COOKIE)
    found = s.live.session(sha256(session_id)) if session_id else None
    if found is None:
        return None
    account, started = found
    request.state.session_started = started
    request.state.session_id = session_id
    return account


def current_account(request: Request, s: Services = Depends(services)) -> Account:
    """Any signed-in account."""
    account = signed_in(request, s)
    if account is None:
        raise HTTPException(401, "Sign in first")
    return account


def consented_account(account: Account = Depends(current_account), s: Services = Depends(services)) -> Account:
    """Signed in, and has agreed to the current consent text."""
    if account.consent_version != s.settings.consent_version:
        raise HTTPException(403, "Agree to the current consent text first")
    return account


def approved_account(account: Account = Depends(consented_account)) -> Account:
    """Consented and approved by an admin."""
    if account.status != "approved":
        raise HTTPException(403, "Your account isn't approved")
    return account


def approved_user(account: Account = Depends(approved_account)) -> Account:
    """Approved, with a profile, which feed, ratings, matches, and safety tools need."""
    if account.user_id is None:
        raise HTTPException(403, "Create your profile first")
    return account


def fresh_sign_in(request: Request, s: Services) -> bool:
    """Whether this session started with a sign-in link recently enough for admin work."""
    started: datetime | None = getattr(request.state, "session_started", None)
    return started is not None and utc_now() - started <= timedelta(minutes=s.settings.admin_step_up_minutes)


def admin_account(request: Request, account: Account = Depends(approved_account), s: Services = Depends(services)) -> Account:
    """An admin who signed in within ADMIN_STEP_UP_MINUTES (a "step-up" sign-in)."""
    if account.role != "admin":
        raise HTTPException(403, "Admins only")
    if not fresh_sign_in(request, s):
        raise HTTPException(403, f"Admins must have signed in within the last {s.settings.admin_step_up_minutes} minutes. Sign in again.")
    return account
