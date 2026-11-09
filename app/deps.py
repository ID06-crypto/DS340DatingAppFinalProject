"""Who may call what. Each dependency adds one requirement to the one before it."""

from fastapi import Depends, HTTPException, Request

from app.security import SESSION_COOKIE
from app.services import Services
from live_store import Account, sha256


def services(request: Request) -> Services:
    return request.app.state.services


def current_account(request: Request, s: Services = Depends(services)) -> Account:
    """Any signed-in account."""
    session_id = request.cookies.get(SESSION_COOKIE)
    account = s.live.account_for_session(sha256(session_id)) if session_id else None
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


def admin_account(account: Account = Depends(approved_account)) -> Account:
    if account.role != "admin":
        raise HTTPException(403, "Admins only")
    return account
