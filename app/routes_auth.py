"""Sign-up and sign-in with a password, consent, sign-out, and account deletion (JSON API)."""

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict

from app import actions
from app.deps import current_account, services
from app.security import SESSION_COOKIE
from app.services import Services
from live_store import Account, sha256

router = APIRouter()

class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str
    password: str


class PasswordChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str | None = None
    new_password: str


class ConsentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str


def set_session_cookie(response: Response, session_id: str, s: Services) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=s.settings.session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=s.settings.cookie_secure,
    )


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _signed_in_reply(s: Services, account: Account) -> dict:
    return {
        "status": account.status,
        "role": account.role,
        "consent_required": account.consent_version != s.settings.consent_version,
        "must_change_password": s.live.must_change_password(account.account_id),
    }


@router.post("/auth/signup", status_code=201)
def signup(body: Credentials, request: Request, response: Response, s: Services = Depends(services)) -> dict:
    account, session_id = actions.sign_up(s, body.email, body.password, _client(request))
    set_session_cookie(response, session_id, s)
    return _signed_in_reply(s, account)


@router.post("/auth/login")
def login(body: Credentials, request: Request, response: Response, s: Services = Depends(services)) -> dict:
    account, session_id = actions.password_sign_in(s, body.email, body.password, _client(request))
    set_session_cookie(response, session_id, s)
    return _signed_in_reply(s, account)


@router.post("/auth/password")
def change_password(body: PasswordChange, request: Request, account: Account = Depends(current_account), s: Services = Depends(services)) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE)
    actions.change_password(s, account, body.current_password, body.new_password, sha256(session_id) if session_id else None)
    return {"changed": True}


@router.get("/auth/consent")
def consent_text(s: Services = Depends(services)) -> dict:
    return {"version": s.settings.consent_version, "text": s.consent_text()}


@router.post("/auth/consent")
def give_consent(body: ConsentRequest, account: Account = Depends(current_account), s: Services = Depends(services)) -> dict:
    actions.give_consent(s, account, body.version)
    return {"consent_version": body.version}


@router.post("/auth/logout")
def logout(request: Request, response: Response, s: Services = Depends(services)) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        s.live.delete_session(sha256(session_id))
    response.delete_cookie(SESSION_COOKIE)
    return {"signed_out": True}


@router.delete("/me")
def delete_me(response: Response, account: Account = Depends(current_account), s: Services = Depends(services)) -> dict:
    """Always allowed once signed in, even without consent or approval."""
    actions.delete_account(s, account)
    response.delete_cookie(SESSION_COOKIE)
    return {"deleted": True}
