"""Magic-link sign-in, consent, sign-out, and account deletion (JSON API)."""

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict

from app import actions
from app.deps import current_account, services
from app.security import SESSION_COOKIE
from app.services import Services
from live_store import Account, sha256

router = APIRouter()

# The same reply for every address, so it can't reveal who has an account or a valid email.
GENERIC_REPLY = {"message": "If that address can sign in, a link is on its way."}


class LinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str


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


@router.post("/auth/request-link")
def request_link(body: LinkRequest, request: Request, s: Services = Depends(services)) -> dict:
    actions.request_login_link(s, body.email, request.client.host if request.client else "unknown")
    return GENERIC_REPLY


@router.get("/auth/verify")
def verify(token: str, response: Response, s: Services = Depends(services)) -> dict:
    account, session_id = actions.sign_in(s, token)
    set_session_cookie(response, session_id, s)
    return {
        "status": account.status,
        "role": account.role,
        "consent_required": account.consent_version != s.settings.consent_version,
    }


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
