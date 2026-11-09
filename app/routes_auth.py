"""Magic-link sign-in, consent, sign-out, and account deletion."""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict

from app.deps import current_account, services
from app.security import SESSION_COOKIE, new_secret, normalize_email
from app.services import Services
from live_store import Account, sha256, utc_now

router = APIRouter()

# The same reply for every address, so it can't reveal who has an account or a valid email.
GENERIC_REPLY = {"message": "If that address can sign in, a link is on its way."}


class LinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str


class ConsentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str


@router.post("/auth/request-link")
def request_link(body: LinkRequest, request: Request, s: Services = Depends(services)) -> dict:
    client_ip = request.client.host if request.client else "unknown"
    if not s.login_per_ip.allow(client_ip):
        raise HTTPException(429, "Too many requests. Try again later.")
    email = normalize_email(body.email, s.settings.allowed_email_domains)
    # A per-email limit that's exceeded still gets the generic reply, so it reveals nothing.
    if email is None or not s.login_per_email.allow(email):
        return GENERIC_REPLY
    token = new_secret()
    expires = utc_now() + timedelta(minutes=s.settings.login_token_minutes)
    s.live.save_login_token(sha256(token), email, expires)
    link = f"{s.settings.site_url}/auth/verify?token={token}"
    s.email_sender.send(
        email,
        "Your sign-in link",
        f"Sign in with this link. It works once and expires in {s.settings.login_token_minutes} minutes.\n\n{link}\n",
    )
    return GENERIC_REPLY


@router.get("/auth/verify")
def verify(token: str, response: Response, s: Services = Depends(services)) -> dict:
    email = s.live.use_login_token(sha256(token))
    if email is None:
        raise HTTPException(400, "This link is invalid, used, or expired. Request a new one.")
    admins = {normalize_email(e) for e in s.settings.admin_emails}
    account, _ = s.live.get_or_create_account(email, admin=email in admins)
    session_id = new_secret()
    s.live.create_session(sha256(session_id), account.account_id, utc_now() + timedelta(days=s.settings.session_days))
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=s.settings.session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=s.settings.cookie_secure,
    )
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
    if body.version != s.settings.consent_version:
        raise HTTPException(409, f"The current consent version is {s.settings.consent_version}")
    s.live.record_consent(account.account_id, body.version)
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
    s.delete_account(account)
    response.delete_cookie(SESSION_COOKIE)
    return {"deleted": True}
