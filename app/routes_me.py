"""A user's own account, profile, and preferences (JSON API).

These work before approval, so a new user can complete everything an admin reviews.
"""

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app import actions
from app.deps import consented_account, current_account, services
from app.services import Services
from live_store import Account
from models import Gender

router = APIRouter()


class ProfileIn(BaseModel):
    """Everything on a profile except user_id and agent, which only the server sets."""

    model_config = ConfigDict(extra="forbid")
    name: str
    age: int
    height_cm: int
    religion: str | None = None
    school: str
    major: str
    job: str
    interests: list[str]
    clubs: list[str]
    gender: Gender
    looking_for: str


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    age: int | None = None
    height_cm: int | None = None
    religion: str | None = None
    school: str | None = None
    major: str | None = None
    job: str | None = None
    interests: list[str] | None = None
    clubs: list[str] | None = None
    gender: Gender | None = None
    looking_for: str | None = None


class PreferencesIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interested_in: list[Gender] | Literal["everyone"] = Field(min_length=1)


@router.get("/me")
def me(account: Account = Depends(current_account), s: Services = Depends(services)) -> dict:
    """Your own account. The only user endpoint that returns your email and preferences."""
    preferences = s.manager.get_preferences(account.user_id) if account.user_id else None
    return {
        "account_id": account.account_id,
        "email": account.email_normalized,
        "status": account.status,
        "role": account.role,
        "consent_version": account.consent_version,
        "consent_current": account.consent_version == s.settings.consent_version,
        "experiment_opt_out": account.experiment_opt_out,
        "notify_on_match": account.notify_on_match,
        "paused": account.paused,
        "wants_best_match": s.live.wants_best_match(account.account_id),
        "profile": s.public_profile(account.user_id, own=True)["profile"] if account.user_id else None,
        "interested_in": sorted(preferences.interested_in) if preferences else None,
        "missing_for_approval": s.missing_for_approval(account),
    }


@router.post("/me/profile", status_code=201)
def create_profile(body: ProfileIn, account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    return s.public_profile(actions.create_profile(s, account, body.model_dump()), own=True)


@router.get("/me/profile")
def get_profile(account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    return s.public_profile(actions.require_profile(account), own=True)


@router.put("/me/profile")
def edit_profile(body: ProfileUpdate, account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    actions.edit_profile(s, account, body.model_dump(exclude_unset=True))
    return s.public_profile(account.user_id, own=True)


@router.get("/me/preferences")
def get_preferences(account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    preferences = s.manager.get_preferences(actions.require_profile(account))
    return {"interested_in": sorted(preferences.interested_in) if preferences else None}


@router.put("/me/preferences")
def set_preferences(body: PreferencesIn, account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    return {"interested_in": sorted(actions.set_preferences(s, account, body.interested_in))}


class BestMatchChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wants_best_match: bool


@router.put("/me/best-match")
def set_best_match_choice(body: BestMatchChoice, account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    """Ask, or stop asking, to see your best match after the study."""
    actions.set_best_match_choice(s, account, body.wants_best_match)
    return {"wants_best_match": s.live.wants_best_match(account.account_id)}


@router.get("/me/best-match")
def best_match(account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    """Your best match, once results are out and if you both asked to see it."""
    found = actions.best_match_for(s, account)
    return {key: found[key] for key in ("state", "opens_text", "profile", "real_name") if key in found}
