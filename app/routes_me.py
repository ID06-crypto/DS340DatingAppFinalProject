"""A user's own account, profile, preferences, and photos, plus serving photos to others.

These work before approval, so a new user can complete everything an admin reviews.
"""

from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.deps import consented_account, current_account, services
from app.photos import MAX_PHOTO_BYTES, MAX_PHOTOS, InvalidPhotoError, clean_photo, new_storage_key
from app.services import Services
from live_store import Account
from models import Gender, GENDERS, LIVE_RUN_ID

router = APIRouter()


class ProfileIn(BaseModel):
    """Everything on a profile except user_id and agent, which only the server sets."""

    model_config = ConfigDict(extra="forbid")
    name: str
    age: int
    height_cm: int
    religion: str
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


class PhotoOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    photo_ids: list[str]


def _require_profile(account: Account) -> str:
    if account.user_id is None:
        raise HTTPException(409, "Create your profile first")
    return account.user_id


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
        "profile": s.public_profile(account.user_id)["profile"] if account.user_id else None,
        "interested_in": sorted(preferences.interested_in) if preferences else None,
        "photo_ids": [p.photo_id for p in s.live.photos_for(account.user_id)] if account.user_id else [],
        "missing_for_approval": s.missing_for_approval(account),
    }


@router.post("/me/profile", status_code=201)
def create_profile(body: ProfileIn, account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    if account.user_id is not None:
        raise HTTPException(409, "You already have a profile. Use PUT to edit it.")
    profile = s.manager.add_profile({**body.model_dump(), "agent": False})  # user_id is generated.
    s.live.link_profile(account.account_id, profile.user_id)
    return s.public_profile(profile.user_id)


@router.get("/me/profile")
def get_profile(account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    return s.public_profile(_require_profile(account))


@router.put("/me/profile")
def edit_profile(body: ProfileUpdate, account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    user_id = _require_profile(account)
    s.manager.edit_profile(user_id, **body.model_dump(exclude_unset=True))
    return s.public_profile(user_id)


@router.get("/me/preferences")
def get_preferences(account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    preferences = s.manager.get_preferences(_require_profile(account))
    return {"interested_in": sorted(preferences.interested_in) if preferences else None}


@router.put("/me/preferences")
def set_preferences(body: PreferencesIn, account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    genders = GENDERS if body.interested_in == "everyone" else body.interested_in
    preferences = s.manager.set_preferences(_require_profile(account), genders)
    return {"interested_in": sorted(preferences.interested_in)}


@router.post("/me/photos", status_code=201)
def upload_photo(file: UploadFile = File(...), account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    user_id = _require_profile(account)
    data = file.file.read(MAX_PHOTO_BYTES + 1)
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(413, "Photos can be at most 5 MB")
    if len(s.live.photos_for(user_id)) >= MAX_PHOTOS:
        raise HTTPException(409, f"You can have at most {MAX_PHOTOS} photos")
    try:
        cleaned = clean_photo(data)
    except InvalidPhotoError as error:
        raise HTTPException(415, str(error)) from None
    key = new_storage_key()
    s.photo_storage.save(key, cleaned)
    photo = s.live.add_photo(user_id, key, MAX_PHOTOS)
    if photo is None:  # Another upload reached the limit first.
        s.photo_storage.delete(key)
        raise HTTPException(409, f"You can have at most {MAX_PHOTOS} photos")
    return {"photo_id": photo.photo_id, "position": photo.position}


@router.delete("/me/photos/{photo_id}")
def delete_photo(photo_id: str, account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    photo = s.live.get_photo(photo_id)
    if photo is None or photo.user_id != account.user_id:
        raise HTTPException(404, "Photo not found")
    s.live.delete_photo(photo_id)
    s.photo_storage.delete(photo.storage_key)
    return {"photo_ids": [p.photo_id for p in s.live.photos_for(photo.user_id)]}


@router.put("/me/photos")
def reorder_photos(body: PhotoOrder, account: Account = Depends(consented_account), s: Services = Depends(services)) -> dict:
    user_id = _require_profile(account)
    if not s.live.reorder_photos(user_id, body.photo_ids):
        raise HTTPException(400, "List each of your photos exactly once")
    return {"photo_ids": body.photo_ids}


@router.get("/photos/{photo_id}")
def get_photo(photo_id: str, account: Account = Depends(consented_account), s: Services = Depends(services)) -> Response:
    """Serve a photo only to its owner, an admin, or an approved user who may see its owner.

    That means the owner is in their eligible pool, has been shown to them, or is an active
    match, with no block either way. Anyone else gets 404, so a photo's existence isn't revealed.
    """
    photo = s.live.get_photo(photo_id)
    if photo is None or not _may_see(account, photo.user_id, s):
        raise HTTPException(404, "Photo not found")
    return Response(s.photo_storage.load(photo.storage_key), media_type="image/jpeg", headers={"Cache-Control": "private"})


def _may_see(account: Account, owner: str, s: Services) -> bool:
    if account.user_id == owner:
        return True
    if account.status != "approved":
        return False
    if account.role == "admin":
        return True
    viewer = account.user_id
    if viewer is None or s.live.blocked_either_way(viewer, owner):
        return False
    return (
        owner in s.manager.eligible_candidates(viewer, LIVE_RUN_ID)
        or s.live.was_shown(viewer, owner)
        or s.live.active_match_between(viewer, owner) is not None
    )
