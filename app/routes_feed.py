"""The feed and ratings (JSON API). Clients never choose who they rate or which matcher served them."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app import actions
from app.deps import approved_user, services
from app.services import Services
from live_store import Account

router = APIRouter()


class RatingIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    impression_id: str
    score: float = Field(ge=actions.MIN_SCORE, le=actions.MAX_SCORE, allow_inf_nan=False)


@router.get("/feed/next")
def next_profile(account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    card = actions.next_impression(s, account)
    if card.impression is None:
        return {"profile": None, "pool_size": card.people_left, "message": card.message}
    return {"impression_id": card.impression.impression_id, **s.public_profile(card.impression.target_id)}


@router.post("/ratings")
def rate(body: RatingIn, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    outcome = actions.rate(s, account, body.impression_id, body.score)
    return {"saved": True, "matched": outcome.matched, "match_id": outcome.match_id}
