"""The feed and ratings. Clients never choose who they rate or which matcher served them."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.deps import approved_user, services
from app.services import Services
from live_store import Account
from matching import PublicView
from models import LIVE_RUN_ID

router = APIRouter()
logger = logging.getLogger(__name__)

MIN_SCORE = 1.0
MAX_SCORE = 10.0


def to_unit_scale(score: float) -> float:
    """Users rate from 1 to 10. Ratings are stored from 0 to 1, like simulated ones."""
    return (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)


class RatingIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    impression_id: str
    score: float = Field(ge=MIN_SCORE, le=MAX_SCORE, allow_inf_nan=False)


@router.get("/feed/next")
def next_profile(account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    viewer = account.user_id
    pool = s.manager.eligible_candidates(viewer, LIVE_RUN_ID)
    if not pool:
        return {"profile": None, "pool_size": 0, "message": "No one left to show right now."}
    with s.matcher_lock:
        name = s.rng.choice(sorted(s.matchers))
        target = s.matchers[name].next_profile(viewer, PublicView(s.manager.store, LIVE_RUN_ID))
    if target is None:
        return {"profile": None, "pool_size": len(pool), "message": "No one left to show right now."}
    if target not in pool:
        logger.error("Matcher %s chose a profile outside the eligible pool", name)
        raise HTTPException(500, "Couldn't choose a profile. Try again.")
    impression = s.live.create_impression(viewer, target, name, len(pool))
    return {"impression_id": impression.impression_id, **s.public_profile(target)}


@router.post("/ratings")
def rate(body: RatingIn, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    """Save a rating, and create a match if both people rated each other highly enough.

    The reply has the same shape either way, so it reveals nothing about whether the other
    person has rated you unless a match was made.
    """
    viewer = account.user_id
    claim = s.live.claim_impression(body.impression_id, viewer)
    if claim == "missing":
        raise HTTPException(404, "Impression not found")
    if claim == "rated":
        raise HTTPException(409, "You've already rated this profile")
    impression = s.live.get_impression(body.impression_id)
    score = to_unit_scale(body.score)
    try:
        rating = s.manager.add_rating(viewer, impression.target_id, score, LIVE_RUN_ID, matcher=impression.matcher)
    except Exception:
        s.live.unclaim_impression(body.impression_id)
        raise
    if impression.matcher in s.matchers:
        with s.matcher_lock:
            s.matchers[impression.matcher].observe(rating)

    match = None
    threshold = to_unit_scale(s.settings.match_threshold) - 1e-9
    reverse = [r for r in s.manager.ratings_by_rater(impression.target_id, LIVE_RUN_ID) if r.target_id == viewer]
    if (
        reverse
        and score >= threshold
        and reverse[0].score >= threshold
        and not s.live.blocked_either_way(viewer, impression.target_id)
    ):
        match = s.live.create_match(viewer, impression.target_id)
    return {"saved": True, "matched": match is not None, "match_id": match.match_id if match else None}
