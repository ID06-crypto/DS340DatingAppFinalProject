"""Matches, chat, blocks, and reports."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.deps import approved_user, services
from app.services import Services
from live_store import Account, Match, ReportReason

router = APIRouter()


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    body: str = Field(min_length=1, max_length=2000)


class BlockIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str


class ReportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reported_id: str
    match_id: str | None = None
    reason: ReportReason
    details: str = Field(default="", max_length=1000)


def _own_match(match_id: str, user_id: str, s: Services) -> Match:
    match = s.live.get_match(match_id)
    if match is None or user_id not in (match.user_a, match.user_b):
        raise HTTPException(404, "Match not found")
    return match


def _open_chat(match_id: str, user_id: str, s: Services) -> Match:
    """A match its participants can still message in: active, and no block either way."""
    match = _own_match(match_id, user_id, s)
    if match.status != "active" or s.live.blocked_either_way(match.user_a, match.user_b):
        raise HTTPException(403, "This match has ended")
    return match


@router.get("/matches")
def list_matches(account: Account = Depends(approved_user), s: Services = Depends(services)) -> list[dict]:
    me = account.user_id
    return [
        {"match_id": m.match_id, "created_at": m.created_at, **s.public_profile(m.other(me))}
        for m in s.live.active_matches_for(me)
    ]


@router.post("/matches/{match_id}/unmatch")
def unmatch(match_id: str, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    match = _own_match(match_id, account.user_id, s)
    if match.status != "active":
        raise HTTPException(409, "This match has already ended")
    s.live.unmatch(match_id, account.user_id)
    return {"unmatched": True}


@router.get("/matches/{match_id}/messages")
def read_messages(match_id: str, after: int = 0, account: Account = Depends(approved_user), s: Services = Depends(services)) -> list[dict]:
    _open_chat(match_id, account.user_id, s)
    return [m.model_dump() for m in s.live.messages_after(match_id, after)]


@router.post("/matches/{match_id}/messages", status_code=201)
def send_message(match_id: str, body: MessageIn, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    _open_chat(match_id, account.user_id, s)
    if not body.body.strip():
        raise HTTPException(422, "Messages can't be blank")
    if not s.messages_per_user.allow(account.user_id):
        raise HTTPException(429, "You're sending messages too quickly")
    return s.live.add_message(match_id, account.user_id, body.body).model_dump()


@router.post("/blocks", status_code=201)
def block(body: BlockIn, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    if body.user_id == account.user_id:
        raise HTTPException(400, "You can't block yourself")
    if not s.manager.store.profile_exists(body.user_id):
        raise HTTPException(404, "User not found")
    s.live.add_block(account.user_id, body.user_id)  # Also ends any active match.
    return {"blocked": body.user_id}


@router.get("/blocks")
def list_blocks(account: Account = Depends(approved_user), s: Services = Depends(services)) -> list[dict]:
    return s.live.blocks_by(account.user_id)


@router.delete("/blocks/{user_id}")
def unblock(user_id: str, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    if not s.live.remove_block(account.user_id, user_id):
        raise HTTPException(404, "You haven't blocked that user")
    return {"unblocked": user_id}


@router.post("/reports", status_code=201)
def report(body: ReportIn, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    me = account.user_id
    if body.reported_id == me:
        raise HTTPException(400, "You can't report yourself")
    if not s.manager.store.profile_exists(body.reported_id):
        raise HTTPException(404, "User not found")
    if body.match_id is not None:
        match = s.live.get_match(body.match_id)
        if match is None or {match.user_a, match.user_b} != {me, body.reported_id}:
            raise HTTPException(400, "That match isn't between you and the reported user")
    created = s.live.add_report(me, body.reported_id, body.match_id, body.reason, body.details)
    return {"report_id": created.report_id}
