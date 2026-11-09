"""Matches, chat, blocks, and reports (JSON API)."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app import actions
from app.deps import approved_user, services
from app.services import Services
from live_store import Account, ReportReason

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


@router.get("/matches")
def list_matches(account: Account = Depends(approved_user), s: Services = Depends(services)) -> list[dict]:
    me = account.user_id
    return [
        {"match_id": m.match_id, "created_at": m.created_at, **s.public_profile(m.other(me))}
        for m in s.live.active_matches_for(me)
    ]


@router.post("/matches/{match_id}/unmatch")
def unmatch(match_id: str, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    actions.unmatch(s, account, match_id)
    return {"unmatched": True}


@router.get("/matches/{match_id}/messages")
def read_messages(match_id: str, after: int = 0, account: Account = Depends(approved_user), s: Services = Depends(services)) -> list[dict]:
    return [m.model_dump() for m in actions.read_messages(s, account, match_id, after)]


@router.post("/matches/{match_id}/messages", status_code=201)
def send_message(match_id: str, body: MessageIn, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    return actions.send_message(s, account, match_id, body.body).model_dump()


@router.post("/blocks", status_code=201)
def block(body: BlockIn, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    actions.block(s, account, body.user_id)
    return {"blocked": body.user_id}


@router.get("/blocks")
def list_blocks(account: Account = Depends(approved_user), s: Services = Depends(services)) -> list[dict]:
    return s.live.blocks_by(account.user_id)


@router.delete("/blocks/{user_id}")
def unblock(user_id: str, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    actions.unblock(s, account, user_id)
    return {"unblocked": user_id}


@router.post("/reports", status_code=201)
def report(body: ReportIn, account: Account = Depends(approved_user), s: Services = Depends(services)) -> dict:
    created = actions.report(s, account, body.reported_id, body.reason, body.details, body.match_id)
    return {"report_id": created.report_id}
