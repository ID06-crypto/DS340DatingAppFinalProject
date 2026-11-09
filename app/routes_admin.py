"""The approval queue and report handling (JSON API). Every action writes to audit_log."""

from typing import Literal

from fastapi import APIRouter, Depends

from app import actions
from app.deps import admin_account, services
from app.services import Services
from live_store import Account

router = APIRouter(prefix="/admin")


@router.get("/accounts")
def list_accounts(status: str | None = "pending", admin: Account = Depends(admin_account), s: Services = Depends(services)) -> list[dict]:
    """Accounts to review, with their email, profile, and photo ids. Photos load via /photos."""
    s.live.audit(admin.account_id, "view_accounts", None, {"status": status})
    result = []
    for account in s.live.list_accounts(status):
        profile = s.public_profile(account.user_id, own=True) if account.user_id else {"profile": None, "photo_ids": []}
        result.append({
            "account_id": account.account_id,
            "email": account.email_normalized,
            "status": account.status,
            "role": account.role,
            "created_at": account.created_at,
            "missing_for_approval": s.missing_for_approval(account),
            **profile,
        })
    return result


@router.post("/accounts/{account_id}/{action}")
def set_status(
    account_id: str,
    action: Literal["approve", "reject", "suspend"],
    admin: Account = Depends(admin_account),
    s: Services = Depends(services),
) -> dict:
    {"approve": actions.approve, "reject": actions.reject, "suspend": actions.suspend}[action](s, admin, account_id)
    return {"account_id": account_id, "status": {"approve": "approved", "reject": "rejected", "suspend": "suspended"}[action]}


@router.get("/reports")
def list_reports(status: str | None = "open", admin: Account = Depends(admin_account), s: Services = Depends(services)) -> list[dict]:
    """Reports, without any message contents. Those are only readable per report, below."""
    s.live.audit(admin.account_id, "list_reports", None, {"status": status})
    return [report.model_dump() for report in s.live.list_reports(status)]


@router.post("/reports/{report_id}/resolve")
def resolve(report_id: str, admin: Account = Depends(admin_account), s: Services = Depends(services)) -> dict:
    actions.resolve_report(s, admin, report_id)
    return {"report_id": report_id, "status": "resolved"}


@router.get("/reports/{report_id}/messages")
def report_messages(report_id: str, admin: Account = Depends(admin_account), s: Services = Depends(services)) -> list[dict]:
    """The only way an admin can read messages, and every read is logged."""
    return [m.model_dump() for m in actions.read_report_messages(s, admin, report_id)]
