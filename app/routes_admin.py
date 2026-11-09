"""The approval queue and report handling. Every action writes to audit_log."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from app.deps import admin_account, services
from app.services import Services
from live_store import Account

router = APIRouter(prefix="/admin")


@router.get("/accounts")
def list_accounts(status: str | None = "pending", admin: Account = Depends(admin_account), s: Services = Depends(services)) -> list[dict]:
    """Accounts to review, with their email, profile, and photo ids. Photos load via /photos."""
    result = []
    for account in s.live.list_accounts(status):
        profile = s.public_profile(account.user_id) if account.user_id else {"profile": None, "photo_ids": []}
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
    account = s.live.get_account(account_id)
    if account is None:
        raise HTTPException(404, "Account not found")
    if action == "approve":
        missing = s.missing_for_approval(account)
        if missing:
            raise HTTPException(409, {"message": "This account isn't complete", "missing": missing})
    status = {"approve": "approved", "reject": "rejected", "suspend": "suspended"}[action]
    s.live.set_status(account_id, status, admin.account_id)
    s.live.audit(admin.account_id, f"account_{action}", account_id, {"previous_status": account.status})
    return {"account_id": account_id, "status": status}


@router.get("/reports")
def list_reports(status: str | None = "open", admin: Account = Depends(admin_account), s: Services = Depends(services)) -> list[dict]:
    """Reports, without any message contents. Those are only readable per report, below."""
    s.live.audit(admin.account_id, "list_reports", None, {"status": status})
    return [report.model_dump() for report in s.live.list_reports(status)]


@router.post("/reports/{report_id}/resolve")
def resolve(report_id: str, admin: Account = Depends(admin_account), s: Services = Depends(services)) -> dict:
    if s.live.get_report(report_id) is None:
        raise HTTPException(404, "Report not found")
    s.live.resolve_report(report_id, admin.account_id)
    s.live.audit(admin.account_id, "resolve_report", report_id, {})
    return {"report_id": report_id, "status": "resolved"}


@router.get("/reports/{report_id}/messages")
def report_messages(report_id: str, admin: Account = Depends(admin_account), s: Services = Depends(services)) -> list[dict]:
    """The only way an admin can read messages, and every read is logged."""
    report = s.live.get_report(report_id)
    if report is None:
        raise HTTPException(404, "Report not found")
    if report.match_id is None:
        raise HTTPException(404, "This report isn't about a match")
    s.live.audit(admin.account_id, "read_report_messages", report_id, {"match_id": report.match_id})
    return [m.model_dump() for m in s.live.messages_after(report.match_id, 0, limit=10_000)]
