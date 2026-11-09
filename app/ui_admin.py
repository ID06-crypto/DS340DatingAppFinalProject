"""Admin pages. Each one needs an admin who signed in recently (step-up), and every view of
sensitive data and every action is written to audit_log. There's deliberately no page for
browsing messages, individual ratings, or anyone's rating history.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from app import actions, analytics
from app.actions import ActionError
from app.deps import services
from app.services import Services
from app.web import page_admin, page_admin_any, redirect, render, require_csrf
from live_store import Account

router = APIRouter(prefix="/ui/admin")
csrf = Depends(require_csrf)


def _page(request: Request, template: str, context: dict, status: int = 200) -> Response:
    return render(request, f"admin/{template}", {"nav": "admin", **context}, status)


@router.get("/step-up")
def step_up(request: Request, admin: Account = Depends(page_admin_any)) -> Response:
    return _page(request, "step_up.html", {})


@router.post("/step-up", dependencies=[csrf])
def step_up_send(request: Request, admin: Account = Depends(page_admin_any), s: Services = Depends(services)) -> Response:
    actions.request_login_link(s, admin.email_normalized, request.client.host if request.client else "unknown")
    return redirect(request, "/login/sent")


@router.get("")
@router.get("/")
def overview(request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    return _page(request, "overview.html", {"data": analytics.overview(s.settings.main_db), "events": s.live.count_events()})


@router.get("/approvals")
def approvals(request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    s.live.audit(admin.account_id, "view_approvals", None, {})
    lists = {}
    for status in ("pending", "suspended", "rejected"):
        lists[status] = [
            {
                "account": a,
                "person": s.public_profile(a.user_id, own=True) if a.user_id else None,
                "consent_current": a.consent_version == s.settings.consent_version,
                "missing": s.missing_for_approval(a),
            }
            for a in s.live.list_accounts(status)
        ]
    return _page(request, "approvals.html", {"lists": lists})


@router.post("/approvals/{account_id}/{action}", dependencies=[csrf])
async def approval_action(account_id: str, action: str, request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    reason = str((await request.form()).get("reason", ""))[:500]
    try:
        if action == "approve":
            actions.approve(s, admin, account_id)
        elif action == "reject":
            actions.reject(s, admin, account_id, reason)
        elif action == "suspend":
            actions.suspend(s, admin, account_id, reason)
        else:
            raise HTTPException(404, "Not found")
    except ActionError as error:
        message = error.message + (": missing " + ", ".join(error.detail["missing"]) if isinstance(error.detail, dict) else "")
        return render(request, "error.html", {"heading": "Couldn't do that", "message": message}, error.status)
    return redirect(request, "/admin/approvals")


@router.get("/reports")
def reports(request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    s.live.audit(admin.account_id, "list_reports", None, {"status": "open"})
    return _page(request, "reports.html", {"reports": s.live.list_reports("open")})


@router.post("/reports/{report_id}/resolve", dependencies=[csrf])
async def resolve(report_id: str, request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    actions.resolve_report(s, admin, report_id, str((await request.form()).get("note", ""))[:1000])
    return redirect(request, "/admin/reports")


@router.get("/reports/{report_id}/conversation")
def conversation_confirm(report_id: str, request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    report = s.live.get_report(report_id)
    if report is None or report.match_id is None:
        raise HTTPException(404, "Not found")
    return render(request, "confirm.html", {
        "nav": "admin",
        "heading": "View this reported conversation?",
        "paragraphs": ["This access will be logged, with your account and the time.", "Only view it if you need it to review this report."],
        "action": f"/admin/reports/{report_id}/conversation",
        "button": "View the conversation",
        "cancel": "/admin/reports",
    })


@router.post("/reports/{report_id}/conversation", dependencies=[csrf])
def conversation(report_id: str, request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    report = s.live.get_report(report_id)
    messages = actions.read_report_messages(s, admin, report_id)  # Writes the audit entry.
    return _page(request, "conversation.html", {"report": report, "messages": messages})


@router.get("/compare")
def compare(request: Request, k: int = 10, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    k = max(1, min(k, 100))
    arms = analytics.compare(s.settings.main_db, s.settings, k)
    position_chart = {
        "type": "line",
        "data": {"datasets": [
            {"label": arm.matcher, "data": [{"x": p, "y": round(m, 2)} for p, m, _ in arm.positions]} for arm in arms
        ]},
        "options": {"scales": {"x": {"type": "linear", "title": {"display": True, "text": "Impression position"}},
                               "y": {"min": 1, "max": 10, "title": {"display": True, "text": "Mean rating"}}}},
    }
    histogram_chart = {
        "type": "bar",
        "data": {"labels": [str(b) for b in range(1, 11)], "datasets": [{"label": arm.matcher, "data": arm.histogram} for arm in arms]},
        "options": {"scales": {"x": {"title": {"display": True, "text": "Rating (rounded down)"}},
                               "y": {"title": {"display": True, "text": "Ratings"}}}},
    }
    return _page(request, "compare.html", {
        "arms": arms,
        "k": k,
        "settings": s.settings,
        "position_chart": json.dumps(position_chart),
        "histogram_chart": json.dumps(histogram_chart),
    })


@router.get("/controls")
def controls(request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    current = s.controls()
    return _page(request, "controls.html", {
        "matchers": [(name, m.description, current.matcher_weights.get(name, 0.0)) for name, m in sorted(s.matchers.items())],
        "controls": current,
        "threshold": s.settings.match_threshold,
        "threshold_locked": s.live.count_live_ratings() > 0,
        "default_matcher": s.settings.default_matcher,
    })


@router.post("/controls", dependencies=[csrf])
async def controls_save(request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    form = await request.form()
    weights = {}
    for name in s.matchers:
        try:
            weight = float(str(form.get(f"weight_{name}", "0")) or 0)
        except ValueError:
            weight = 0.0
        weights[name] = weight if form.get(f"enabled_{name}") == "yes" else 0.0
    actions.set_controls(s, admin, weights, form.get("signups_paused") == "yes", form.get("feed_paused") == "yes")
    return redirect(request, "/admin/controls")


@router.get("/audit")
def audit(request: Request, action: str | None = None, actor: str | None = None, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    entries = s.live.audit_entries(action or None, actor or None, limit=500)
    return _page(request, "audit.html", {"entries": list(reversed(entries)), "action": action or "", "actor": actor or ""})


def _export(s: Services, admin: Account, kind: str) -> list[dict]:
    if not s.settings.export_salt:
        raise ActionError(409, "Set the EXPORT_SALT environment variable before exporting.")
    rows = analytics.export_rows(s.settings.main_db, s.settings.export_salt)
    s.live.audit(admin.account_id, "export", kind, {"rows": len(rows)})
    return rows


@router.get("/export.csv")
def export_csv(admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    return Response(analytics.to_csv(_export(s, admin, "csv")), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="impressions.csv"'})


@router.get("/export.json")
def export_json(admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    return Response(json.dumps(_export(s, admin, "json"), indent=1), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="impressions.json"'})
