"""Admin pages. Each one needs an admin who signed in recently (step-up), and every view of
sensitive data and every action is written to audit_log. Ratings are shown only on the
Users page, with people as codes rather than names, and only for people who haven't
opted out.
"""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from app import actions, analytics, insights
from app.actions import ActionError
from app.deps import services
from app.services import Services
from app.web import page_admin, page_admin_any, redirect, render, require_csrf
from app.routes_auth import set_session_cookie
from live_store import Account, utc_now

router = APIRouter(prefix="/ui/admin")
csrf = Depends(require_csrf)


def _page(request: Request, template: str, context: dict, status: int = 200) -> Response:
    return render(request, f"admin/{template}", {"nav": "admin", **context}, status)


@router.get("/login")
def password_login_page(request: Request) -> Response:
    return _page(request, "login.html", {"error": None, "email": ""})


@router.post("/login")
async def password_login(request: Request, s: Services = Depends(services)) -> Response:
    form = await request.form()
    email = str(form.get("email", ""))[:320]
    try:
        account, session_id = actions.password_sign_in(s, email, str(form.get("password", "")), request.client.host if request.client else "unknown")
    except ActionError as error:
        return _page(request, "login.html", {"error": error.message, "email": email}, error.status)
    destination = "/admin" if account.consent_version == s.settings.consent_version else "/onboarding/consent"
    response = redirect(request, destination)
    set_session_cookie(response, session_id, s)
    return response


@router.get("/password")
def password_page(request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    has_password = s.live.password_hash(admin.account_id) is not None
    return _page(request, "password.html", {"error": None, "done": request.query_params.get("done") == "1", "has_password": has_password})


@router.post("/password", dependencies=[csrf])
async def password_save(request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    form = await request.form()
    new, again = str(form.get("new_password", "")), str(form.get("new_password_again", ""))
    has_password = s.live.password_hash(admin.account_id) is not None
    try:
        if new != again:
            raise ActionError(422, "The two new passwords don't match.")
        actions.set_admin_password(s, admin, str(form.get("current_password", "")), new)
    except ActionError as error:
        return _page(request, "password.html", {"error": error.message, "done": False, "has_password": has_password}, error.status)
    return redirect(request, "/admin/password?done=1")


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


@router.get("/compare")
def compare(request: Request, k: int = 10, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    k = max(1, min(k, 100))
    arms = analytics.compare(s.settings.main_db, s.settings, k)
    position_chart = {
        "type": "line",
        "data": {"datasets": [
            {"label": arm.matcher, "data": [{"x": p, "y": round(m, 3)} for p, m, _ in arm.positions]} for arm in arms
        ]},
        "options": {"scales": {"x": {"type": "linear", "title": {"display": True, "text": "Impression position"}},
                               "y": {"min": 0, "max": 1, "title": {"display": True, "text": "Mean score (0 to 1)"}}}},
    }
    histogram_chart = {
        "type": "bar",
        "data": {"labels": [f"{b / 10:.1f}–{(b + 1) / 10:.1f}" for b in range(10)], "datasets": [{"label": arm.matcher, "data": arm.histogram} for arm in arms]},
        "options": {"scales": {"x": {"title": {"display": True, "text": "Score (0 to 1)"}},
                               "y": {"title": {"display": True, "text": "Ratings"}}}},
    }
    return _page(request, "compare.html", {
        "arms": arms,
        "k": k,
        "settings": s.settings,
        "position_chart": json.dumps(position_chart),
        "histogram_chart": json.dumps(histogram_chart),
    })


def _bar_chart(effect: insights.Effect, horizontal: bool) -> dict:
    datasets = [{"label": name, "data": values} for name, values in effect.series.items()]
    value_axis = {"min": 0, "max": 1, "title": {"display": True, "text": "Mean score (0 to 1)"}}
    category_axis = {"title": {"display": False}}
    chart_type = "line" if len(datasets) > 1 else "bar"
    return {
        "type": chart_type,
        "data": {"labels": effect.labels, "datasets": datasets},
        "options": {
            "indexAxis": "y" if horizontal and chart_type == "bar" else "x",
            "plugins": {"legend": {"display": len(datasets) > 1}},
            "scales": {"x": value_axis, "y": category_axis} if horizontal and chart_type == "bar" else {"x": category_axis, "y": value_axis},
        },
    }


def _distribution_chart(data: dict, axis_title: str) -> dict:
    return {
        "type": "line",
        "data": {"labels": data["labels"], "datasets": [{"label": name, "data": values} for name, values in data["series"].items()]},
        "options": {"scales": {"x": {"title": {"display": True, "text": "Score (0 to 1)"}},
                               "y": {"min": 0, "title": {"display": True, "text": axis_title}}}},
    }


@router.get("/users")
def users(request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    """Every participant's scores for every other, as heat maps, with descriptive charts.
    People appear as codes, never names, and each view is audited."""
    threshold = analytics.to_unit(s.settings.positive_threshold)
    data = insights.load(s.settings.main_db, threshold, s.settings.export_salt)
    s.live.audit(admin.account_id, "view_user_insights", None, {"people": len(data.people), "ratings": len(data.ratings)})
    effects = [
        {"effect": e, "chart": json.dumps(_bar_chart(e, horizontal=len(e.series) == 1 and len(e.labels) > 5)), "tall": len(e.labels) > 8}
        for e in data.effects
    ]
    return _page(request, "users.html", {
        "data": data,
        "threshold": threshold,
        "threshold_10": s.settings.positive_threshold,
        "has_salt": bool(s.settings.export_salt),
        "interest_map": json.dumps(data.interest_map),
        "reciprocal_map": json.dumps(data.reciprocal_map),
        "given_chart": json.dumps(_distribution_chart(data.gender_given, "Share of ratings given")),
        "received_chart": json.dumps(_distribution_chart(data.gender_received, "Share of ratings received")),
        "effects": effects,
    })


@router.get("/controls")
def controls(request: Request, admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    current = s.controls()
    return _page(request, "controls.html", {
        "matchers": [(name, m.description, current.matcher_weights.get(name, 0.0)) for name, m in sorted(s.matchers.items())],
        "controls": current,
        "opens_value": f"{s.local_time(current.rating_opens_at):%Y-%m-%dT%H:%M}" if current.rating_opens_at else "",
        "opens_text": s.describe_time(current.rating_opens_at) if current.rating_opens_at else None,
        "rating_open": actions.rating_not_open_message(s) is None,
        "timezone": s.settings.study_timezone,
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
    opens_text = str(form.get("rating_opens_at", "")).strip()
    try:
        opens = datetime.fromisoformat(opens_text).replace(tzinfo=ZoneInfo(s.settings.study_timezone)) if opens_text else None
    except ValueError:
        raise HTTPException(422, "Enter the date and time rating opens, or leave it empty.") from None
    actions.set_controls(s, admin, weights, form.get("signups_paused") == "yes", form.get("feed_paused") == "yes", opens)
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


@router.get("/truth.zip")
def truth_zip(admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    """The ground truth: every score between participants, by participant_id. See the README."""
    if not s.settings.export_salt:
        raise ActionError(409, "Set the EXPORT_SALT environment variable before exporting.")
    data = insights.load(s.settings.main_db, analytics.to_unit(s.settings.positive_threshold), s.settings.export_salt)
    s.live.audit(admin.account_id, "export", "truth", {"people": len(data.people), "rows": len(data.ratings)})
    stamp = utc_now().strftime("%Y%m%d-%H%M")
    return Response(insights.truth_zip(data), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="ground-truth-{stamp}.zip"'})


@router.get("/export.csv")
def export_csv(admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    return Response(analytics.to_csv(_export(s, admin, "csv")), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="impressions.csv"'})


@router.get("/export.json")
def export_json(admin: Account = Depends(page_admin), s: Services = Depends(services)) -> Response:
    return Response(json.dumps(_export(s, admin, "json"), indent=1), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="impressions.json"'})
