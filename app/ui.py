"""The web pages for users. Every change goes through app/actions.py, like the JSON API.

These routes are mounted under /ui, but browsers see them at their plain URLs, because
PageRouting (app/web.py) sends page requests here. Forms use Post/Redirect/Get, and HTMX
requests get partial templates back.
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from app import actions
from app.actions import ActionError
from app.deps import services, signed_in
from app.photos import MAX_PHOTO_BYTES
from app.routes_auth import set_session_cookie
from app.security import SESSION_COOKIE
from app.services import Services, first_name
from app.web import (
    is_htmx,
    next_onboarding_step,
    page_account,
    page_consented,
    page_member,
    redirect,
    render,
    require_csrf,
)
from live_store import Account, sha256
from models import Profile, generate_user_id

router = APIRouter(prefix="/ui")
csrf = Depends(require_csrf)

DESIGN_RULES = [
    "No streaks, badges, or counts meant to pull you back.",
    "No \"someone liked you\" teasers. You only find out about a match.",
    "No infinite scroll. You see one person at a time.",
    "No online, typing, or read indicators.",
    "No reminder or re-engagement emails. The only emails are sign-in links and one email per new match.",
    "No skip button. Every rating counts, so take your time.",
    "No ads and no tracking scripts.",
]
PROFILE_TEXT_FIELDS = ("name", "school", "major", "job", "religion", "gender", "looking_for")


def _safe_back(path: str | None, default: str) -> str:
    return path if path and path.startswith("/") and not path.startswith("//") else default


def _plain(error: dict[str, Any]) -> str:
    """Pydantic's message, in plainer words where it's technical."""
    kind = error.get("type", "")
    if kind in ("missing", "string_too_short"):
        return "This is required."
    if kind == "literal_error":
        return "Choose one of the options."
    if kind == "too_short":
        return "Add at least one."
    message = str(error.get("msg", "This isn't valid.")).removeprefix("Value error, ")
    return message[0].upper() + message[1:] + ("" if message.endswith(".") else ".")


# Public pages


@router.get("/")
def landing(request: Request, s: Services = Depends(services)) -> Response:
    account = signed_in(request, s)
    if account is not None:
        return redirect(request, next_onboarding_step(s, account) or "/discover")
    return render(request, "landing.html")


@router.post("/login")
async def login(request: Request, s: Services = Depends(services)) -> Response:
    form = await request.form()
    try:
        actions.request_login_link(s, str(form.get("email", "")), request.client.host if request.client else "unknown")
    except ActionError as error:
        return render(request, "error.html", {"heading": "Please wait", "message": error.message}, error.status)
    return redirect(request, "/login/sent")


@router.get("/login/sent")
def login_sent(request: Request, s: Services = Depends(services)) -> Response:
    return render(request, "login_sent.html", {"minutes": s.settings.login_token_minutes})


@router.get("/auth/verify")
def verify(token: str, request: Request, s: Services = Depends(services)) -> Response:
    try:
        account, session_id = actions.sign_in(s, token)
    except ActionError as error:
        return render(request, "error.html", {"heading": "That link didn't work", "message": error.message}, error.status)
    destination = "/admin" if account.role == "admin" and account.status == "approved" else next_onboarding_step(s, account) or "/discover"
    response = redirect(request, destination)
    set_session_cookie(response, session_id, s)
    return response


@router.post("/logout", dependencies=[csrf])
def logout(request: Request, s: Services = Depends(services)) -> Response:
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        s.live.delete_session(sha256(session_id))
    response = redirect(request, "/")
    response.delete_cookie(SESSION_COOKIE)
    return response


@router.get("/transparency")
def transparency(request: Request, s: Services = Depends(services)) -> Response:
    signed_in(request, s)
    weights = s.controls().matcher_weights
    algorithms = [(name, m.description) for name, m in sorted(s.matchers.items()) if weights.get(name, 0) > 0]
    return render(request, "transparency.html", {
        "algorithms": algorithms,
        "design_rules": DESIGN_RULES,
        "source_url": s.settings.source_repo_url,
    })


# Onboarding


@router.get("/onboarding/consent")
def consent_page(request: Request, account: Account = Depends(page_account), s: Services = Depends(services)) -> Response:
    return _consent(request, s)


def _consent(request: Request, s: Services, error: str | None = None) -> Response:
    paragraphs = [p.strip().lstrip("#- ").strip() for p in s.consent_text().split("\n\n") if p.strip()]
    return render(request, "onboarding_consent.html", {
        "version": s.settings.consent_version,
        "consent_paragraphs": paragraphs,
        "error": error,
    }, 422 if error else 200)


@router.post("/onboarding/consent", dependencies=[csrf])
async def consent_submit(request: Request, account: Account = Depends(page_account), s: Services = Depends(services)) -> Response:
    form = await request.form()
    if form.get("agree") != "yes":
        return _consent(request, s, "Tick the box to agree before continuing.")
    actions.give_consent(s, account, str(form.get("version", "")))
    return redirect(request, "/onboarding/profile")


@router.get("/onboarding/profile")
def profile_page(request: Request, account: Account = Depends(page_consented), s: Services = Depends(services)) -> Response:
    if account.user_id is not None:
        return redirect(request, next_onboarding_step(s, account) or "/discover")
    return render(request, "profile_form.html", {"values": {}, "errors": {}, "action": "/onboarding/profile", "editing": False})


@router.post("/onboarding/profile", dependencies=[csrf])
async def profile_submit(request: Request, account: Account = Depends(page_consented), s: Services = Depends(services)) -> Response:
    return await _save_profile(request, account, s, editing=False)


async def _save_profile(request: Request, account: Account, s: Services, editing: bool) -> Response:
    form = await request.form()
    values = {k: str(form.get(k, "")).strip() for k in (*PROFILE_TEXT_FIELDS, "age", "height_cm", "interests", "clubs")}
    errors: dict[str, str] = {}
    fields: dict[str, Any] = {k: values[k] for k in PROFILE_TEXT_FIELDS}
    for number in ("age", "height_cm"):
        try:
            fields[number] = int(values[number])
        except ValueError:
            errors[number] = "Enter a whole number."
    fields["interests"] = [item for item in values["interests"].split(",") if item.strip()]
    fields["clubs"] = [item for item in values["clubs"].split(",") if item.strip()]
    for choice in ("gender", "looking_for"):
        if not fields[choice]:
            errors[choice] = "Choose one of the options."
    # Check every field at once, so all the problems show together, not one at a time.
    try:
        Profile.model_validate({**{k: v for k, v in fields.items() if k not in errors}, "user_id": generate_user_id(), "agent": False})
    except ValidationError as error:
        for item in error.errors():
            name = str(item["loc"][0]) if item["loc"] else "name"
            errors.setdefault(name, _plain(item))
    if not errors:
        try:
            if editing:
                actions.edit_profile(s, account, fields)
            else:
                actions.create_profile(s, account, fields)
        except ValidationError as error:
            for item in error.errors():
                errors.setdefault(str(item["loc"][0]) if item["loc"] else "name", _plain(item))
        except ActionError as error:
            errors["name"] = error.message
    if errors:
        action = "/me/profile" if editing else "/onboarding/profile"
        return render(request, "profile_form.html", {"values": values, "errors": errors, "action": action, "editing": editing}, 422)
    return redirect(request, "/me" if editing else "/onboarding/preferences")


@router.get("/onboarding/preferences")
def preferences_page(request: Request, account: Account = Depends(page_consented), s: Services = Depends(services)) -> Response:
    if account.user_id is None:
        return redirect(request, "/onboarding/profile")
    return _preferences(request, s, account, "/onboarding/preferences", editing=False)


def _preferences(request: Request, s: Services, account: Account, action: str, editing: bool, error: str | None = None) -> Response:
    current = s.manager.get_preferences(account.user_id)
    return render(request, "preferences_form.html", {
        "chosen": current.interested_in if current else set(),
        "action": action,
        "editing": editing,
        "error": error,
    }, 422 if error else 200)


async def _save_preferences(request: Request, account: Account, s: Services, editing: bool) -> Response:
    chosen = [str(v) for v in (await request.form()).getlist("interested_in")]
    action = "/me/preferences" if editing else "/onboarding/preferences"
    if not chosen:
        return _preferences(request, s, account, action, editing, "Choose at least one.")
    actions.set_preferences(s, account, "everyone" if "everyone" in chosen else chosen)
    return redirect(request, "/me" if editing else "/onboarding/photos")


@router.post("/onboarding/preferences", dependencies=[csrf])
async def preferences_submit(request: Request, account: Account = Depends(page_consented), s: Services = Depends(services)) -> Response:
    return await _save_preferences(request, account, s, editing=False)


def _photos(request: Request, s: Services, account: Account, base: str, editing: bool, error: str | None = None) -> Response:
    profile = s.manager.get_profile(account.user_id)
    return render(request, "photos.html", {
        "photos": s.live.photos_for(account.user_id),
        "first": first_name(profile.name),
        "base": base,
        "editing": editing,
        "error": error,
    }, 422 if error else 200)


@router.get("/onboarding/photos")
def photos_page(request: Request, account: Account = Depends(page_consented), s: Services = Depends(services)) -> Response:
    if account.user_id is None:
        return redirect(request, "/onboarding/profile")
    return _photos(request, s, account, "/onboarding/photos", editing=False)


async def _upload(request: Request, account: Account, s: Services, base: str, editing: bool) -> Response:
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        return _photos(request, s, account, base, editing, "Choose a photo to upload.")
    try:
        actions.upload_photo(s, account, await upload.read(MAX_PHOTO_BYTES + 1), MAX_PHOTO_BYTES)
    except ActionError as error:
        return _photos(request, s, account, base, editing, error.message)
    return redirect(request, base)


@router.post("/onboarding/photos", dependencies=[csrf])
async def photos_upload(request: Request, account: Account = Depends(page_consented), s: Services = Depends(services)) -> Response:
    return await _upload(request, account, s, "/onboarding/photos", editing=False)


def _move(s: Services, account: Account, photo_id: str, direction: str) -> None:
    order = [p.photo_id for p in s.live.photos_for(account.user_id)]
    if photo_id not in order:
        raise ActionError(404, "Photo not found")
    i = order.index(photo_id)
    j = i - 1 if direction == "up" else i + 1
    if 0 <= j < len(order):
        order[i], order[j] = order[j], order[i]
        actions.reorder_photos(s, account, order)


@router.post("/onboarding/photos/{photo_id}/move", dependencies=[csrf])
async def photos_move(photo_id: str, request: Request, account: Account = Depends(page_consented), s: Services = Depends(services)) -> Response:
    _move(s, account, photo_id, str((await request.form()).get("direction")))
    return redirect(request, "/onboarding/photos")


@router.post("/onboarding/photos/{photo_id}/delete", dependencies=[csrf])
def photos_delete(photo_id: str, request: Request, account: Account = Depends(page_consented), s: Services = Depends(services)) -> Response:
    actions.delete_photo(s, account, photo_id)
    return redirect(request, "/onboarding/photos")


@router.get("/pending")
def pending(request: Request, account: Account = Depends(page_consented), s: Services = Depends(services)) -> Response:
    step = next_onboarding_step(s, account)
    if step is None:
        return redirect(request, "/discover")
    if step != "/pending":
        return redirect(request, step)
    return render(request, "pending.html", {"status": account.status})


# Discover


def _discover_context(s: Services, account: Account, notice: str | None = None, error: str | None = None) -> dict[str, Any]:
    card = actions.current_card(s, account)
    person = s.public_profile(card.impression.target_id) if card.impression else None
    survey = None
    pool_empty = card.impression is None and card.message == actions.EMPTY_POOL_MESSAGE
    if actions.session_survey_due(s, account, pool_empty):
        actions.mark_survey_shown(s, account, "session")
        survey = {"kind": "session", "question": "Was this worth your time?", "match_id": None, "back": "/discover"}
    return {"card": card, "person": person, "survey": survey, "notice": notice, "error": error, "nav": "discover"}


@router.get("/discover")
def discover(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    return render(request, "discover.html", _discover_context(s, account))


@router.post("/discover/rate", dependencies=[csrf])
async def discover_rate(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    form = await request.form()
    notice = error = None
    try:
        score = float(str(form.get("score", "")))
        outcome = actions.rate(s, account, str(form.get("impression_id", "")), score)
        if outcome.matched:
            notice = "It's a match. You can message each other in Matches."
    except ValueError:
        error = "Enter a rating from 1.0 to 10.0."
    except ActionError as failure:
        error = failure.message
    if not is_htmx(request):
        return redirect(request, "/discover")
    return render(request, "partials/discover_region.html", _discover_context(s, account, notice, error), 422 if error else 200)


@router.post("/survey", dependencies=[csrf])
async def survey_answer(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    form = await request.form()
    try:
        match_id = str(form["match_id"]) if form.get("match_id") else None
        actions.answer_survey(s, account, str(form.get("kind", "")), int(str(form.get("score", ""))), match_id)
    except (ValueError, ActionError):
        raise HTTPException(422, "That answer couldn't be saved.") from None
    if is_htmx(request):
        return Response('<p class="notice" role="status">Thanks for answering.</p>', media_type="text/html")
    return redirect(request, _safe_back(str(form.get("back", "")), "/discover"))


@router.post("/survey/dismiss", dependencies=[csrf])
async def survey_dismiss(request: Request, account: Account = Depends(page_member)) -> Response:
    # The prompt was recorded when shown, so dismissing needs nothing else.
    if is_htmx(request):
        return Response("", media_type="text/html")
    return redirect(request, _safe_back(str((await request.form()).get("back", "")), "/discover"))


# Blocking and reporting


def _visible_person(s: Services, account: Account, user_id: str) -> str:
    """The first name of someone this user may see, or 404."""
    if not s.manager.store.profile_exists(user_id) or user_id == account.user_id or not actions.may_see(s, account, user_id):
        raise HTTPException(404, "Not found")
    return first_name(s.manager.get_profile(user_id).name)


@router.get("/people/{user_id}/block")
def block_confirm(user_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    name = _visible_person(s, account, user_id)
    return render(request, "confirm.html", {
        "heading": f"Block {name}?",
        "paragraphs": [
            f"You and {name} won't see each other again, and any match between you ends.",
            "They aren't told. You can unblock them later from your Me page.",
        ],
        "action": f"/people/{user_id}/block",
        "button": f"Block {name}",
        "cancel": "/discover",
    })


@router.post("/people/{user_id}/block", dependencies=[csrf])
def block(user_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    _visible_person(s, account, user_id)
    actions.block(s, account, user_id)
    return redirect(request, "/discover")


@router.get("/people/{user_id}/report")
def report_form(user_id: str, request: Request, match_id: str | None = None, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    name = _visible_person(s, account, user_id)
    back = f"/matches/{match_id}" if match_id else "/discover"
    return render(request, "report_form.html", {"name": name, "user_id": user_id, "match_id": match_id, "back": back})


@router.post("/people/{user_id}/report", dependencies=[csrf])
async def report(user_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    name = _visible_person(s, account, user_id)
    form = await request.form()
    match_id = str(form["match_id"]) if form.get("match_id") else None
    try:
        actions.report(s, account, user_id, str(form.get("reason", "")), str(form.get("details", "")), match_id)
    except ActionError as error:
        return render(request, "report_form.html", {"name": name, "user_id": user_id, "match_id": match_id, "back": "/discover", "error": error.message}, error.status)
    return redirect(request, f"/matches/{match_id}" if match_id else "/discover")


# Matches and chat


@router.get("/matches")
def matches(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    rows = []
    for match in s.live.active_matches_for(account.user_id):
        other = match.other(account.user_id)
        photos = s.live.photos_for(other)
        rows.append({
            "match_id": match.match_id,
            "name": first_name(s.manager.get_profile(other).name),
            "photo_id": photos[0].photo_id if photos else None,
            "last_message_at": s.live.last_message_at(match.match_id),
        })
    return render(request, "matches.html", {"matches": rows, "threshold": s.settings.match_threshold, "nav": "matches"})


def _chat_context(s: Services, account: Account, match_id: str, error: str | None = None) -> dict[str, Any]:
    match = actions.open_chat(s, account, match_id)
    other = match.other(account.user_id)
    return {
        "match_id": match_id,
        "other_id": other,
        "name": first_name(s.manager.get_profile(other).name),
        "messages": s.live.messages_after(match_id, 0),
        "me": account.user_id,
        "error": error,
        "nav": "matches",
    }


@router.get("/matches/{match_id}")
def chat(match_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    try:
        context = _chat_context(s, account, match_id)
    except ActionError as error:
        if error.status == 403:
            return render(request, "chat_ended.html", {"nav": "matches"})
        raise HTTPException(error.status, error.message) from None
    match = s.live.get_match(match_id)
    if actions.post_match_survey_due(s, account, match):
        actions.mark_survey_shown(s, account, "post_match", match_id)
        context["survey"] = {"kind": "post_match", "question": "Was this match worth your time?", "match_id": match_id, "back": f"/matches/{match_id}"}
    return render(request, "chat.html", context)


@router.get("/matches/{match_id}/log")
def chat_log(match_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    try:
        context = _chat_context(s, account, match_id)
    except ActionError:
        # HTMX stops polling on status 286.
        return Response("<p>This match has ended.</p>", status_code=286, media_type="text/html")
    return render(request, "partials/messages.html", context)


@router.post("/matches/{match_id}/send", dependencies=[csrf])
async def chat_send(match_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    form = await request.form()
    error = None
    try:
        actions.send_message(s, account, match_id, str(form.get("body", "")))
    except ActionError as failure:
        if failure.status in (403, 404):
            return redirect(request, f"/matches/{match_id}")
        error = failure.message
    if not is_htmx(request):
        return redirect(request, f"/matches/{match_id}")
    return render(request, "partials/chat.html", _chat_context(s, account, match_id, error), 422 if error else 200)


@router.get("/matches/{match_id}/unmatch")
def unmatch_confirm(match_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    try:
        name = _chat_context(s, account, match_id)["name"]
    except ActionError:
        raise HTTPException(404, "Not found") from None
    return render(request, "confirm.html", {
        "heading": f"Unmatch {name}?",
        "paragraphs": ["Your conversation ends for both of you, and you won't see each other again."],
        "action": f"/matches/{match_id}/unmatch",
        "button": "Unmatch",
        "cancel": f"/matches/{match_id}",
    })


@router.post("/matches/{match_id}/unmatch", dependencies=[csrf])
def unmatch(match_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    try:
        actions.unmatch(s, account, match_id)
    except ActionError as error:
        if error.status == 404:
            raise HTTPException(404, "Not found") from None
    return redirect(request, "/matches")


# Me


@router.get("/me")
def me(request: Request, account: Account = Depends(page_member)) -> Response:
    notice = {"saved": "Settings saved.", "paused": "Your account is paused.", "unpaused": "Your account is active again."}.get(
        request.query_params.get("done", "")
    )
    return render(request, "me.html", {"notice": notice, "nav": "me"})


@router.get("/me/profile")
def me_profile(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    profile = s.manager.get_profile(account.user_id)
    values = {k: getattr(profile, k) or "" for k in PROFILE_TEXT_FIELDS}
    values.update(age=profile.age, height_cm=profile.height_cm, interests=", ".join(profile.interests), clubs=", ".join(profile.clubs))
    return render(request, "profile_form.html", {"values": values, "errors": {}, "action": "/me/profile", "editing": True, "nav": "me"})


@router.post("/me/profile", dependencies=[csrf])
async def me_profile_save(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    return await _save_profile(request, account, s, editing=True)


@router.get("/me/preferences")
def me_preferences(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    return _preferences(request, s, account, "/me/preferences", editing=True)


@router.post("/me/preferences", dependencies=[csrf])
async def me_preferences_save(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    return await _save_preferences(request, account, s, editing=True)


@router.get("/me/photos")
def me_photos(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    return _photos(request, s, account, "/me/photos", editing=True)


@router.post("/me/photos", dependencies=[csrf])
async def me_photos_upload(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    return await _upload(request, account, s, "/me/photos", editing=True)


@router.post("/me/photos/{photo_id}/move", dependencies=[csrf])
async def me_photos_move(photo_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    _move(s, account, photo_id, str((await request.form()).get("direction")))
    return redirect(request, "/me/photos")


@router.post("/me/photos/{photo_id}/delete", dependencies=[csrf])
def me_photos_delete(photo_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    if len(s.live.photos_for(account.user_id)) <= 1 and s.settings.require_photo:
        return _photos(request, s, account, "/me/photos", editing=True, error="Add another photo before deleting your last one.")
    actions.delete_photo(s, account, photo_id)
    return redirect(request, "/me/photos")


@router.get("/me/blocked")
def me_blocked(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    blocked = [
        {"user_id": b["user_id"], "name": first_name(s.manager.get_profile(b["user_id"]).name)}
        for b in s.live.blocks_by(account.user_id)
    ]
    return render(request, "blocked.html", {"blocked": blocked, "nav": "me"})


@router.post("/me/blocked/{user_id}/unblock", dependencies=[csrf])
def me_unblock(user_id: str, request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    actions.unblock(s, account, user_id)
    return redirect(request, "/me/blocked")


@router.post("/me/settings", dependencies=[csrf])
async def me_settings(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    form = await request.form()
    actions.set_account_flags(
        s, account, notify_on_match=form.get("notify_on_match") == "yes", experiment_opt_out=form.get("experiment_opt_out") == "yes"
    )
    return redirect(request, "/me?done=saved")


@router.post("/me/pause", dependencies=[csrf])
def me_pause(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    actions.set_account_flags(s, account, paused=True)
    return redirect(request, "/me?done=paused")


@router.post("/me/unpause", dependencies=[csrf])
def me_unpause(request: Request, account: Account = Depends(page_member), s: Services = Depends(services)) -> Response:
    actions.set_account_flags(s, account, paused=False)
    return redirect(request, "/me?done=unpaused")


@router.get("/me/data.json")
def me_data(account: Account = Depends(page_account), s: Services = Depends(services)) -> Response:
    body = json.dumps(actions.export_my_data(s, account), indent=2)
    return Response(body, media_type="application/json", headers={"Content-Disposition": 'attachment; filename="my-data.json"'})


@router.get("/me/delete")
def me_delete_confirm(request: Request, account: Account = Depends(page_account)) -> Response:
    return render(request, "confirm.html", {
        "heading": "Delete your account?",
        "paragraphs": [
            "This permanently erases your account, profile, preferences, photos, ratings, matches, messages, and blocks, and the reports you've made. It can't be undone.",
            "Records of administrator actions are kept, with your identity removed.",
        ],
        "action": "/me/delete",
        "button": "Delete everything",
        "cancel": "/me",
    })


@router.post("/me/delete", dependencies=[csrf])
def me_delete(request: Request, account: Account = Depends(page_account), s: Services = Depends(services)) -> Response:
    actions.delete_account(s, account)
    response = redirect(request, "/")
    response.delete_cookie(SESSION_COOKIE)
    return response
