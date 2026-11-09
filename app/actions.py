"""Everything users and admins can do, shared by the JSON API and the web pages.

Routes only translate between HTTP and these functions, so the rules live in one place.
A function that refuses raises ActionError with an HTTP status and a plain message.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.photos import MAX_PHOTOS, InvalidPhotoError, clean_photo, new_storage_key
from app.security import new_secret, normalize_email
from app.services import Services
from live_store import Account, Impression, Match, Message, Photo, Report, sha256, utc_now
from matching import PublicView
from models import GENDERS, LIVE_RUN_ID

MIN_SCORE = 1.0
MAX_SCORE = 10.0
SESSION_SURVEY_EVERY = 25  # Ratings between session surveys.
SESSION_SURVEY_GAP = timedelta(hours=24)  # At most one session survey per day.
POST_MATCH_SURVEY_AFTER = timedelta(days=3)
EMPTY_POOL_MESSAGE = "You've seen everyone available to you right now. New people may join."


class ActionError(Exception):
    def __init__(self, status: int, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.detail = detail if detail is not None else message


def to_unit_scale(score: float) -> float:
    """People rate from 1 to 10. Ratings are stored from 0 to 1, like simulated ones."""
    return (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)


def to_rating_scale(unit: float) -> float:
    return MIN_SCORE + unit * (MAX_SCORE - MIN_SCORE)


# Sign-in


def request_login_link(s: Services, raw_email: str, client_ip: str) -> None:
    """Send a sign-in link if allowed. Callers always give the same reply, whatever happens."""
    if not s.login_per_ip.allow(client_ip):
        raise ActionError(429, "Too many requests. Try again later.")
    email = normalize_email(raw_email, s.settings.allowed_email_domains)
    if email is None or not s.login_per_email.allow(email):
        return
    admins = {normalize_email(e) for e in s.settings.admin_emails}
    if s.controls().signups_paused and email not in admins and s.live.account_for_email(email) is None:
        return  # New signups are paused. Existing accounts can still sign in.
    token = new_secret()
    s.live.save_login_token(sha256(token), email, utc_now() + timedelta(minutes=s.settings.login_token_minutes))
    link = f"{s.settings.site_url}/auth/verify?token={token}"
    s.send_email(
        email,
        "Your sign-in link",
        f"Sign in with this link. It works once and expires in {s.settings.login_token_minutes} minutes.\n\n{link}\n",
    )


def sign_in(s: Services, token: str) -> tuple[Account, str]:
    """Use a sign-in link. Returns the account and a new session id for the cookie."""
    email = s.live.use_login_token(sha256(token))
    if email is None:
        raise ActionError(400, "This link is invalid, used, or expired. Request a new one.")
    admins = {normalize_email(e) for e in s.settings.admin_emails}
    account, _ = s.live.get_or_create_account(email, admin=email in admins)
    session_id = new_secret()
    s.live.create_session(sha256(session_id), account.account_id, utc_now() + timedelta(days=s.settings.session_days))
    return account, session_id


def give_consent(s: Services, account: Account, version: str) -> None:
    if version != s.settings.consent_version:
        raise ActionError(409, f"The current consent version is {s.settings.consent_version}")
    s.live.record_consent(account.account_id, version)


# Profile, preferences, and photos


def require_profile(account: Account) -> str:
    if account.user_id is None:
        raise ActionError(409, "Create your profile first")
    return account.user_id


def create_profile(s: Services, account: Account, fields: dict[str, Any]) -> str:
    """Create the caller's profile. The server sets user_id and agent, never the client."""
    if account.user_id is not None:
        raise ActionError(409, "You already have a profile. Use PUT to edit it.")
    profile = s.manager.add_profile({**fields, "agent": False})
    s.live.link_profile(account.account_id, profile.user_id)
    return profile.user_id


def edit_profile(s: Services, account: Account, changes: dict[str, Any]) -> None:
    s.manager.edit_profile(require_profile(account), **changes)


def set_preferences(s: Services, account: Account, interested_in: list[str] | str) -> frozenset[str]:
    genders = GENDERS if interested_in == "everyone" else interested_in
    return s.manager.set_preferences(require_profile(account), genders).interested_in


def upload_photo(s: Services, account: Account, data: bytes, max_bytes: int) -> Photo:
    user_id = require_profile(account)
    if len(data) > max_bytes:
        raise ActionError(413, "Photos can be at most 5 MB")
    if len(s.live.photos_for(user_id)) >= MAX_PHOTOS:
        raise ActionError(409, f"You can have at most {MAX_PHOTOS} photos")
    try:
        cleaned = clean_photo(data)
    except InvalidPhotoError as error:
        raise ActionError(415, str(error)) from None
    key = new_storage_key()
    s.photo_storage.save(key, cleaned)
    photo = s.live.add_photo(user_id, key, MAX_PHOTOS)
    if photo is None:  # Another upload reached the limit first.
        s.photo_storage.delete(key)
        raise ActionError(409, f"You can have at most {MAX_PHOTOS} photos")
    return photo


def delete_photo(s: Services, account: Account, photo_id: str) -> None:
    photo = s.live.get_photo(photo_id)
    if photo is None or photo.user_id != account.user_id:
        raise ActionError(404, "Photo not found")
    s.live.delete_photo(photo_id)
    s.photo_storage.delete(photo.storage_key)


def reorder_photos(s: Services, account: Account, photo_ids: list[str]) -> None:
    if not s.live.reorder_photos(require_profile(account), photo_ids):
        raise ActionError(400, "List each of your photos exactly once")


def may_see(s: Services, account: Account, owner: str) -> bool:
    """Whether this account may see another user's profile and photos.

    The owner, admins, and approved users who have the owner in their pool, have been
    shown them, or have an active match with them, as long as neither has blocked the other.
    """
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


def photo_bytes(s: Services, account: Account, photo_id: str) -> bytes:
    photo = s.live.get_photo(photo_id)
    if photo is None or not may_see(s, account, photo.user_id):
        raise ActionError(404, "Photo not found")  # 404, so a photo's existence isn't revealed.
    return s.photo_storage.load(photo.storage_key)


# The feed and ratings


@dataclass
class Card:
    """What Discover shows next: an impression, or why there isn't one."""

    impression: Impression | None
    people_left: int  # Eligible and not yet rated, including the person on this card.
    message: str | None = None


def choose_matcher(s: Services, account: Account) -> str:
    """Opted-out users get the default matcher. Others get a weighted random enabled one."""
    if account.experiment_opt_out:
        return s.settings.default_matcher
    weights = {name: w for name, w in s.controls().matcher_weights.items() if w > 0 and name in s.matchers}
    if not weights:
        return s.settings.default_matcher
    names = sorted(weights)
    return s.rng.choices(names, weights=[weights[n] for n in names])[0]


def _unavailable_message(s: Services, account: Account) -> str | None:
    if s.controls().feed_paused:
        return "Discover is paused for maintenance. Please check back later."
    if account.paused:
        return "Your account is paused, so you're hidden from others and can't see new people. You can unpause it on your Me page."
    return None


def next_impression(s: Services, account: Account) -> Card:
    """Choose the next profile with a matcher, and log the impression."""
    viewer = require_profile(account)
    message = _unavailable_message(s, account)
    if message:
        return Card(None, 0, message)
    pool = s.manager.eligible_candidates(viewer, LIVE_RUN_ID)
    if not pool:
        return Card(None, 0, EMPTY_POOL_MESSAGE)
    name = choose_matcher(s, account)
    view = PublicView(s.manager.store, LIVE_RUN_ID)
    with s.matcher_lock:
        matcher = s.matchers[name]
        scores = matcher.score_candidates(viewer, view, pool)
        target = matcher.next_profile(viewer, view)
    if target is None:
        return Card(None, len(pool), EMPTY_POOL_MESSAGE)
    if target not in pool:
        s.live.record_event("matcher_error")
        raise ActionError(500, "Couldn't choose a profile. Try again.")
    impression = s.live.create_impression(
        viewer,
        target,
        name,
        len(pool),
        predicted_score=scores.get(target) if scores else None,
        candidate_scores={c: scores[c] for c in pool if c in scores} if scores else None,
    )
    return Card(impression, len(pool))


def current_card(s: Services, account: Account) -> Card:
    """Discover's card: the unrated profile already shown, if there is one, else the next.

    Reloading the page shows the same person again, so reloading can't be used to skip.
    """
    viewer = require_profile(account)
    message = _unavailable_message(s, account)
    if message:
        return Card(None, 0, message)
    pending = s.live.pending_impression(viewer)
    if pending is not None and _still_showable(s, viewer, pending.target_id):
        return Card(pending, len(s.manager.eligible_candidates(viewer, LIVE_RUN_ID)) + 1)
    return next_impression(s, account)


def _still_showable(s: Services, viewer: str, target: str) -> bool:
    target_account = s.live.account_for_user(target)
    return (
        target_account is not None
        and target_account.status == "approved"
        and not target_account.paused
        and not s.live.blocked_either_way(viewer, target)
        and s.live.active_match_between(viewer, target) is None
    )


@dataclass
class RatingOutcome:
    matched: bool
    match_id: str | None


def rate(s: Services, account: Account, impression_id: str, score: float) -> RatingOutcome:
    """Save a rating, and make a match if both people rated each other highly enough.

    The target and matcher come from the impression, never the client. The outcome has
    the same shape either way, so it reveals nothing about the other person unless they matched.
    """
    viewer = require_profile(account)
    if not MIN_SCORE <= score <= MAX_SCORE:
        raise ActionError(422, "Ratings go from 1 to 10")
    claim = s.live.claim_impression(impression_id, viewer)
    if claim == "missing":
        raise ActionError(404, "Impression not found")
    if claim == "rated":
        raise ActionError(409, "You've already rated this profile")
    impression = s.live.get_impression(impression_id)
    unit = to_unit_scale(score)
    try:
        rating = s.manager.add_rating(viewer, impression.target_id, unit, LIVE_RUN_ID, matcher=impression.matcher)
    except Exception:
        s.live.unclaim_impression(impression_id)
        raise
    if impression.matcher in s.matchers:
        with s.matcher_lock:
            s.matchers[impression.matcher].observe(rating)

    threshold = to_unit_scale(s.settings.match_threshold) - 1e-9
    reverse = [r for r in s.manager.ratings_by_rater(impression.target_id, LIVE_RUN_ID) if r.target_id == viewer]
    if not (reverse and unit >= threshold and reverse[0].score >= threshold):
        return RatingOutcome(False, None)
    if s.live.blocked_either_way(viewer, impression.target_id):
        return RatingOutcome(False, None)
    existing = s.live.active_match_between(viewer, impression.target_id)
    match = existing or s.live.create_match(viewer, impression.target_id)
    if existing is None:
        _email_new_match(s, match)
    return RatingOutcome(True, match.match_id)


def _email_new_match(s: Services, match: Match) -> None:
    """One email per participant, with no names or photos, only if they want it."""
    for user_id in (match.user_a, match.user_b):
        account = s.live.account_for_user(user_id)
        if account is not None and account.notify_on_match:
            s.send_email(
                account.email_normalized,
                "You have a new match",
                f"You have a new match.\n\n{s.settings.site_url}/matches\n",
            )


# Surveys


def session_survey_due(s: Services, account: Account, pool_empty: bool) -> bool:
    """Due when the pool runs out, or after every 25th rating, and at most once a day."""
    user_id = account.user_id
    if user_id is None:
        return False
    last = s.live.last_survey_prompt(user_id, "session")
    if last is not None and utc_now() - last < SESSION_SURVEY_GAP:
        return False
    rated = s.live.count_rated(user_id)
    return pool_empty or (rated > 0 and rated % SESSION_SURVEY_EVERY == 0)


def post_match_survey_due(s: Services, account: Account, match: Match) -> bool:
    """Due once per match, once it's at least 3 days old."""
    age = utc_now() - datetime.fromisoformat(match.created_at)
    return age >= POST_MATCH_SURVEY_AFTER and s.live.last_survey_prompt(account.user_id, "post_match", match.match_id) is None


def mark_survey_shown(s: Services, account: Account, kind: str, match_id: str | None = None) -> None:
    s.live.record_survey_prompt(require_profile(account), kind, match_id)


def answer_survey(s: Services, account: Account, kind: str, score: int, match_id: str | None = None) -> None:
    user_id = require_profile(account)
    if kind not in ("session", "post_match") or not 1 <= score <= 5:
        raise ActionError(422, "Answers go from 1 to 5")
    if kind == "post_match":
        match = s.live.get_match(match_id or "")
        if match is None or user_id not in (match.user_a, match.user_b):
            raise ActionError(404, "Match not found")
    s.live.add_survey(user_id, kind, score, match_id if kind == "post_match" else None)


# Matches, chat, blocks, and reports


def own_match(s: Services, account: Account, match_id: str) -> Match:
    match = s.live.get_match(match_id)
    if match is None or account.user_id not in (match.user_a, match.user_b):
        raise ActionError(404, "Match not found")
    return match


def open_chat(s: Services, account: Account, match_id: str) -> Match:
    """A match its participants can still message in: active, and no block either way."""
    match = own_match(s, account, match_id)
    if match.status != "active" or s.live.blocked_either_way(match.user_a, match.user_b):
        raise ActionError(403, "This match has ended")
    return match


def unmatch(s: Services, account: Account, match_id: str) -> None:
    match = own_match(s, account, match_id)
    if match.status != "active":
        raise ActionError(409, "This match has already ended")
    s.live.unmatch(match_id, account.user_id)


def read_messages(s: Services, account: Account, match_id: str, after: int = 0) -> list[Message]:
    open_chat(s, account, match_id)
    return s.live.messages_after(match_id, after)


def send_message(s: Services, account: Account, match_id: str, body: str) -> Message:
    open_chat(s, account, match_id)
    if not body.strip():
        raise ActionError(422, "Messages can't be blank")
    if len(body) > 2000:
        raise ActionError(422, "Messages can be at most 2,000 characters")
    if not s.messages_per_user.allow(account.user_id):
        raise ActionError(429, "You're sending messages too quickly")
    return s.live.add_message(match_id, account.user_id, body)


def block(s: Services, account: Account, user_id: str) -> None:
    me = require_profile(account)
    if user_id == me:
        raise ActionError(400, "You can't block yourself")
    if not s.manager.store.profile_exists(user_id):
        raise ActionError(404, "User not found")
    s.live.add_block(me, user_id)  # Also ends any active match.


def unblock(s: Services, account: Account, user_id: str) -> None:
    if not s.live.remove_block(require_profile(account), user_id):
        raise ActionError(404, "You haven't blocked that user")


def report(
    s: Services, account: Account, reported_id: str, reason: str, details: str, match_id: str | None
) -> Report:
    me = require_profile(account)
    if reported_id == me:
        raise ActionError(400, "You can't report yourself")
    if not s.manager.store.profile_exists(reported_id):
        raise ActionError(404, "User not found")
    if reason not in ("harassment", "spam", "fake", "inappropriate", "other"):
        raise ActionError(422, "Choose a reason")
    if len(details) > 1000:
        raise ActionError(422, "Details can be at most 1,000 characters")
    if match_id is not None:
        match = s.live.get_match(match_id)
        if match is None or {match.user_a, match.user_b} != {me, reported_id}:
            raise ActionError(400, "That match isn't between you and the reported user")
    return s.live.add_report(me, reported_id, match_id, reason, details)


# Account settings


def set_account_flags(s: Services, account: Account, **flags: bool) -> None:
    s.live.set_flags(account.account_id, **flags)


def export_my_data(s: Services, account: Account) -> dict[str, Any]:
    """Everything about the caller, and nothing about anyone else beyond their user ids."""
    user_id = account.user_id
    preferences = s.manager.get_preferences(user_id) if user_id else None
    profile = s.manager.get_profile(user_id).model_dump(exclude={"agent"}) if user_id else None
    return {
        "account": {
            "email": account.email_normalized,
            "status": account.status,
            "created_at": account.created_at,
            "consent_version": account.consent_version,
            "consent_at": account.consent_at,
            "experiment_opt_out": account.experiment_opt_out,
            "notify_on_match": account.notify_on_match,
            "paused": account.paused,
        },
        "profile": profile,
        "interested_in": sorted(preferences.interested_in) if preferences else None,
        "ratings": [
            {"target_id": r.target_id, "score": round(to_rating_scale(r.score), 2), "rated_at": r.timestamp.isoformat()}
            for r in (s.manager.ratings_by_rater(user_id, LIVE_RUN_ID) if user_id else [])
        ],
        "matches": [
            {"match_id": m.match_id, "with_user_id": m.other(user_id), "created_at": m.created_at, "status": m.status}
            for m in (s.live.matches_for(user_id) if user_id else [])
        ],
        "messages_sent": [
            {"match_id": m.match_id, "body": m.body, "created_at": m.created_at}
            for m in (s.live.messages_sent_by(user_id) if user_id else [])
        ],
    }


def delete_account(s: Services, account: Account) -> None:
    s.delete_account(account)


# Admin


def approve(s: Services, admin: Account, account_id: str) -> None:
    account = _account(s, account_id)
    missing = s.missing_for_approval(account)
    if missing:
        raise ActionError(409, "This account isn't complete", {"message": "This account isn't complete", "missing": missing})
    s.live.set_status(account_id, "approved", admin.account_id)
    s.live.audit(admin.account_id, "account_approve", account_id, {"previous_status": account.status})


def reject(s: Services, admin: Account, account_id: str, reason: str = "") -> None:
    account = _account(s, account_id)
    s.live.set_status(account_id, "rejected", admin.account_id)
    s.live.audit(admin.account_id, "account_reject", account_id, {"previous_status": account.status, "reason": reason})


def suspend(s: Services, admin: Account, account_id: str, reason: str = "") -> None:
    account = _account(s, account_id)
    s.live.set_status(account_id, "suspended", admin.account_id)
    s.live.audit(admin.account_id, "account_suspend", account_id, {"previous_status": account.status, "reason": reason})


def _account(s: Services, account_id: str) -> Account:
    account = s.live.get_account(account_id)
    if account is None:
        raise ActionError(404, "Account not found")
    return account


def resolve_report(s: Services, admin: Account, report_id: str, note: str = "") -> None:
    if s.live.get_report(report_id) is None:
        raise ActionError(404, "Report not found")
    s.live.resolve_report(report_id, admin.account_id)
    s.live.audit(admin.account_id, "resolve_report", report_id, {"note": note})


def read_report_messages(s: Services, admin: Account, report_id: str) -> list[Message]:
    """The only way an admin can read messages. Every read is logged."""
    report_row = s.live.get_report(report_id)
    if report_row is None:
        raise ActionError(404, "Report not found")
    if report_row.match_id is None:
        raise ActionError(404, "This report isn't about a match")
    s.live.audit(admin.account_id, "read_report_messages", report_id, {"match_id": report_row.match_id})
    return s.live.messages_after(report_row.match_id, 0, limit=10_000)


def set_controls(
    s: Services,
    admin: Account,
    matcher_weights: dict[str, float],
    signups_paused: bool,
    feed_paused: bool,
) -> None:
    unknown = set(matcher_weights) - set(s.matchers)
    if unknown or any(weight < 0 for weight in matcher_weights.values()):
        raise ActionError(422, "Weights must be zero or more, for registered matchers")
    s.live.set_setting("matcher_weights", matcher_weights)
    s.live.set_setting("signups_paused", signups_paused)
    s.live.set_setting("feed_paused", feed_paused)
    s.live.audit(
        admin.account_id,
        "set_controls",
        None,
        {"matcher_weights": matcher_weights, "signups_paused": signups_paused, "feed_paused": feed_paused},
    )

