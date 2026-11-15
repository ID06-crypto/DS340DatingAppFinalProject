"""Everything users and admins can do, shared by the JSON API and the web pages.

Routes only translate between HTTP and these functions, so the rules live in one place.
A function that refuses raises ActionError with an HTTP status and a plain message.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.security import hash_password, new_secret, normalize_email, password_problem, temporary_password, verify_password
from app.services import Services
from live_store import Account, Impression, Match, Report, sha256, utc_now
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


# Sign-in. Everyone signs in with an email and password. The app sends no email: admins
# review each new account by hand, and reset forgotten passwords.

WRONG_PASSWORD = "That email and password don't match an account."
TOO_MANY = "Too many attempts. Wait 15 minutes and try again."




def _new_session(s: Services, account: Account) -> str:
    session_id = new_secret()
    s.live.create_session(sha256(session_id), account.account_id, utc_now() + timedelta(days=s.settings.session_days))
    return session_id


def sign_up(s: Services, raw_email: str, password: str, client_ip: str) -> tuple[Account, str]:
    """Create an account with a password, and sign it in. It waits for an admin to approve it.

    Joining never makes anyone an admin, even with an ADMIN_EMAILS address: nothing checks
    that people own the email they give, so anyone could otherwise claim an admin address.
    Admins are made with scripts/set_admin_password.py, on the server.
    """
    if s.login_per_ip.exceeded(client_ip):
        raise ActionError(429, TOO_MANY)
    try:
        email = normalize_email(raw_email, s.settings.allowed_email_domains)
        if email is None:
            raise ActionError(422, "Enter a valid email address.")
        problem = password_problem(password)
        if problem:
            raise ActionError(422, problem)
        admins = {normalize_email(e) for e in s.settings.admin_emails}
        if s.controls().signups_paused and email not in admins:
            raise ActionError(403, "New sign-ups are closed right now.")
        account, created = s.live.get_or_create_account(email, admin=False)
        if not created:
            raise ActionError(409, "There's already an account with that email. Sign in instead, or ask the research team to reset your password.")
    except ActionError:
        s.login_per_ip.record(client_ip)  # Only failures count, so a class can join together.
        raise
    s.live.set_password_hash(account.account_id, hash_password(password))
    return account, _new_session(s, account)


def password_sign_in(s: Services, raw_email: str, password: str, client_ip: str) -> tuple[Account, str]:
    """Sign in with an email and password.

    Every failure gives the same message, and takes about as long, so it doesn't reveal
    which emails have accounts. Wrong attempts are limited per email and per address.
    """
    email = normalize_email(raw_email, s.settings.allowed_email_domains)
    if s.login_per_ip.exceeded(client_ip) or (email is not None and s.password_per_email.exceeded(email)):
        raise ActionError(429, TOO_MANY)
    account = s.live.account_for_email(email) if email else None
    stored = s.live.password_hash(account.account_id) if account else None
    if not verify_password(password, stored) or account is None:
        # Only failures count, so people sharing a campus network aren't locked out.
        s.login_per_ip.record(client_ip)
        if email is not None:
            s.password_per_email.record(email)
        raise ActionError(401, WRONG_PASSWORD)
    return account, _new_session(s, account)


def confirm_password(s: Services, account: Account, password: str, client_ip: str) -> str:
    """Re-check the signed-in person's password, for admin step-up. Returns a fresh session."""
    if s.login_per_ip.exceeded(client_ip) or s.password_per_email.exceeded(account.email_normalized):
        raise ActionError(429, TOO_MANY)
    if not verify_password(password, s.live.password_hash(account.account_id)):
        s.login_per_ip.record(client_ip)
        s.password_per_email.record(account.email_normalized)
        raise ActionError(401, "That password isn't right.")
    return _new_session(s, account)


def change_password(s: Services, account: Account, current: str | None, new: str, keep_session_hash: str | None) -> None:
    """Change your own password. It needs the current one, unless an admin just reset it.
    Every other session for the account is signed out."""
    stored = s.live.password_hash(account.account_id)
    reset = s.live.must_change_password(account.account_id)
    if stored is not None and not reset and not verify_password(current or "", stored):
        raise ActionError(403, "Your current password is wrong.")
    problem = password_problem(new)
    if problem:
        raise ActionError(422, problem)
    s.live.set_password_hash(account.account_id, hash_password(new))
    s.live.end_other_sessions(account.account_id, keep_session_hash)
    s.live.audit(account.account_id, "password_change", account.account_id, {})


def reset_password(s: Services, admin: Account, account_id: str) -> str:
    """Give someone a temporary password, which they must change when they next sign in.
    Signs them out everywhere. Returns the temporary password, to show the admin once."""
    target = s.live.get_account(account_id)
    if target is None:
        raise ActionError(404, "Account not found")
    temporary = temporary_password()
    s.live.set_password_hash(account_id, hash_password(temporary), must_change=True)
    s.live.end_other_sessions(account_id)
    s.live.audit(admin.account_id, "password_reset", account_id, {})
    return temporary


def give_consent(s: Services, account: Account, version: str, wants_best_match: bool | None = None) -> None:
    """Record consent, and the optional best-match choice when the form asked for it."""
    if version != s.settings.consent_version:
        raise ActionError(409, f"The current consent version is {s.settings.consent_version}")
    s.live.record_consent(account.account_id, version)
    if wants_best_match is not None:
        set_best_match_choice(s, account, wants_best_match)


# Best matches
#
# After the study, the team runs its algorithm offline and uploads one best-match pair per
# person. Someone sees their best match, with that person's real name, only when both of
# them opted in, the reveal time has passed, both are still approved participants, and
# neither has blocked the other. Every one of these is checked each time the page loads,
# so opting out later, deleting an account, or a block hides the pair straight away.

BEST_MATCH_COLUMNS = ("participant_a", "participant_b")


def set_best_match_choice(s: Services, account: Account, opted_in: bool) -> None:
    if account.role == "admin":
        return  # Admins don't take part.
    if s.live.wants_best_match(account.account_id) == opted_in:
        return
    s.live.set_wants_best_match(account.account_id, opted_in)
    s.live.audit(account.account_id, "best_match_choice", account.account_id, {"opted_in": opted_in})


def results_open(s: Services) -> bool:
    shown = s.controls().results_open_at
    return shown is not None and utc_now() >= shown


def best_match_for(s: Services, account: Account) -> dict[str, Any]:
    """What this person may see about their best match.

    state is "not_opted_in", "waiting" (results aren't out), "none" (no pair they can see),
    or "revealed", which comes with the other person's profile and real name.
    """
    shown = s.controls().results_open_at
    result: dict[str, Any] = {"state": "not_opted_in", "opens_text": s.describe_time(shown) if shown else None}
    if account.user_id is None or not s.live.wants_best_match(account.account_id):
        return result
    if not results_open(s):
        return {**result, "state": "waiting"}
    other = s.live.best_match_of(account.user_id)
    other_account = s.live.account_for_user(other) if other else None
    if (
        other is None
        or other_account is None
        or other_account.role != "user"
        or other_account.status != "approved"
        or account.status != "approved"
        or not s.live.wants_best_match(other_account.account_id)
        or s.live.blocked_either_way(account.user_id, other)
    ):
        return {**result, "state": "none"}
    profile = s.public_profile(other)["profile"]
    return {**result, "state": "revealed", "user_id": other, "profile": profile, "real_name": s.live.real_name(other) or profile["name"]}


def participant_ids(s: Services) -> dict[str, str]:
    """participant_id -> user_id for every real participant, as the exports name them."""
    from app.analytics import pseudonym

    with_profiles = [a.user_id for a in s.live.list_accounts(None) if a.role == "user" and a.user_id]
    return {pseudonym(user_id, s.settings.export_salt): user_id for user_id in with_profiles}


def upload_best_matches(s: Services, admin: Account, text: str) -> int:
    """Replace the best-match pairs with a CSV of participant_a,participant_b rows, using
    the participant_ids from the ground-truth download. Returns the number of pairs.

    The whole file is refused, listing every problem, if any row is wrong: an unknown id,
    someone paired with themselves or listed twice, or someone who hasn't opted in.
    """
    import csv
    import io

    if not s.settings.export_salt:
        raise ActionError(409, "Set the EXPORT_SALT environment variable first. The pairs use the ids it produces.")
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    rows = [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]
    if not rows or tuple(cell.lower() for cell in rows[0][:2]) != BEST_MATCH_COLUMNS:
        raise ActionError(422, "The first line must be the column names: participant_a,participant_b")
    known = participant_ids(s)
    wanting = s.live.users_wanting_best_match()
    problems: list[str] = []
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for line, row in enumerate(rows[1:], start=2):
        if len(row) < 2 or not row[0] or not row[1]:
            problems.append(f"Line {line}: needs two participant ids.")
            continue
        a, b = row[0].lower(), row[1].lower()
        line_problems = [f"Line {line}: {pid} isn't a participant." for pid in (a, b) if pid not in known]
        if a == b:
            line_problems.append(f"Line {line}: {a} is paired with themselves.")
        line_problems += [f"Line {line}: {pid} is already in an earlier pair." for pid in (a, b) if pid in seen]
        line_problems += [
            f"Line {line}: {pid} hasn't asked to see their best match." for pid in (a, b) if pid in known and known[pid] not in wanting
        ]
        seen.update((a, b))
        if line_problems:
            problems += line_problems
        else:
            pairs.append((known[a], known[b]))
    if problems:
        shown = problems[:20] + ([f"...and {len(problems) - 20} more."] if len(problems) > 20 else [])
        raise ActionError(422, "Nothing was saved. Fix these and upload the file again: " + " ".join(shown), detail=problems)
    s.live.replace_best_matches(pairs)
    s.live.audit(admin.account_id, "upload_best_matches", None, {"pairs": len(pairs)})
    return len(pairs)


def clear_best_matches(s: Services, admin: Account) -> None:
    s.live.replace_best_matches([])
    s.live.audit(admin.account_id, "clear_best_matches", None, {})


def set_results_open(s: Services, admin: Account, when: datetime | None) -> None:
    shown = when.astimezone(timezone.utc).isoformat() if when else None
    s.live.set_setting("results_open_at", shown)
    s.live.audit(admin.account_id, "set_results_open", None, {"results_open_at": shown})


# Profile and preferences


def require_profile(account: Account) -> str:
    if account.user_id is None:
        raise ActionError(409, "Create your profile first")
    return account.user_id


def create_profile(s: Services, account: Account, fields: dict[str, Any]) -> str:
    """Create the caller's profile. The server sets user_id and agent, never the client."""
    if account.role == "admin":
        raise ActionError(403, "Administrators run the study and don't take part in it.")
    if account.user_id is not None:
        raise ActionError(409, "You already have a profile. Use PUT to edit it.")
    # Others see an assigned display name. The real name is kept privately.
    real_name = str(fields.get("name", "")).strip()
    if not real_name:
        raise ActionError(422, "Enter your name.")
    display = s.pick_display_name(str(fields.get("gender", "")))
    profile = s.manager.add_profile({**fields, "name": display, "agent": False})
    s.live.set_real_name(profile.user_id, real_name)
    s.live.link_profile(account.account_id, profile.user_id)
    return profile.user_id


def edit_profile(s: Services, account: Account, changes: dict[str, Any]) -> None:
    """Edit the caller's profile. A new name is their private real name. Changing gender
    gives them a new display name from that gender's list."""
    user_id = require_profile(account)
    changes = dict(changes)
    real_name = changes.pop("name", None)
    current = s.manager.get_profile(user_id)
    if "gender" in changes and changes["gender"] != current.gender:
        changes["name"] = s.pick_display_name(str(changes["gender"]))
    if changes:
        s.manager.edit_profile(user_id, **changes)
    if real_name is not None:
        if not str(real_name).strip():
            raise ActionError(422, "Enter your name.")
        s.live.set_real_name(user_id, str(real_name).strip())


def set_preferences(s: Services, account: Account, interested_in: list[str] | str) -> frozenset[str]:
    genders = GENDERS if interested_in == "everyone" else interested_in
    return s.manager.set_preferences(require_profile(account), genders).interested_in


def may_see(s: Services, account: Account, owner: str) -> bool:
    """Whether this account may see another user's profile.

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


def rating_not_open_message(s: Services) -> str | None:
    opens = s.controls().rating_opens_at
    if opens is not None and utc_now() < opens:
        return f"Rating opens on {s.describe_time(opens)}. Please come back then."
    return None


def _unavailable_message(s: Services, account: Account) -> str | None:
    if s.controls().feed_paused:
        return "Discover is paused for maintenance. Please check back later."
    if (not_open := rating_not_open_message(s)) is not None:
        return not_open
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
    """The card to show: an unrated profile already shown, if there is one, else the next.

    Reloading, or asking the API again, shows the same person, so nobody can be skipped.
    Someone shown but not rated, because they paused or were blocked meanwhile, comes back
    once they can be shown again, so everyone ends up rated.
    """
    viewer = require_profile(account)
    message = _unavailable_message(s, account)
    if message:
        return Card(None, 0, message)
    waiting = [i for i in s.live.unrated_impressions(viewer) if _still_showable(s, viewer, i.target_id)]
    if waiting:
        targets = {i.target_id for i in waiting}
        return Card(waiting[0], len(s.manager.eligible_candidates(viewer, LIVE_RUN_ID)) + len(targets))
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
    if (not_open := rating_not_open_message(s)) is not None:
        raise ActionError(403, not_open)
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
    return RatingOutcome(True, match.match_id)


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


# Matches, blocks, and reports


def own_match(s: Services, account: Account, match_id: str) -> Match:
    match = s.live.get_match(match_id)
    if match is None or account.user_id not in (match.user_a, match.user_b):
        raise ActionError(404, "Match not found")
    return match


def unmatch(s: Services, account: Account, match_id: str) -> None:
    match = own_match(s, account, match_id)
    if match.status != "active":
        raise ActionError(409, "This match has already ended")
    s.live.unmatch(match_id, account.user_id)


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
    profile = s.public_profile(user_id, own=True)["profile"] if user_id else None
    return {
        "account": {
            "email": account.email_normalized,
            "status": account.status,
            "created_at": account.created_at,
            "consent_version": account.consent_version,
            "consent_at": account.consent_at,
            "experiment_opt_out": account.experiment_opt_out,
            "paused": account.paused,
            "wants_best_match": s.live.wants_best_match(account.account_id),
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


def set_controls(
    s: Services,
    admin: Account,
    matcher_weights: dict[str, float],
    signups_paused: bool,
    feed_paused: bool,
    rating_opens_at: datetime | None = None,
) -> None:
    unknown = set(matcher_weights) - set(s.matchers)
    if unknown or any(weight < 0 for weight in matcher_weights.values()):
        raise ActionError(422, "Weights must be zero or more, for registered matchers")
    s.live.set_setting("matcher_weights", matcher_weights)
    s.live.set_setting("signups_paused", signups_paused)
    s.live.set_setting("feed_paused", feed_paused)
    opens = rating_opens_at.astimezone(timezone.utc).isoformat() if rating_opens_at else None
    s.live.set_setting("rating_opens_at", opens)
    s.live.audit(
        admin.account_id,
        "set_controls",
        None,
        {"matcher_weights": matcher_weights, "signups_paused": signups_paused, "feed_paused": feed_paused, "rating_opens_at": opens},
    )

