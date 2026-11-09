import csv
import io
from html import unescape
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import actions
from app import matchers as matcher_registry
from app.config import Settings
from app.emails import ConsoleEmailSender
from app.main import create_app
from live_store import to_text, utc_now
from matching import Matcher, PublicView
from models import LIVE_RUN_ID
from raters import build_rater_input
from test_api import ADMIN_EMAIL, SITE, Env, jpeg, profile_body

HTML = {"Accept": "text/html"}


def make_env(tmp_path: Path, **overrides: Any) -> Env:
    settings = Settings(
        site_url=SITE,
        admin_emails=(ADMIN_EMAIL,),
        main_db=tmp_path / "main.db",
        persona_db=tmp_path / "personas.db",
        photo_dir=tmp_path / "photos",
        cookie_secure=False,
        login_requests_per_email=50,
        login_requests_per_ip=10_000,
        export_salt="test-salt",
        **overrides,
    )
    sender = ConsoleEmailSender(stream=None)
    return Env(create_app(settings, email_sender=sender, rng=random.Random(1)), settings, sender)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return make_env(tmp_path)


def page(env: Env, api_client: TestClient | None = None) -> TestClient:
    """A browser-like client: asks for HTML, and doesn't follow redirects."""
    client = TestClient(env.app, headers=HTML, follow_redirects=False)
    if api_client is not None:
        client.cookies.update(api_client.cookies)
    return client


def csrf_of(html: str) -> str:
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def htmx(client: TestClient, html: str) -> dict[str, str]:
    return {"HX-Request": "true", "X-CSRF-Token": csrf_of(html)}


def member(env: Env, email: str, gender: str = "man", interested_in: Any = ("woman",), name: str = "Sam Smith", **kw: Any) -> tuple[TestClient, str]:
    api, user_id = env.user(email, gender, interested_in, name=name, **kw)
    return page(env, api), user_id


def impression_of(env: Env, viewer_id: str):
    return env.s.live.pending_impression(viewer_id)


# Access


@pytest.mark.parametrize("path", ["/discover", "/matches", "/me", "/admin", "/onboarding/profile"])
def test_signed_out_visitors_are_sent_to_sign_in(env: Env, path: str) -> None:
    response = page(env).get(path)
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_onboarding_order_and_pending_lock(env: Env) -> None:
    api = env.login("new@bu.edu")
    browser = page(env, api)
    for path in ("/discover", "/matches", "/me", "/onboarding/profile"):
        assert browser.get(path).headers["location"] == "/onboarding/consent"
    csrf = csrf_of(browser.get("/onboarding/consent").text)
    assert browser.post("/onboarding/consent", data={"csrf_token": csrf, "version": "1"}).status_code == 422
    assert browser.post("/onboarding/consent", data={"csrf_token": csrf, "version": "1", "agree": "yes"}).headers["location"] == "/onboarding/profile"
    assert browser.get("/discover").headers["location"] == "/onboarding/profile"
    form = {**{k: v for k, v in profile_body("Jane Doe", "woman").items() if not isinstance(v, list)},
            "interests": "Hiking,  chess", "clubs": "", "csrf_token": csrf, "religion": "  "}
    form["age"] = str(form["age"]); form["height_cm"] = str(form["height_cm"])
    assert browser.post("/onboarding/profile", data=form).headers["location"] == "/onboarding/preferences"
    me = api.get("/me").json()["profile"]
    assert me["interests"] == ["hiking", "chess"]
    assert me["religion"] is None
    assert browser.get("/discover").headers["location"] == "/onboarding/preferences"
    browser.post("/onboarding/preferences", data={"csrf_token": csrf, "interested_in": "everyone"})
    assert browser.get("/discover").headers["location"] == "/onboarding/photos"
    browser.post("/onboarding/photos", data={"csrf_token": csrf}, files={"file": ("a.jpg", jpeg(), "image/jpeg")})
    assert browser.get("/discover").headers["location"] == "/pending"
    pending = browser.get("/pending")
    assert "Your profile is under review." in pending.text
    assert 'href="/discover"' not in pending.text


def test_profile_errors_show_inline_in_plain_words(env: Env) -> None:
    browser = page(env, env.consented("jane@bu.edu"))
    csrf = csrf_of(browser.get("/onboarding/profile").text)
    form = {**{k: str(v) for k, v in profile_body().items() if not isinstance(v, list)}, "interests": "", "clubs": "", "csrf_token": csrf, "age": "17", "height_cm": "abc"}
    response = browser.post("/onboarding/profile", data=form)
    assert response.status_code == 422
    assert 'id="age-error"' in response.text and "18" in response.text
    assert 'id="height_cm-error"' in response.text and "Enter a whole number." in response.text
    assert 'id="interests-error"' in response.text


# Discover


def test_discover_card_has_no_matcher_no_default_and_no_engagement_tricks(env: Env) -> None:
    browser, viewer = member(env, "v@bu.edu")
    member(env, "w@bu.edu", "woman", ("man",), name="Jane Doe")
    html = browser.get("/discover").text
    impression = impression_of(env, viewer)
    assert impression.matcher not in re.sub(r"<[^>]+>", " ", html).split() and "matcher" not in html.lower()
    slider = re.search(r'<input id="score-slider"[^>]*>', html).group(0)
    assert "value=" not in slider
    number = re.search(r'<input id="score"[^>]*>', html).group(0)
    assert "value=" not in number
    assert re.search(r'<button type="submit" disabled>Submit rating</button>', html)
    # Visible text and button labels, apart from the accessibility "Skip to content" link.
    visible = re.sub(r"<[^>]+>", " ", html).lower().replace("skip to content", "")
    for word in ("skip", "streak", "badge", "liked you", "online", "typing"):
        assert word not in visible
    assert "1 person left for you right now." in html
    assert "Jane" in html and "Doe" not in html


def test_discover_shows_people_left_and_a_calm_end_state(env: Env) -> None:
    browser, _ = member(env, "v@bu.edu")
    for i in range(3):
        member(env, f"w{i}@bu.edu", "woman", ("man",))
    html = browser.get("/discover").text
    assert "3 people left for you right now." in html
    for left in (2, 1):
        html = browser.post("/discover/rate", data={"csrf_token": csrf_of(html), "impression_id": re.search(r'name="impression_id" value="([^"]+)"', html).group(1), "score": "5"}, headers={"HX-Request": "true"}).text
        assert f"{left} {'person' if left == 1 else 'people'} left for you right now." in html
    html = browser.post("/discover/rate", data={"csrf_token": csrf_of(html), "impression_id": re.search(r'name="impression_id" value="([^"]+)"', html).group(1), "score": "5"}, headers={"HX-Request": "true"}).text
    assert "You've seen everyone available to you right now. New people may join." in unescape(html)
    assert "come back" not in html.lower()


def test_reloading_shows_the_same_person_so_it_cannot_skip(env: Env) -> None:
    browser, viewer = member(env, "v@bu.edu")
    member(env, "a@bu.edu", "woman", ("man",))
    member(env, "b@bu.edu", "woman", ("man",))
    first = re.search(r'name="impression_id" value="([^"]+)"', browser.get("/discover").text).group(1)
    second = re.search(r'name="impression_id" value="([^"]+)"', browser.get("/discover").text).group(1)
    assert first == second


def test_rating_with_htmx_saves_what_the_impression_says(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(matcher_registry.REGISTRY, "scored", ScoringMatcher)
    env = make_env(tmp_path / "scored", enabled_matchers=("scored",))
    browser, viewer = member(env, "v@bu.edu")
    _, target = member(env, "w@bu.edu", "woman", ("man",))
    member(env, "x@bu.edu", "woman", ("man",))
    html = browser.get("/discover").text
    impression = impression_of(env, viewer)
    assert (impression.matcher, impression.position, impression.predicted_score) == ("scored", 1, 0.75)
    assert set(env.s.live.candidate_scores(impression.impression_id)) == set(env.s.manager.eligible_candidates(viewer)) | {impression.target_id}
    response = browser.post("/discover/rate", data={"impression_id": impression.impression_id, "score": "8.3"}, headers=htmx(browser, html))
    assert response.status_code == 200
    (rating,) = env.s.manager.ratings_by_rater(viewer, LIVE_RUN_ID)
    assert (rating.target_id, rating.matcher, round(rating.score, 4)) == (impression.target_id, "scored", round(7.3 / 9, 4))
    assert impression_of(env, viewer).position == 2  # The next card is the viewer's second impression.


class ScoringMatcher(Matcher):
    name = "scored"
    description = "Picks the first eligible person and scores everyone 0.75."

    def __init__(self, rng: random.Random) -> None:
        pass

    def score_candidates(self, user_id: str, view: PublicView, candidates: list[str]) -> dict[str, float]:
        return {c: 0.75 for c in candidates}

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        pool = view.eligible_candidates(user_id)
        return pool[0] if pool else None


def test_rating_needs_csrf_and_the_callers_own_impression(env: Env) -> None:
    browser, viewer = member(env, "v@bu.edu")
    other, other_id = member(env, "o@bu.edu")
    member(env, "w@bu.edu", "woman", ("man",))
    html = browser.get("/discover").text
    impression = impression_of(env, viewer)
    bad = browser.post("/discover/rate", data={"impression_id": impression.impression_id, "score": "5"}, headers={"HX-Request": "true", "X-CSRF-Token": "wrong"})
    assert bad.status_code == 403
    foreign_origin = browser.post("/discover/rate", data={"impression_id": impression.impression_id, "score": "5"}, headers={**htmx(browser, html), "Origin": "https://evil.example"})
    assert foreign_origin.status_code == 403
    other_html = other.get("/discover").text
    stolen = other.post("/discover/rate", data={"impression_id": impression.impression_id, "score": "5"}, headers=htmx(other, other_html))
    assert "Impression not found" in stolen.text
    assert env.s.manager.ratings_by_rater(viewer, LIVE_RUN_ID) == []
    assert env.s.manager.ratings_by_rater(other_id, LIVE_RUN_ID) == []


# Surveys


def test_session_survey_at_exhaustion_once_per_day(env: Env) -> None:
    browser, viewer = member(env, "v@bu.edu")
    html = browser.get("/discover").text
    assert "Was this worth your time?" in html  # Nobody is available, so the pool is exhausted.
    assert "Was this worth your time?" not in browser.get("/discover").text
    with sqlite3.connect(env.settings.main_db) as conn:
        conn.execute("UPDATE survey_prompts SET shown_at = '2000-01-01T00:00:00.000000+00:00'")
    html = browser.get("/discover").text
    assert "Was this worth your time?" in html
    assert browser.post("/survey", data={"kind": "session", "score": "4"}, headers=htmx(browser, html)).status_code == 200
    with sqlite3.connect(env.settings.main_db) as conn:
        assert conn.execute("SELECT kind, score FROM surveys").fetchall() == [("session", 4)]


def test_session_survey_after_every_25th_rating(env: Env) -> None:
    browser, viewer = member(env, "v@bu.edu")
    for i in range(27):
        member(env, f"w{i}@bu.edu", "woman", ("man",))
    html = browser.get("/discover").text
    for n in range(1, 27):
        assert ("Was this worth your time?" in html) == (n == 26), n  # After the 25th rating.
        html = browser.post("/discover/rate", data={"impression_id": impression_of(env, viewer).impression_id, "score": "5"}, headers=htmx(browser, html)).text


def test_post_match_survey_only_after_three_days_and_once(env: Env) -> None:
    a, a_id = member(env, "a@bu.edu")
    _, b_id = member(env, "b@bu.edu", "woman", ("man",))
    match = env.s.live.create_match(a_id, b_id)
    assert "Was this match worth your time?" not in a.get(f"/matches/{match.match_id}").text
    with sqlite3.connect(env.settings.main_db) as conn:
        conn.execute("UPDATE matches SET created_at = ?", (to_text(utc_now().replace(year=2000)),))
    assert "Was this match worth your time?" in a.get(f"/matches/{match.match_id}").text
    assert "Was this match worth your time?" not in a.get(f"/matches/{match.match_id}").text


# Matches, chat, names, and religion


def matched(env: Env) -> tuple[TestClient, str, TestClient, str, str]:
    a, a_id = member(env, "a@bu.edu", name="Alex Smith")
    b, b_id = member(env, "b@bu.edu", "woman", ("man",), name="Bea Jones")
    for client in (a, b):
        html = client.get("/discover").text
        user = a_id if client is a else b_id
        client.post("/discover/rate", data={"impression_id": impression_of(env, user).impression_id, "score": "9"}, headers=htmx(client, html))
    return a, a_id, b, b_id, env.s.live.active_match_between(a_id, b_id).match_id


def test_first_names_only_and_religion_only_when_given(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched(env)
    env.s.manager.edit_profile(b_id, religion=None)
    for html in (a.get("/matches").text, a.get(f"/matches/{match_id}").text):
        assert "Bea" in html and "Jones" not in html
    c, _ = member(env, "c@bu.edu")
    html = c.get("/discover").text
    assert "Religion" not in html
    env.s.manager.edit_profile(b_id, religion="buddhist")
    d, _ = member(env, "d@bu.edu")
    assert "buddhist" in d.get("/discover").text


def test_chat_polls_every_5_seconds_and_sends_with_htmx(env: Env) -> None:
    a, _, b, _, match_id = matched(env)
    html = a.get(f"/matches/{match_id}").text
    assert 'hx-trigger="every 5s"' in html and 'maxlength="2000"' in html
    sent = a.post(f"/matches/{match_id}/send", data={"body": "<b>hi</b>"}, headers=htmx(a, html))
    assert "&lt;b&gt;hi&lt;/b&gt;" in sent.text
    assert "&lt;b&gt;hi&lt;/b&gt;" in b.get(f"/matches/{match_id}/log", headers={"HX-Request": "true"}).text
    for word in ("read", "seen", "typing", "online"):
        assert f">{word}" not in html.lower()


def test_match_email_once_per_participant_with_no_names(env: Env) -> None:
    a, a_id = member(env, "a@bu.edu", name="Alex Smith")
    b, b_id = member(env, "b@bu.edu", "woman", ("man",), name="Bea Jones")
    env.s.live.set_flags(env.s.live.account_for_user(b_id).account_id, notify_on_match=False)
    before = len(env.sender.outbox)
    for client, user in ((a, a_id), (b, b_id)):
        html = client.get("/discover").text
        client.post("/discover/rate", data={"impression_id": impression_of(env, user).impression_id, "score": "9"}, headers=htmx(client, html))
    new = env.sender.outbox[before:]
    assert [m["to"] for m in new] == ["a@bu.edu"]
    assert new[0]["subject"] == "You have a new match"
    for secret in ("Alex", "Bea", "Smith", "Jones", "/photos/"):
        assert secret not in new[0]["body"]
    assert {m["subject"] for m in env.sender.outbox} <= {"Your sign-in link", "You have a new match"}


# Account settings


def test_opted_out_users_get_the_default_matcher_and_leave_analytics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(matcher_registry.REGISTRY, "scored", ScoringMatcher)
    env = make_env(tmp_path, enabled_matchers=("scored",), default_matcher="random")
    browser, viewer = member(env, "v@bu.edu")
    others = [member(env, f"w{i}@bu.edu", "woman", ("man",))[1] for i in range(3)]
    html = browser.get("/me").text
    browser.post("/me/settings", data={"csrf_token": csrf_of(html), "experiment_opt_out": "yes", "notify_on_match": "yes"})
    for _ in range(3):
        html = browser.get("/discover").text
        browser.post("/discover/rate", data={"impression_id": impression_of(env, viewer).impression_id, "score": "5"}, headers=htmx(browser, html))
    assert {r.matcher for r in env.s.manager.ratings_by_rater(viewer, LIVE_RUN_ID)} == {"random"}
    from app import analytics

    assert analytics.load_rows(env.settings.main_db) == []
    assert analytics.export_rows(env.settings.main_db, "salt") == []


def test_paused_users_leave_pools_and_can_unpause(env: Env) -> None:
    viewer, viewer_id = member(env, "v@bu.edu")
    paused, paused_id = member(env, "p@bu.edu", "woman", ("man",))
    html = paused.get("/me").text
    paused.post("/me/pause", data={"csrf_token": csrf_of(html)})
    assert paused_id not in env.s.manager.eligible_candidates(viewer_id)
    assert "Your account is paused" in paused.get("/discover").text
    paused.post("/me/unpause", data={"csrf_token": csrf_of(html)})
    assert paused_id in env.s.manager.eligible_candidates(viewer_id)


def test_download_my_data_is_only_the_callers(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched(env)
    html = a.get(f"/matches/{match_id}").text
    a.post(f"/matches/{match_id}/send", data={"body": "from a"}, headers=htmx(a, html))
    html = b.get(f"/matches/{match_id}").text
    b.post(f"/matches/{match_id}/send", data={"body": "from b"}, headers=htmx(b, html))
    data = a.get("/me/data.json").json()
    assert data["account"]["email"] == "a@bu.edu"
    assert data["profile"]["name"] == "Alex Smith"
    assert [m["body"] for m in data["messages_sent"]] == ["from a"]
    assert [r["target_id"] for r in data["ratings"]] == [b_id]
    text = json.dumps(data)
    assert "b@bu.edu" not in text and "Bea" not in text and "from b" not in text


def test_delete_my_account_from_the_page_cascades(env: Env) -> None:
    a, a_id, _, _, match_id = matched(env)
    confirm = a.get("/me/delete")
    assert "can't be undone" in unescape(confirm.text)
    response = a.post("/me/delete", data={"csrf_token": csrf_of(confirm.text)})
    assert response.headers["location"] == "/"
    assert not env.s.manager.store.profile_exists(a_id)
    assert env.s.live.get_match(match_id) is None


# Admin


def admin_page(env: Env) -> TestClient:
    return page(env, env.admin())


def test_non_admins_get_403_on_admin_pages(env: Env) -> None:
    browser, _ = member(env, "v@bu.edu")
    for path in ("/admin", "/admin/approvals", "/admin/compare", "/admin/export.csv", "/admin/audit"):
        assert browser.get(path).status_code == 403


def test_stale_admin_sign_in_must_step_up(env: Env) -> None:
    admin = admin_page(env)
    assert admin.get("/admin").status_code == 200
    with sqlite3.connect(env.settings.main_db) as conn:
        conn.execute("UPDATE sessions SET created_at = '2000-01-01T00:00:00.000000+00:00'")
    assert admin.get("/admin/approvals").headers["location"] == "/admin/step-up"
    assert TestClient(env.app, cookies=dict(admin.cookies)).get("/admin/accounts").status_code == 403
    html = admin.get("/admin/step-up").text
    assert admin.post("/admin/step-up", data={"csrf_token": csrf_of(html)}).headers["location"] == "/login/sent"


def test_viewing_a_reported_conversation_needs_confirmation_and_is_audited(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched(env)
    html = a.get(f"/matches/{match_id}").text
    a.post(f"/matches/{match_id}/send", data={"body": "PRIVATE"}, headers=htmx(a, html))
    form = b.get(f"/people/{a_id}/report?match_id={match_id}").text
    b.post(f"/people/{a_id}/report", data={"csrf_token": csrf_of(form), "reason": "harassment", "details": "rude", "match_id": match_id})
    report_id = env.s.live.list_reports("open")[0].report_id
    admin = admin_page(env)
    listing = admin.get("/admin/reports").text
    assert "PRIVATE" not in listing
    confirm = admin.get(f"/admin/reports/{report_id}/conversation").text
    assert "PRIVATE" not in confirm and "will be logged" in confirm
    before = [e for e in env.s.live.audit_entries() if e["action"] == "read_report_messages"]
    shown = admin.post(f"/admin/reports/{report_id}/conversation", data={"csrf_token": csrf_of(confirm)})
    assert "PRIVATE" in shown.text
    assert len([e for e in env.s.live.audit_entries() if e["action"] == "read_report_messages"]) == len(before) + 1


def test_no_admin_route_lists_messages_or_ratings(env: Env) -> None:
    admin_paths = sorted({getattr(r, "path", "") for r in env.app.router.routes if "admin" in getattr(r, "path", "")})
    for path in admin_paths:
        assert "message" not in path or path.endswith("/reports/{report_id}/messages"), path
        assert "rating" not in path, path
        assert "conversation" not in path or "{report_id}" in path, path


def test_admin_pages_render_and_controls_are_audited(env: Env) -> None:
    a, *_ = matched(env)
    admin = admin_page(env)
    for path in ("/admin", "/admin/approvals", "/admin/reports", "/admin/compare?k=5", "/admin/controls", "/admin/audit?action=account_approve"):
        response = admin.get(path)
        assert response.status_code == 200, path
    html = admin.get("/admin/controls").text
    assert "Locked for experiment integrity" in html
    admin.post("/admin/controls", data={"csrf_token": csrf_of(html), "enabled_random": "yes", "weight_random": "2", "feed_paused": "yes"})
    assert env.s.controls().feed_paused
    assert "paused for maintenance" in a.get("/discover").text
    assert any(e["action"] == "set_controls" for e in env.s.live.audit_entries())


def test_export_is_pseudonymized_stable_and_audited(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched(env)
    admin = admin_page(env)
    first = admin.get("/admin/export.csv")
    assert first.status_code == 200
    rows = list(csv.DictReader(io.StringIO(first.text)))
    assert len(rows) == 2
    text = first.text + admin.get("/admin/export.json").text
    for secret in ("@bu.edu", "Alex", "Bea", a_id, b_id, "hiking", "religion", "/photos/"):
        assert secret not in text, secret
    assert all(r["viewer_age_band"] == "21-23" for r in rows)
    assert {r["produced_match"] for r in rows} == {"True"}
    again = list(csv.DictReader(io.StringIO(admin.get("/admin/export.csv").text)))
    assert sorted(r["viewer"] for r in rows) == sorted(r["viewer"] for r in again)
    assert [e["details"] for e in env.s.live.audit_entries(action="export")][0] == '{"rows": 2}'


# Pages in general


def test_pages_have_strict_headers_and_no_external_assets(env: Env) -> None:
    browser, _ = member(env, "v@bu.edu")
    for path in ("/", "/transparency", "/discover", "/matches", "/me"):
        response = (page(env) if path in ("/", "/transparency") else browser).get(path)
        assert response.status_code == 200, path
        headers = response.headers
        assert "script-src 'self'" in headers["content-security-policy"]
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["cache-control"] == "no-store"
        assert headers["x-content-type-options"] == "nosniff"
        assert not re.search(r'(src|href)="(https?:)?//', response.text.replace(SITE, "")), path
        assert "<script>" not in response.text and "style=" not in response.text
    assert page(env).get("/static/app.css").headers["cache-control"].startswith("public")


def test_transparency_lists_algorithms_and_design_rules(env: Env) -> None:
    html = page(env).get("/transparency").text
    assert "Shows you someone at random" in html
    assert "No skip button" in html and "No ads and no tracking scripts" in html


def test_friendly_error_pages_leak_nothing(env: Env) -> None:
    response = page(env).get("/no/such/page")
    assert response.status_code == 404 and "Page not found" in response.text and "Traceback" not in response.text


def test_religion_none_is_left_out_of_rater_input(env: Env) -> None:
    rater = env.s.manager.add_profile({**profile_body("A"), "religion": None, "agent": True})
    target = env.s.manager.add_profile({**profile_body("B"), "religion": "catholic", "agent": True})
    persona = env.s.manager.add_persona(rater.user_id, "Likes reading.")
    rater_input = build_rater_input(rater, persona, target)
    assert "religion" not in rater_input["rater_profile"]
    assert rater_input["target_profile"]["religion"] == "catholic"
    assert rater_input["rater_profile"]["gender"] == "man"
