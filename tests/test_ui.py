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
from app.main import create_app
from live_store import to_text, utc_now
from matching import Matcher, PublicView
from models import LIVE_RUN_ID
from raters import build_rater_input
from test_api import ADMIN_EMAIL, PASSWORD, SITE, Env, profile_body, shown_ids

HTML = {"Accept": "text/html"}


def make_env(tmp_path: Path, **overrides: Any) -> Env:
    settings = Settings(
        site_url=SITE,
        admin_emails=(ADMIN_EMAIL,),
        main_db=tmp_path / "main.db",
        persona_db=tmp_path / "personas.db",
        cookie_secure=False,
        login_requests_per_ip=10_000,
        export_salt="test-salt",
        **overrides,
    )
    return Env(create_app(settings, rng=random.Random(1)), settings)


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
    assert browser.post("/onboarding/consent", data={"csrf_token": csrf, "version": "3"}).status_code == 422
    assert browser.post("/onboarding/consent", data={"csrf_token": csrf, "version": "3", "agree": "yes"}).headers["location"] == "/onboarding/profile"
    assert browser.get("/discover").headers["location"] == "/onboarding/profile"
    form = {**{k: v for k, v in profile_body("Jane Doe", "woman").items() if not isinstance(v, list)},
            "interests": "Hiking,  chess", "clubs": "", "csrf_token": csrf, "religion": "  "}
    form["age"] = str(form["age"]); form["height_cm"] = str(form["height_cm"]); form["height_unit"] = "metric"
    assert browser.post("/onboarding/profile", data=form).headers["location"] == "/onboarding/preferences"
    me = api.get("/me").json()["profile"]
    assert me["interests"] == ["hiking", "chess"]
    assert me["religion"] is None
    assert browser.get("/discover").headers["location"] == "/onboarding/preferences"
    saved = browser.post("/onboarding/preferences", data={"csrf_token": csrf, "interested_in": "everyone"})
    assert saved.headers["location"] == "/pending"
    assert browser.get("/discover").headers["location"] == "/pending"
    pending = browser.get("/pending")
    assert "Your profile is under review." in pending.text
    assert 'href="/discover"' not in pending.text


def test_profile_errors_show_inline_in_plain_words(env: Env) -> None:
    browser = page(env, env.consented("jane@bu.edu"))
    csrf = csrf_of(browser.get("/onboarding/profile").text)
    form = {**{k: str(v) for k, v in profile_body().items() if not isinstance(v, list)}, "interests": "", "clubs": "", "csrf_token": csrf, "age": "17", "height_unit": "metric", "height_cm": "abc"}
    response = browser.post("/onboarding/profile", data=form)
    assert response.status_code == 422
    assert 'id="age-error"' in response.text and "18" in response.text
    assert 'id="height_cm-error"' in response.text and "whole centimeters" in response.text
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
    shown_as = env.s.manager.get_profile(impression.target_id).name
    assert shown_as in html and "Jane" not in html and "Doe" not in html


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
    env.s.live.create_match(a_id, b_id)
    question = f"Was your match with {env.s.manager.get_profile(b_id).name} worth your time?"
    assert question not in a.get("/matches").text
    with sqlite3.connect(env.settings.main_db) as conn:
        conn.execute("UPDATE matches SET created_at = ?", (to_text(utc_now().replace(year=2000)),))
    assert question in a.get("/matches").text
    assert question not in a.get("/matches").text


# Matches, names, and religion


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
    html = a.get("/matches").text
    assert env.s.manager.get_profile(b_id).name in html and "Bea" not in html and "Jones" not in html
    for user_id in (a_id, b_id):
        env.s.manager.edit_profile(user_id, religion=None)
    c, c_id = member(env, "c@bu.edu")
    html = c.get("/discover").text
    assert "Religion" not in html
    shown = impression_of(env, c_id).target_id
    env.s.manager.edit_profile(shown, religion="buddhist")
    assert "buddhist" in c.get("/discover").text  # The same card again, now with religion.


def test_the_app_never_sends_email(env: Env) -> None:
    import app.services as services_module

    assert not hasattr(env.s, "send_email") and not hasattr(env.s, "email_sender")
    assert "smtplib" not in open(services_module.__file__).read()


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
    data = a.get("/me/data.json").json()
    assert data["account"]["email"] == "a@bu.edu"
    assert data["profile"]["name"] == "Alex Smith"
    assert "messages_sent" not in data
    assert [r["target_id"] for r in data["ratings"]] == [b_id]
    text = json.dumps(data)
    assert "b@bu.edu" not in text and "Bea" not in text


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
    for path in ("/admin", "/admin/approvals", "/admin/compare", "/admin/users", "/admin/export.csv", "/admin/audit"):
        assert browser.get(path).status_code == 403


def test_stale_admin_sign_in_must_step_up(env: Env) -> None:
    admin = admin_page(env)
    assert admin.get("/admin").status_code == 200
    with sqlite3.connect(env.settings.main_db) as conn:
        conn.execute("UPDATE sessions SET created_at = '2000-01-01T00:00:00.000000+00:00'")
    assert admin.get("/admin/approvals").headers["location"] == "/admin/step-up"
    assert TestClient(env.app, cookies=dict(admin.cookies)).get("/admin/accounts").status_code == 403
    html = admin.get("/admin/step-up").text
    assert admin.post("/admin/step-up", data={"csrf_token": csrf_of(html), "password": "wrong"}).status_code == 401
    confirmed = admin.post("/admin/step-up", data={"csrf_token": csrf_of(html), "password": PASSWORD})
    assert confirmed.headers["location"] == "/admin"
    admin.cookies.update(confirmed.cookies)
    assert admin.get("/admin/approvals").status_code == 200


def test_reports_show_no_conversation_and_admin_routes_list_no_ratings(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched(env)
    form = b.get(f"/people/{a_id}/report?match_id={match_id}").text
    b.post(f"/people/{a_id}/report", data={"csrf_token": csrf_of(form), "reason": "harassment", "details": "rude", "match_id": match_id})
    report_id = env.s.live.list_reports("open")[0].report_id
    admin = admin_page(env)
    assert "rude" in admin.get("/admin/reports").text
    assert admin.get(f"/admin/reports/{report_id}/conversation").status_code == 404
    admin_paths = {getattr(r, "path", "") for r in env.app.router.routes if "admin" in getattr(r, "path", "")}
    for path in admin_paths:
        for word in ("message", "conversation", "rating", "photo"):
            assert word not in path, path


def test_no_page_mentions_photos_or_chat(env: Env) -> None:
    a, *_ = matched(env)
    for path in ("/discover", "/matches", "/me", "/transparency"):
        html = a.get(path).text.lower()
        for word in ("photo", "chat", "/send", "messages"):
            assert word not in html, (path, word)


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
    for secret in ("@bu.edu", "Alex", "Bea", a_id, b_id, "hiking", "religion"):
        assert secret not in text, secret
    assert all(r["viewer_age_band"] == "21-23" for r in rows)
    assert {r["produced_match"] for r in rows} == {"True"}
    again = list(csv.DictReader(io.StringIO(admin.get("/admin/export.csv").text)))
    assert sorted(r["viewer"] for r in rows) == sorted(r["viewer"] for r in again)
    assert [e["details"] for e in env.s.live.audit_entries(action="export")][0] == '{"rows": 2}'
    # Scores are exported on the 0 to 1 scale they're stored on: a 9 is (9 - 1) / 9.
    assert {r["score"] for r in rows} == {"0.8889"}
    assert "rating" not in rows[0]


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


def test_admin_sets_when_rating_opens_and_discover_waits_for_it(env: Env) -> None:
    browser, _ = member(env, "v@bu.edu")
    member(env, "w@bu.edu", "woman", ("man",))
    admin = admin_page(env)
    html = admin.get("/admin/controls").text
    form = {"csrf_token": csrf_of(html), "enabled_random": "yes", "weight_random": "1", "rating_opens_at": "2999-01-04T09:30"}
    assert admin.post("/admin/controls", data=form).status_code == 303
    assert "January 4 at 9:30 AM" in admin.get("/admin/controls").text
    page_html = browser.get("/discover").text
    assert "Rating opens on " in page_html and "January 4 at 9:30 AM" in page_html
    assert 'name="score"' not in page_html
    assert admin.post("/admin/controls", data={**form, "rating_opens_at": "not a date"}).status_code == 422
    assert admin.post("/admin/controls", data={**form, "rating_opens_at": ""}).status_code == 303
    assert 'name="score"' in browser.get("/discover").text


def test_users_page_shows_codes_not_names_leaves_out_opted_out_and_is_audited(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched(env)
    c, c_id = member(env, "c@bu.edu", "woman", ("man",), name="Cora Quinn")
    env.s.live.set_flags(env.s.live.account_for_user(c_id).account_id, experiment_opt_out=True)
    admin = admin_page(env)
    response = admin.get("/admin/users")
    assert response.status_code == 200
    html = unescape(response.text)
    for secret in ("Alex", "Bea", "Cora", "Smith", "Jones", "@bu.edu", a_id, b_id, c_id):
        assert secret not in html, secret
    interest = json.loads(re.search(r'data-heatmap="([^"]+)" data-kind="interest"', response.text).group(1).replace("&#34;", '"').replace("&quot;", '"'))
    assert sorted(interest["labels"]) == ["M01", "W01"]  # Cora opted out, so she isn't here.
    assert {v for row in interest["values"] for v in row} == {None, 0.889}
    reciprocal = json.loads(unescape(re.search(r'data-heatmap="([^"]+)" data-kind="reciprocal"', response.text).group(1)))
    assert {v for row in reciprocal["values"] for v in row} == {None, 0.79}  # 0.889 × 0.889
    assert "Pairs who both liked each other</dt><dd>1 " in response.text
    assert "How these numbers are calculated" in html and "Pearson correlation" in html and "η²" in html
    entries = env.s.live.audit_entries(action="view_user_insights")
    assert entries and entries[-1]["details"] == '{"people": 2, "ratings": 2}'


def test_admins_skip_consent_and_never_fill_in_a_profile(env: Env) -> None:
    api = env.login(ADMIN_EMAIL)
    browser = page(env, api)
    signed_in = browser.post("/login", data={"email": ADMIN_EMAIL, "password": PASSWORD})
    assert signed_in.headers["location"] == "/admin"
    for path in ("/onboarding/consent", "/onboarding/profile", "/onboarding/preferences", "/pending", "/discover", "/me"):
        response = browser.get(path)
        assert response.status_code == 303 and response.headers["location"] == "/admin", path
    assert browser.get("/admin/users").status_code == 200
    assert api.post("/me/profile", json=profile_body()).status_code == 403
    assert api.get("/feed/next").status_code == 403


def test_an_admin_with_an_old_profile_is_not_a_participant(env: Env) -> None:
    """Someone who joined as a participant and was later made an admin keeps their profile
    row, but nobody is shown it and it's left out of every result."""
    member(env, "a@bu.edu", "man", ("woman",))
    old_client, old_id = member(env, "later-admin@bu.edu", "woman", ("man",), name="Dana Old")
    a_client = env.login("a@bu.edu")
    env.s.live.make_admin(env.s.live.account_for_user(old_id).account_id, "test")
    assert old_id not in shown_ids(a_client)
    admin = admin_page(env)
    assert admin.get("/admin/users").status_code == 200
    from app.insights import load
    insights = load(env.settings.main_db, env.settings.positive_threshold)
    assert old_id not in {p.user_id for p in insights.people}
    approvals = admin.get("/admin/approvals").text
    assert "later-admin@bu.edu" not in approvals and ADMIN_EMAIL not in approvals


def test_users_page_with_nobody_yet(env: Env) -> None:
    html = admin_page(env).get("/admin/users").text
    assert "Nobody's here yet" in html


def test_ground_truth_download_matches_the_export_ids_and_leaves_out_names(env: Env) -> None:
    import zipfile

    a, a_id, b, b_id, match_id = matched(env)
    _, c_id = member(env, "c@bu.edu", "woman", ("man",), name="Cora Quinn")
    env.s.live.set_flags(env.s.live.account_for_user(c_id).account_id, experiment_opt_out=True)
    admin = admin_page(env)
    assert 'href="/admin/truth.zip"' in admin.get("/admin/users").text
    response = admin.get("/admin/truth.zip")
    assert response.status_code == 200 and response.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    files = {name.split("/")[-1]: archive.read(name).decode() for name in archive.namelist()}
    assert set(files) == {"participants.csv", "scores.csv", "matrix.csv", "README.txt"}
    for text in files.values():
        for secret in ("Alex", "Bea", "Cora", "@bu.edu", a_id, b_id, c_id):
            assert secret not in text, secret
    scores = list(csv.DictReader(io.StringIO(files["scores.csv"])))
    assert {r["score"] for r in scores} == {"0.888889"}  # Both rated 9: (9 - 1) / 9.
    export_ids = {r["viewer"] for r in csv.DictReader(io.StringIO(admin.get("/admin/export.csv").text))}
    assert {r["rater_id"] for r in scores} == export_ids  # The same ids as the impressions export.
    matrix = list(csv.reader(io.StringIO(files["matrix.csv"])))
    assert len(matrix) == 3 and len(matrix[0]) == 3  # Two people; the opted-out one is left out.
    assert env.s.live.audit_entries(action="export")[-2]["target"] == "truth"


def test_export_truth_script_saves_a_dated_snapshot(env: Env, tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    matched(env)
    folder = tmp_path / "snapshot"
    environment = {**os.environ, "APP_MAIN_DB": str(env.settings.main_db), "APP_PERSONA_DB": str(env.settings.persona_db), "EXPORT_SALT": "test-salt"}
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, str(root / "scripts" / "export_truth.py"), str(folder)], env=environment, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in folder.iterdir()) == ["README.txt", "matrix.csv", "participants.csv", "scores.csv"]
    assert "2 participants, 2 ratings" in done.stdout
    again = subprocess.run([sys.executable, str(root / "scripts" / "export_truth.py"), str(folder)], env=environment, capture_output=True, text=True)
    assert again.returncode != 0  # It never overwrites a snapshot.
    missing_salt = subprocess.run([sys.executable, str(root / "scripts" / "export_truth.py"), str(tmp_path / "x")], env={**environment, "EXPORT_SALT": ""}, capture_output=True, text=True)
    assert missing_salt.returncode != 0 and "EXPORT_SALT" in missing_salt.stderr


# Display names


def test_everyone_gets_a_display_name_from_their_genders_list_and_keeps_their_real_name_private(env: Env) -> None:
    from app.fake_names import MEN, WOMEN

    a, a_id, b, b_id, match_id = matched(env)  # Alex Smith, a man, and Bea Jones, a woman.
    shown_a, shown_b = env.s.manager.get_profile(a_id).name, env.s.manager.get_profile(b_id).name
    assert shown_a in MEN and shown_b in WOMEN and shown_a != shown_b
    assert (env.s.live.real_name(a_id), env.s.live.real_name(b_id)) == ("Alex Smith", "Bea Jones")
    # Others, and matchers, only ever see the display name.
    assert "Bea" not in a.get("/matches").text
    assert PublicView(env.s.manager.store, LIVE_RUN_ID).get_profile(b_id).name == shown_b
    # The person sees their real name, and what others see.
    me = env.client()
    me.cookies.update(b.cookies)
    profile = me.get("/me/profile").json()["profile"]
    assert (profile["name"], profile["shown_as"]) == ("Bea Jones", shown_b)
    assert "Bea Jones" in b.get("/me/profile").text and shown_b in b.get("/me/profile").text
    data = b.get("/me/data.json").json()
    assert data["profile"]["name"] == "Bea Jones"
    # Admins reviewing accounts see real names.
    accounts = env.admin().get("/admin/accounts", params={"status": "approved"}).json()
    assert {a["profile"]["name"] for a in accounts if a["profile"]} >= {"Alex Smith", "Bea Jones"}


def test_editing_keeps_the_display_name_unless_gender_changes(env: Env) -> None:
    from app.fake_names import WOMEN

    api, user_id = env.user("x@bu.edu", "man", ["woman"], name="Sam Real")
    before = env.s.manager.get_profile(user_id).name
    assert api.put("/me/profile", json=profile_body("Samuel Real", "man")).status_code == 200
    assert env.s.manager.get_profile(user_id).name == before
    assert env.s.live.real_name(user_id) == "Samuel Real"
    assert api.put("/me/profile", json=profile_body("Samuel Real", "woman")).status_code == 200
    assert env.s.manager.get_profile(user_id).name in WOMEN


def test_display_names_avoid_repeats_until_a_list_runs_out() -> None:
    import random as random_module

    from app.fake_names import MEN, WOMEN, pick_name

    rng = random_module.Random(0)
    used: set[str] = set()
    for _ in range(len(MEN)):
        used.add(pick_name("man", used, rng))
    assert used == set(MEN)
    assert pick_name("man", used, rng) in MEN  # Repeats only once every name is taken.
    assert pick_name("nonbinary", set(), rng) in set(MEN) | set(WOMEN)


def test_people_who_joined_before_display_names_get_one_at_startup(env: Env) -> None:
    from app.fake_names import WOMEN

    _, user_id = env.user("old@bu.edu", "woman", ["man"], name="Olga Old")
    with sqlite3.connect(env.settings.main_db) as conn:  # As if made by the old code.
        conn.execute("DELETE FROM private_names WHERE user_id = ?", (user_id,))
        conn.execute("UPDATE profiles SET name = 'Olga Old' WHERE user_id = ?", (user_id,))
    create_app(env.settings)
    assert env.s.live.real_name(user_id) == "Olga Old"
    assert env.s.manager.get_profile(user_id).name in WOMEN


# Passwords


def set_password_with_script(env: Env, email: str, password: str):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    environment = {**os.environ, "APP_MAIN_DB": str(env.settings.main_db), "APP_PERSONA_DB": str(env.settings.persona_db)}
    return subprocess.run([sys.executable, str(root / "scripts" / "set_admin_password.py"), email], input=password + "\n",
                          env=environment, capture_output=True, text=True)


def sign_in_page(env: Env, email: str, password: str):
    return page(env).post("/login", data={"email": email, "password": password})


def browser_with(env: Env, response) -> TestClient:
    browser = page(env)
    browser.cookies.update(response.cookies)
    return browser


def test_joining_from_the_page_creates_a_pending_account_and_signs_in(env: Env) -> None:
    form = {"email": "new@bu.edu", "password": "a good password", "password_again": "a good password"}
    assert page(env).post("/join", data={**form, "password_again": "different"}).status_code == 422
    joined = page(env).post("/join", data=form)
    assert joined.headers["location"] == "/onboarding/consent"
    assert env.s.live.account_for_email("new@bu.edu").status == "pending"
    assert page(env).post("/join", data=form).status_code == 409
    assert "Sign in" in page(env).get("/").text and 'href="/join"' in page(env).get("/").text


def test_signing_in_from_the_page(env: Env) -> None:
    env.login("jane@bu.edu")
    wrong = sign_in_page(env, "jane@bu.edu", "wrong password")
    unknown = sign_in_page(env, "nobody@bu.edu", PASSWORD)
    assert wrong.status_code == unknown.status_code == 401
    assert "don't match an account" in unescape(wrong.text) and 'value="jane@bu.edu"' in wrong.text
    assert sign_in_page(env, "jane@bu.edu", PASSWORD).headers["location"] == "/onboarding/consent"


def test_the_admin_script_sets_a_password_and_admins_land_on_admin(env: Env) -> None:
    done = set_password_with_script(env, "lead@bu.edu", "correct horse")
    assert done.returncode == 0, done.stderr
    response = sign_in_page(env, "Lead@BU.edu", "correct horse")
    assert response.headers["location"] == "/admin"  # No consent or profile for admins.
    browser = browser_with(env, response)
    assert browser.get("/admin/users").status_code == 200
    assert sign_in_page(env, "lead@bu.edu", "correct horse").headers["location"] == "/admin"
    assert set_password_with_script(env, "x@bu.edu", "short").returncode != 0


def test_changing_your_password_needs_the_current_one_and_signs_out_elsewhere(env: Env) -> None:
    api = env.login("jane@bu.edu")
    other_device = sign_in_page(env, "jane@bu.edu", PASSWORD)
    browser = browser_with(env, sign_in_page(env, "jane@bu.edu", PASSWORD))
    html = browser.get("/password").text
    form = {"csrf_token": csrf_of(html), "current_password": "wrong", "new_password": "a new password", "new_password_again": "a new password"}
    assert browser.post("/password", data=form).status_code == 403
    assert browser.post("/password", data={**form, "current_password": PASSWORD, "new_password_again": "x"}).status_code == 422
    assert browser.post("/password", data={**form, "current_password": PASSWORD, "new_password": "short", "new_password_again": "short"}).status_code == 422
    assert browser.post("/password", data={**form, "current_password": PASSWORD}).headers["location"] == "/password?done=1"
    assert browser.get("/password").status_code == 200  # Still signed in here.
    assert browser_with(env, other_device).get("/password").headers["location"] == "/"  # Signed out there.
    assert api.get("/me").status_code == 401
    assert sign_in_page(env, "jane@bu.edu", "a new password").status_code == 303


def test_admins_reset_forgotten_passwords_to_a_temporary_one(env: Env) -> None:
    member_browser, user_id = member(env, "jane@bu.edu", "woman", ("man",))
    account_id = env.s.live.account_for_user(user_id).account_id
    admin = admin_page(env)
    html = admin.get("/admin/approvals").text
    assert f"/admin/approvals/{account_id}/reset-password" in html  # Approved people are listed too.
    shown = admin.post(f"/admin/approvals/{account_id}/reset-password", data={"csrf_token": csrf_of(html)})
    temporary = re.search(r"<code>([a-z0-9-]+)</code>", shown.text).group(1)
    assert re.fullmatch(r"[a-z2-9]{4}-[a-z2-9]{4}-[a-z2-9]{4}", temporary)
    assert temporary not in "\n".join(sqlite3.connect(env.settings.main_db).iterdump())
    assert member_browser.get("/discover").headers["location"] == "/"  # Signed out everywhere.
    assert sign_in_page(env, "jane@bu.edu", PASSWORD).status_code == 401
    response = sign_in_page(env, "jane@bu.edu", temporary)
    assert response.headers["location"] == "/password"
    browser = browser_with(env, response)
    assert browser.get("/discover").headers["location"] == "/password"  # Nothing else until it's changed.
    api = env.client()
    api.cookies.update(response.cookies)
    assert api.get("/me").status_code == 403
    html = browser.get("/password").text
    assert "current_password" not in html
    done = browser.post("/password", data={"csrf_token": csrf_of(html), "new_password": "my own password", "new_password_again": "my own password"})
    assert done.headers["location"] == "/discover"
    assert browser.get("/discover").status_code == 200
    assert {"password_reset", "password_change"} <= {e["action"] for e in env.s.live.audit_entries()}


def test_admins_download_a_full_backup_and_it_is_audited(env: Env, tmp_path: Path) -> None:
    matched(env)
    browser, _ = member(env, "v@bu.edu")
    assert browser.get("/admin/backup.db").status_code == 403
    response = admin_page(env).get("/admin/backup.db")
    assert response.status_code == 200 and response.headers["content-disposition"].startswith('attachment; filename="study-backup-')
    copy = tmp_path / "copy.db"
    copy.write_bytes(response.content)
    with sqlite3.connect(copy) as conn:
        assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] >= 3
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert env.s.live.audit_entries(action="download_backup")


@pytest.mark.parametrize("people", [1, 2])
def test_users_page_works_before_anyone_has_rated(env: Env, people: int) -> None:
    for i in range(people):
        member(env, f"p{i}@bu.edu", "woman" if i else "man", ("man", "woman"))
    response = admin_page(env).get("/admin/users")
    assert response.status_code == 200, response.text[:500]
    assert "No ratings yet" in response.text


@pytest.mark.parametrize("path", ["/admin", "/admin/approvals", "/admin/users", "/admin/compare", "/admin/reports",
                                  "/admin/controls", "/admin/audit", "/admin/export.csv", "/admin/truth.zip", "/admin/backup.db",
                                  "/admin/best-matches"])
def test_every_admin_page_works_with_people_but_no_ratings(env: Env, path: str) -> None:
    member(env, "a@bu.edu", "man", ("woman",))
    member(env, "b@bu.edu", "woman", ("man",))
    assert admin_page(env).get(path).status_code == 200, path


@pytest.mark.parametrize("step", ["/onboarding/consent", "/onboarding/profile", "/onboarding/preferences", "/pending", "/onboarding/best-match"])
def test_people_partway_through_signing_up_can_sign_out(env: Env, step: str) -> None:
    """A browser stays signed in for SESSION_DAYS, so coming back resumes sign-up. Every
    sign-up step must offer Sign out, or nobody else can sign in on that browser."""
    client = env.login("half@bu.edu")
    if step != "/onboarding/consent":
        client.post("/auth/consent", json={"version": "3"})
    if step in ("/onboarding/preferences", "/pending", "/onboarding/best-match"):
        assert client.post("/me/profile", json=profile_body()).status_code in (200, 201)
    if step in ("/pending", "/onboarding/best-match"):
        assert client.put("/me/preferences", json={"interested_in": ["woman"]}).status_code in (200, 201)
    if step == "/onboarding/best-match":  # Asked once approved.
        account_id = env.s.live.account_for_email("half@bu.edu").account_id
        assert env.admin().post(f"/admin/accounts/{account_id}/approve").status_code == 200
    page = client.get(step, headers={"Accept": "text/html"})
    assert page.status_code == 200 and str(page.url).endswith(step), (step, page.url)
    assert "Sign out" in page.text
    out = client.post("/logout", data={"csrf_token": csrf_of(page.text)}, headers={"Accept": "text/html"})
    assert out.status_code == 200
    landing = client.get("/", headers={"Accept": "text/html"})
    assert 'action="/login"' in landing.text


@pytest.mark.parametrize("feet, inches, cm", [(5, 9, 175), (5, 0, 152), (6, 2, 188), (4, 11, 150)])
def test_height_in_feet_and_inches_is_stored_in_cm_and_edits_back_unchanged(env: Env, feet: int, inches: int, cm: int) -> None:
    api = env.consented("tall@bu.edu")
    browser = page(env, api)
    html = browser.get("/onboarding/profile").text
    assert 'value="imperial" checked' in html  # Feet and inches is the default.
    form = {**{k: str(v) for k, v in profile_body().items() if not isinstance(v, list)},
            "interests": "chess", "clubs": "", "csrf_token": csrf_of(html),
            "height_unit": "imperial", "height_ft": str(feet), "height_in": str(inches), "height_cm": ""}
    assert browser.post("/onboarding/profile", data=form).headers["location"] == "/onboarding/preferences"
    assert api.get("/me").json()["profile"]["height_cm"] == cm
    from app.ui import feet_and_inches, height_values
    assert feet_and_inches(cm) == (feet, inches)
    assert height_values(cm)["height_ft"] == feet and height_values(cm)["height_in"] == inches


def test_every_whole_inch_survives_the_round_trip_through_cm() -> None:
    from app.ui import feet_and_inches, parse_height
    for total in range(48, 91):
        cm, problem = parse_height({"height_unit": "imperial", "height_ft": str(total // 12), "height_in": str(total % 12)})
        assert problem is None and feet_and_inches(cm) == (total // 12, total % 12), total


@pytest.mark.parametrize("ft, inch, message", [("5", "12", "0 to 11"), ("8", "0", "7 ft 6 in"), ("", "3", "whole feet")])
def test_bad_feet_and_inches_explain_themselves(ft: str, inch: str, message: str) -> None:
    from app.ui import parse_height
    cm, problem = parse_height({"height_unit": "imperial", "height_ft": ft, "height_in": inch})
    assert cm is None and message in problem


@pytest.mark.parametrize("status", ["pending", "suspended"])
def test_an_admin_whose_account_isnt_approved_still_goes_straight_to_admin(env: Env, status: str) -> None:
    """An account made by an older version can have the admin role without being approved.
    The role alone decides: no consent, no profile form, straight to the admin pages."""
    api = env.login(ADMIN_EMAIL)
    account = env.s.live.account_for_email(ADMIN_EMAIL)
    with sqlite3.connect(env.settings.main_db) as conn:
        conn.execute("UPDATE accounts SET status = ?, consent_version = NULL WHERE account_id = ?", (status, account.account_id))
    browser = page(env, api)
    assert browser.post("/login", data={"email": ADMIN_EMAIL, "password": PASSWORD}).headers["location"] == "/admin"
    for path in ("/", "/onboarding/consent", "/onboarding/profile", "/discover"):
        assert browser.get(path).headers["location"] == "/admin", path
    assert browser.get("/admin").status_code == 200
    assert browser.get("/admin/users").status_code == 200



# Starting matching


def test_approved_people_wait_until_an_admin_starts_matching(env: Env) -> None:
    env.s.live.set_setting("matching_started", False)
    browser, _ = member(env, "a@bu.edu", "man", ("woman",))
    api, _ = env.user("b@bu.edu", "woman", ("man",))
    html = browser.get("/discover").text
    assert "Waiting for the site administrators to start matching." in html
    assert 'hx-get="/discover/region"' in html and 'hx-trigger="every 30s"' in html
    assert api.get("/feed/next").json()["profile"] is None
    admin = admin_page(env)
    controls = admin.get("/admin/controls").text
    assert "Not started. 2 approved participants are waiting." in controls
    assert admin.post("/admin/controls/start", data={"csrf_token": csrf_of(controls)}).status_code == 303
    region = browser.get("/discover/region").text
    assert "Waiting for the site administrators" not in region and "hx-trigger" not in region
    assert 'name="impression_id"' in region
    assert "Matching has started." in admin.get("/admin/controls").text
    assert [e["action"] for e in env.s.live.audit_entries("start_matching", None, limit=5)] == ["start_matching"]


def test_ratings_are_refused_before_matching_starts_and_after_it_stops(env: Env) -> None:
    api, _ = env.user("a@bu.edu", "man", ("woman",))
    env.user("b@bu.edu", "woman", ("man",))
    card = api.get("/feed/next").json()
    admin = admin_page(env)
    admin.post("/admin/controls/stop", data={"csrf_token": csrf_of(admin.get("/admin/controls").text)})
    response = api.post("/ratings", json={"impression_id": card["impression_id"], "score": 5})
    assert response.status_code == 403 and "start matching" in response.text


def test_a_new_site_starts_out_waiting(tmp_path: Path) -> None:
    settings = Settings(site_url=SITE, admin_emails=(ADMIN_EMAIL,), main_db=tmp_path / "m.db", persona_db=tmp_path / "p.db", cookie_secure=False)
    app = create_app(settings)
    assert app.state.services.controls().matching_started is False


def test_users_page_opens_with_people_scores_and_progress_charts(env: Env) -> None:
    a, a_id = member(env, "a@bu.edu", "man", ("woman",))
    b, b_id = member(env, "b@bu.edu", "woman", ("man",))
    actions.rate(env.s, env.s.live.account_for_user(a_id), actions.next_impression(env.s, env.s.live.account_for_user(a_id)).impression.impression_id, 7.3)
    html = admin_page(env).get("/admin/users").text
    assert "Pairs who rated each other" not in html
    charts = {m.group(1): json.loads(unescape(m.group(2))) for m in re.finditer(r'<canvas (?:id="([^"]*)" )?data-chart="([^"]+)"', html) if m.group(1)}
    score = charts["score-chart"]
    counts = dict(zip(score["data"]["labels"], score["views"]["all"]["data"]))
    assert counts["7.3"] == 1 and sum(counts.values()) == 1
    assert sum(score["views"]["interested"]["data"]) == 1  # A straight man rating a woman.
    assert 'data-switch-for="score-chart"' in html and 'value="interested"' in html
    assert "1 of 2 ratings done" in html
    assert "Rating progress" in html and "People</h2>" in html
