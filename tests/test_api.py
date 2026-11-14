import json
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import matchers as matcher_registry
from app.config import Settings
from app.main import create_app
from app.security import normalize_email
from live_store import sha256
from matching import Matcher, PublicView, RandomMatcher
from models import LIVE_RUN_ID

SITE = "http://testserver"
ADMIN_EMAIL = "admin@bu.edu"
EVERYONE = ["man", "woman", "nonbinary"]
PASSWORD = "test password"


def profile_body(name: str = "Sam", gender: str = "man", **overrides: Any) -> dict[str, Any]:
    body = {
        "name": name,
        "age": 21,
        "height_cm": 175,
        "religion": "none",
        "school": "BU",
        "major": "biology",
        "job": "tutor",
        "interests": ["hiking"],
        "clubs": [],
        "gender": gender,
        "looking_for": "long-term",
    }
    return {**body, **overrides}


@dataclass
class Env:
    app: Any
    settings: Settings
    _admin: TestClient | None = None

    @property
    def s(self) -> Any:
        return self.app.state.services

    def client(self, origin: str = SITE) -> TestClient:
        return TestClient(self.app, headers={"Origin": origin})

    def login(self, email: str) -> TestClient:
        client = self.client()
        body = {"email": email, "password": PASSWORD}
        response = client.post("/auth/login", json=body)
        if response.status_code == 401:
            response = client.post("/auth/signup", json=body)
        assert response.status_code in (200, 201), response.text
        return client

    def consented(self, email: str) -> TestClient:
        client = self.login(email)
        assert client.post("/auth/consent", json={"version": "2"}).status_code == 200
        return client

    def admin(self) -> TestClient:
        """Signed in once per test."""
        if self._admin is None:
            self._admin = self.consented(ADMIN_EMAIL)
        return self._admin

    def account_id(self, client: TestClient) -> str:
        return client.get("/me").json()["account_id"]

    def user(
        self,
        email: str,
        gender: str = "man",
        interested_in: Any = ("woman",),
        approve: bool = True,
        name: str = "Sam",
    ) -> tuple[TestClient, str]:
        client = self.consented(email)
        created = client.post("/me/profile", json=profile_body(name, gender))
        assert created.status_code == 201, created.text
        user_id = created.json()["profile"]["user_id"]
        body = {"interested_in": interested_in if isinstance(interested_in, str) else list(interested_in)}
        assert client.put("/me/preferences", json=body).status_code == 200
        if approve:
            response = self.admin().post(f"/admin/accounts/{self.account_id(client)}/approve")
            assert response.status_code == 200, response.text
        return client, user_id


@pytest.fixture
def env(tmp_path: Path) -> Env:
    settings = Settings(
        site_url=SITE,
        admin_emails=(ADMIN_EMAIL,),
        main_db=tmp_path / "main.db",
        persona_db=tmp_path / "personas.db",
        cookie_secure=False,  # TestClient uses plain http.
        login_requests_per_ip=1000,
    )
    return Env(create_app(settings, rng=random.Random(1)), settings)


def shown_ids(client: TestClient, limit: int = 50) -> list[str]:
    """Walk the whole feed, rating everyone it shows 1 (so nobody matches), in order."""
    shown = []
    for _ in range(limit):
        response = client.get("/feed/next").json()
        if response["profile"] is None:
            return shown
        shown.append(response["profile"]["user_id"])
        assert client.post("/ratings", json={"impression_id": response["impression_id"], "score": 1}).status_code == 200
    raise AssertionError("the feed never ran out")


def rate_next(client: TestClient, score: float) -> dict[str, Any]:
    feed = client.get("/feed/next").json()
    response = client.post("/ratings", json={"impression_id": feed["impression_id"], "score": score})
    assert response.status_code == 200, response.text
    return response.json()


# Email and login links


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Jane@BU.edu", "jane@bu.edu"),
        ("  jane+dating@bu.edu ", "jane@bu.edu"),
        ("jane.doe+a+b@bu.edu", "jane.doe@bu.edu"),
        ("jane@gmail.com", "jane@gmail.com"),
        ("jane@sub.bu.edu", "jane@sub.bu.edu"),
        ("jane@localhost", None),
        ("+tag@bu.edu", None),
        ("no-at-sign", None),
        ("a@b@bu.edu", None),
        ("", None),
    ],
)
def test_email_normalization(raw: str, expected: str | None) -> None:
    assert normalize_email(raw) == expected


def test_an_allowed_domain_list_still_restricts_when_set() -> None:
    assert normalize_email("jane@bu.edu", ["bu.edu"]) == "jane@bu.edu"
    assert normalize_email("jane@gmail.com", ["bu.edu"]) is None
    assert normalize_email("jane@bu.edu.evil.com", ["bu.edu"]) is None
    assert normalize_email("jane@sub.bu.edu", ["bu.edu"]) is None


def test_sign_up_accepts_any_domain_unless_a_list_is_configured(env: Env, tmp_path: Path) -> None:
    assert env.client().post("/auth/signup", json={"email": "not an email", "password": PASSWORD}).status_code == 422
    response = env.client().post("/auth/signup", json={"email": "Jane+x@Gmail.com", "password": PASSWORD})
    assert response.status_code == 201 and response.json()["status"] == "pending"
    assert env.s.live.account_for_email("jane@gmail.com") is not None
    settings = Settings(site_url=SITE, main_db=tmp_path / "m3.db", persona_db=tmp_path / "p3.db", allowed_email_domains=("bu.edu",))
    client = TestClient(create_app(settings), headers={"Origin": SITE})
    assert client.post("/auth/signup", json={"email": "jane@gmail.com", "password": PASSWORD}).status_code == 422
    assert client.post("/auth/signup", json={"email": "Jane+x@BU.edu", "password": PASSWORD}).status_code == 201


def test_sign_up_rules(env: Env) -> None:
    assert env.client().post("/auth/signup", json={"email": "a@bu.edu", "password": "short"}).status_code == 422
    assert env.client().post("/auth/signup", json={"email": "a@bu.edu", "password": PASSWORD}).status_code == 201
    taken = env.client().post("/auth/signup", json={"email": "A@bu.edu", "password": "another one"})
    assert taken.status_code == 409 and "already an account" in taken.text
    admin = env.client().post("/auth/signup", json={"email": ADMIN_EMAIL, "password": PASSWORD}).json()
    assert (admin["role"], admin["status"]) == ("admin", "approved")
    env.s.live.set_setting("signups_paused", True)
    assert env.client().post("/auth/signup", json={"email": "late@bu.edu", "password": PASSWORD}).status_code == 403


def test_passwords_are_hashed_and_wrong_ones_get_one_answer(env: Env) -> None:
    env.client().post("/auth/signup", json={"email": "jane@bu.edu", "password": PASSWORD})
    with sqlite3.connect(env.settings.main_db) as conn:
        dump = "\n".join(conn.iterdump())
    assert PASSWORD not in dump and "scrypt$" in dump
    wrong = env.client().post("/auth/login", json={"email": "jane@bu.edu", "password": "wrong password"})
    unknown = env.client().post("/auth/login", json={"email": "nobody@bu.edu", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    assert env.client().post("/auth/login", json={"email": "JANE@bu.edu", "password": PASSWORD}).status_code == 200


def test_failed_sign_ins_are_limited_per_email_and_per_address(env: Env, tmp_path: Path) -> None:
    env.client().post("/auth/signup", json={"email": "jane@bu.edu", "password": PASSWORD})
    codes = [env.client().post("/auth/login", json={"email": "jane@bu.edu", "password": "wrong"}).status_code for _ in range(6)]
    assert codes == [401] * 5 + [429]
    assert env.client().post("/auth/login", json={"email": "jane@bu.edu", "password": PASSWORD}).status_code == 429
    settings = Settings(site_url=SITE, main_db=tmp_path / "m2.db", persona_db=tmp_path / "p2.db", login_requests_per_ip=2)
    limited = TestClient(create_app(settings), headers={"Origin": SITE})
    codes = [limited.post("/auth/login", json={"email": f"u{i}@bu.edu", "password": "x"}).status_code for i in range(3)]
    assert codes == [401, 401, 429]


def test_successful_sign_ins_never_count_toward_the_limit(env: Env) -> None:
    env.client().post("/auth/signup", json={"email": "jane@bu.edu", "password": PASSWORD})
    for _ in range(10):
        assert env.client().post("/auth/login", json={"email": "jane@bu.edu", "password": PASSWORD}).status_code == 200


def test_sessions_are_hashed_and_cookies_are_locked_down(env: Env) -> None:
    client = env.client()
    response = client.post("/auth/signup", json={"email": "jane@bu.edu", "password": PASSWORD})
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    session_id = client.cookies["session"]
    with sqlite3.connect(env.settings.main_db) as conn:
        stored = [row[0] for row in conn.execute("SELECT session_hash FROM sessions")]
    assert stored == [sha256(session_id)]
    assert response.json()["status"] == "pending"


def test_secure_cookie_flag_is_set_when_configured(tmp_path: Path) -> None:
    settings = Settings(site_url=SITE, main_db=tmp_path / "m.db", persona_db=tmp_path / "p.db", cookie_secure=True)
    env = Env(create_app(settings), settings)
    response = env.client().post("/auth/signup", json={"email": "jane@bu.edu", "password": PASSWORD})
    assert "Secure" in response.headers["set-cookie"]
    assert "Strict-Transport-Security" in response.headers


def test_logout_ends_the_session(env: Env) -> None:
    client = env.login("jane@bu.edu")
    assert client.get("/me").status_code == 200
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/me").status_code == 401


def test_admin_emails_get_admin_role_and_auto_approval(env: Env) -> None:
    client = env.login("Admin+anything@bu.edu")
    me = client.get("/me").json()
    assert (me["role"], me["status"]) == ("admin", "approved")
    assert env.login("jane@bu.edu").get("/me").json()["role"] == "user"


# Consent and approval


def test_consent_is_required_for_everything_but_auth(env: Env) -> None:
    client = env.login("jane@bu.edu")
    assert client.get("/auth/consent").json()["version"] == "2"
    assert "aren't used to train" in client.get("/auth/consent").json()["text"]
    assert client.post("/me/profile", json=profile_body()).status_code == 403
    assert client.get("/feed/next").status_code == 403
    assert client.post("/auth/consent", json={"version": "0"}).status_code == 409
    assert client.post("/auth/consent", json={"version": "2"}).status_code == 200
    assert client.post("/me/profile", json=profile_body()).status_code == 201


def test_unapproved_accounts_get_403_on_feed_ratings_and_matches(env: Env) -> None:
    client, _ = env.user("jane@bu.edu", approve=False)
    assert client.get("/feed/next").status_code == 403
    assert client.post("/ratings", json={"impression_id": "x", "score": 5}).status_code == 403
    assert client.get("/matches").status_code == 403
    assert client.post("/blocks", json={"user_id": "x"}).status_code == 403
    # Their own profile and preferences still work.
    assert client.get("/me/profile").status_code == 200
    assert client.get("/me/preferences").status_code == 200


def test_approval_requires_consent_profile_and_preferences(env: Env) -> None:
    admin = env.admin()
    client = env.login("jane@bu.edu")
    account_id = env.account_id(client)

    def missing() -> list[str]:
        response = admin.post(f"/admin/accounts/{account_id}/approve")
        return response.json()["detail"]["missing"] if response.status_code == 409 else []

    assert missing() == ["consent", "profile"]
    client.post("/auth/consent", json={"version": "2"})
    client.post("/me/profile", json=profile_body())
    assert missing() == ["preferences"]
    client.put("/me/preferences", json={"interested_in": "everyone"})
    assert missing() == []
    assert client.get("/me").json()["status"] == "approved"
    actions = [e["action"] for e in env.s.live.audit_entries()]
    assert actions.count("account_approve") == 1


def test_reject_and_suspend_block_access_and_are_audited(env: Env) -> None:
    client, _ = env.user("jane@bu.edu")
    admin = env.admin()
    account_id = env.account_id(client)
    assert client.get("/feed/next").status_code == 200
    assert admin.post(f"/admin/accounts/{account_id}/suspend").status_code == 200
    assert client.get("/feed/next").status_code == 403
    assert admin.post(f"/admin/accounts/{account_id}/reject").json()["status"] == "rejected"
    assert {"account_suspend", "account_reject"} <= {e["action"] for e in env.s.live.audit_entries()}


def test_admin_endpoints_are_admins_only(env: Env) -> None:
    client, _ = env.user("jane@bu.edu")
    for path in ("/admin/accounts", "/admin/reports"):
        assert client.get(path).status_code == 403
    assert client.post("/admin/accounts/x/approve").status_code == 403


def test_admin_queue_lists_pending_accounts_with_email_and_profile(env: Env) -> None:
    client, _ = env.user("jane@bu.edu", approve=False)
    queue = env.admin().get("/admin/accounts", params={"status": "pending"}).json()
    assert [a["email"] for a in queue] == ["jane@bu.edu"]
    assert queue[0]["profile"]["name"] == "Sam"


# Profiles and preferences


def test_clients_cannot_set_user_id_or_agent(env: Env) -> None:
    client = env.consented("jane@bu.edu")
    assert client.post("/me/profile", json={**profile_body(), "user_id": "00000000-0000-4000-8000-000000000000"}).status_code == 422
    assert client.post("/me/profile", json={**profile_body(), "agent": True}).status_code == 422
    created = client.post("/me/profile", json=profile_body())
    assert created.status_code == 201
    assert "agent" not in created.json()["profile"]
    assert env.s.manager.get_profile(created.json()["profile"]["user_id"]).agent is False
    assert client.put("/me/profile", json={"user_id": "x"}).status_code == 422
    assert client.post("/me/profile", json=profile_body()).status_code == 409


def test_profile_validation_errors_are_422(env: Env) -> None:
    client = env.consented("jane@bu.edu")
    assert client.post("/me/profile", json=profile_body(age=17)).status_code == 422
    assert client.post("/me/profile", json=profile_body(gender="men")).status_code == 422
    assert client.post("/me/profile", json=profile_body()).status_code == 201
    assert client.put("/me/profile", json={"age": 17}).status_code == 422
    assert client.put("/me/profile", json={"major": "physics"}).json()["profile"]["major"] == "physics"


def test_preferences_accept_everyone_and_are_only_shown_to_their_owner(env: Env) -> None:
    client = env.consented("jane@bu.edu")
    client.post("/me/profile", json=profile_body())
    assert client.put("/me/preferences", json={"interested_in": "everyone"}).json() == {"interested_in": sorted(EVERYONE)}
    assert client.put("/me/preferences", json={"interested_in": []}).status_code == 422
    assert client.put("/me/preferences", json={"interested_in": ["men"]}).status_code == 422
    assert client.get("/me/preferences").json() == {"interested_in": sorted(EVERYONE)}
    assert client.get("/me").json()["interested_in"] == sorted(EVERYONE)


# The eligible pool


def test_live_pool_shows_everyone_whatever_their_gender_but_simulations_keep_the_filter(env: Env) -> None:
    man_for_women, man_id = env.user("m1@bu.edu", "man", ["woman"])
    woman_for_women, ww_id = env.user("w1@bu.edu", "woman", ["woman"])
    _, wm_id = env.user("w2@bu.edu", "woman", ["man"])
    _, nb_id = env.user("n1@bu.edu", "nonbinary", ["nonbinary"])
    assert sorted(env.s.manager.eligible_candidates(ww_id, "sim-run")) == []
    assert env.s.manager.eligible_candidates(man_id, "sim-run") == [wm_id]
    assert sorted(shown_ids(man_for_women)) == sorted([ww_id, wm_id, nb_id])
    assert sorted(shown_ids(woman_for_women)) == sorted([man_id, wm_id, nb_id])


def test_humans_are_never_shown_agents(env: Env) -> None:
    viewer, viewer_id = env.user("v@bu.edu", "man", "everyone")
    agent = env.s.manager.add_profile({**profile_body("Agent", "woman"), "agent": True})
    env.s.manager.set_preferences(agent.user_id, ["man"])
    # Even an agent that somehow had an approved account stays out of the live pool.
    with sqlite3.connect(env.settings.main_db) as conn:
        conn.execute(
            "INSERT INTO accounts (account_id, email_normalized, user_id, status, role, created_at) VALUES ('x', 'agent@x.org', ?, 'approved', 'user', '2026-01-01')",
            (agent.user_id,),
        )
    assert agent.user_id not in env.s.manager.eligible_candidates(viewer_id, LIVE_RUN_ID)
    assert shown_ids(viewer) == []


def test_the_app_refuses_to_start_on_a_database_with_agents(env: Env, tmp_path: Path) -> None:
    env.s.manager.add_profile({**profile_body("Agent", "woman"), "agent": True})
    with pytest.raises(RuntimeError, match="simulated agents"):
        create_app(env.settings)
    import paths

    settings = Settings(main_db=paths.MAIN_DB, persona_db=tmp_path / "p.db")
    with pytest.raises(RuntimeError, match="simulated-agent data"):
        create_app(settings)


def test_asking_the_feed_again_shows_the_same_person_so_nobody_is_skipped(env: Env) -> None:
    viewer, _ = env.user("v@bu.edu", "man", "everyone")
    for i in range(3):
        env.user(f"w{i}@bu.edu", "woman", "everyone")
    first = viewer.get("/feed/next").json()
    again = viewer.get("/feed/next").json()
    assert first["impression_id"] == again["impression_id"]


def test_someone_paused_while_on_screen_comes_back_later(env: Env) -> None:
    viewer, _ = env.user("v@bu.edu", "man", "everyone")
    others = {env.user(f"w{i}@bu.edu", "woman", "everyone")[1] for i in range(3)}
    first = viewer.get("/feed/next").json()["profile"]["user_id"]
    paused_account = env.s.live.account_for_user(first).account_id
    env.s.live.set_flags(paused_account, paused=True)
    rated = shown_ids(viewer)
    assert first not in rated and len(rated) == 2
    env.s.live.set_flags(paused_account, paused=False)
    assert shown_ids(viewer) == [first]
    assert set(rated) | {first} == others


def test_nobody_can_rate_before_rating_opens(env: Env) -> None:
    from datetime import timedelta

    from app import actions
    from live_store import utc_now

    viewer, _ = env.user("v@bu.edu", "man", "everyone")
    env.user("w@bu.edu", "woman", "everyone")
    admin_account = env.s.live.account_for_email(ADMIN_EMAIL)
    impression_id = viewer.get("/feed/next").json()["impression_id"]
    actions.set_controls(env.s, admin_account, {"random": 1.0}, False, False, utc_now() + timedelta(days=1))
    closed = viewer.get("/feed/next").json()
    assert closed["profile"] is None and closed["message"].startswith("Rating opens on ")
    refused = viewer.post("/ratings", json={"impression_id": impression_id, "score": 5})
    assert refused.status_code == 403
    actions.set_controls(env.s, admin_account, {"random": 1.0}, False, False, utc_now() - timedelta(minutes=1))
    assert viewer.post("/ratings", json={"impression_id": impression_id, "score": 5}).status_code == 200
    assert env.s.live.audit_entries()[-1]["action"] == "set_controls"


def test_self_blocked_and_unapproved_users_never_appear(env: Env) -> None:
    viewer, viewer_id = env.user("v@bu.edu", "man", "everyone")
    blocked_client, blocked_id = env.user("b@bu.edu", "woman", "everyone")
    blocker_client, blocker_id = env.user("c@bu.edu", "woman", "everyone")
    _, pending_id = env.user("p@bu.edu", "woman", "everyone", approve=False)
    _, visible_id = env.user("ok@bu.edu", "woman", "everyone")
    viewer.post("/blocks", json={"user_id": blocked_id})
    blocker_client.post("/blocks", json={"user_id": viewer_id})
    assert shown_ids(viewer) == [visible_id]
    assert viewer_id not in shown_ids(blocked_client)
    assert pending_id not in env.s.manager.eligible_candidates(viewer_id, LIVE_RUN_ID)


def test_rated_profiles_never_repeat(env: Env) -> None:
    viewer, _ = env.user("v@bu.edu", "man", ["woman"])
    ids = {env.user(f"w{i}@bu.edu", "woman", ["man"])[1] for i in range(3)}
    seen = []
    for score in (5, 6, 7):
        feed = viewer.get("/feed/next").json()
        seen.append(feed["profile"]["user_id"])
        viewer.post("/ratings", json={"impression_id": feed["impression_id"], "score": score})
    assert set(seen) == ids
    empty = viewer.get("/feed/next").json()
    assert empty == {"profile": None, "pool_size": 0, "message": "You've seen everyone available to you right now. New people may join."}


def test_active_matches_leave_the_pool_and_simulations_skip_unapproved_accounts(env: Env) -> None:
    a, a_id = env.user("a@bu.edu", "man", ["woman"])
    b, b_id = env.user("b@bu.edu", "woman", ["man"])
    env.s.live.create_match(a_id, b_id)
    assert env.s.manager.eligible_candidates(a_id, LIVE_RUN_ID) == []
    # In a simulation run, accounts must be approved too, but agents without accounts count.
    agent = env.s.manager.add_profile({**profile_body("Agent", "woman"), "agent": True})
    env.s.manager.set_preferences(agent.user_id, ["man"])
    _, pending = env.user("p@bu.edu", "woman", ["man"], approve=False)
    assert env.s.manager.eligible_candidates(a_id, "sim-run") == [agent.user_id]
    assert agent.user_id not in env.s.manager.eligible_candidates(a_id, LIVE_RUN_ID)
    assert pending not in env.s.manager.eligible_candidates(a_id, "sim-run")


# Feed and ratings


def test_feed_returns_only_public_data_and_records_the_impression(env: Env) -> None:
    viewer, viewer_id = env.user("v@bu.edu", "man", ["woman"])
    _, target_id = env.user("t@bu.edu", "woman", ["man"])
    env.s.manager.add_persona(target_id, "SECRET_PERSONA_TEXT")
    response = viewer.get("/feed/next")
    body = response.json()
    assert set(body) == {"impression_id", "profile"}
    assert set(body["profile"]) == {"user_id", "name", "age", "height_cm", "religion", "school", "major", "job", "interests", "clubs", "gender", "looking_for"}
    text = response.text
    for secret in ("t@bu.edu", "interested_in", "SECRET_PERSONA_TEXT", "agent", "score"):
        assert secret not in text
    impression = env.s.live.get_impression(body["impression_id"])
    assert (impression.viewer_id, impression.target_id, impression.matcher, impression.pool_size) == (viewer_id, target_id, "random", 1)


class FirstMatcher(Matcher):
    name = "first"

    def __init__(self, rng: random.Random) -> None:
        pass

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        pool = view.eligible_candidates(user_id)
        return pool[0] if pool else None


def test_registry_accepts_any_matcher_and_picks_with_a_seeded_rng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(matcher_registry.REGISTRY, "first", FirstMatcher)

    runs = iter(range(100))

    def chosen(seed: int) -> list[str]:
        folder = tmp_path / f"run{next(runs)}"
        settings = Settings(site_url=SITE, admin_emails=(ADMIN_EMAIL,), main_db=folder / "m.db", persona_db=folder / "p.db", cookie_secure=False, enabled_matchers=("random", "first"))
        env = Env(create_app(settings, rng=random.Random(seed)), settings)
        viewer, _ = env.user("v@bu.edu", "man", ["woman"])
        for i in range(8):
            env.user(f"w{i}@bu.edu", "woman", ["man"])
        names = []
        for _ in range(8):
            feed = viewer.get("/feed/next").json()
            names.append(env.s.live.get_impression(feed["impression_id"]).matcher)
            viewer.post("/ratings", json={"impression_id": feed["impression_id"], "score": 1})
        return names

    assert chosen(3) == chosen(3)
    assert set(chosen(3)) == {"random", "first"}


def test_unknown_matcher_names_are_refused(tmp_path: Path) -> None:
    settings = Settings(main_db=tmp_path / "m.db", persona_db=tmp_path / "p.db", enabled_matchers=("elo",))
    with pytest.raises(ValueError, match="ENABLED_MATCHERS"):
        create_app(settings)


class OutsidePoolMatcher(RandomMatcher):
    name = "rogue"

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        return user_id  # Never eligible.


def test_a_pick_outside_the_pool_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(matcher_registry.REGISTRY, "rogue", OutsidePoolMatcher)
    settings = Settings(site_url=SITE, admin_emails=(ADMIN_EMAIL,), main_db=tmp_path / "m.db", persona_db=tmp_path / "p.db", cookie_secure=False, enabled_matchers=("rogue",))
    env = Env(create_app(settings), settings)
    viewer, _ = env.user("v@bu.edu", "man", ["woman"])
    env.user("w@bu.edu", "woman", ["man"])
    assert viewer.get("/feed/next").status_code == 500


def test_ratings_need_your_own_unrated_impression(env: Env) -> None:
    viewer, viewer_id = env.user("v@bu.edu", "man", ["woman"])
    other, _ = env.user("o@bu.edu", "man", ["woman"])
    env.user("t@bu.edu", "woman", ["man"])
    feed = viewer.get("/feed/next").json()
    target_id = feed["profile"]["user_id"]
    assert other.post("/ratings", json={"impression_id": feed["impression_id"], "score": 5}).status_code == 404
    assert viewer.post("/ratings", json={"impression_id": "made-up", "score": 5}).status_code == 404
    assert viewer.post("/ratings", json={"impression_id": feed["impression_id"], "score": 5}).status_code == 200
    assert viewer.post("/ratings", json={"impression_id": feed["impression_id"], "score": 9}).status_code == 409
    rating = env.s.manager.ratings_by_rater(viewer_id, LIVE_RUN_ID)
    assert [(r.target_id, round(r.score, 6)) for r in rating] == [(target_id, round(4 / 9, 6))]


@pytest.mark.parametrize("score", [0.9, 10.1, -1, "high", None, float("inf")])
def test_score_range_is_enforced(env: Env, score: Any) -> None:
    viewer, _ = env.user("v@bu.edu", "man", ["woman"])
    env.user("t@bu.edu", "woman", ["man"])
    feed = viewer.get("/feed/next").json()
    payload = json.dumps({"impression_id": feed["impression_id"], "score": score}).replace("Infinity", "1e999")
    response = viewer.post("/ratings", content=payload, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert env.s.live.get_impression(feed["impression_id"]).rated_at is None


def test_clients_cannot_choose_the_target_or_matcher(env: Env) -> None:
    viewer, viewer_id = env.user("v@bu.edu", "man", ["woman"])
    _, target_id = env.user("t@bu.edu", "woman", ["man"])
    _, other_id = env.user("x@bu.edu", "woman", ["man"])
    feed = viewer.get("/feed/next").json()
    for extra in ({"matcher": "elo"}, {"target_id": other_id}):
        response = viewer.post("/ratings", json={"impression_id": feed["impression_id"], "score": 5, **extra})
        assert response.status_code == 422
    viewer.post("/ratings", json={"impression_id": feed["impression_id"], "score": 5})
    (rating,) = env.s.manager.ratings_by_rater(viewer_id, LIVE_RUN_ID)
    assert rating.target_id == feed["profile"]["user_id"]
    assert rating.matcher == "random"


def test_a_match_needs_both_scores_at_the_threshold(env: Env) -> None:
    a, _ = env.user("a@bu.edu", "man", ["woman"])
    b, _ = env.user("b@bu.edu", "woman", ["man"])
    first = rate_next(a, 7.0)
    assert first == {"saved": True, "matched": False, "match_id": None}
    second = rate_next(b, 7.0)
    assert second["matched"] is True
    assert a.get("/matches").json()[0]["match_id"] == second["match_id"]


def test_no_match_below_the_threshold_and_replies_reveal_nothing(env: Env) -> None:
    a, _ = env.user("a@bu.edu", "man", ["woman"])
    b, _ = env.user("b@bu.edu", "woman", ["man"])
    c, _ = env.user("c@bu.edu", "man", ["woman"])
    assert rate_next(a, 9.5) == {"saved": True, "matched": False, "match_id": None}
    # b is rated highly by a but rates a low. c hasn't rated b at all. Both replies are identical.
    replies = [rate_next(b, 6.9), rate_next(b, 6.9)]
    assert replies == [{"saved": True, "matched": False, "match_id": None}] * 2
    assert a.get("/matches").json() == []


def test_no_match_if_a_block_exists(env: Env) -> None:
    a, a_id = env.user("a@bu.edu", "man", ["woman"])
    b, b_id = env.user("b@bu.edu", "woman", ["man"])
    rate_next(a, 9)
    feed = b.get("/feed/next").json()
    a.post("/blocks", json={"user_id": b_id})
    response = b.post("/ratings", json={"impression_id": feed["impression_id"], "score": 9}).json()
    assert response["matched"] is False


# Matches and safety


def matched_pair(env: Env) -> tuple[TestClient, str, TestClient, str, str]:
    a, a_id = env.user("a@bu.edu", "man", ["woman"], name="Al")
    b, b_id = env.user("b@bu.edu", "woman", ["man"], name="Bea")
    rate_next(a, 8)
    match_id = rate_next(b, 8)["match_id"]
    return a, a_id, b, b_id, match_id


def test_matches_show_the_other_persons_public_profile(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched_pair(env)
    (match,) = a.get("/matches").json()
    assert match["profile"]["user_id"] == b_id
    # Others see the assigned display name, never the real one.
    assert match["profile"]["name"] == env.s.manager.get_profile(b_id).name != "Bea"
    assert "Bea" not in a.get("/matches").text


def test_unmatch_removes_the_match_for_both(env: Env) -> None:
    a, _, b, _, match_id = matched_pair(env)
    outsider, _ = env.user("c@bu.edu", "man", ["woman"])
    assert outsider.post(f"/matches/{match_id}/unmatch").status_code == 404
    assert b.post(f"/matches/{match_id}/unmatch").status_code == 200
    assert a.get("/matches").json() == []
    assert b.post(f"/matches/{match_id}/unmatch").status_code == 409


def test_block_ends_the_match(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched_pair(env)
    assert b.post("/blocks", json={"user_id": a_id}).status_code == 201
    assert env.s.live.get_match(match_id).status == "unmatched"
    assert a.get("/matches").json() == []
    assert b.get("/blocks").json()[0]["user_id"] == a_id
    assert b.delete(f"/blocks/{a_id}").status_code == 200
    assert b.delete(f"/blocks/{a_id}").status_code == 404
    assert b.post("/blocks", json={"user_id": b_id}).status_code == 400


def test_reports_check_the_match_and_admins_resolve_them(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched_pair(env)
    _, c_id = env.user("c@bu.edu", "man", ["woman"])
    assert b.post("/reports", json={"reported_id": c_id, "match_id": match_id, "reason": "spam"}).status_code == 400
    assert b.post("/reports", json={"reported_id": a_id, "reason": "rude"}).status_code == 422
    assert b.post("/reports", json={"reported_id": a_id, "reason": "other", "details": "x" * 1001}).status_code == 422
    report_id = b.post("/reports", json={"reported_id": a_id, "match_id": match_id, "reason": "harassment", "details": "rude"}).json()["report_id"]

    admin = env.admin()
    assert [r["report_id"] for r in admin.get("/admin/reports").json()] == [report_id]
    assert b.post(f"/admin/reports/{report_id}/resolve").status_code == 403
    assert admin.post(f"/admin/reports/{report_id}/resolve").json()["status"] == "resolved"
    assert admin.get("/admin/reports").json() == []


# Deletion, privacy, and request safety


def test_deleting_an_account_removes_everything(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched_pair(env)
    a.post("/reports", json={"reported_id": b_id, "match_id": match_id, "reason": "spam"})
    b.post("/reports", json={"reported_id": a_id, "match_id": match_id, "reason": "fake"})
    _, c_id = env.user("c@bu.edu", "woman", ["man"])
    a.post("/blocks", json={"user_id": c_id})
    a_account = env.account_id(a)
    env.admin().get(f"/admin/reports")

    assert a.delete("/me").status_code == 200

    assert a.get("/me").status_code == 401
    with sqlite3.connect(env.settings.main_db) as conn:
        def count(sql: str, *args: str) -> int:
            return conn.execute(sql, args).fetchone()[0]

        assert count("SELECT COUNT(*) FROM accounts WHERE account_id = ? OR email_normalized = 'a@bu.edu'", a_account) == 0
        assert count("SELECT COUNT(*) FROM profiles WHERE user_id = ?", a_id) == 0
        assert count("SELECT COUNT(*) FROM preferences WHERE user_id = ?", a_id) == 0
        assert count("SELECT COUNT(*) FROM ratings WHERE rater_id = ? OR target_id = ?", a_id, a_id) == 0
        assert count("SELECT COUNT(*) FROM impressions WHERE viewer_id = ? OR target_id = ?", a_id, a_id) == 0
        assert count("SELECT COUNT(*) FROM matches WHERE user_a = ? OR user_b = ?", a_id, a_id) == 0
        assert count("SELECT COUNT(*) FROM blocks WHERE blocker_id = ? OR blocked_id = ?", a_id, a_id) == 0
        assert count("SELECT COUNT(*) FROM reports WHERE reporter_id = ?", a_id) == 0
        assert count("SELECT COUNT(*) FROM sessions WHERE account_id = ?", a_account) == 0
        assert count("SELECT COUNT(*) FROM passwords WHERE account_id = ?", a_account) == 0
        # The report about them is kept as a safety record, with them removed from it.
        assert count("SELECT COUNT(*) FROM reports WHERE reporter_id = ? AND reported_id IS NULL", b_id) == 1
        dump = "\n".join(conn.iterdump())
    assert a_id not in dump
    assert a_account not in dump
    assert "a@bu.edu" not in dump


def test_deletion_keeps_audit_rows_without_the_users_ids(env: Env) -> None:
    client, user_id = env.user("jane@bu.edu")
    account_id = env.account_id(client)
    assert any(e["target"] == account_id for e in env.s.live.audit_entries())
    client.delete("/me")
    entries = env.s.live.audit_entries()
    assert any(e["action"] == "account_approve" and e["target"] == "[deleted]" for e in entries)
    assert account_id not in json.dumps(entries)


def test_a_new_signup_can_delete_without_a_profile(env: Env) -> None:
    client = env.login("new@bu.edu")
    assert client.delete("/me").status_code == 200
    assert env.s.live.list_accounts(None) == []


def test_no_response_leaks_emails_preferences_or_personas(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched_pair(env)
    viewer, _ = env.user("v@bu.edu", "man", ["woman"])
    for user_id in (a_id, b_id):
        env.s.manager.add_persona(user_id, f"SECRET_PERSONA_{user_id}")
    a.post("/reports", json={"reported_id": b_id, "match_id": match_id, "reason": "spam"})
    responses = [
        a.get("/matches"),
        a.get("/blocks"),
        a.get("/me/profile"),
    ]
    feed = viewer.get("/feed/next")
    viewer.post("/ratings", json={"impression_id": feed.json()["impression_id"], "score": 5})
    responses += [
        feed,
        viewer.get("/matches"),
        env.admin().get("/admin/reports"),
        a.get("/auth/consent"),
    ]
    for response in responses:
        assert response.status_code == 200, response.text
        for secret in ("@bu.edu", "interested_in", "SECRET_PERSONA", "nonbinary\", \"woman"):
            assert secret not in response.text, (response.url, secret)
    # The two exceptions: your own /me, and the admin accounts list.
    assert "a@bu.edu" in a.get("/me").text
    assert "interested_in" in a.get("/me").text
    assert "@bu.edu" in env.admin().get("/admin/accounts", params={"status": "approved"}).text


def test_state_changing_requests_need_a_matching_origin(env: Env) -> None:
    client = env.login("jane@bu.edu")
    cookies = dict(client.cookies)
    for headers in ({"Origin": "https://evil.example"}, {"Referer": "https://evil.example/page"}, {}):
        hostile = TestClient(env.app, headers=headers, cookies=cookies)
        assert hostile.post("/auth/consent", json={"version": "2"}).status_code == 403
        assert hostile.delete("/me").status_code == 403
    assert client.get("/me").status_code == 200  # Not deleted.
    referer_only = TestClient(env.app, headers={"Referer": f"{SITE}/settings"}, cookies=cookies)
    assert referer_only.post("/auth/consent", json={"version": "2"}).status_code == 200


def test_security_headers_are_on_every_response(env: Env) -> None:
    for response in (env.client().get("/auth/consent"), env.client().get("/feed/next"), env.client("https://evil.example").post("/auth/logout")):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "default-src 'self'" in response.headers["content-security-policy"]
        assert "script-src 'self'" in response.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/matches/x/messages"),
        ("post", "/matches/x/messages"),
        ("post", "/me/photos"),
        ("get", "/photos/x"),
        ("get", "/admin/reports/x/messages"),
        ("post", "/auth/request-link"),
        ("get", "/auth/verify"),
    ],
)
def test_photo_and_message_routes_are_gone(env: Env, method: str, path: str) -> None:
    client, _ = env.user("jane@bu.edu")
    assert getattr(client, method)(path).status_code in (404, 405)


def test_a_whole_class_can_join_from_one_network(tmp_path: Path) -> None:
    settings = Settings(site_url=SITE, main_db=tmp_path / "m.db", persona_db=tmp_path / "p.db", login_requests_per_ip=3, cookie_secure=False)
    client = TestClient(create_app(settings), headers={"Origin": SITE})
    for i in range(10):
        assert client.post("/auth/signup", json={"email": f"s{i}@bu.edu", "password": PASSWORD}).status_code == 201
    codes = [client.post("/auth/signup", json={"email": "bad", "password": PASSWORD}).status_code for _ in range(4)]
    assert codes == [422, 422, 422, 429]
