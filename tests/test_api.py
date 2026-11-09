import io
import json
import logging
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import matchers as matcher_registry
from app.config import Settings
from app.emails import ConsoleEmailSender
from app.main import create_app
from app.photos import MAX_PHOTO_BYTES
from app.security import RedactTokens, normalize_email
from live_store import sha256
from matching import Matcher, PublicView, RandomMatcher
from models import LIVE_RUN_ID

SITE = "http://testserver"
ADMIN_EMAIL = "admin@bu.edu"
EVERYONE = ["man", "woman", "nonbinary"]


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


def jpeg(color: str = "red", size: tuple[int, int] = (64, 48), exif: bool = False) -> bytes:
    image = Image.new("RGB", size, color)
    output = io.BytesIO()
    if exif:
        data = Image.Exif()
        data[0x010F] = "SECRETCAMERA"  # Make
        data[0x8825] = {1: "N", 2: (42.0, 21.0, 0.0), 3: "W", 4: (71.0, 6.0, 0.0)}  # GPS
        image.save(output, "JPEG", exif=data)
    else:
        image.save(output, "JPEG")
    return output.getvalue()


@dataclass
class Env:
    app: Any
    settings: Settings
    sender: ConsoleEmailSender
    _admin: TestClient | None = None

    @property
    def s(self) -> Any:
        return self.app.state.services

    def client(self, origin: str = SITE) -> TestClient:
        return TestClient(self.app, headers={"Origin": origin})

    def link_token(self, email: str) -> str:
        to = normalize_email(email)
        message = [m for m in self.sender.outbox if m["to"] == to][-1]
        return re.search(r"token=(\S+)", message["body"]).group(1)

    def login(self, email: str) -> TestClient:
        client = self.client()
        assert client.post("/auth/request-link", json={"email": email}).status_code == 200
        assert client.get("/auth/verify", params={"token": self.link_token(email)}).status_code == 200
        return client

    def consented(self, email: str) -> TestClient:
        client = self.login(email)
        assert client.post("/auth/consent", json={"version": "1"}).status_code == 200
        return client

    def admin(self) -> TestClient:
        """Signed in once per test, because sign-in links are rate limited per email."""
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
        upload = client.post("/me/photos", files={"file": ("me.jpg", jpeg(), "image/jpeg")})
        assert upload.status_code == 201, upload.text
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
        photo_dir=tmp_path / "photos",
        cookie_secure=False,  # TestClient uses plain http.
        login_requests_per_email=3,
        login_requests_per_ip=1000,
    )
    sender = ConsoleEmailSender(stream=None)
    return Env(create_app(settings, email_sender=sender, rng=random.Random(1)), settings, sender)


def shown_ids(client: TestClient, limit: int = 50) -> list[str]:
    """Walk the whole feed, returning everyone it shows, without rating anyone."""
    shown = []
    for _ in range(limit):
        response = client.get("/feed/next").json()
        if response["profile"] is None:
            return shown
        shown.append(response["profile"]["user_id"])
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
        ("jane@gmail.com", None),
        ("jane@bu.edu.evil.com", None),
        ("jane@sub.bu.edu", None),
        ("+tag@bu.edu", None),
        ("no-at-sign", None),
        ("a@b@bu.edu", None),
        ("", None),
    ],
)
def test_email_normalization(raw: str, expected: str | None) -> None:
    assert normalize_email(raw, ["bu.edu"]) == expected


@pytest.mark.parametrize("email", ["someone@gmail.com", "not an email", "x@bu.edu.com", "unknown@bu.edu"])
def test_request_link_gives_the_same_reply_for_any_address(env: Env, email: str) -> None:
    response = env.client().post("/auth/request-link", json={"email": email})
    assert response.status_code == 200
    assert response.json() == {"message": "If that address can sign in, a link is on its way."}


def test_links_are_only_sent_to_allowed_domains(env: Env) -> None:
    env.client().post("/auth/request-link", json={"email": "jane@gmail.com"})
    assert env.sender.outbox == []
    env.client().post("/auth/request-link", json={"email": "Jane+x@BU.edu"})
    assert [m["to"] for m in env.sender.outbox] == ["jane@bu.edu"]


def test_login_tokens_are_hashed_single_use_and_expire(env: Env) -> None:
    client = env.client()
    client.post("/auth/request-link", json={"email": "jane@bu.edu"})
    token = env.link_token("jane@bu.edu")
    with sqlite3.connect(env.settings.main_db) as conn:
        stored = [row[0] for row in conn.execute("SELECT token_hash FROM login_tokens")]
    assert stored == [sha256(token)]
    assert token not in env.settings.main_db.read_bytes().decode("latin-1")

    assert client.get("/auth/verify", params={"token": token}).status_code == 200
    assert env.client().get("/auth/verify", params={"token": token}).status_code == 400

    env.client().post("/auth/request-link", json={"email": "late@bu.edu"})
    late = env.link_token("late@bu.edu")
    with sqlite3.connect(env.settings.main_db) as conn:
        conn.execute("UPDATE login_tokens SET expires_at = '2000-01-01T00:00:00.000000+00:00' WHERE token_hash = ?", (sha256(late),))
    assert env.client().get("/auth/verify", params={"token": late}).status_code == 400
    assert env.client().get("/auth/verify", params={"token": "made-up"}).status_code == 400


def test_sessions_are_hashed_and_cookies_are_locked_down(env: Env) -> None:
    client = env.client()
    client.post("/auth/request-link", json={"email": "jane@bu.edu"})
    response = client.get("/auth/verify", params={"token": env.link_token("jane@bu.edu")})
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    session_id = client.cookies["session"]
    with sqlite3.connect(env.settings.main_db) as conn:
        stored = [row[0] for row in conn.execute("SELECT session_hash FROM sessions")]
    assert stored == [sha256(session_id)]
    assert response.json()["status"] == "pending"


def test_secure_cookie_flag_is_set_when_configured(tmp_path: Path) -> None:
    settings = Settings(site_url=SITE, main_db=tmp_path / "m.db", persona_db=tmp_path / "p.db", photo_dir=tmp_path / "ph", cookie_secure=True)
    sender = ConsoleEmailSender(stream=None)
    env = Env(create_app(settings, email_sender=sender), settings, sender)
    client = env.client()
    client.post("/auth/request-link", json={"email": "jane@bu.edu"})
    response = client.get("/auth/verify", params={"token": env.link_token("jane@bu.edu")})
    assert "Secure" in response.headers["set-cookie"]
    assert "Strict-Transport-Security" in response.headers


def test_per_email_limit_is_silent_and_per_ip_limit_is_429(env: Env, tmp_path: Path) -> None:
    for _ in range(5):
        assert env.client().post("/auth/request-link", json={"email": "jane@bu.edu"}).status_code == 200
    assert len(env.sender.outbox) == env.settings.login_requests_per_email

    settings = Settings(site_url=SITE, main_db=tmp_path / "m2.db", persona_db=tmp_path / "p2.db", photo_dir=tmp_path / "ph2", login_requests_per_ip=2)
    limited = TestClient(create_app(settings, email_sender=ConsoleEmailSender(stream=None)), headers={"Origin": SITE})
    codes = [limited.post("/auth/request-link", json={"email": f"u{i}@bu.edu"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


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
    assert client.get("/auth/consent").json()["version"] == "1"
    assert "aren't\nused to train new models" in client.get("/auth/consent").json()["text"]
    assert client.post("/me/profile", json=profile_body()).status_code == 403
    assert client.get("/feed/next").status_code == 403
    assert client.post("/auth/consent", json={"version": "0"}).status_code == 409
    assert client.post("/auth/consent", json={"version": "1"}).status_code == 200
    assert client.post("/me/profile", json=profile_body()).status_code == 201


def test_unapproved_accounts_get_403_on_feed_ratings_matches_and_chat(env: Env) -> None:
    client, _ = env.user("jane@bu.edu", approve=False)
    assert client.get("/feed/next").status_code == 403
    assert client.post("/ratings", json={"impression_id": "x", "score": 5}).status_code == 403
    assert client.get("/matches").status_code == 403
    assert client.get("/matches/x/messages").status_code == 403
    assert client.post("/matches/x/messages", json={"body": "hi"}).status_code == 403
    assert client.post("/blocks", json={"user_id": "x"}).status_code == 403
    # Their own profile, preferences, and photos still work.
    assert client.get("/me/profile").status_code == 200
    assert client.get("/me/preferences").status_code == 200
    photo_id = client.get("/me").json()["photo_ids"][0]
    assert client.get(f"/photos/{photo_id}").status_code == 200


def test_approval_requires_consent_profile_preferences_and_photo(env: Env) -> None:
    admin = env.admin()
    client = env.login("jane@bu.edu")
    account_id = env.account_id(client)

    def missing() -> list[str]:
        response = admin.post(f"/admin/accounts/{account_id}/approve")
        return response.json()["detail"]["missing"] if response.status_code == 409 else []

    assert missing() == ["consent", "profile", "photo"]
    client.post("/auth/consent", json={"version": "1"})
    client.post("/me/profile", json=profile_body())
    assert missing() == ["preferences", "photo"]
    client.put("/me/preferences", json={"interested_in": "everyone"})
    assert missing() == ["photo"]
    client.post("/me/photos", files={"file": ("a.jpg", jpeg(), "image/jpeg")})
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


def test_admin_queue_lists_pending_accounts_with_email_and_photos(env: Env) -> None:
    client, _ = env.user("jane@bu.edu", approve=False)
    queue = env.admin().get("/admin/accounts", params={"status": "pending"}).json()
    assert [a["email"] for a in queue] == ["jane@bu.edu"]
    assert queue[0]["profile"]["name"] == "Sam"
    assert len(queue[0]["photo_ids"]) == 1
    assert env.admin().get(f"/photos/{queue[0]['photo_ids'][0]}").status_code == 200


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


def test_mutual_gender_filter(env: Env) -> None:
    man_for_women, man_id = env.user("m1@bu.edu", "man", ["woman"])
    woman_for_women, ww_id = env.user("w1@bu.edu", "woman", ["woman"])
    woman_for_men, wm_id = env.user("w2@bu.edu", "woman", ["man"])
    assert shown_ids(man_for_women) == [wm_id]
    assert shown_ids(woman_for_women) == []
    assert shown_ids(woman_for_men) == [man_id]


def test_everyone_setting_sees_all_three_genders(env: Env) -> None:
    viewer, _ = env.user("v@bu.edu", "man", "everyone")
    others = {
        env.user("a@bu.edu", "man", ["man"])[1],
        env.user("b@bu.edu", "woman", ["man"])[1],
        env.user("c@bu.edu", "nonbinary", "everyone")[1],
    }
    env.user("d@bu.edu", "woman", ["woman"])  # Not interested in men.
    assert set(shown_ids(viewer)) == others


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


def test_already_shown_or_rated_profiles_never_repeat(env: Env) -> None:
    viewer, _ = env.user("v@bu.edu", "man", ["woman"])
    ids = {env.user(f"w{i}@bu.edu", "woman", ["man"])[1] for i in range(3)}
    seen = []
    for score in (5, 6, 7):
        feed = viewer.get("/feed/next").json()
        seen.append(feed["profile"]["user_id"])
        viewer.post("/ratings", json={"impression_id": feed["impression_id"], "score": score})
    assert set(seen) == ids
    empty = viewer.get("/feed/next").json()
    assert empty == {"profile": None, "pool_size": 0, "message": "No one left to show right now."}


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
    assert set(body) == {"impression_id", "profile", "photo_ids"}
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
        settings = Settings(site_url=SITE, admin_emails=(ADMIN_EMAIL,), main_db=folder / "m.db", persona_db=folder / "p.db", photo_dir=folder / "ph", cookie_secure=False, enabled_matchers=("random", "first"))
        sender = ConsoleEmailSender(stream=None)
        env = Env(create_app(settings, email_sender=sender, rng=random.Random(seed)), settings, sender)
        viewer, _ = env.user("v@bu.edu", "man", ["woman"])
        for i in range(8):
            env.user(f"w{i}@bu.edu", "woman", ["man"])
        names = []
        for _ in range(8):
            feed = viewer.get("/feed/next").json()
            names.append(env.s.live.get_impression(feed["impression_id"]).matcher)
        return names

    assert chosen(3) == chosen(3)
    assert set(chosen(3)) == {"random", "first"}


def test_unknown_matcher_names_are_refused(tmp_path: Path) -> None:
    settings = Settings(main_db=tmp_path / "m.db", persona_db=tmp_path / "p.db", photo_dir=tmp_path / "ph", enabled_matchers=("elo",))
    with pytest.raises(ValueError, match="ENABLED_MATCHERS"):
        create_app(settings, email_sender=ConsoleEmailSender(stream=None))


class OutsidePoolMatcher(RandomMatcher):
    name = "rogue"

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        return user_id  # Never eligible.


def test_a_pick_outside_the_pool_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(matcher_registry.REGISTRY, "rogue", OutsidePoolMatcher)
    settings = Settings(site_url=SITE, admin_emails=(ADMIN_EMAIL,), main_db=tmp_path / "m.db", persona_db=tmp_path / "p.db", photo_dir=tmp_path / "ph", cookie_secure=False, enabled_matchers=("rogue",))
    sender = ConsoleEmailSender(stream=None)
    env = Env(create_app(settings, email_sender=sender), settings, sender)
    viewer, _ = env.user("v@bu.edu", "man", ["woman"])
    env.user("w@bu.edu", "woman", ["man"])
    assert viewer.get("/feed/next").status_code == 500


def test_ratings_need_your_own_unrated_impression(env: Env) -> None:
    viewer, viewer_id = env.user("v@bu.edu", "man", ["woman"])
    other, _ = env.user("o@bu.edu", "man", ["woman"])
    _, target_id = env.user("t@bu.edu", "woman", ["man"])
    feed = viewer.get("/feed/next").json()
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


# Matches, chat, and safety


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
    assert match["profile"]["name"] == "Bea"
    assert len(match["photo_ids"]) == 1
    assert a.get(f"/photos/{match['photo_ids'][0]}").status_code == 200


def test_chat_is_only_for_participants_of_an_active_match(env: Env) -> None:
    a, _, b, _, match_id = matched_pair(env)
    outsider, _ = env.user("c@bu.edu", "man", ["woman"])
    assert a.post(f"/matches/{match_id}/messages", json={"body": "hi"}).status_code == 201
    reply = b.post(f"/matches/{match_id}/messages", json={"body": "hello"}).json()
    assert [m["body"] for m in b.get(f"/matches/{match_id}/messages").json()] == ["hi", "hello"]
    assert a.get(f"/matches/{match_id}/messages", params={"after": reply["message_id"] - 1}).json()[0]["body"] == "hello"
    assert outsider.get(f"/matches/{match_id}/messages").status_code == 404
    assert outsider.post(f"/matches/{match_id}/messages", json={"body": "x"}).status_code == 404


@pytest.mark.parametrize("body", ["", "   ", "x" * 2001])
def test_message_length_is_enforced(env: Env, body: str) -> None:
    a, _, _, _, match_id = matched_pair(env)
    assert a.post(f"/matches/{match_id}/messages", json={"body": body}).status_code == 422


def test_unmatch_stops_messaging(env: Env) -> None:
    a, _, b, _, match_id = matched_pair(env)
    assert b.post(f"/matches/{match_id}/unmatch").status_code == 200
    assert a.post(f"/matches/{match_id}/messages", json={"body": "hi"}).status_code == 403
    assert a.get(f"/matches/{match_id}/messages").status_code == 403
    assert a.get("/matches").json() == []
    assert b.post(f"/matches/{match_id}/unmatch").status_code == 409


def test_block_ends_the_match_and_stops_messaging(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched_pair(env)
    assert b.post("/blocks", json={"user_id": a_id}).status_code == 201
    assert env.s.live.get_match(match_id).status == "unmatched"
    assert a.post(f"/matches/{match_id}/messages", json={"body": "hi"}).status_code == 403
    assert b.get("/blocks").json()[0]["user_id"] == a_id
    assert b.delete(f"/blocks/{a_id}").status_code == 200
    assert b.delete(f"/blocks/{a_id}").status_code == 404
    assert b.post("/blocks", json={"user_id": b_id}).status_code == 400


def test_message_rate_limit(env: Env) -> None:
    a, _, _, _, match_id = matched_pair(env)
    codes = [a.post(f"/matches/{match_id}/messages", json={"body": str(i)}).status_code for i in range(env.settings.messages_per_minute + 1)]
    assert codes[-1] == 429
    assert codes[:-1] == [201] * env.settings.messages_per_minute


def test_reports_check_the_match_and_only_admins_read_messages_with_audit(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched_pair(env)
    _, c_id = env.user("c@bu.edu", "man", ["woman"])
    a.post(f"/matches/{match_id}/messages", json={"body": "PRIVATE MESSAGE"})
    assert b.post("/reports", json={"reported_id": c_id, "match_id": match_id, "reason": "spam"}).status_code == 400
    assert b.post("/reports", json={"reported_id": a_id, "reason": "rude"}).status_code == 422
    assert b.post("/reports", json={"reported_id": a_id, "reason": "other", "details": "x" * 1001}).status_code == 422
    report_id = b.post("/reports", json={"reported_id": a_id, "match_id": match_id, "reason": "harassment", "details": "rude"}).json()["report_id"]

    admin = env.admin()
    reports = admin.get("/admin/reports")
    assert "PRIVATE MESSAGE" not in reports.text
    assert b.get(f"/admin/reports/{report_id}/messages").status_code == 403
    before = len([e for e in env.s.live.audit_entries() if e["action"] == "read_report_messages"])
    messages = admin.get(f"/admin/reports/{report_id}/messages").json()
    assert [m["body"] for m in messages] == ["PRIVATE MESSAGE"]
    reads = [e for e in env.s.live.audit_entries() if e["action"] == "read_report_messages"]
    assert len(reads) == before + 1
    assert reads[-1]["target"] == report_id
    assert admin.post(f"/admin/reports/{report_id}/resolve").json()["status"] == "resolved"
    assert admin.get("/admin/reports").json() == []


# Photos


def test_photos_are_reencoded_without_exif_and_stored_under_random_names(env: Env) -> None:
    client, user_id = env.user("jane@bu.edu")
    original = jpeg("blue", (2400, 1200), exif=True)
    assert Image.open(io.BytesIO(original)).getexif().get_ifd(0x8825)  # The test file has GPS.
    upload = client.post("/me/photos", files={"file": ("jane_holiday.jpg", original, "image/jpeg")})
    assert upload.status_code == 201
    photo = env.s.live.get_photo(upload.json()["photo_id"])
    assert re.fullmatch(r"[0-9a-f]{32}\.jpg", photo.storage_key)
    assert user_id not in photo.storage_key
    served = client.get(f"/photos/{photo.photo_id}")
    assert served.headers["cache-control"] == "private"
    assert served.headers["content-type"] == "image/jpeg"
    image = Image.open(io.BytesIO(served.content))
    assert dict(image.getexif()) == {}
    assert max(image.size) == 1600
    assert b"SECRETCAMERA" not in served.content
    on_disk = (env.settings.photo_dir / photo.storage_key).read_bytes()
    assert b"SECRETCAMERA" not in on_disk


@pytest.mark.parametrize(
    ("content", "status"),
    [(b"not an image at all", 415), (b"GIF89a" + b"\x00" * 50, 415), (b"\x89PNG\r\n\x1a\nbroken", 415)],
)
def test_non_images_are_rejected(env: Env, content: bytes, status: int) -> None:
    client, _ = env.user("jane@bu.edu")
    response = client.post("/me/photos", files={"file": ("x.jpg", content, "image/jpeg")})
    assert response.status_code == status
    assert len(client.get("/me").json()["photo_ids"]) == 1


def test_oversized_uploads_and_too_many_photos_are_rejected(env: Env) -> None:
    client, _ = env.user("jane@bu.edu")
    big = b"\xff\xd8" + b"\x00" * MAX_PHOTO_BYTES
    assert client.post("/me/photos", files={"file": ("big.jpg", big, "image/jpeg")}).status_code == 413
    for _ in range(4):
        assert client.post("/me/photos", files={"file": ("a.jpg", jpeg(), "image/jpeg")}).status_code == 201
    assert client.post("/me/photos", files={"file": ("a.jpg", jpeg(), "image/jpeg")}).status_code == 409
    assert len(list(env.settings.photo_dir.iterdir())) == 5


def test_photos_can_be_reordered_and_deleted(env: Env) -> None:
    client, _ = env.user("jane@bu.edu")
    client.post("/me/photos", files={"file": ("a.jpg", jpeg("green"), "image/jpeg")})
    first, second = client.get("/me").json()["photo_ids"]
    assert client.put("/me/photos", json={"photo_ids": [second, first]}).status_code == 200
    assert client.get("/me").json()["photo_ids"] == [second, first]
    assert client.put("/me/photos", json={"photo_ids": [second]}).status_code == 400
    key = env.s.live.get_photo(second).storage_key
    assert client.delete(f"/me/photos/{second}").json()["photo_ids"] == [first]
    assert not (env.settings.photo_dir / key).exists()
    other, _ = env.user("o@bu.edu")
    assert other.delete(f"/me/photos/{first}").status_code == 404


def test_photo_route_denies_users_who_may_not_see_the_owner(env: Env) -> None:
    owner, owner_id = env.user("w@bu.edu", "woman", ["man"])
    photo_id = owner.get("/me").json()["photo_ids"][0]
    viewer, _ = env.user("m@bu.edu", "man", ["woman"])
    stranger, _ = env.user("x@bu.edu", "woman", ["woman"])
    unapproved, _ = env.user("p@bu.edu", "man", ["woman"], approve=False)
    assert viewer.get(f"/photos/{photo_id}").status_code == 200  # In their pool.
    assert stranger.get(f"/photos/{photo_id}").status_code == 404
    assert unapproved.get(f"/photos/{photo_id}").status_code == 404
    assert env.client().get(f"/photos/{photo_id}").status_code == 401
    viewer.get("/feed/next")
    assert viewer.get(f"/photos/{photo_id}").status_code == 200  # Shown, so out of the pool now.
    owner.post("/blocks", json={"user_id": viewer.get("/me").json()["profile"]["user_id"]})
    assert viewer.get(f"/photos/{photo_id}").status_code == 404


# Deletion, privacy, and request safety


def test_deleting_an_account_removes_everything_including_photo_files(env: Env) -> None:
    a, a_id, b, b_id, match_id = matched_pair(env)
    a.post(f"/matches/{match_id}/messages", json={"body": "hello"})
    a.post("/reports", json={"reported_id": b_id, "match_id": match_id, "reason": "spam"})
    b.post("/reports", json={"reported_id": a_id, "match_id": match_id, "reason": "fake"})
    _, c_id = env.user("c@bu.edu", "woman", ["man"])
    a.post("/blocks", json={"user_id": c_id})
    a_account = env.account_id(a)
    env.admin().get(f"/admin/reports")
    keys = [p.storage_key for p in env.s.live.photos_for(a_id)]
    assert all((env.settings.photo_dir / k).exists() for k in keys)

    assert a.delete("/me").status_code == 200

    assert a.get("/me").status_code == 401
    assert not any((env.settings.photo_dir / k).exists() for k in keys)
    with sqlite3.connect(env.settings.main_db) as conn:
        def count(sql: str, *args: str) -> int:
            return conn.execute(sql, args).fetchone()[0]

        assert count("SELECT COUNT(*) FROM accounts WHERE account_id = ? OR email_normalized = 'a@bu.edu'", a_account) == 0
        assert count("SELECT COUNT(*) FROM profiles WHERE user_id = ?", a_id) == 0
        assert count("SELECT COUNT(*) FROM preferences WHERE user_id = ?", a_id) == 0
        assert count("SELECT COUNT(*) FROM ratings WHERE rater_id = ? OR target_id = ?", a_id, a_id) == 0
        assert count("SELECT COUNT(*) FROM impressions WHERE viewer_id = ? OR target_id = ?", a_id, a_id) == 0
        assert count("SELECT COUNT(*) FROM matches WHERE user_a = ? OR user_b = ?", a_id, a_id) == 0
        assert count("SELECT COUNT(*) FROM messages WHERE sender_id = ?", a_id) == 0
        assert count("SELECT COUNT(*) FROM blocks WHERE blocker_id = ? OR blocked_id = ?", a_id, a_id) == 0
        assert count("SELECT COUNT(*) FROM reports WHERE reporter_id = ?", a_id) == 0
        assert count("SELECT COUNT(*) FROM sessions WHERE account_id = ?", a_account) == 0
        assert count("SELECT COUNT(*) FROM login_tokens WHERE email_normalized = 'a@bu.edu'") == 0
        assert count("SELECT COUNT(*) FROM photos WHERE user_id = ?", a_id) == 0
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
    a.post(f"/matches/{match_id}/messages", json={"body": "hi"})
    report_id = a.post("/reports", json={"reported_id": b_id, "match_id": match_id, "reason": "spam"}).json()["report_id"]
    responses = [
        a.get("/matches"),
        a.get(f"/matches/{match_id}/messages"),
        a.get("/blocks"),
        a.get("/me/profile"),
    ]
    feed = viewer.get("/feed/next")
    viewer.post("/ratings", json={"impression_id": feed.json()["impression_id"], "score": 5})
    responses += [
        feed,
        viewer.get("/matches"),
        env.admin().get("/admin/reports"),
        env.admin().get(f"/admin/reports/{report_id}/messages"),
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
        assert hostile.post("/auth/consent", json={"version": "1"}).status_code == 403
        assert hostile.delete("/me").status_code == 403
    assert client.get("/me").status_code == 200  # Not deleted.
    referer_only = TestClient(env.app, headers={"Referer": f"{SITE}/settings"}, cookies=cookies)
    assert referer_only.post("/auth/consent", json={"version": "1"}).status_code == 200


def test_security_headers_are_on_every_response(env: Env) -> None:
    for response in (env.client().get("/auth/consent"), env.client().get("/feed/next"), env.client("https://evil.example").post("/auth/logout")):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "default-src 'none'" in response.headers["content-security-policy"]
        assert response.headers["cache-control"] == "no-store"


def test_tokens_are_redacted_from_access_logs() -> None:
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s"', ("1.2.3.4", "GET", "/auth/verify?token=abc123&x=1"), None)
    RedactTokens().filter(record)
    assert "abc123" not in record.getMessage()
    assert "token=[redacted]" in record.getMessage()
