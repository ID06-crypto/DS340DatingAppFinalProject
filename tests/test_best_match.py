"""Best matches: opting in, uploading pairs, and who may see whose real name, and when."""

import io
from html import unescape
import zipfile
from datetime import timedelta
from pathlib import Path

import pytest

from app import actions
from app.analytics import pseudonym
from live_store import utc_now
from test_api import ADMIN_EMAIL, Env
from test_ui import admin_page, csrf_of, make_env, page


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return make_env(tmp_path)


def pid(user_id: str) -> str:
    return pseudonym(user_id, "test-salt")


def person(env: Env, email: str, name: str, gender: str = "man", wants: bool = True):
    api, user_id = env.user(email, gender, ("woman", "man"), name=name)
    if wants:
        assert api.put("/me/best-match", json={"wants_best_match": True}).json() == {"wants_best_match": True}
    return api, user_id


def upload(env: Env, text: str):
    admin = admin_page(env)
    html = admin.get("/admin/best-matches").text
    return admin.post(
        "/admin/best-matches/upload",
        data={"csrf_token": csrf_of(html)},
        files={"pairs": ("pairs.csv", text.encode(), "text/csv")},
    )


def reveal(env: Env, when=None) -> None:
    admin = env.s.live.account_for_email(ADMIN_EMAIL)
    actions.set_results_open(env.s, admin, when or utc_now() - timedelta(minutes=1))


def matches_page(env: Env, api) -> str:
    response = page(env, api).get("/matches")
    assert response.status_code == 200
    return response.text


@pytest.fixture
def pair(env: Env):
    """Two people who both asked, paired by the algorithm."""
    a, a_id = person(env, "a@bu.edu", "Alexandra Realname", "woman")
    b, b_id = person(env, "b@bu.edu", "Benjamin Truename", "man")
    response = upload(env, f"participant_a,participant_b\n{pid(a_id)},{pid(b_id)}\n")
    assert response.status_code == 303, response.text
    return a, a_id, b, b_id


# Opting in


def test_the_consent_form_offers_it_unticked_and_records_the_choice(env: Env) -> None:
    for email, ticked in (("yes@bu.edu", True), ("no@bu.edu", False)):
        api = env.login(email)
        browser = page(env, api)
        html = browser.get("/onboarding/consent").text
        assert 'name="best_match"' in html and 'name="best_match" value="yes" aria-describedby="best-match-help" >' in html
        form = {"csrf_token": csrf_of(html), "version": "3", "agree": "yes"}
        if ticked:
            form["best_match"] = "yes"
        assert browser.post("/onboarding/consent", data=form).status_code == 303
        assert api.get("/me").json()["wants_best_match"] is ticked


def test_it_can_be_changed_on_the_me_page_and_is_audited(env: Env) -> None:
    api, _ = person(env, "a@bu.edu", "Ann Lee", wants=False)
    browser = page(env, api)
    html = browser.get("/me").text
    browser.post("/me/settings", data={"csrf_token": csrf_of(html), "best_match": "yes"})
    assert api.get("/me").json()["wants_best_match"] is True
    browser.post("/me/settings", data={"csrf_token": csrf_of(html)})
    assert api.get("/me").json()["wants_best_match"] is False
    actions_logged = [e["action"] for e in env.s.live.audit_entries("best_match_choice", None, limit=10)]
    assert len(actions_logged) == 2


def test_the_ground_truth_says_who_asked(env: Env) -> None:
    _, yes_id = person(env, "a@bu.edu", "Ann Lee", "woman")
    _, no_id = person(env, "b@bu.edu", "Bob Ray", wants=False)
    data = admin_page(env).get("/admin/truth.zip").content
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        name = next(n for n in archive.namelist() if n.endswith("participants.csv"))
        rows = archive.read(name).decode().splitlines()
    assert rows[0].endswith(",wants_best_match")
    flags = {row.split(",")[0]: row.split(",")[-1] for row in rows[1:]}
    assert flags == {pid(yes_id): "1", pid(no_id): "0"}


# Uploading pairs


def test_a_bad_file_saves_nothing_and_lists_every_problem(env: Env) -> None:
    _, a_id = person(env, "a@bu.edu", "Ann Lee", "woman")
    _, b_id = person(env, "b@bu.edu", "Bob Ray")
    _, c_id = person(env, "c@bu.edu", "Cy Fox")
    _, no_id = person(env, "d@bu.edu", "Di Noe", wants=False)
    text = (
        "participant_a,participant_b\n"
        f"{pid(a_id)},{pid(b_id)}\n"
        f"{pid(b_id)},{pid(c_id)}\n"  # b twice
        f"{pid(c_id)},{pid(c_id)}\n"  # with themselves
        f"{pid(no_id)},ffffffffffffffff\n"  # didn't ask, and unknown
    )
    response = upload(env, text)
    assert response.status_code == 422
    for expected in ("already in an earlier pair", "paired with themselves", "hasn't asked", "isn't a participant", "Nothing was saved"):
        assert expected in unescape(response.text), expected
    assert env.s.live.best_match_pairs() == []


def test_the_header_is_required(env: Env) -> None:
    _, a_id = person(env, "a@bu.edu", "Ann Lee", "woman")
    _, b_id = person(env, "b@bu.edu", "Bob Ray")
    assert upload(env, f"{pid(a_id)},{pid(b_id)}\n").status_code == 422


def test_a_new_upload_replaces_the_old_pairs(env: Env, pair) -> None:
    _, a_id, _, b_id = pair
    _, c_id = person(env, "c@bu.edu", "Cy Fox")
    assert upload(env, f"participant_a,participant_b\n{pid(a_id)},{pid(c_id)}\n").status_code == 303
    assert env.s.live.best_match_of(a_id) == c_id and env.s.live.best_match_of(b_id) is None
    assert [e["action"] for e in env.s.live.audit_entries("upload_best_matches", None, limit=10)] == ["upload_best_matches"] * 2


def test_the_admin_page_lists_pairs_and_can_clear_them(env: Env, pair) -> None:
    admin = admin_page(env)
    html = admin.get("/admin/best-matches").text
    assert "2 of 2 participants have asked" in html
    assert pid(pair[1]) in html and "Alexandra" not in html  # Display names, not real names.
    admin.post("/admin/best-matches/clear", data={"csrf_token": csrf_of(html)})
    assert env.s.live.best_match_pairs() == []


# Who sees what, and when


def test_nothing_shows_before_results_come_out(env: Env, pair) -> None:
    a, _, _, _ = pair
    html = matches_page(env, a)
    assert "Results are hidden" not in html and "Results come out after the study ends" in html
    assert "Truename" not in html
    reveal(env, utc_now() + timedelta(days=2))
    html = matches_page(env, a)
    assert "Results come out on" in html and "Truename" not in html
    assert a.get("/me/best-match").json()["state"] == "waiting"


def test_once_out_both_see_each_others_real_name(env: Env, pair) -> None:
    a, a_id, b, b_id = pair
    reveal(env)
    assert "Benjamin Truename" in matches_page(env, a)
    assert "Alexandra Realname" in matches_page(env, b)
    found = a.get("/me/best-match").json()
    assert found["state"] == "revealed" and found["real_name"] == "Benjamin Truename"
    assert "email" not in str(found) and "b@bu.edu" not in str(found)


def test_if_either_changes_their_mind_neither_sees_the_other(env: Env, pair) -> None:
    a, _, b, _ = pair
    reveal(env)
    b.put("/me/best-match", json={"wants_best_match": False})
    for client, hidden in ((a, "Truename"), (b, "Realname")):
        html = matches_page(env, client)
        assert hidden not in html
    assert a.get("/me/best-match").json()["state"] == "none"
    assert b.get("/me/best-match").json()["state"] == "not_opted_in"
    assert "no longer wants" in admin_page(env).get("/admin/best-matches").text


def test_someone_who_never_asked_sees_nothing_even_if_paired(env: Env) -> None:
    """Uploading checks this, but the page checks again, in case it changes later."""
    a, a_id = person(env, "a@bu.edu", "Ann Realname", "woman")
    b, b_id = person(env, "b@bu.edu", "Bob Truename", wants=False)
    env.s.live.replace_best_matches([(a_id, b_id)])
    reveal(env)
    assert "Truename" not in matches_page(env, a)
    assert "Realname" not in matches_page(env, b)


def test_a_block_hides_the_pair(env: Env, pair) -> None:
    a, a_id, b, b_id = pair
    reveal(env)
    env.s.live.add_block(b_id, a_id)
    assert "Truename" not in matches_page(env, a)
    assert "Realname" not in matches_page(env, b)


def test_a_suspended_partner_is_hidden(env: Env, pair) -> None:
    a, _, _, b_id = pair
    reveal(env)
    env.s.live.set_status(env.s.live.account_for_user(b_id).account_id, "suspended", "test")
    assert "Truename" not in matches_page(env, a)


def test_deleting_an_account_removes_the_pair(env: Env, pair) -> None:
    a, a_id, b, _ = pair
    reveal(env)
    assert b.delete("/me").status_code == 200
    assert env.s.live.best_match_of(a_id) is None
    assert a.get("/me/best-match").json()["state"] == "none"


def test_admins_cannot_opt_in(env: Env) -> None:
    api = env.admin()
    api.put("/me/best-match", json={"wants_best_match": True})
    assert env.s.live.users_wanting_best_match() == set()
