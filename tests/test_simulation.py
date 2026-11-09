import json
import logging
import random
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

import raters
from manager import ProfileManager
from matching import (
    CompatibleRandomMatcher,
    Matcher,
    PublicView,
    RandomMatcher,
)
from metrics import learning_curve
from run_metrics import compute_metrics
from models import Persona, Profile, Rating, generate_user_id, mutually_interested
from personas import PersonaStore
from raters import (
    JEV_MODEL,
    JEV_QUESTION,
    FakeRater,
    JevRater,
    Rater,
    RatingResult,
    build_jev_state,
    build_rater_input,
)
from simulation import (
    MAX_CONSECUTIVE_FAILURES,
    NoEligibleUsersError,
    RunAbortedError,
    run_simulation,
)
from store import DuplicateRunError, Store

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN = "run-1"
SEED = 42


def profile_data(name: str = "Sam") -> dict[str, Any]:
    return {
        "name": name,
        "age": 30,
        "height_cm": 175,
        "religion": "none",
        "school": "BU",
        "major": "biology",
        "job": "engineer",
        "interests": ["hiking"],
        "clubs": ["chess club"],
        "gender": "man",
        "looking_for": "long-term",
        "agent": True,
    }


@pytest.fixture
def manager(tmp_path: Path) -> ProfileManager:
    return ProfileManager(tmp_path / "main.db", tmp_path / "personas.db")


EVERYONE = ["man", "woman", "nonbinary"]


def add_users(manager: ProfileManager, count: int, persona: bool = True) -> list[str]:
    """Users who are all interested in everyone, so each is in every other's pool."""
    users = []
    for _ in range(count):
        user_id = manager.add_profile(profile_data()).user_id
        manager.set_preferences(user_id, EVERYONE)
        if persona:
            manager.add_persona(user_id, f"SECRET_PERSONA_{user_id}")
        users.append(user_id)
    return users


def seeded_run(
    manager: ProfileManager, run_id: str = RUN, n_steps: int = 10, seed: int = SEED
) -> Any:
    return run_simulation(
        manager,
        RandomMatcher(random.Random(seed)),
        FakeRater(random.Random(seed)),
        run_id,
        n_steps,
        seed,
    )


class ScriptedRater(Rater):
    """Returns or raises whatever is scripted for each call, in order."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = iter(outcomes)

    def rate(
        self, rater_profile: Profile, rater_persona: Persona, target_profile: Profile
    ) -> RatingResult:
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome()
        return RatingResult(score=outcome)


class RecordingMatcher(RandomMatcher):
    name = "recording"

    def __init__(self, rng: random.Random) -> None:
        super().__init__(rng)
        self.observed: list[Rating] = []

    def observe(self, rating: Rating) -> None:
        self.observed.append(rating)


# RandomMatcher


def test_random_matcher_skips_self_and_never_repeats(manager: ProfileManager) -> None:
    users = add_users(manager, 6)
    view = PublicView(manager.store, RUN)
    matcher = RandomMatcher(random.Random(SEED))
    user = users[0]

    shown = []
    for _ in range(len(users) - 1):
        target = matcher.next_profile(user, view)
        assert target is not None
        shown.append(target)
        manager.add_rating(user, target, 0.5, RUN)

    assert user not in shown
    assert len(set(shown)) == len(shown)
    assert set(shown) == set(users) - {user}


def test_random_matcher_returns_none_when_user_has_seen_everyone(
    manager: ProfileManager,
) -> None:
    users = add_users(manager, 3)
    view = PublicView(manager.store, RUN)
    matcher = RandomMatcher(random.Random(SEED))
    for target in users[1:]:
        manager.add_rating(users[0], target, 0.5, RUN)
    assert matcher.next_profile(users[0], view) is None


def test_random_matcher_returns_none_when_user_is_alone(
    manager: ProfileManager,
) -> None:
    user = add_users(manager, 1)[0]
    view = PublicView(manager.store, RUN)
    assert RandomMatcher(random.Random(SEED)).next_profile(user, view) is None


def test_random_matcher_only_counts_current_run(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 0.5, "other-run")
    view = PublicView(manager.store, RUN)
    assert RandomMatcher(random.Random(SEED)).next_profile(alice, view) == bob


def picks(manager: ProfileManager, user: str, seed: int) -> list[str | None]:
    view = PublicView(manager.store, RUN)
    matcher = RandomMatcher(random.Random(seed))
    return [matcher.next_profile(user, view) for _ in range(30)]


def test_random_matcher_is_reproducible_and_seed_dependent(
    manager: ProfileManager,
) -> None:
    users = add_users(manager, 10)
    assert picks(manager, users[0], 1) == picks(manager, users[0], 1)
    assert picks(manager, users[0], 1) != picks(manager, users[0], 2)


def test_random_matcher_ignores_profile_storage_order(tmp_path: Path) -> None:
    first = ProfileManager(tmp_path / "a.db", tmp_path / "a_p.db")
    second = ProfileManager(tmp_path / "b.db", tmp_path / "b_p.db")
    ids = sorted(first.add_profile(profile_data()).user_id for _ in range(6))
    for user_id in reversed(ids):
        second.add_profile({**profile_data(), "user_id": user_id})
    assert picks(first, ids[0], 3) == picks(second, ids[0], 3)


def test_matcher_must_implement_next_profile() -> None:
    with pytest.raises(TypeError):
        Matcher()  # type: ignore[abstract]


# Ratings across runs


def test_same_pair_can_be_rated_in_different_runs(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    seeded_run(manager, "run-a", n_steps=2)
    seeded_run(manager, "run-b", n_steps=2)
    assert manager.already_rated_pairs("run-a") == {(alice, bob), (bob, alice)}
    assert manager.already_rated_pairs("run-b") == {(alice, bob), (bob, alice)}
    assert manager.count_ratings() == 4


# Full runs


def test_full_run_stores_expected_ratings(manager: ProfileManager) -> None:
    add_users(manager, 4)
    expected = 4 * 3

    summary = seeded_run(manager, n_steps=expected)

    assert summary.run_id == RUN
    assert summary.steps_completed == expected
    assert summary.failures == 0
    ratings = manager.ratings_in_run(RUN)
    assert len(ratings) == expected
    assert [r.step for r in ratings] == list(range(1, expected + 1))
    assert {r.run_id for r in ratings} == {RUN}
    assert {r.model_version for r in ratings} == {FakeRater.MODEL_VERSION}
    transcripts = manager.transcripts_in_run(RUN)
    assert len(transcripts) == expected
    assert all(t.raw_output and t.input_snapshot for t in transcripts)
    assert all(0.0 <= r.score <= 1.0 for r in ratings)
    assert summary.mean_score == pytest.approx(
        sum(r.score for r in ratings) / expected
    )


def test_run_stops_at_n_steps(manager: ProfileManager) -> None:
    add_users(manager, 5)
    summary = seeded_run(manager, n_steps=7)
    assert summary.steps_completed == 7
    assert manager.count_ratings(RUN) == 7


def test_run_stops_early_when_all_users_are_exhausted(
    manager: ProfileManager,
) -> None:
    add_users(manager, 3)
    summary = seeded_run(manager, n_steps=1000)
    assert summary.steps_completed == 3 * 2
    assert summary.users_exhausted == 3
    assert summary.failures == 0
    assert manager.get_run(RUN).finished_at is not None


def test_exhausted_users_do_not_use_up_steps(manager: ProfileManager) -> None:
    # With exactly enough steps, every pair must be rated for any seed.
    add_users(manager, 4)
    for seed in range(20):
        run_id = f"run-{seed}"
        assert seeded_run(manager, run_id, 12, seed).steps_completed == 12


def test_same_seed_gives_same_run(tmp_path: Path) -> None:
    results = []
    for name in ("a", "b"):
        manager = ProfileManager(
            tmp_path / f"{name}.db",
            tmp_path / f"{name}_p.db",
            rng=random.Random(9),
        )
        add_users(manager, 5)
        seeded_run(manager, n_steps=12)
        results.append(
            [
                (r.rater_id, r.target_id, r.score, r.step)
                for r in manager.ratings_in_run(RUN)
            ]
        )
    assert results[0] == results[1]


def test_different_seed_gives_different_run(manager: ProfileManager) -> None:
    add_users(manager, 5)
    seeded_run(manager, "run-a", n_steps=12, seed=1)
    seeded_run(manager, "run-b", n_steps=12, seed=2)
    run_a = [(r.rater_id, r.target_id) for r in manager.ratings_in_run("run-a")]
    run_b = [(r.rater_id, r.target_id) for r in manager.ratings_in_run("run-b")]
    assert run_a != run_b


def test_run_records_metadata(manager: ProfileManager) -> None:
    add_users(manager, 3)
    seeded_run(manager, n_steps=2)
    run = manager.get_run(RUN)
    assert run.matcher_name == RandomMatcher.name
    assert run.seed == SEED
    assert run.config == {"n_steps": 2, "rater": "FakeRater", "eligible_users": 3, "n_workers": 1}
    assert run.finished_at is not None
    assert run.finished_at >= run.started_at


def test_invalid_n_steps_rejected_without_starting_a_run(
    manager: ProfileManager,
) -> None:
    add_users(manager, 2)
    with pytest.raises(ValueError, match="n_steps"):
        seeded_run(manager, n_steps=0)
    assert manager.store.count_ratings() == 0


# Failures and clamping


def test_failing_rater_is_counted_and_run_finishes(
    manager: ProfileManager, caplog: pytest.LogCaptureFixture
) -> None:
    add_users(manager, 4)
    rater = ScriptedRater([0.7, RuntimeError("boom"), 0.8, RuntimeError("boom"), 0.9])

    with caplog.at_level(logging.ERROR):
        summary = run_simulation(
            manager, RandomMatcher(random.Random(SEED)), rater, RUN, 5, SEED
        )

    assert summary.steps_completed == 3
    assert summary.failures == 2
    assert summary.mean_score == pytest.approx(0.8)
    assert [r.step for r in manager.ratings_in_run(RUN)] == [1, 3, 5]
    assert manager.get_run(RUN).finished_at is not None
    assert "boom" in caplog.text


def test_run_where_every_rating_fails_has_no_mean(manager: ProfileManager) -> None:
    add_users(manager, 3)
    rater = ScriptedRater([RuntimeError("boom")] * 3)
    summary = run_simulation(
        manager, RandomMatcher(random.Random(SEED)), rater, RUN, 3, SEED
    )
    assert summary.steps_completed == 0
    assert summary.failures == 3
    assert summary.mean_score is None
    assert manager.get_run(RUN).finished_at is not None


@pytest.mark.parametrize("raw, saved", [(1.5, 1.0), (-0.2, 0.0), (0.735, 0.735)])
def test_scores_are_clamped_when_saved(
    manager: ProfileManager, raw: float, saved: float
) -> None:
    add_users(manager, 2)
    summary = run_simulation(
        manager, RandomMatcher(random.Random(SEED)), ScriptedRater([raw]), RUN, 1, SEED
    )
    assert summary.failures == 0
    assert summary.mean_score == saved
    assert manager.ratings_in_run(RUN)[0].score == saved


@pytest.mark.parametrize("bad_output", ["not a result", None, {"score": "high"}])
def test_invalid_rater_output_counts_as_failure(
    manager: ProfileManager, bad_output: Any
) -> None:
    add_users(manager, 2)
    rater = ScriptedRater([lambda: bad_output])
    summary = run_simulation(
        manager, RandomMatcher(random.Random(SEED)), rater, RUN, 1, SEED
    )
    assert summary.failures == 1
    assert manager.count_ratings(RUN) == 0


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "7.5", True, None])
def test_rating_result_rejects_non_finite_or_non_numeric_scores(score: Any) -> None:
    with pytest.raises(ValidationError):
        RatingResult(score=score)


class RepeatingMatcher(Matcher):
    """A buggy matcher that always returns the same target."""

    name = "repeating"

    def __init__(self, target: str) -> None:
        self.target = target

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        return self.target


def test_matcher_repeating_a_profile_cannot_overwrite_a_rating(
    manager: ProfileManager,
) -> None:
    # The target has no persona, so the only eligible user rates on every step.
    rater = add_users(manager, 1)[0]
    target = add_users(manager, 1, persona=False)[0]
    summary = run_simulation(
        manager, RepeatingMatcher(target), ScriptedRater([0.4, 0.9, 0.9]), RUN, 3, SEED
    )
    assert summary.steps_completed == 1
    assert summary.failures == 2
    assert summary.mean_score == 0.4
    stored = manager.ratings_by_rater(rater, RUN)
    assert [(r.target_id, r.score, r.step) for r in stored] == [(target, 0.4, 1)]


def test_matcher_returning_unknown_profile_counts_as_failure(
    manager: ProfileManager,
) -> None:
    add_users(manager, 2)
    summary = run_simulation(
        manager, RepeatingMatcher("missing"), FakeRater(random.Random(SEED)), RUN, 3, SEED
    )
    assert summary.failures == 3
    assert manager.get_run(RUN).finished_at is not None


class BrokenMatcher(Matcher):
    name = "broken"

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        raise RuntimeError("matcher bug")


def test_matcher_exception_propagates_and_leaves_run_unfinished(
    manager: ProfileManager,
) -> None:
    add_users(manager, 2)
    with pytest.raises(RuntimeError, match="matcher bug"):
        run_simulation(
            manager, BrokenMatcher(), FakeRater(random.Random(SEED)), RUN, 3, SEED
        )
    assert manager.get_run(RUN).finished_at is None


# observe()


def test_observe_is_called_after_each_saved_rating(manager: ProfileManager) -> None:
    add_users(manager, 4)
    matcher = RecordingMatcher(random.Random(SEED))
    rater = ScriptedRater([0.5, RuntimeError("boom"), 0.6, 0.7])

    run_simulation(manager, matcher, rater, RUN, 4, SEED)

    stored = manager.ratings_in_run(RUN)
    assert [r.step for r in matcher.observed] == [1, 3, 4]
    assert matcher.observed == stored


def test_observe_sees_rating_already_saved(manager: ProfileManager) -> None:
    add_users(manager, 3)
    counts = []

    class CountingMatcher(RandomMatcher):
        def observe(self, rating: Rating) -> None:
            counts.append(manager.count_ratings(RUN))

    run_simulation(
        manager,
        CountingMatcher(random.Random(SEED)),
        FakeRater(random.Random(SEED)),
        RUN,
        3,
        SEED,
    )
    assert counts == [1, 2, 3]


# Eligibility


def test_users_without_persona_are_skipped_with_warning(
    manager: ProfileManager, caplog: pytest.LogCaptureFixture
) -> None:
    with_persona = add_users(manager, 2)
    without_persona = add_users(manager, 1, persona=False)[0]

    with caplog.at_level(logging.WARNING):
        seeded_run(manager, n_steps=100)

    ratings = manager.ratings_in_run(RUN)
    assert {r.rater_id for r in ratings} == set(with_persona)
    assert without_persona in {r.target_id for r in ratings}
    assert without_persona in caplog.text


def test_zero_eligible_users_raises_without_starting_a_run(
    manager: ProfileManager,
) -> None:
    add_users(manager, 3, persona=False)
    with pytest.raises(NoEligibleUsersError, match="persona"):
        seeded_run(manager)
    # The run_id is still free once personas exist.
    for profile in manager.list_profiles():
        manager.add_persona(profile.user_id, "description")
    assert seeded_run(manager, n_steps=1).steps_completed == 1


def test_duplicate_run_id_in_simulation_raises(manager: ProfileManager) -> None:
    add_users(manager, 3)
    seeded_run(manager, n_steps=2)
    with pytest.raises(DuplicateRunError, match=RUN):
        seeded_run(manager, n_steps=2)
    assert manager.count_ratings(RUN) == 2


# PublicView


def view_after_run(manager: ProfileManager) -> PublicView:
    seeded_run(manager, n_steps=100)
    return PublicView(manager.store, RUN)


def test_view_public_interface_is_only_public_data() -> None:
    public = {name for name in dir(PublicView) if not name.startswith("_")}
    assert public == {
        "run_id",
        "list_profiles",
        "get_profile",
        "ratings_by_rater",
        "ratings_of_target",
        "already_rated",
        "eligible_candidates",  # Only user_ids, with preferences applied but not revealed.
    }


def test_view_holds_only_the_main_store(manager: ProfileManager) -> None:
    view = PublicView(manager.store, RUN)
    held = [getattr(view, slot) for slot in PublicView.__slots__]
    assert not any(isinstance(value, (ProfileManager, PersonaStore)) for value in held)
    assert {type(value) for value in held} == {Store, str}
    assert not any(
        isinstance(value, (ProfileManager, PersonaStore))
        for value in vars(manager.store).values()
    )
    with pytest.raises(AttributeError):
        view.manager = manager  # type: ignore[attr-defined]


def test_view_works_with_no_persona_database(tmp_path: Path) -> None:
    store = Store(tmp_path / "main.db")
    profile = Profile(user_id=generate_user_id(), **profile_data())
    store.insert_profile(profile)
    public = PublicView(store, RUN).list_profiles()
    assert [p.model_dump() for p in public] == [profile.model_dump(exclude={"agent"})]
    assert not (tmp_path / "personas.db").exists()


def test_view_never_shows_the_agent_flag(manager: ProfileManager) -> None:
    user = add_users(manager, 3)[0]
    view = view_after_run(manager)
    profiles = [*view.list_profiles(), view.get_profile(user)]
    assert len(profiles) == 4
    for profile in profiles:
        assert not hasattr(profile, "agent")
        assert "agent" not in profile.model_dump()


def test_matching_module_cannot_import_persona_code() -> None:
    code = (
        "import sys, matching; "
        "assert 'personas' not in sys.modules; "
        "assert 'manager' not in sys.modules; "
        "assert 'raters' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True)


def test_everything_the_view_returns_is_free_of_persona_text(
    manager: ProfileManager,
) -> None:
    users = add_users(manager, 4)
    view = view_after_run(manager)

    returned: list[Any] = [view.list_profiles()]
    for user in users:
        returned += [
            view.get_profile(user),
            view.ratings_by_rater(user),
            view.ratings_of_target(user),
            view.already_rated(user),
        ]
    assert "SECRET_PERSONA" not in repr(returned)


def test_observe_receives_no_persona_text(manager: ProfileManager) -> None:
    add_users(manager, 3)
    matcher = RecordingMatcher(random.Random(SEED))
    run_simulation(manager, matcher, FakeRater(random.Random(SEED)), RUN, 6, SEED)
    assert len(matcher.observed) == 6
    assert "SECRET_PERSONA" not in repr(matcher.observed)


def test_main_database_holds_no_persona_text_after_a_run(
    manager: ProfileManager, tmp_path: Path
) -> None:
    add_users(manager, 4)
    seeded_run(manager, n_steps=12)
    # Read raw bytes, including the write-ahead log, so a new column can't slip past.
    main_files = list(tmp_path.glob("main.db*"))
    assert main_files
    for path in main_files:
        assert b"SECRET_PERSONA" not in path.read_bytes()
    # The check above only means something if the text was produced and stored.
    assert "SECRET_PERSONA" in manager.transcripts_in_run(RUN)[0].input_snapshot


def test_view_ratings_only_cover_its_run(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 0.5, "other-run")
    view = PublicView(manager.store, RUN)
    assert view.ratings_by_rater(alice) == []
    assert view.ratings_of_target(bob) == []
    assert view.already_rated(alice) == set()


# Raters


def make_people() -> tuple[Profile, Persona, Profile, Persona]:
    rater_id, target_id = generate_user_id(), generate_user_id()
    return (
        Profile(user_id=rater_id, **profile_data("Rae")),
        Persona(
            user_id=rater_id,
            description="RATER_PERSONA_TEXT",
            generation_seeds={"hidden": "RATER_SEED_VALUE"},
        ),
        Profile(user_id=target_id, **profile_data("Tess")),
        Persona(
            user_id=target_id,
            description="TARGET_PERSONA_TEXT",
            generation_seeds={"hidden": "TARGET_SEED_VALUE"},
        ),
    )


def test_build_rater_input_has_expected_structure() -> None:
    rater, persona, target, _ = make_people()
    rater_input = build_rater_input(rater, persona, target)
    assert rater_input["rater_profile"] == rater.model_dump(exclude={"agent"})
    assert rater_input["rater_persona"] == "RATER_PERSONA_TEXT"
    assert rater_input["target_profile"] == target.model_dump(exclude={"agent"})
    assert "agent" not in rater_input["rater_profile"]
    assert "agent" not in rater_input["target_profile"]
    assert "scale" not in rater_input
    json.dumps(rater_input)


def test_build_rater_input_never_contains_target_persona() -> None:
    rater, persona, target, _ = make_people()
    text = json.dumps(build_rater_input(rater, persona, target))
    assert "TARGET_PERSONA_TEXT" not in text
    assert "TARGET_SEED_VALUE" not in text
    assert "RATER_SEED_VALUE" not in text


def test_saved_snapshots_hold_rater_persona_and_never_the_targets(
    manager: ProfileManager,
) -> None:
    add_users(manager, 4)
    seeded_run(manager, n_steps=12)
    transcripts = manager.transcripts_in_run(RUN)
    assert len(transcripts) == 12
    for transcript in transcripts:
        assert f"SECRET_PERSONA_{transcript.rater_id}" in transcript.input_snapshot
        assert f"SECRET_PERSONA_{transcript.target_id}" not in transcript.input_snapshot


def test_fake_rater_scores_are_in_range_and_rounded() -> None:
    rater, persona, target, _ = make_people()
    fake = FakeRater(random.Random(SEED))
    for _ in range(200):
        score = fake.rate(rater, persona, target).score
        assert 0.0 <= score <= 1.0
        assert score == round(score, 2)


def test_fake_rater_is_reproducible() -> None:
    rater, persona, target, _ = make_people()
    first = FakeRater(random.Random(SEED))
    second = FakeRater(random.Random(SEED))
    for _ in range(5):
        assert first.rate(rater, persona, target) == second.rate(rater, persona, target)


def test_jev_rater_only_needs_call_jev_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rater, persona, target, _ = make_people()
    received = []

    def fake_call(rater_input: dict[str, Any]) -> tuple[float, str]:
        received.append(rater_input)
        return 0.75, '{"noul": 0.75}'

    monkeypatch.setattr(raters, "_call_jev", fake_call)
    result = JevRater().rate(rater, persona, target)

    assert result.score == 0.75
    assert result.raw_output == '{"noul": 0.75}'
    assert json.loads(result.input_snapshot) == received[0]
    assert received[0] == build_rater_input(rater, persona, target)


# Jev. These use the real SDK with a fake network layer, so no API call can be made.


def jev_reply(noul: float = 0.65, status_code: int = 200) -> httpx2.Response:
    return httpx2.Response(
        status_code,
        json={
            "model": JEV_MODEL,
            "answers": {"like": {"type": "noul", "noul": noul}},
            "usage": {"input_tokens": 700, "output_tokens": 18},
        },
    )


def use_fake_jev(monkeypatch: pytest.MonkeyPatch, replies: list[httpx2.Response]) -> list[Any]:
    """Send Jev calls to the given replies, and return the list of request bodies sent."""
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return replies.pop(0)

    client = TypeSafeClient(
        api_key="test-key",
        model=JEV_MODEL,
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=2, backoff_initial=0, backoff_max=0, backoff_jitter=0),
        compression=False,
    )
    monkeypatch.setattr(raters, "_jev_client", lambda: client)
    return requests


@pytest.mark.parametrize("noul", [0.0, 0.42, 1.0])
def test_jev_probability_is_saved_as_the_score(
    monkeypatch: pytest.MonkeyPatch, noul: float
) -> None:
    use_fake_jev(monkeypatch, [jev_reply(noul)])
    rater, persona, target, _ = make_people()
    assert JevRater().rate(rater, persona, target).score == noul


def test_jev_request_and_saved_output(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = use_fake_jev(monkeypatch, [jev_reply()])
    rater, persona, target, _ = make_people()

    result = JevRater().rate(rater, persona, target)

    assert requests[0]["model"] == JEV_MODEL
    assert requests[0]["questions"]["like"]["type"] == "noul"
    assert requests[0]["questions"]["like"]["instructions"] == JEV_QUESTION.instructions
    assert requests[0]["state"]["rater_persona"] == "RATER_PERSONA_TEXT"
    assert result.model_version == JEV_MODEL
    assert json.loads(result.raw_output) == {
        "model": JEV_MODEL,
        "noul": 0.65,
        "input_tokens": 700,
    }


def test_jev_request_holds_nothing_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = use_fake_jev(monkeypatch, [jev_reply()])
    rater, persona, target, target_persona = make_people()
    JevRater().rate(rater, persona, target)
    sent = json.dumps(requests[0])
    assert "TARGET_PERSONA_TEXT" not in sent
    assert "TARGET_SEED_VALUE" not in sent
    assert "RATER_SEED_VALUE" not in sent
    assert rater.user_id not in sent
    assert target.user_id not in sent
    assert '"agent"' not in sent


def test_jev_state_leaves_out_user_ids() -> None:
    rater, persona, target, _ = make_people()
    state = build_jev_state(build_rater_input(rater, persona, target))
    assert state["shown_profile"]["name"] == "Tess"
    assert "user_id" not in state["rater"]
    assert "user_id" not in state["shown_profile"]


def test_jev_rate_limits_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = use_fake_jev(monkeypatch, [httpx2.Response(429, json={}), jev_reply(0.4)])
    rater, persona, target, _ = make_people()
    assert JevRater().rate(rater, persona, target).score == 0.4
    assert len(requests) == 2


def test_jev_failure_is_counted_and_the_run_continues(
    manager: ProfileManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_users(manager, 3)
    use_fake_jev(monkeypatch, [httpx2.Response(400, json={}), jev_reply(), jev_reply()])
    summary = run_simulation(
        manager, RandomMatcher(random.Random(SEED)), JevRater(), RUN, 3, SEED
    )
    assert summary.steps_completed == 2
    assert summary.failures == 1
    assert manager.ratings_in_run(RUN)[0].model_version == JEV_MODEL


def test_jev_without_an_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    raters._jev_client.cache_clear()
    rater, persona, target, _ = make_people()
    with pytest.raises(TypeSafeError, match="API key"):
        JevRater().rate(rater, persona, target)
    raters._jev_client.cache_clear()


# Compatibility


PLURAL_TO_GENDER = {"men": "man", "women": "woman"}


def public_user(manager: ProfileManager, gender: str, interested_in: str) -> str:
    """A user interested in one gender, written as "men" or "women" for readability."""
    user_id = manager.add_profile({**profile_data(), "gender": gender}).user_id
    manager.set_preferences(user_id, [PLURAL_TO_GENDER[interested_in]])
    manager.add_persona(user_id, f"SECRET_PERSONA_{user_id}")
    return user_id


def compatible(manager: ProfileManager, a: str, b: str) -> bool:
    gender = {p.user_id: p.gender for p in manager.list_profiles()}
    interested_in = manager.store.all_preferences()
    return mutually_interested(gender[a], interested_in[a], gender[b], interested_in[b])


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (("man", "women"), ("woman", "men"), True),
        (("man", "men"), ("man", "men"), True),
        (("woman", "women"), ("woman", "women"), True),
        (("man", "women"), ("man", "men"), False),
        (("man", "women"), ("woman", "women"), False),
        (("man", "men"), ("woman", "men"), False),
    ],
)
def test_eligible_pool_needs_interest_both_ways(
    manager: ProfileManager,
    first: tuple[str, str],
    second: tuple[str, str],
    expected: bool,
) -> None:
    view = PublicView(manager.store, RUN)
    a = public_user(manager, *first)
    b = public_user(manager, *second)
    assert (b in view.eligible_candidates(a)) is expected
    assert (a in view.eligible_candidates(b)) is expected
    assert compatible(manager, a, b) is expected


def test_compatible_matcher_only_shows_compatible_profiles(
    manager: ProfileManager,
) -> None:
    for gender, interested_in in [
        ("man", "women"),
        ("man", "women"),
        ("woman", "men"),
        ("woman", "men"),
        ("man", "men"),
        ("man", "men"),
        ("woman", "women"),
    ]:
        public_user(manager, gender, interested_in)

    summary = run_simulation(
        manager,
        CompatibleRandomMatcher(random.Random(SEED)),
        FakeRater(random.Random(SEED)),
        RUN,
        50,
        SEED,
    )

    # 2 x 2 opposite-gender pairs in each direction, plus one pair of men both ways. The
    # woman interested in women has nobody compatible, so uses no steps.
    assert summary.steps_completed == 10
    assert summary.failures == 0
    assert summary.users_exhausted == 7
    for rating in manager.ratings_in_run(RUN):
        assert compatible(manager, rating.rater_id, rating.target_id)


def test_compatible_matcher_runs_are_reproducible(tmp_path: Path) -> None:
    runs = []
    for name in ("a", "b"):
        manager = ProfileManager(
            tmp_path / f"{name}.db",
            tmp_path / f"{name}_personas.db",
            rng=random.Random(SEED),
        )
        for gender, interested_in in [("man", "women"), ("woman", "men")] * 3:
            public_user(manager, gender, interested_in)
        run_simulation(
            manager,
            CompatibleRandomMatcher(random.Random(SEED)),
            FakeRater(random.Random(SEED)),
            RUN,
            8,
            SEED,
        )
        runs.append(
            [(r.rater_id, r.target_id, r.score) for r in manager.ratings_in_run(RUN)]
        )
    assert runs[0] == runs[1]


# Metrics


def test_learning_curve_averages_windows_in_order() -> None:
    assert learning_curve([1.0, 3.0, 5.0, 7.0], 2) == [2.0, 6.0]
    assert learning_curve([2.0, 4.0], 10) == [2.0, 4.0]
    assert learning_curve([], 10) == []


def test_metrics_count_compatibility_and_match_chances(manager: ProfileManager) -> None:
    man = public_user(manager, "man", "women")
    woman = public_user(manager, "woman", "men")
    other_woman = public_user(manager, "woman", "men")
    other_man = public_user(manager, "man", "women")
    manager.start_run(RUN, "test", SEED, {})
    manager.add_rating(man, woman, 0.8, RUN, step=1)  # Rated both ways.
    manager.add_rating(woman, man, 0.7, RUN, step=2)
    manager.add_rating(man, other_woman, 0.9, RUN, step=3)  # Rated both ways.
    manager.add_rating(other_woman, man, 0.5, RUN, step=4)
    manager.add_rating(man, other_man, 0.1, RUN, step=5)  # One way, and not compatible.

    metrics = compute_metrics(manager.store, RUN, n_windows=5)

    assert metrics.n_ratings == 5
    assert metrics.mean_score == pytest.approx(3.0 / 5)
    assert metrics.compatible_share == pytest.approx(4 / 5)
    assert metrics.pairs_rated_both_ways == 2
    assert metrics.expected_matches == pytest.approx(0.8 * 0.7 + 0.9 * 0.5)
    assert metrics.match_rate == pytest.approx((0.8 * 0.7 + 0.9 * 0.5) / 2)
    assert metrics.learning_curve == [0.8, 0.7, 0.9, 0.5, 0.1]


def test_metrics_ignore_other_runs(manager: ProfileManager) -> None:
    man = public_user(manager, "man", "women")
    woman = public_user(manager, "woman", "men")
    manager.start_run(RUN, "test", SEED, {})
    manager.start_run("other", "test", SEED, {})
    manager.add_rating(man, woman, 0.9, RUN, step=1)
    manager.add_rating(woman, man, 0.9, "other", step=1)
    metrics = compute_metrics(manager.store, RUN)
    assert metrics.n_ratings == 1
    assert metrics.pairs_rated_both_ways == 0
    assert metrics.expected_matches == 0
    assert metrics.match_rate is None


def test_metrics_of_an_empty_run(manager: ProfileManager) -> None:
    add_users(manager, 2)
    manager.start_run(RUN, "test", SEED, {})
    metrics = compute_metrics(manager.store, RUN)
    assert metrics.n_ratings == 0
    assert metrics.mean_score is None
    assert metrics.compatible_share is None
    assert metrics.match_rate is None
    assert metrics.learning_curve == []


# Resuming


class StopsAtCall(RandomMatcher):
    """Raises on a chosen call to next_profile, like a run being interrupted."""

    def __init__(self, rng: random.Random, stop_at: int) -> None:
        super().__init__(rng)
        self.calls = 0
        self.stop_at = stop_at
        self.observed: list[Rating] = []

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        self.calls += 1
        if self.calls == self.stop_at:
            raise KeyboardInterrupt
        return super().next_profile(user_id, view)

    def observe(self, rating: Rating) -> None:
        self.observed.append(rating)


def fake_run(manager: ProfileManager, matcher: Matcher, n_steps: int = 12, **kwargs: Any) -> Any:
    return run_simulation(manager, matcher, FakeRater(random.Random(SEED)), RUN, n_steps, SEED, **kwargs)


def test_interrupted_run_resumes_to_the_total_without_repeats(manager: ProfileManager) -> None:
    add_users(manager, 6)
    with pytest.raises(KeyboardInterrupt):
        fake_run(manager, StopsAtCall(random.Random(SEED), stop_at=6))
    assert manager.count_ratings(RUN) == 5
    assert manager.get_run(RUN).finished_at is None
    before = manager.ratings_in_run(RUN)

    summary = fake_run(manager, RandomMatcher(random.Random(SEED + 1)), resume=True)

    ratings = manager.ratings_in_run(RUN)
    assert summary.steps_completed == 12
    assert summary.failures == 0
    assert [r.step for r in ratings] == list(range(1, 13))
    assert ratings[:5] == before
    assert len({(r.rater_id, r.target_id) for r in ratings}) == 12
    assert summary.mean_score == pytest.approx(sum(r.score for r in ratings) / 12)
    assert manager.get_run(RUN).finished_at is not None


def test_resume_replays_saved_ratings_to_the_matcher(manager: ProfileManager) -> None:
    add_users(manager, 6)
    with pytest.raises(KeyboardInterrupt):
        fake_run(manager, StopsAtCall(random.Random(SEED), stop_at=4))
    saved = manager.ratings_in_run(RUN)

    matcher = StopsAtCall(random.Random(SEED + 1), stop_at=-1)
    fake_run(manager, matcher, n_steps=5, resume=True)

    assert matcher.observed[:3] == saved
    assert len(matcher.observed) == 5


def test_resume_starts_a_new_run_if_there_is_none(manager: ProfileManager) -> None:
    add_users(manager, 4)
    summary = fake_run(manager, RandomMatcher(random.Random(SEED)), n_steps=3, resume=True)
    assert summary.steps_completed == 3
    assert manager.get_run(RUN).finished_at is not None


def test_resume_refuses_a_finished_run(manager: ProfileManager) -> None:
    add_users(manager, 4)
    fake_run(manager, RandomMatcher(random.Random(SEED)), n_steps=3)
    with pytest.raises(ValueError, match="already finished"):
        fake_run(manager, RandomMatcher(random.Random(SEED)), n_steps=5, resume=True)
    assert manager.count_ratings(RUN) == 3


def test_resume_refuses_a_different_matcher(manager: ProfileManager) -> None:
    add_users(manager, 6)
    with pytest.raises(KeyboardInterrupt):
        fake_run(manager, StopsAtCall(random.Random(SEED), stop_at=3))
    with pytest.raises(ValueError, match="used matcher"):
        fake_run(manager, CompatibleRandomMatcher(random.Random(SEED)), resume=True)


def test_without_resume_an_existing_run_id_is_still_rejected(manager: ProfileManager) -> None:
    add_users(manager, 4)
    fake_run(manager, RandomMatcher(random.Random(SEED)), n_steps=2)
    with pytest.raises(DuplicateRunError):
        fake_run(manager, RandomMatcher(random.Random(SEED)), n_steps=2)


def test_outage_aborts_the_run_so_it_can_be_resumed(manager: ProfileManager) -> None:
    add_users(manager, 8)
    outage = [RuntimeError("down")] * MAX_CONSECUTIVE_FAILURES
    rater = ScriptedRater([0.5, *outage])
    with pytest.raises(RunAbortedError, match="in a row"):
        run_simulation(manager, RandomMatcher(random.Random(SEED)), rater, RUN, 500, SEED)
    assert manager.count_ratings(RUN) == 1
    assert manager.get_run(RUN).finished_at is None

    summary = fake_run(manager, RandomMatcher(random.Random(SEED)), n_steps=5, resume=True)
    assert summary.steps_completed == 5
    assert summary.failures == 0


def test_scattered_failures_do_not_abort_the_run(manager: ProfileManager) -> None:
    add_users(manager, 12)
    below_limit = [RuntimeError("blip")] * (MAX_CONSECUTIVE_FAILURES - 1)
    rater = ScriptedRater([*below_limit, 0.5, *below_limit, 0.6])
    summary = run_simulation(
        manager, RandomMatcher(random.Random(SEED)), rater, RUN, 2 * MAX_CONSECUTIVE_FAILURES, SEED
    )
    assert summary.steps_completed == 2
    assert summary.failures == 2 * MAX_CONSECUTIVE_FAILURES - 2


# Parallel ratings


class SlowFakeRater(FakeRater):
    """Only succeeds if four ratings are in progress at the same time."""

    def __init__(self, rng: random.Random) -> None:
        super().__init__(rng)
        self.barrier = threading.Barrier(4, timeout=10)

    def rate(self, rater_profile: Profile, rater_persona: Persona, target_profile: Profile) -> RatingResult:
        self.barrier.wait()
        return super().rate(rater_profile, rater_persona, target_profile)


def test_ratings_really_run_at_the_same_time(manager: ProfileManager) -> None:
    add_users(manager, 6)
    summary = run_simulation(
        manager, RandomMatcher(random.Random(SEED)), SlowFakeRater(random.Random(SEED)), RUN, 8, SEED, n_workers=4
    )
    assert summary.steps_completed == 8
    assert summary.failures == 0


@pytest.mark.parametrize("n_workers", [2, 8])
def test_parallel_run_saves_every_rating_once_and_in_order(manager: ProfileManager, n_workers: int) -> None:
    add_users(manager, 10)
    matcher = RecordingMatcher(random.Random(SEED))
    summary = run_simulation(matcher=matcher, manager=manager, rater=FakeRater(random.Random(SEED)), run_id=RUN, n_steps=40, seed=SEED, n_workers=n_workers)
    ratings = manager.ratings_in_run(RUN)
    assert summary.steps_completed == 40
    assert [r.step for r in ratings] == list(range(1, 41))
    assert len({(r.rater_id, r.target_id) for r in ratings}) == 40
    assert [r.step for r in matcher.observed] == list(range(1, 41))
    assert len(manager.transcripts_in_run(RUN)) == 40
    assert manager.get_run(RUN).finished_at is not None


def test_batches_never_go_past_n_steps(manager: ProfileManager) -> None:
    add_users(manager, 8)
    summary = fake_run(manager, RandomMatcher(random.Random(SEED)), n_steps=11, n_workers=8)
    assert summary.steps_completed == 11
    assert manager.count_ratings(RUN) == 11


def test_parallel_run_finishes_when_almost_nothing_is_left(manager: ProfileManager) -> None:
    add_users(manager, 2)
    summary = fake_run(manager, RandomMatcher(random.Random(SEED)), n_steps=50, n_workers=8)
    assert summary.steps_completed == 2
    assert summary.users_exhausted == 2


def test_parallel_failures_leave_gaps_and_the_run_continues(manager: ProfileManager) -> None:
    add_users(manager, 8)

    class FailsSometimes(FakeRater):
        def rate(self, rater_profile: Profile, rater_persona: Persona, target_profile: Profile) -> RatingResult:
            if target_profile.user_id == failing_target:
                raise RuntimeError("boom")
            return super().rate(rater_profile, rater_persona, target_profile)

    failing_target = manager.list_profiles()[0].user_id
    summary = run_simulation(
        manager, RandomMatcher(random.Random(SEED)), FailsSometimes(random.Random(SEED)), RUN, 30, SEED, n_workers=4
    )
    ratings = manager.ratings_in_run(RUN)
    assert summary.failures > 0
    assert summary.steps_completed + summary.failures == 30
    assert all(r.target_id != failing_target for r in ratings)
    assert [r.step for r in ratings] == sorted(r.step for r in ratings)


def test_parallel_outage_aborts_and_can_be_resumed(manager: ProfileManager) -> None:
    add_users(manager, 10)

    class AlwaysDown(FakeRater):
        def rate(self, rater_profile: Profile, rater_persona: Persona, target_profile: Profile) -> RatingResult:
            raise RuntimeError("down")

    with pytest.raises(RunAbortedError):
        run_simulation(manager, RandomMatcher(random.Random(SEED)), AlwaysDown(random.Random(SEED)), RUN, 500, SEED, n_workers=8)
    assert manager.get_run(RUN).finished_at is None
    summary = fake_run(manager, RandomMatcher(random.Random(SEED)), n_steps=20, resume=True, n_workers=8)
    assert summary.steps_completed == 20


def test_one_worker_gives_the_same_run_as_before_batching(tmp_path: Path) -> None:
    runs = []
    for name in ("a", "b"):
        manager = ProfileManager(tmp_path / f"{name}.db", tmp_path / f"{name}_p.db", rng=random.Random(SEED))
        add_users(manager, 5)
        fake_run(manager, CompatibleRandomMatcher(random.Random(SEED)), n_steps=10)
        runs.append([(r.rater_id, r.target_id, r.score, r.step) for r in manager.ratings_in_run(RUN)])
    assert runs[0] == runs[1]


def test_n_workers_must_be_positive(manager: ProfileManager) -> None:
    add_users(manager, 2)
    with pytest.raises(ValueError, match="n_workers"):
        fake_run(manager, RandomMatcher(random.Random(SEED)), n_workers=0)


def test_compatible_random_is_random_under_its_old_name() -> None:
    assert CompatibleRandomMatcher.name == "compatible-random"
    assert issubclass(CompatibleRandomMatcher, RandomMatcher)


def test_live_run_id_is_reserved_for_real_people(manager: ProfileManager) -> None:
    add_users(manager, 2)
    with pytest.raises(ValueError, match="reserved"):
        run_simulation(manager, RandomMatcher(random.Random(SEED)), FakeRater(random.Random(SEED)), "live", 2, SEED)


def test_simulation_ratings_record_the_matcher(manager: ProfileManager) -> None:
    add_users(manager, 3)
    fake_run(manager, CompatibleRandomMatcher(random.Random(SEED)), n_steps=3)
    assert {r.matcher for r in manager.ratings_in_run(RUN)} == {"compatible-random"}


def test_rater_input_has_both_genders_and_no_preferences(manager: ProfileManager) -> None:
    rater_id, target_id = add_users(manager, 2)
    manager.edit_profile(target_id, gender="nonbinary")
    rater_input = build_rater_input(
        manager.get_profile(rater_id), manager.get_persona(rater_id), manager.get_profile(target_id)
    )
    assert rater_input["rater_profile"]["gender"] == "man"
    assert rater_input["target_profile"]["gender"] == "nonbinary"
    assert "interested_in" not in json.dumps(rater_input)


def test_compatible_matcher_never_repeats_or_picks_self(manager: ProfileManager) -> None:
    men = [public_user(manager, "man", "women") for _ in range(3)]
    women = [public_user(manager, "woman", "men") for _ in range(3)]
    fake_run(manager, CompatibleRandomMatcher(random.Random(SEED)), n_steps=100, n_workers=4)
    ratings = manager.ratings_in_run(RUN)
    assert len(ratings) == 18
    assert {(r.rater_id, r.target_id) for r in ratings} == {
        *((a, b) for a in men for b in women),
        *((b, a) for a in men for b in women),
    }


def test_baseline_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_baseline.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "steps_completed=15" in result.stdout
