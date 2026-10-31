import json
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import raters
from manager import ProfileManager
from matching import Matcher, PublicView, RandomMatcher
from models import Persona, Profile, Rating, generate_user_id
from personas import PersonaStore
from raters import FakeRater, JevRater, Rater, RatingResult, build_rater_input
from simulation import NoEligibleUsersError, run_simulation
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
        "job": "engineer",
        "interests": ["hiking"],
    }


@pytest.fixture
def manager(tmp_path: Path) -> ProfileManager:
    return ProfileManager(tmp_path / "main.db", tmp_path / "personas.db")


def add_users(manager: ProfileManager, count: int, persona: bool = True) -> list[str]:
    users = []
    for _ in range(count):
        user_id = manager.add_profile(profile_data()).user_id
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
        manager.add_rating(user, target, 5.0, RUN)

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
        manager.add_rating(users[0], target, 5.0, RUN)
    assert matcher.next_profile(users[0], view) is None


def test_random_matcher_returns_none_when_user_is_alone(
    manager: ProfileManager,
) -> None:
    user = add_users(manager, 1)[0]
    view = PublicView(manager.store, RUN)
    assert RandomMatcher(random.Random(SEED)).next_profile(user, view) is None


def test_random_matcher_only_counts_current_run(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 5.0, "other-run")
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
    assert all(1.0 <= r.score <= 10.0 for r in ratings)
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
    assert run.config == {"n_steps": 2, "rater": "FakeRater", "eligible_users": 3}
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
    rater = ScriptedRater([7.0, RuntimeError("boom"), 8.0, RuntimeError("boom"), 9.0])

    with caplog.at_level(logging.ERROR):
        summary = run_simulation(
            manager, RandomMatcher(random.Random(SEED)), rater, RUN, 5, SEED
        )

    assert summary.steps_completed == 3
    assert summary.failures == 2
    assert summary.mean_score == pytest.approx(8.0)
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


@pytest.mark.parametrize("raw, saved", [(12.5, 10.0), (0.2, 1.0), (7.35, 7.35)])
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
        manager, RepeatingMatcher(target), ScriptedRater([4.0, 9.0, 9.0]), RUN, 3, SEED
    )
    assert summary.steps_completed == 1
    assert summary.failures == 2
    assert summary.mean_score == 4.0
    stored = manager.ratings_by_rater(rater, RUN)
    assert [(r.target_id, r.score, r.step) for r in stored] == [(target, 4.0, 1)]


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
    rater = ScriptedRater([5.0, RuntimeError("boom"), 6.0, 7.0])

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
    assert PublicView(store, RUN).list_profiles() == [profile]
    assert not (tmp_path / "personas.db").exists()


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
    manager.add_rating(alice, bob, 5.0, "other-run")
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
    assert rater_input["rater_profile"] == rater.model_dump()
    assert rater_input["rater_persona"] == "RATER_PERSONA_TEXT"
    assert rater_input["target_profile"] == target.model_dump()
    assert rater_input["scale"]["anchors"] == {
        "1": "no interest",
        "5": "neutral",
        "10": "would message immediately",
    }
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
        assert 1.0 <= score <= 10.0
        assert score == round(score, 2)


def test_fake_rater_is_reproducible() -> None:
    rater, persona, target, _ = make_people()
    first = FakeRater(random.Random(SEED))
    second = FakeRater(random.Random(SEED))
    for _ in range(5):
        assert first.rate(rater, persona, target) == second.rate(rater, persona, target)


def test_jev_rater_is_a_clearly_marked_stub() -> None:
    rater, persona, target, _ = make_people()
    with pytest.raises(NotImplementedError, match="Jev API integration is pending"):
        JevRater().rate(rater, persona, target)


def test_jev_rater_only_needs_call_jev_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rater, persona, target, _ = make_people()
    received = []

    def fake_call(rater_input: dict[str, Any]) -> tuple[float, str]:
        received.append(rater_input)
        return 7.5, '{"score": 7.5}'

    monkeypatch.setattr(raters, "_call_jev", fake_call)
    result = JevRater().rate(rater, persona, target)

    assert result.score == 7.5
    assert result.raw_output == '{"score": 7.5}'
    assert json.loads(result.input_snapshot) == received[0]
    assert received[0] == build_rater_input(rater, persona, target)


def test_baseline_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_baseline.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "steps_completed=15" in result.stdout
