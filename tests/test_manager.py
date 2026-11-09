import json
import multiprocessing
import random
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from manager import ProfileManager
from models import Rating, generate_user_id
from personas import PersonaStore
from store import (
    DuplicateProfileError,
    DuplicateRunError,
    MutualPair,
    ProfileNotFoundError,
    RunNotFoundError,
    Store,
)

RUN = "run-1"


def profile_data(**overrides: Any) -> dict[str, Any]:
    data = {
        "name": "Sam",
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
    return {**data, **overrides}


@pytest.fixture
def main_db(tmp_path: Path) -> Path:
    return tmp_path / "main.db"


@pytest.fixture
def persona_db(tmp_path: Path) -> Path:
    return tmp_path / "personas.db"


@pytest.fixture
def manager(main_db: Path, persona_db: Path) -> ProfileManager:
    return ProfileManager(main_db, persona_db)


def add_users(manager: ProfileManager, count: int) -> list[str]:
    return [manager.add_profile(profile_data()).user_id for _ in range(count)]


# Creating profiles


def test_add_profile_generates_id_and_persists(manager: ProfileManager) -> None:
    profile = manager.add_profile(profile_data())
    assert uuid.UUID(profile.user_id).version == 4
    assert manager.get_profile(profile.user_id) == profile
    assert manager.list_profiles() == [profile]
    assert manager.count_profiles() == 1


def test_add_profile_keeps_supplied_id(manager: ProfileManager) -> None:
    user_id = str(uuid.uuid4())
    assert manager.add_profile(profile_data(user_id=user_id)).user_id == user_id
    assert manager.get_profile(user_id).user_id == user_id


def test_add_profile_rejects_duplicate_id(manager: ProfileManager) -> None:
    user_id = str(uuid.uuid4())
    manager.add_profile(profile_data(user_id=user_id, name="First"))
    with pytest.raises(DuplicateProfileError, match=user_id):
        manager.add_profile(profile_data(user_id=user_id, name="Second"))
    assert manager.get_profile(user_id).name == "First"
    assert manager.count_profiles() == 1


@pytest.mark.parametrize(
    "user_id",
    [
        "not-a-uuid",
        "",
        str(uuid.uuid1()),
        str(uuid.uuid4()).upper(),
        uuid.uuid4().hex,
        None,
    ],
)
def test_add_profile_rejects_invalid_supplied_id(
    manager: ProfileManager, user_id: Any
) -> None:
    with pytest.raises(ValidationError, match="user_id"):
        manager.add_profile(profile_data(user_id=user_id))
    assert manager.count_profiles() == 0


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "age",
        "height_cm",
        "religion",
        "school",
        "major",
        "job",
        "interests",
        "clubs",
        "gender",
        "looking_for",
        "agent",
    ],
)
def test_add_profile_rejects_missing_field(
    manager: ProfileManager, field: str
) -> None:
    data = profile_data()
    del data[field]
    with pytest.raises(ValidationError, match=field):
        manager.add_profile(data)
    assert manager.count_profiles() == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"age": 17},
        {"age": 101},
        {"age": "30"},
        {"age": True},
        {"age": 30.0},
        {"height_cm": 119},
        {"height_cm": 231},
        {"name": ""},
        {"name": "   "},
        {"religion": ""},
        {"school": "  "},
        {"job": ""},
        {"interests": []},
        {"interests": ["hiking", "  "]},
        {"interests": "hiking"},
        {"major": ""},
        {"major": "   "},
        {"major": None},
        {"clubs": "chess club"},
        {"clubs": ["chess club", "  "]},
        {"clubs": None},
        {"gender": "men"},
        {"gender": "Man"},
        {"gender": None},
        {"gender": "everyone"},
        {"interested_in": ["woman"]},  # Private, so never part of a profile.
        {"looking_for": "marriage"},
        {"looking_for": "Long-Term"},
        {"looking_for": ""},
        {"looking_for": None},
        {"agent": 1},
        {"agent": "true"},
        {"agent": None},
        {"nickname": "unknown field"},
    ],
)
def test_add_profile_rejects_invalid_values(
    manager: ProfileManager, overrides: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        manager.add_profile(profile_data(**overrides))
    assert manager.count_profiles() == 0


@pytest.mark.parametrize(
    "overrides",
    [{"age": 18}, {"age": 100}, {"height_cm": 120}, {"height_cm": 230}],
)
def test_add_profile_accepts_bounds(
    manager: ProfileManager, overrides: dict[str, Any]
) -> None:
    manager.add_profile(profile_data(**overrides))
    assert manager.count_profiles() == 1


@pytest.mark.parametrize("looking_for", ["short-term", "long-term", "just a date"])
def test_looking_for_round_trips(manager: ProfileManager, looking_for: str) -> None:
    profile = manager.add_profile(profile_data(looking_for=looking_for))
    assert manager.get_profile(profile.user_id).looking_for == looking_for


def test_clubs_are_normalized_and_round_trip(manager: ProfileManager) -> None:
    profile = manager.add_profile(
        profile_data(clubs=[" Chess Club", "DEBATE ", "chess club"])
    )
    assert profile.clubs == ["chess club", "debate"]
    assert manager.get_profile(profile.user_id).clubs == ["chess club", "debate"]


def test_clubs_can_be_empty(manager: ProfileManager) -> None:
    profile = manager.add_profile(profile_data(clubs=[]))
    assert manager.get_profile(profile.user_id).clubs == []


def test_major_round_trips(manager: ProfileManager) -> None:
    profile = manager.add_profile(profile_data(major=" Computer Science "))
    assert manager.get_profile(profile.user_id).major == "Computer Science"


@pytest.mark.parametrize("gender", ["man", "woman", "nonbinary"])
def test_gender_round_trips(manager: ProfileManager, gender: str) -> None:
    profile = manager.add_profile(profile_data(gender=gender))
    assert manager.get_profile(profile.user_id).gender == gender


# Preferences


@pytest.mark.parametrize(
    "interested_in",
    [["man"], ["woman", "nonbinary"], ["man", "woman", "nonbinary"], ("woman", "woman")],
)
def test_preferences_round_trip(manager: ProfileManager, interested_in: Any) -> None:
    user = add_users(manager, 1)[0]
    manager.set_preferences(user, interested_in)
    assert manager.get_preferences(user).interested_in == frozenset(interested_in)


@pytest.mark.parametrize("interested_in", [[], ["men"], ["everyone"], [None]])
def test_invalid_preferences_are_rejected(manager: ProfileManager, interested_in: Any) -> None:
    user = add_users(manager, 1)[0]
    with pytest.raises(ValidationError):
        manager.set_preferences(user, interested_in)
    assert manager.get_preferences(user) is None


def test_preferences_replace_and_need_a_profile(manager: ProfileManager) -> None:
    user = add_users(manager, 1)[0]
    manager.set_preferences(user, ["man", "woman"])
    manager.set_preferences(user, ["nonbinary"])
    assert manager.get_preferences(user).interested_in == {"nonbinary"}
    with pytest.raises(ProfileNotFoundError):
        manager.set_preferences(str(uuid.uuid4()), ["man"])


def test_preferences_are_never_part_of_a_profile_or_export(
    manager: ProfileManager, tmp_path: Path
) -> None:
    user = add_users(manager, 1)[0]
    manager.set_preferences(user, ["nonbinary"])
    assert "interested_in" not in manager.get_profile(user).model_dump()
    manager.store.export_profiles_json(tmp_path / "profiles.json")
    exported = (tmp_path / "profiles.json").read_text()
    assert "interested_in" not in exported
    assert "nonbinary" not in exported


def test_deleting_a_profile_deletes_its_preferences(manager: ProfileManager) -> None:
    user = add_users(manager, 1)[0]
    manager.set_preferences(user, ["man"])
    manager.delete_profile(user)
    assert manager.get_preferences(user) is None
    assert manager.store.all_preferences() == {}


def test_eligible_pool_needs_mutual_interest_and_excludes_self_and_rated(
    manager: ProfileManager,
) -> None:
    me, man, woman, nonbinary = (
        manager.add_profile(profile_data(gender=g)).user_id
        for g in ("woman", "man", "woman", "nonbinary")
    )
    manager.set_preferences(me, ["man", "nonbinary"])
    manager.set_preferences(man, ["woman"])
    manager.set_preferences(woman, ["man", "woman", "nonbinary"])
    manager.set_preferences(nonbinary, ["man"])  # Not interested in women.

    assert manager.eligible_candidates(me, RUN) == [man]
    assert woman not in manager.eligible_candidates(me, RUN)
    assert me not in manager.eligible_candidates(me, RUN)

    manager.add_rating(me, man, 0.5, RUN)
    assert manager.eligible_candidates(me, RUN) == []
    assert manager.eligible_candidates(me, "other-run") == [man]


def test_users_without_preferences_are_in_nobodys_pool(manager: ProfileManager) -> None:
    me, other = add_users(manager, 2)
    manager.set_preferences(me, ["man"])
    assert manager.eligible_candidates(me, RUN) == []
    assert manager.eligible_candidates(other, RUN) == []


@pytest.mark.parametrize("agent", [True, False])
def test_agent_flag_round_trips(manager: ProfileManager, agent: bool) -> None:
    profile = manager.add_profile(profile_data(agent=agent))
    assert manager.get_profile(profile.user_id).agent is agent
    assert manager.list_profiles()[0].agent is agent


def test_interests_are_normalized_and_round_trip(manager: ProfileManager) -> None:
    profile = manager.add_profile(
        profile_data(interests=[" Hiking", "CHESS ", "hiking", "chess", "Art"])
    )
    assert profile.interests == ["hiking", "chess", "art"]
    assert manager.get_profile(profile.user_id).interests == ["hiking", "chess", "art"]


# Generated IDs


def test_seeded_generate_user_id_is_reproducible() -> None:
    first = random.Random(42)
    second = random.Random(42)
    first_ids = [generate_user_id(first) for _ in range(3)]
    assert [generate_user_id(second) for _ in range(3)] == first_ids
    assert len(set(first_ids)) == 3
    assert all(uuid.UUID(user_id).version == 4 for user_id in first_ids)


def test_unseeded_ids_are_unique() -> None:
    assert len({generate_user_id() for _ in range(1000)}) == 1000


def test_seeded_manager_is_reproducible(tmp_path: Path) -> None:
    first = ProfileManager(
        tmp_path / "a.db", tmp_path / "a_personas.db", rng=random.Random(7)
    )
    second = ProfileManager(
        tmp_path / "b.db", tmp_path / "b_personas.db", rng=random.Random(7)
    )
    assert add_users(first, 3) == add_users(second, 3)


# Editing profiles


def test_edit_profile_applies_and_persists(manager: ProfileManager) -> None:
    profile = manager.add_profile(profile_data())
    edited = manager.edit_profile(
        profile.user_id, job="teacher", interests=["Art", "ART"]
    )
    assert edited.job == "teacher"
    assert edited.interests == ["art"]
    assert edited.name == profile.name
    assert manager.get_profile(profile.user_id) == edited


@pytest.mark.parametrize(
    "changes",
    [
        {"age": 17},
        {"name": None},
        {"name": ""},
        {"job": "   "},
        {"interests": []},
        {"major": "   "},
        {"clubs": ["  "]},
        {"gender": "men"},
        {"interested_in": ["man"]},
        {"looking_for": "marriage"},
        {"nickname": "unknown field"},
    ],
)
def test_failed_edit_leaves_profile_unchanged(
    manager: ProfileManager, changes: dict[str, Any]
) -> None:
    profile = manager.add_profile(profile_data())
    with pytest.raises(ValidationError):
        manager.edit_profile(profile.user_id, **changes)
    assert manager.get_profile(profile.user_id) == profile


def test_edit_cannot_change_user_id(manager: ProfileManager) -> None:
    profile = manager.add_profile(profile_data())
    with pytest.raises(ValueError, match="immutable"):
        manager.edit_profile(profile.user_id, user_id=str(uuid.uuid4()))
    assert manager.get_profile(profile.user_id) == profile


def test_edit_unknown_profile_raises(manager: ProfileManager) -> None:
    with pytest.raises(ProfileNotFoundError):
        manager.edit_profile("missing", job="teacher")


# Ratings


def test_add_rating_stores_all_fields(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    rating = manager.add_rating(
        alice,
        bob,
        0.75,
        RUN,
        model_version="model-1",
        raw_output='{"score": 0.75}',
        input_snapshot="what the rater saw",
    )
    assert rating.timestamp.utcoffset() is not None
    assert rating.model_version == "model-1"
    assert manager.ratings_by_rater(alice) == [rating]
    assert manager.ratings_of_target(bob) == [rating]
    assert manager.ratings_by_rater(bob) == []
    assert manager.ratings_of_target(alice) == []
    transcript = manager.get_transcript(alice, bob, RUN)
    assert transcript.raw_output == '{"score": 0.75}'
    assert transcript.input_snapshot == "what the rater saw"
    assert manager.transcripts_in_run(RUN) == [transcript]
    assert manager.transcripts_in_run("other-run") == []


@pytest.mark.parametrize("score", [0.735, 0.0, 1.0, 0.33333333333333335, 0.0000001])
def test_float_scores_round_trip_exactly(
    manager: ProfileManager, score: float
) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, score, RUN)
    assert manager.ratings_by_rater(alice)[0].score == score


def test_whole_number_score_is_stored_as_float(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 1, RUN)
    stored = manager.ratings_by_rater(alice)[0].score
    assert stored == 1.0
    assert isinstance(stored, float)


@pytest.mark.parametrize(
    "score", [-0.1, 1.1, 2, -1.0, float("nan"), float("inf"), "0.75", True, None]
)
def test_add_rating_rejects_bad_score(manager: ProfileManager, score: Any) -> None:
    alice, bob = add_users(manager, 2)
    with pytest.raises(ValidationError):
        manager.add_rating(alice, bob, score, RUN)
    assert manager.count_ratings() == 0


def test_self_rating_rejected(manager: ProfileManager) -> None:
    alice = add_users(manager, 1)[0]
    with pytest.raises(ValidationError, match="rate themselves"):
        manager.add_rating(alice, alice, 0.5, RUN)
    assert manager.count_ratings() == 0


def test_rating_nonexistent_user_rejected(manager: ProfileManager) -> None:
    alice = add_users(manager, 1)[0]
    with pytest.raises(ProfileNotFoundError):
        manager.add_rating(alice, "missing", 0.5, RUN)
    with pytest.raises(ProfileNotFoundError):
        manager.add_rating("missing", alice, 0.5, RUN)
    assert manager.count_ratings() == 0


def test_rerating_overwrites_within_a_run(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 0.3, RUN, model_version="old", raw_output="old")
    newer = manager.add_rating(alice, bob, 0.9, RUN, model_version="new")
    assert manager.ratings_by_rater(alice) == [newer]
    # The old transcript belonged to the replaced rating, so it mustn't linger.
    assert manager.get_transcript(alice, bob, RUN) is None
    assert manager.count_ratings() == 1


def test_rerating_replaces_transcript(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 0.3, RUN, raw_output="old", input_snapshot="old")
    manager.add_rating(alice, bob, 0.9, RUN, raw_output="new")
    transcript = manager.get_transcript(alice, bob, RUN)
    assert transcript.raw_output == "new"
    assert transcript.input_snapshot is None


def test_rating_without_transcript_data_stores_no_transcript(
    manager: ProfileManager,
) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 0.5, RUN)
    assert manager.get_transcript(alice, bob, RUN) is None


def test_invalid_rating_stores_no_transcript(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    with pytest.raises(ValidationError):
        manager.add_rating(alice, bob, 1.1, RUN, raw_output="x")
    with pytest.raises(ProfileNotFoundError):
        manager.add_rating(alice, "missing", 0.5, RUN, raw_output="x")
    assert manager.transcripts_in_run(RUN) == []


def test_saved_rating_always_has_its_transcript(
    manager: ProfileManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice, bob = add_users(manager, 2)

    def fail(transcript: Any) -> None:
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(manager.personas, "save_transcript", fail)
    with pytest.raises(sqlite3.OperationalError):
        manager.add_rating(alice, bob, 0.5, RUN, raw_output="x")
    assert manager.count_ratings() == 0


def test_transcripts_are_not_in_main_database(
    manager: ProfileManager, main_db: Path
) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(
        alice, bob, 0.5, RUN, raw_output="SECRET_RAW", input_snapshot="SECRET_SNAPSHOT"
    )
    for path in main_db.parent.glob("main.db*"):
        assert b"SECRET" not in path.read_bytes()
    assert manager.get_transcript(alice, bob, RUN) is not None


def test_different_run_stores_a_second_row(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 0.3, "run-1")
    manager.add_rating(alice, bob, 0.5, "run-2")
    assert [r.score for r in manager.ratings_by_rater(alice, "run-1")] == [0.3]
    assert [r.score for r in manager.ratings_by_rater(alice, "run-2")] == [0.5]
    assert [r.score for r in manager.ratings_of_target(bob, "run-2")] == [0.5]
    assert manager.ratings_of_target(bob, "run-3") == []
    assert manager.count_ratings() == 2
    assert manager.count_ratings("run-1") == 1


def test_already_rated_pairs(manager: ProfileManager) -> None:
    alice, bob, carol = add_users(manager, 3)
    manager.add_rating(alice, bob, 0.5, "run-1")
    manager.add_rating(bob, alice, 0.5, "run-1")
    manager.add_rating(alice, carol, 0.5, "run-2")
    assert manager.already_rated_pairs("run-1") == {(alice, bob), (bob, alice)}
    assert manager.already_rated_pairs("run-2") == {(alice, carol)}
    assert manager.already_rated_pairs("run-3") == set()


@pytest.mark.parametrize(
    "rater, target, score",
    [
        ("a", "b", 1.05),  # Score out of range.
        ("a", "a", 0.5),  # Self-rating.
        ("a", "missing", 0.5),  # Foreign key.
    ],
)
def test_database_constraints_reject_bad_rows(
    main_db: Path, rater: str, target: str, score: float
) -> None:
    Store(main_db)
    conn = sqlite3.connect(main_db)
    conn.execute("PRAGMA foreign_keys = ON")
    for user_id in ("a", "b"):
        conn.execute(
            "INSERT INTO profiles VALUES "
            "(?, 'n', 30, 170, 'r', 's', 'm', 'j', '[]', '[]', 'man', 'long-term', 1)",
            (user_id,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ratings (rater_id, target_id, score, run_id, timestamp) "
            "VALUES (?, ?, ?, 'r', 't')",
            (rater, target, score),
        )
    conn.close()


# Mutual pairs


def expected_pair(
    a: str, b: str, a_to_b: float, b_to_a: float, run_id: str
) -> MutualPair:
    if a < b:
        return MutualPair(a, b, a_to_b, b_to_a, run_id)
    return MutualPair(b, a, b_to_a, a_to_b, run_id)


def test_mutual_pairs_threshold_is_inclusive_and_needs_both(
    manager: ProfileManager,
) -> None:
    alice, bob, carol = add_users(manager, 3)
    manager.add_rating(alice, bob, 0.75, RUN)
    manager.add_rating(bob, alice, 0.825, RUN)
    manager.add_rating(alice, carol, 0.9, RUN)
    manager.add_rating(carol, alice, 0.6, RUN)
    assert manager.mutual_pairs(0.75) == [expected_pair(alice, bob, 0.75, 0.825, RUN)]
    assert manager.mutual_pairs(0.76) == []


def test_mutual_pairs_ignore_one_way_ratings(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 1.0, RUN)
    assert manager.mutual_pairs(0.1) == []


def test_mutual_pairs_filter_by_run(manager: ProfileManager) -> None:
    alice, bob, carol = add_users(manager, 3)
    manager.add_rating(alice, bob, 0.8, "run-1")
    manager.add_rating(bob, alice, 0.9, "run-1")
    manager.add_rating(alice, carol, 0.7, "run-2")
    manager.add_rating(carol, alice, 0.7, "run-2")
    run_1 = expected_pair(alice, bob, 0.8, 0.9, "run-1")
    run_2 = expected_pair(alice, carol, 0.7, 0.7, "run-2")
    assert manager.mutual_pairs(0.5, "run-1") == [run_1]
    assert manager.mutual_pairs(0.5, "run-2") == [run_2]
    assert manager.mutual_pairs(0.5) == [run_1, run_2]


def test_mutual_pairs_never_combine_scores_across_runs(
    manager: ProfileManager,
) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 1.0, "run-1")
    manager.add_rating(bob, alice, 1.0, "run-2")
    assert manager.mutual_pairs(0.1) == []


def test_mutual_pair_in_two_runs_appears_once_per_run(
    manager: ProfileManager,
) -> None:
    alice, bob = add_users(manager, 2)
    for run_id, score in (("run-1", 0.6), ("run-2", 0.8)):
        manager.add_rating(alice, bob, score, run_id)
        manager.add_rating(bob, alice, score, run_id)
    assert [pair.run_id for pair in manager.mutual_pairs(0.5)] == ["run-1", "run-2"]


# Deleting profiles


def test_delete_profile_removes_it(manager: ProfileManager) -> None:
    profile = manager.add_profile(profile_data())
    manager.delete_profile(profile.user_id)
    with pytest.raises(ProfileNotFoundError):
        manager.get_profile(profile.user_id)
    assert manager.list_profiles() == []


def test_delete_nonexistent_profile_raises(manager: ProfileManager) -> None:
    with pytest.raises(ProfileNotFoundError):
        manager.delete_profile("missing")


def test_delete_removes_ratings_as_rater_and_target(manager: ProfileManager) -> None:
    alice, bob, carol = add_users(manager, 3)
    manager.add_rating(alice, bob, 0.7, "run-1")
    manager.add_rating(bob, alice, 0.7, "run-2")
    kept = manager.add_rating(bob, carol, 0.7, "run-1")

    manager.delete_profile(alice)

    assert manager.ratings_by_rater(alice) == []
    assert manager.ratings_of_target(alice) == []
    assert manager.ratings_by_rater(bob) == [kept]
    assert manager.ratings_of_target(bob) == []
    assert manager.count_ratings() == 1


def test_delete_removes_persona(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_persona(alice, "alice persona")
    manager.add_persona(bob, "bob persona")

    manager.delete_profile(alice)

    assert manager.get_persona(alice) is None
    assert manager.get_persona(bob) is not None


def test_delete_removes_transcripts_as_rater_and_target(
    manager: ProfileManager,
) -> None:
    alice, bob, carol = add_users(manager, 3)
    manager.add_rating(alice, bob, 0.7, "run-1", raw_output="a to b")
    manager.add_rating(bob, alice, 0.7, "run-2", raw_output="b to a")
    manager.add_rating(bob, carol, 0.7, "run-1", raw_output="b to c")

    manager.delete_profile(alice)

    assert manager.get_transcript(alice, bob, "run-1") is None
    assert manager.get_transcript(bob, alice, "run-2") is None
    assert manager.get_transcript(bob, carol, "run-1") is not None


def test_delete_works_without_a_persona(manager: ProfileManager) -> None:
    alice = add_users(manager, 1)[0]
    manager.delete_profile(alice)
    assert manager.count_profiles() == 0


def test_interrupted_delete_can_be_retried(
    manager: ProfileManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = add_users(manager, 1)[0]
    manager.add_persona(alice, "persona")

    def fail(user_id: str) -> None:
        raise sqlite3.OperationalError("simulated failure")

    with monkeypatch.context() as patch:
        patch.setattr(manager.store, "delete_profile", fail)
        with pytest.raises(sqlite3.OperationalError):
            manager.delete_profile(alice)

    manager.delete_profile(alice)
    assert manager.count_profiles() == 0
    assert manager.get_persona(alice) is None


# Personas


def test_add_persona_round_trips(manager: ProfileManager) -> None:
    alice = add_users(manager, 1)[0]
    seeds = {"openness": 0.8, "tags": ["a", "b"], "nested": {"x": 1}}
    persona = manager.add_persona(alice, "likes long walks", seeds)
    assert persona.generation_seeds == seeds
    assert persona.created_at.utcoffset() is not None
    assert manager.get_persona(alice) == persona


def test_add_persona_defaults_seeds_and_upserts(manager: ProfileManager) -> None:
    alice = add_users(manager, 1)[0]
    assert manager.add_persona(alice, "first").generation_seeds == {}
    manager.add_persona(alice, "second")
    assert manager.get_persona(alice).description == "second"
    assert len(manager.personas.list_personas()) == 1


def test_add_persona_requires_existing_profile(manager: ProfileManager) -> None:
    with pytest.raises(ProfileNotFoundError):
        manager.add_persona("missing", "description")
    assert manager.personas.list_personas() == []


def test_add_persona_rejects_blank_description(manager: ProfileManager) -> None:
    alice = add_users(manager, 1)[0]
    with pytest.raises(ValidationError):
        manager.add_persona(alice, "   ")


def test_get_persona_returns_none_when_absent(manager: ProfileManager) -> None:
    alice = add_users(manager, 1)[0]
    assert manager.get_persona(alice) is None


def test_personas_live_in_a_separate_file(
    manager: ProfileManager, main_db: Path, persona_db: Path
) -> None:
    alice = add_users(manager, 1)[0]
    manager.add_persona(alice, "SECRET_DESCRIPTION", {"seed": "SECRET_SEED"})
    assert main_db != persona_db

    conn = sqlite3.connect(main_db)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    conn.close()
    assert "personas" not in tables

    # Check the raw bytes, including the write-ahead log.
    for path in main_db.parent.glob("main.db*"):
        assert b"SECRET" not in path.read_bytes()
    assert PersonaStore(persona_db).get_persona(alice) is not None


def test_same_file_for_both_databases_rejected(main_db: Path) -> None:
    with pytest.raises(ValueError, match="separate"):
        ProfileManager(main_db, main_db)


def test_store_has_no_persona_access(manager: ProfileManager) -> None:
    assert not [name for name in dir(manager.store) if "persona" in name.lower()]


# Exports


def test_export_profiles_json(manager: ProfileManager, tmp_path: Path) -> None:
    profiles = [manager.add_profile(profile_data(name=n)) for n in ("Ann", "Ben")]
    manager.add_persona(profiles[0].user_id, "SECRET_DESCRIPTION")
    path = tmp_path / "profiles.json"
    manager.store.export_profiles_json(path)
    assert json.loads(path.read_text()) == [p.model_dump() for p in profiles]
    assert "SECRET" not in path.read_text()


def test_export_personas_json(manager: ProfileManager, tmp_path: Path) -> None:
    alice = add_users(manager, 1)[0]
    manager.add_persona(alice, "description", {"seed": 1})
    path = tmp_path / "personas.json"
    manager.personas.export_personas_json(path)
    exported = json.loads(path.read_text())
    assert [(p["user_id"], p["description"]) for p in exported] == [
        (alice, "description")
    ]
    assert exported[0]["generation_seeds"] == {"seed": 1}


# Database setup and concurrency


@pytest.mark.parametrize("db_name", ["main.db", "personas.db"])
def test_wal_mode_is_active(manager: ProfileManager, tmp_path: Path, db_name: str) -> None:
    conn = sqlite3.connect(tmp_path / db_name)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()


def test_data_survives_reopening(main_db: Path, persona_db: Path) -> None:
    first = ProfileManager(main_db, persona_db)
    profile = first.add_profile(profile_data())
    persona = first.add_persona(profile.user_id, "description")
    second = ProfileManager(main_db, persona_db)
    assert second.get_profile(profile.user_id) == profile
    assert second.get_persona(profile.user_id) == persona


def test_in_memory_database_rejected() -> None:
    with pytest.raises(ValueError, match="in-memory"):
        Store(":memory:")


def test_sql_metacharacters_are_stored_literally(manager: ProfileManager) -> None:
    hostile = "x'); DROP TABLE profiles; --"
    profile = manager.add_profile(profile_data(name=hostile))
    assert manager.get_profile(profile.user_id).name == hostile
    assert manager.ratings_by_rater(hostile) == []


def test_concurrent_threads_writing_ratings(manager: ProfileManager) -> None:
    users = add_users(manager, 8)
    pairs = [(a, b) for a in users for b in users if a != b]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda pair: manager.add_rating(*pair, 0.55, RUN), pairs))

    assert manager.count_ratings(RUN) == len(pairs)
    assert len(manager.mutual_pairs(0.55, RUN)) == len(pairs) // 2


def test_concurrent_edits_to_different_fields_are_not_lost(
    manager: ProfileManager,
) -> None:
    user_id = manager.add_profile(profile_data()).user_id
    edits = [
        {"name": "New name"},
        {"age": 44},
        {"height_cm": 190},
        {"religion": "new religion"},
        {"school": "new school"},
        {"job": "new job"},
        {"interests": ["new interest"]},
        {"looking_for": "just a date"},
        {"agent": False},
        {"major": "new major"},
        {"clubs": ["new club"]},
        {"gender": "nonbinary"},
    ]

    with ThreadPoolExecutor(max_workers=len(edits)) as pool:
        list(pool.map(lambda edit: manager.edit_profile(user_id, **edit), edits))

    final = manager.get_profile(user_id)
    for edit in edits:
        for field, value in edit.items():
            assert getattr(final, field) == value


def open_manager_after_barrier(main_db: Path, persona_db: Path, barrier: Any) -> None:
    barrier.wait()
    ProfileManager(main_db, persona_db)


def test_concurrent_creation_of_new_databases(tmp_path: Path) -> None:
    # Needs real processes: threads in one process don't reproduce the lock failure.
    for attempt in range(3):
        barrier = multiprocessing.Barrier(8)
        processes = [
            multiprocessing.Process(
                target=open_manager_after_barrier,
                args=(
                    tmp_path / f"main{attempt}.db",
                    tmp_path / f"personas{attempt}.db",
                    barrier,
                ),
            )
            for _ in range(8)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        assert [process.exitcode for process in processes] == [0] * 8


def test_rating_model_defaults() -> None:
    rating = Rating(rater_id="a", target_id="b", score=0.5, run_id=RUN)
    assert rating.timestamp.utcoffset() is not None
    assert rating.model_version is None
    assert rating.step is None


# Steps


def test_step_round_trips_and_orders_a_run(manager: ProfileManager) -> None:
    alice, bob, carol = add_users(manager, 3)
    second = manager.add_rating(alice, bob, 0.5, RUN, step=2)
    first = manager.add_rating(alice, carol, 0.6, RUN, step=1)
    assert manager.ratings_in_run(RUN) == [first, second]
    assert manager.ratings_in_run("other-run") == []


@pytest.mark.parametrize("step", [0, -1, "1", 1.5])
def test_invalid_step_rejected(manager: ProfileManager, step: Any) -> None:
    alice, bob = add_users(manager, 2)
    with pytest.raises(ValidationError):
        manager.add_rating(alice, bob, 0.5, RUN, step=step)


# Runs


def test_start_and_finish_run(manager: ProfileManager) -> None:
    config = {"n_steps": 5, "nested": {"a": [1, 2]}}
    run = manager.start_run(RUN, "random", 42, config)
    assert run.finished_at is None
    assert manager.get_run(RUN) == run

    finished = manager.finish_run(RUN)

    assert finished.finished_at is not None
    assert finished.finished_at >= finished.started_at
    assert finished.config == config
    assert finished.seed == 42
    assert finished.matcher_name == "random"
    assert manager.get_run(RUN) == finished


def test_seed_and_config_are_optional(manager: ProfileManager) -> None:
    run = manager.start_run(RUN, "random")
    assert run.seed is None
    assert run.config == {}
    assert manager.get_run(RUN) == run


def test_duplicate_run_id_raises(manager: ProfileManager) -> None:
    original = manager.start_run(RUN, "random", 1, {"a": 1})
    with pytest.raises(DuplicateRunError, match=RUN):
        manager.start_run(RUN, "other", 2, {"b": 2})
    assert manager.get_run(RUN) == original


def test_unknown_run_raises(manager: ProfileManager) -> None:
    with pytest.raises(RunNotFoundError):
        manager.get_run("missing")
    with pytest.raises(RunNotFoundError):
        manager.finish_run("missing")


def test_finishing_a_run_twice_raises(manager: ProfileManager) -> None:
    manager.start_run(RUN, "random")
    finished = manager.finish_run(RUN)
    with pytest.raises(ValueError, match="already finished"):
        manager.finish_run(RUN)
    assert manager.get_run(RUN) == finished
