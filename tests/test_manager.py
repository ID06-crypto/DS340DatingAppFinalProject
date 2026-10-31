import json
import multiprocessing
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from manager import ProfileManager
from models import ANONYMIZED_ID, Persona, Profile, generate_user_id
from store import MutualPair, ProfileNotFoundError, SchemaVersionError, Store

RUN = "run-1"


def profile_data(**overrides: Any) -> dict[str, Any]:
    data = {
        "name": "Sam",
        "age": 30,
        "height_cm": 175,
        "religion": "none",
        "school": "BU",
        "job": "engineer",
        "interests": ["hiking"],
    }
    return {**data, **overrides}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def manager(db_path: Path) -> ProfileManager:
    return ProfileManager(db_path)


def add_users(manager: ProfileManager, count: int) -> list[str]:
    return [manager.add_profile(profile_data()).user_id for _ in range(count)]


# Creating profiles


def test_add_profile_generates_id_and_persists(manager: ProfileManager) -> None:
    profile = manager.add_profile(profile_data())
    assert profile.user_id
    assert manager.get_profile(profile.user_id) == profile
    assert manager.list_profiles() == [profile]


def test_add_profile_rejects_supplied_user_id(manager: ProfileManager) -> None:
    with pytest.raises(ValueError, match="user_id"):
        manager.add_profile(profile_data(user_id="chosen"))
    assert manager.list_profiles() == []


@pytest.mark.parametrize(
    "field",
    ["name", "age", "height_cm", "religion", "school", "job", "interests"],
)
def test_add_profile_rejects_missing_field(
    manager: ProfileManager, field: str
) -> None:
    data = profile_data()
    del data[field]
    with pytest.raises(ValidationError, match=field):
        manager.add_profile(data)
    assert manager.list_profiles() == []


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
        {"job": "  "},
        {"interests": []},
        {"interests": ["hiking", "  "]},
        {"interests": "hiking"},
        {"nickname": "typo field"},
    ],
)
def test_add_profile_rejects_invalid_values(
    manager: ProfileManager, overrides: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        manager.add_profile(profile_data(**overrides))
    assert manager.list_profiles() == []


def test_interests_are_normalized_and_round_trip(manager: ProfileManager) -> None:
    profile = manager.add_profile(
        profile_data(interests=[" Hiking", "CHESS ", "hiking", "chess", "Art"])
    )
    assert profile.interests == ["hiking", "chess", "art"]
    assert manager.get_profile(profile.user_id).interests == ["hiking", "chess", "art"]


# Seeded IDs


def test_seeded_ids_are_reproducible(tmp_path: Path) -> None:
    first = ProfileManager(tmp_path / "a.db", rng=42)
    second = ProfileManager(tmp_path / "b.db", rng=42)
    first_ids = add_users(first, 3)
    assert add_users(second, 3) == first_ids
    assert len(set(first_ids)) == 3


def test_manager_accepts_random_instance(tmp_path: Path) -> None:
    first = ProfileManager(tmp_path / "a.db", rng=random.Random(7))
    second = ProfileManager(tmp_path / "b.db", rng=7)
    assert add_users(first, 2) == add_users(second, 2)


def test_different_seeds_give_different_ids(tmp_path: Path) -> None:
    first = ProfileManager(tmp_path / "a.db", rng=1)
    second = ProfileManager(tmp_path / "b.db", rng=2)
    assert add_users(first, 1) != add_users(second, 1)


def test_generate_user_id_accepts_seed_or_random() -> None:
    assert generate_user_id(5) == generate_user_id(5)
    assert generate_user_id(random.Random(5)) == generate_user_id(5)
    assert generate_user_id() != generate_user_id()


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
    [{"age": 17}, {"name": None}, {"name": ""}, {"interests": []}, {"nickname": "x"}],
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
        manager.edit_profile(profile.user_id, user_id="other")
    assert manager.get_profile(profile.user_id) == profile


def test_edit_unknown_profile_raises(manager: ProfileManager) -> None:
    with pytest.raises(ProfileNotFoundError):
        manager.edit_profile("missing", job="teacher")


# Ratings


def test_add_rating_is_queryable_both_ways(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    rating = manager.add_rating(alice, bob, 7, RUN)
    assert manager.ratings_by(alice) == [rating]
    assert manager.ratings_of(bob) == [rating]
    assert manager.ratings_by(bob) == []
    assert manager.ratings_of(alice) == []


@pytest.mark.parametrize("score", [0, 11, -1, "7", 7.5, 7.0, True, None])
def test_add_rating_rejects_bad_score(manager: ProfileManager, score: Any) -> None:
    alice, bob = add_users(manager, 2)
    with pytest.raises(ValidationError):
        manager.add_rating(alice, bob, score, RUN)
    assert manager.ratings_by(alice) == []


@pytest.mark.parametrize("score", [1, 10])
def test_add_rating_accepts_score_bounds(manager: ProfileManager, score: int) -> None:
    alice, bob = add_users(manager, 2)
    assert manager.add_rating(alice, bob, score, RUN).score == score


def test_self_rating_rejected(manager: ProfileManager) -> None:
    alice = add_users(manager, 1)[0]
    with pytest.raises(ValidationError, match="rate themselves"):
        manager.add_rating(alice, alice, 5, RUN)
    assert manager.ratings_by(alice) == []


def test_rating_unknown_user_rejected(manager: ProfileManager) -> None:
    alice = add_users(manager, 1)[0]
    with pytest.raises(ProfileNotFoundError):
        manager.add_rating(alice, "missing", 5, RUN)
    with pytest.raises(ProfileNotFoundError):
        manager.add_rating("missing", alice, 5, RUN)


def test_database_enforces_rating_references(db_path: Path) -> None:
    Store(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ratings (rater_id, target_id, score, timestamp, run_id) "
            "VALUES ('x', 'y', 5, '2026-01-01T00:00:00+00:00', 'r')"
        )
    conn.close()


def test_rerating_replaces_within_run_only(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 3, "run-1")
    manager.add_rating(alice, bob, 9, "run-1")
    manager.add_rating(alice, bob, 5, "run-2")
    assert [r.score for r in manager.ratings_by(alice, "run-1")] == [9]
    assert [r.score for r in manager.ratings_by(alice, "run-2")] == [5]
    assert len(manager.ratings_by(alice)) == 2


def test_rating_queries_filter_by_run(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 3, "run-1")
    manager.add_rating(alice, bob, 5, "run-2")
    assert [r.run_id for r in manager.ratings_of(bob, "run-2")] == ["run-2"]
    assert manager.ratings_of(bob, "run-3") == []


def test_timestamps_are_stored_as_utc_and_ordered(manager: ProfileManager) -> None:
    alice, bob, carol = add_users(manager, 3)
    eastern = timezone(timedelta(hours=-5))
    # 09:00 at UTC-5 is 14:00 UTC, so it's later than 10:00 UTC despite the clock time.
    later = manager.add_rating(
        alice, bob, 5, RUN, timestamp=datetime(2026, 1, 1, 9, tzinfo=eastern)
    )
    earlier = manager.add_rating(
        alice, carol, 5, RUN, timestamp=datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    )
    assert manager.ratings_by(alice) == [earlier, later]
    assert manager.ratings_by(alice)[1].timestamp.utcoffset() == timedelta(0)


def test_naive_timestamp_rejected(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    with pytest.raises(ValidationError):
        manager.add_rating(alice, bob, 5, RUN, timestamp=datetime(2026, 1, 1))


# Mutual pairs


def test_mutual_pairs_threshold_is_inclusive_and_needs_both(
    manager: ProfileManager,
) -> None:
    alice, bob, carol = add_users(manager, 3)
    manager.add_rating(alice, bob, 7, RUN)
    manager.add_rating(bob, alice, 8, RUN)
    manager.add_rating(alice, carol, 9, RUN)
    manager.add_rating(carol, alice, 6, RUN)

    first, second = sorted([alice, bob])
    first_score, second_score = (7, 8) if first == alice else (8, 7)
    assert manager.mutual_pairs(RUN, 7) == [
        MutualPair(first, second, first_score, second_score)
    ]
    assert manager.mutual_pairs(RUN, 9) == []


def test_mutual_pairs_ignore_one_way_and_other_runs(manager: ProfileManager) -> None:
    alice, bob, carol = add_users(manager, 3)
    manager.add_rating(alice, carol, 10, RUN)
    manager.add_rating(alice, bob, 10, "run-1")
    manager.add_rating(bob, alice, 10, "run-2")
    assert manager.mutual_pairs("run-1", 1) == []
    assert manager.mutual_pairs("run-2", 1) == []


# Deleting profiles


def test_delete_profile_removes_it(manager: ProfileManager) -> None:
    profile = manager.add_profile(profile_data())
    manager.delete_profile(profile.user_id)
    with pytest.raises(ProfileNotFoundError):
        manager.get_profile(profile.user_id)
    assert manager.list_profiles() == []


def test_delete_unknown_profile_raises(manager: ProfileManager) -> None:
    with pytest.raises(ProfileNotFoundError):
        manager.delete_profile("missing")


def test_delete_cleans_up_ratings_in_all_runs(
    manager: ProfileManager, db_path: Path
) -> None:
    alice, bob, carol = add_users(manager, 3)
    manager.add_rating(alice, bob, 7, "run-1")
    manager.add_rating(bob, alice, 7, "run-2")
    manager.add_rating(bob, carol, 7, "run-1")

    manager.delete_profile(alice)

    assert manager.ratings_by(alice) == []
    assert manager.ratings_of(alice) == []
    assert manager.ratings_of(bob) == []
    assert [r.target_id for r in manager.ratings_by(bob)] == [carol]
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] == 1
    conn.close()


def test_anonymizing_delete_keeps_scores_without_orphans(
    manager: ProfileManager, db_path: Path
) -> None:
    alice, bob = add_users(manager, 2)
    manager.add_rating(alice, bob, 7, RUN)
    manager.add_rating(bob, alice, 4, RUN)

    manager.delete_profile(alice, anonymize_ratings=True)

    received = manager.ratings_of(bob)
    given = manager.ratings_by(bob)
    assert [(r.rater_id, r.score) for r in received] == [(ANONYMIZED_ID, 7)]
    assert [(r.target_id, r.score) for r in given] == [(ANONYMIZED_ID, 4)]
    assert manager.mutual_pairs(RUN, 1) == []
    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM ratings WHERE ? IN (rater_id, target_id)", (alice,)
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_anonymizing_both_sides_removes_rating(
    manager: ProfileManager, db_path: Path
) -> None:
    alice, bob, carol = add_users(manager, 3)
    manager.add_rating(alice, bob, 7, RUN)
    manager.add_rating(alice, carol, 3, RUN)

    manager.delete_profile(alice, anonymize_ratings=True)
    manager.delete_profile(bob, anonymize_ratings=True)

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] == 1
    conn.close()
    assert [r.score for r in manager.ratings_of(carol)] == [3]


def test_anonymizing_unknown_profile_changes_nothing(manager: ProfileManager) -> None:
    alice, bob = add_users(manager, 2)
    rating = manager.add_rating(alice, bob, 7, RUN)
    with pytest.raises(ProfileNotFoundError):
        manager.delete_profile("missing", anonymize_ratings=True)
    assert manager.ratings_by(alice) == [rating]


@pytest.mark.parametrize("anonymize", [False, True])
def test_delete_removes_persona(
    manager: ProfileManager, db_path: Path, anonymize: bool
) -> None:
    alice = add_users(manager, 1)[0]
    store = Store(db_path)
    store.save_persona(Persona(user_id=alice, valued_traits={"kindness": 0.9}))
    assert store.get_persona(alice) is not None

    manager.delete_profile(alice, anonymize_ratings=anonymize)

    assert store.get_persona(alice) is None


# Personas stay separate from public data


def test_persona_round_trips_and_updates(
    manager: ProfileManager, db_path: Path
) -> None:
    alice = add_users(manager, 1)[0]
    store = Store(db_path)
    store.save_persona(Persona(user_id=alice, notes="first"))
    updated = Persona(user_id=alice, valued_traits={"humor": 0.5}, notes="second")
    store.save_persona(updated)
    assert store.get_persona(alice) == updated


def test_persona_requires_existing_profile(db_path: Path) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        Store(db_path).save_persona(Persona(user_id="missing"))


def test_persona_is_not_exposed_by_public_interface(
    manager: ProfileManager, db_path: Path, tmp_path: Path
) -> None:
    alice = add_users(manager, 1)[0]
    Store(db_path).save_persona(
        Persona(user_id=alice, valued_traits={"secret_trait": 1.0}, notes="hidden note")
    )
    export_path = tmp_path / "profiles.json"
    manager.export_profiles_json(export_path)

    assert "secret_trait" not in export_path.read_text()
    assert "hidden note" not in export_path.read_text()
    assert "secret_trait" not in manager.get_profile(alice).model_dump_json()
    assert not [name for name in dir(manager) if "persona" in name.lower()]
    assert not (set(Persona.model_fields) - {"user_id"}) & set(Profile.model_fields)


# Export


def test_export_profiles_json(manager: ProfileManager, tmp_path: Path) -> None:
    profiles = [manager.add_profile(profile_data(name=n)) for n in ("Ann", "Ben")]
    path = tmp_path / "profiles.json"
    manager.export_profiles_json(path)
    assert json.loads(path.read_text()) == [p.model_dump() for p in profiles]


# Database setup and concurrency


def test_wal_mode_is_active(manager: ProfileManager, db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()


def test_data_survives_reopening(db_path: Path) -> None:
    profile = ProfileManager(db_path).add_profile(profile_data())
    assert ProfileManager(db_path).get_profile(profile.user_id) == profile


def test_schema_version_mismatch_raises(db_path: Path) -> None:
    Store(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA user_version = 999")
    conn.close()
    with pytest.raises(SchemaVersionError, match="999"):
        Store(db_path)


def test_in_memory_database_rejected() -> None:
    with pytest.raises(ValueError, match="in-memory"):
        Store(":memory:")


def open_store_after_barrier(db_path: Path, barrier: Any) -> None:
    barrier.wait()
    Store(db_path)


def test_concurrent_creation_of_new_database(tmp_path: Path) -> None:
    # Needs real processes: threads in one process don't reproduce the lock failure.
    for attempt in range(3):
        barrier = multiprocessing.Barrier(8)
        processes = [
            multiprocessing.Process(
                target=open_store_after_barrier,
                args=(tmp_path / f"new{attempt}.db", barrier),
            )
            for _ in range(8)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        assert [process.exitcode for process in processes] == [0] * 8


def test_sql_metacharacters_are_stored_literally(manager: ProfileManager) -> None:
    hostile = "x'); DROP TABLE profiles; --"
    profile = manager.add_profile(profile_data(name=hostile))
    assert manager.get_profile(profile.user_id).name == hostile
    assert manager.ratings_by(hostile) == []


def test_concurrent_rating_writes_are_all_kept(manager: ProfileManager) -> None:
    users = add_users(manager, 8)
    pairs = [(a, b) for a in users for b in users if a != b]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda pair: manager.add_rating(*pair, 5, RUN), pairs))

    assert sum(len(manager.ratings_by(user)) for user in users) == len(pairs)
    assert len(manager.mutual_pairs(RUN, 5)) == len(pairs) // 2


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
    ]

    with ThreadPoolExecutor(max_workers=len(edits)) as pool:
        list(pool.map(lambda edit: manager.edit_profile(user_id, **edit), edits))

    final = manager.get_profile(user_id)
    for edit in edits:
        for field, value in edit.items():
            assert getattr(final, field) == value
