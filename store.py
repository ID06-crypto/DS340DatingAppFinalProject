"""SQLite persistence for profiles, preferences, ratings, and runs (main.db).

This module can't reach personas.db, and nothing stored here contains persona text.
main.db also holds the app's tables (see live_schema.py), because a user's eligible pool
depends on accounts, blocks, matches, and impressions.
"""

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from db import connect, initialize, transaction
from live_schema import LIVE_SCHEMA
from models import LIVE_RUN_ID, Preferences, Profile, Rating, Run

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS profiles (
        user_id       TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        age           INTEGER NOT NULL,
        height_cm     INTEGER NOT NULL,
        religion      TEXT,
        school        TEXT NOT NULL,
        major         TEXT NOT NULL,
        job           TEXT NOT NULL,
        interests     TEXT NOT NULL,
        clubs         TEXT NOT NULL,
        gender        TEXT NOT NULL,
        looking_for   TEXT NOT NULL,
        agent         INTEGER NOT NULL CHECK (agent IN (0, 1))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ratings (
        rater_id       TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        target_id      TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        score          REAL NOT NULL CHECK (score BETWEEN 0.0 AND 1.0),
        run_id         TEXT NOT NULL,
        timestamp      TEXT NOT NULL,
        model_version  TEXT,
        step           INTEGER,
        matcher        TEXT,
        PRIMARY KEY (rater_id, target_id, run_id),
        CHECK (rater_id <> target_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ratings_target ON ratings (target_id, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_ratings_run ON ratings (run_id)",
    """
    CREATE TABLE IF NOT EXISTS preferences (
        user_id TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        gender  TEXT NOT NULL CHECK (gender IN ('man', 'woman', 'nonbinary')),
        PRIMARY KEY (user_id, gender)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_preferences_gender ON preferences (gender, user_id)",
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id       TEXT PRIMARY KEY,
        matcher_name TEXT NOT NULL,
        seed         INTEGER,
        config       TEXT NOT NULL,
        started_at   TEXT NOT NULL,
        finished_at  TEXT
    )
    """,
)

PROFILE_COLUMNS = tuple(Profile.model_fields)
EDITABLE_PROFILE_COLUMNS = tuple(c for c in PROFILE_COLUMNS if c != "user_id")
RATING_COLUMNS = tuple(Rating.model_fields)
RATING_KEY_COLUMNS = ("rater_id", "target_id", "run_id")
RATING_VALUE_COLUMNS = tuple(c for c in RATING_COLUMNS if c not in RATING_KEY_COLUMNS)

_INSERT_PROFILE = (
    f"INSERT INTO profiles ({', '.join(PROFILE_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in PROFILE_COLUMNS)})"
)
_UPDATE_PROFILE = (
    f"UPDATE profiles SET {', '.join(f'{c} = ?' for c in EDITABLE_PROFILE_COLUMNS)} "
    "WHERE user_id = ?"
)
_UPSERT_RATING = (
    f"INSERT INTO ratings ({', '.join(RATING_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in RATING_COLUMNS)}) "
    f"ON CONFLICT ({', '.join(RATING_KEY_COLUMNS)}) DO UPDATE SET "
    f"{', '.join(f'{c} = excluded.{c}' for c in RATING_VALUE_COLUMNS)}"
)


class ProfileNotFoundError(LookupError):
    """The requested user_id has no profile."""


class DuplicateProfileError(ValueError):
    """A profile with this user_id already exists."""


class RunNotFoundError(LookupError):
    """The requested run_id has no run."""


class DuplicateRunError(ValueError):
    """A run with this run_id already exists."""


class OutdatedSchemaError(RuntimeError):
    """The database uses an older schema. Run scripts/migrate_schema.py on it first."""


class MutualPair(NamedTuple):
    """Two users who both scored each other at or above a threshold. user_a < user_b."""

    user_a: str
    user_b: str
    score_a_to_b: float
    score_b_to_a: float
    run_id: str


def _profile_to_row(profile: Profile) -> dict[str, object]:
    row = profile.model_dump()
    row["interests"] = json.dumps(profile.interests)
    row["clubs"] = json.dumps(profile.clubs)
    return row


def _row_to_profile(row: sqlite3.Row) -> Profile:
    data = dict(row)
    data["interests"] = json.loads(data["interests"])
    data["clubs"] = json.loads(data["clubs"])
    data["agent"] = bool(data["agent"])  # SQLite stores booleans as 0 and 1.
    return Profile.model_validate(data)


def _rating_to_row(rating: Rating) -> dict[str, object]:
    row = rating.model_dump()
    # Fixed precision, so timestamps sort correctly as text.
    row["timestamp"] = rating.timestamp.isoformat(timespec="microseconds")
    return row


def _not_found(user_id: str) -> ProfileNotFoundError:
    return ProfileNotFoundError(f"no profile with user_id {user_id!r}")


def _row_to_run(row: sqlite3.Row) -> Run:
    data = dict(row)
    data["config"] = json.loads(data["config"])
    return Run.model_validate(data)


# A user's eligible pool. Each condition is one rule, so the rules can be read directly:
# mutual gender interest, approved accounts, no blocks either way, no active match, and
# nobody already rated in this run (or already shown, in the live app).
_ELIGIBLE_CANDIDATES = """
    SELECT c.user_id
    FROM profiles AS me
    JOIN profiles AS c ON c.user_id <> me.user_id
    WHERE me.user_id = :me
        AND EXISTS (SELECT 1 FROM preferences WHERE user_id = :me AND gender = c.gender)
        AND EXISTS (SELECT 1 FROM preferences WHERE user_id = c.user_id AND gender = me.gender)
        AND CASE WHEN :live
            THEN EXISTS (
                SELECT 1 FROM accounts
                WHERE user_id = c.user_id AND status = 'approved' AND paused = 0
            )
            -- Simulated users have no account, and count as approved in simulations.
            ELSE NOT EXISTS (
                SELECT 1 FROM accounts WHERE user_id = c.user_id AND status <> 'approved'
            )
        END
        AND NOT EXISTS (
            SELECT 1 FROM blocks
            WHERE (blocker_id = :me AND blocked_id = c.user_id)
                OR (blocker_id = c.user_id AND blocked_id = :me)
        )
        AND NOT EXISTS (
            SELECT 1 FROM matches
            WHERE status = 'active'
                AND user_a = MIN(:me, c.user_id) AND user_b = MAX(:me, c.user_id)
        )
        AND NOT EXISTS (
            SELECT 1 FROM ratings
            WHERE rater_id = :me AND target_id = c.user_id AND run_id = :run_id
        )
        AND NOT (:live AND EXISTS (
            SELECT 1 FROM impressions WHERE viewer_id = :me AND target_id = c.user_id
        ))
    ORDER BY c.user_id
"""


# Columns added after a table was first created. CREATE TABLE IF NOT EXISTS can't add them.
REQUIRED_COLUMNS = {
    "ratings": {"matcher"},
    "accounts": {"experiment_opt_out", "notify_on_match", "paused"},
    "impressions": {"predicted_score", "position"},
}


def _check_schema(db_path: str | Path) -> None:
    """Refuse a database made with an older schema, and say how to migrate it."""
    if not Path(db_path).exists():
        return
    with connect(db_path) as conn:
        columns = {
            table: {row["name"]: row for row in conn.execute(f"PRAGMA table_info({table})")}
            for table in ("profiles", *REQUIRED_COLUMNS)
        }
    profiles = columns["profiles"]
    outdated = "interested_in" in profiles or (profiles and profiles["religion"]["notnull"])
    for table, required in REQUIRED_COLUMNS.items():
        if columns[table] and not required <= set(columns[table]):
            outdated = True
    if outdated:
        raise OutdatedSchemaError(
            f"{db_path} uses an older schema. Run: python scripts/migrate_schema.py {db_path}"
        )


class Store:
    def __init__(self, db_path: str | Path = "main.db") -> None:
        self.db_path = db_path
        _check_schema(db_path)
        initialize(db_path, SCHEMA + LIVE_SCHEMA)

    # Profiles

    def insert_profile(self, profile: Profile) -> None:
        row = _profile_to_row(profile)
        try:
            with transaction(self.db_path) as conn:
                conn.execute(_INSERT_PROFILE, [row[c] for c in PROFILE_COLUMNS])
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY:
                raise DuplicateProfileError(
                    f"a profile with user_id {profile.user_id!r} already exists"
                ) from error
            raise

    def get_profile(self, user_id: str) -> Profile:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            raise _not_found(user_id)
        return _row_to_profile(row)

    def profile_exists(self, user_id: str) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row is not None

    def update_profile(
        self, user_id: str, apply: Callable[[Profile], Profile]
    ) -> Profile:
        """Replace a profile with apply(current), holding the write lock throughout.

        Holding the lock means concurrent edits to different fields can't overwrite
        each other, and a failure in apply leaves the stored row unchanged.
        """
        with transaction(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                raise _not_found(user_id)
            updated = apply(_row_to_profile(row))
            if updated.user_id != user_id:
                raise ValueError("user_id is immutable")
            new_row = _profile_to_row(updated)
            conn.execute(
                _UPDATE_PROFILE,
                [new_row[c] for c in EDITABLE_PROFILE_COLUMNS] + [user_id],
            )
        return updated

    def delete_profile(self, user_id: str) -> None:
        """Delete a profile. ON DELETE CASCADE removes its ratings in every run."""
        with transaction(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
            if cursor.rowcount == 0:
                raise _not_found(user_id)

    def list_all_profiles(self) -> list[Profile]:
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM profiles ORDER BY rowid").fetchall()
        return [_row_to_profile(row) for row in rows]

    def count_profiles(self) -> int:
        with connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]

    def export_profiles_json(self, path: str | Path) -> None:
        profiles = [profile.model_dump() for profile in self.list_all_profiles()]
        Path(path).write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    # Preferences (private)

    def set_preferences(self, preferences: Preferences) -> None:
        """Replace a user's preferences."""
        with transaction(self.db_path) as conn:
            if conn.execute(
                "SELECT 1 FROM profiles WHERE user_id = ?", (preferences.user_id,)
            ).fetchone() is None:
                raise _not_found(preferences.user_id)
            conn.execute("DELETE FROM preferences WHERE user_id = ?", (preferences.user_id,))
            conn.executemany(
                "INSERT INTO preferences (user_id, gender) VALUES (?, ?)",
                [(preferences.user_id, gender) for gender in sorted(preferences.interested_in)],
            )

    def get_preferences(self, user_id: str) -> Preferences | None:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT gender FROM preferences WHERE user_id = ?", (user_id,)
            ).fetchall()
        if not rows:
            return None
        return Preferences(user_id=user_id, interested_in=frozenset(row["gender"] for row in rows))

    def all_preferences(self) -> dict[str, frozenset[str]]:
        """Everyone's preferences at once, for analysis code that needs many."""
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT user_id, gender FROM preferences").fetchall()
        preferences: dict[str, set[str]] = {}
        for row in rows:
            preferences.setdefault(row["user_id"], set()).add(row["gender"])
        return {user: frozenset(genders) for user, genders in preferences.items()}

    def eligible_candidates(self, user_id: str, run_id: str = LIVE_RUN_ID) -> list[str]:
        """The user_ids this user may be shown next in this run, sorted.

        Only approved, unpaused accounts, never themselves, nobody blocked either way, nobody they
        have an active match with, mutual gender interest, and nobody they've already
        rated in this run. In the live app, also nobody they've already been shown.
        Users without preferences have an empty pool and appear in nobody's.
        """
        with connect(self.db_path) as conn:
            rows = conn.execute(
                _ELIGIBLE_CANDIDATES,
                {"me": user_id, "run_id": run_id, "live": run_id == LIVE_RUN_ID},
            ).fetchall()
        return [row["user_id"] for row in rows]

    # Ratings

    def upsert_rating(self, rating: Rating) -> None:
        """Store a rating, replacing all fields of any earlier one for this key."""
        row = _rating_to_row(rating)
        with transaction(self.db_path) as conn:
            conn.execute(_UPSERT_RATING, [row[c] for c in RATING_COLUMNS])

    def ratings_by_rater(self, rater_id: str, run_id: str | None = None) -> list[Rating]:
        """Ratings this user gave, oldest first, optionally within one run."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM ratings
                WHERE rater_id = ? AND (? IS NULL OR run_id = ?)
                ORDER BY timestamp, target_id, run_id
                """,
                (rater_id, run_id, run_id),
            ).fetchall()
        return [Rating.model_validate(dict(row)) for row in rows]

    def ratings_of_target(
        self, target_id: str, run_id: str | None = None
    ) -> list[Rating]:
        """Ratings this user received, oldest first, optionally within one run."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM ratings
                WHERE target_id = ? AND (? IS NULL OR run_id = ?)
                ORDER BY timestamp, rater_id, run_id
                """,
                (target_id, run_id, run_id),
            ).fetchall()
        return [Rating.model_validate(dict(row)) for row in rows]

    def mutual_pairs(
        self, min_score: float, run_id: str | None = None
    ) -> list[MutualPair]:
        """Pairs who both scored each other at min_score or higher, in the same run.

        Scores are never combined across runs. With run_id=None, a pair that
        qualifies in several runs appears once for each of those runs.
        """
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT a.rater_id, a.target_id, a.score, b.score, a.run_id
                FROM ratings AS a
                JOIN ratings AS b
                    ON a.run_id = b.run_id
                    AND a.rater_id = b.target_id
                    AND a.target_id = b.rater_id
                WHERE a.rater_id < a.target_id
                    AND a.score >= ?
                    AND b.score >= ?
                    AND (? IS NULL OR a.run_id = ?)
                ORDER BY a.run_id, a.rater_id, a.target_id
                """,
                (min_score, min_score, run_id, run_id),
            ).fetchall()
        return [MutualPair(*row) for row in rows]

    def ratings_in_run(self, run_id: str) -> list[Rating]:
        """All ratings in this run, in the order they were produced."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM ratings
                WHERE run_id = ?
                ORDER BY step, timestamp, rater_id, target_id
                """,
                (run_id,),
            ).fetchall()
        return [Rating.model_validate(dict(row)) for row in rows]

    def already_rated_pairs(self, run_id: str) -> set[tuple[str, str]]:
        """The (rater_id, target_id) pairs already scored in this run."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT rater_id, target_id FROM ratings WHERE run_id = ?", (run_id,)
            ).fetchall()
        return {(row["rater_id"], row["target_id"]) for row in rows}

    def count_ratings(self, run_id: str | None = None) -> int:
        with connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM ratings WHERE ? IS NULL OR run_id = ?",
                (run_id, run_id),
            ).fetchone()[0]

    # Runs

    def insert_run(self, run: Run) -> None:
        try:
            with transaction(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, matcher_name, seed, config, started_at, finished_at)
                    VALUES (?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        run.run_id,
                        run.matcher_name,
                        run.seed,
                        json.dumps(run.config),
                        run.started_at.isoformat(timespec="microseconds"),
                    ),
                )
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY:
                raise DuplicateRunError(
                    f"a run with run_id {run.run_id!r} already exists"
                ) from error
            raise

    def get_run(self, run_id: str) -> Run:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFoundError(f"no run with run_id {run_id!r}")
        return _row_to_run(row)

    def finish_run(self, run_id: str) -> Run:
        """Record that a run has finished. A run can only be finished once."""
        finished_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with transaction(self.db_path) as conn:
            row = conn.execute(
                "SELECT finished_at FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(f"no run with run_id {run_id!r}")
            if row["finished_at"] is not None:
                raise ValueError(f"run {run_id!r} has already finished")
            conn.execute(
                "UPDATE runs SET finished_at = ? WHERE run_id = ?",
                (finished_at, run_id),
            )
        return self.get_run(run_id)
