"""SQLite persistence layer.

Each operation opens its own short-lived connection, so a Store can be shared
across threads and the same database file can be used from multiple processes.

Application code should use manager.py. This module also holds persona access,
which must stay out of reach of the matching algorithm.
"""

import json
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from models import ANONYMIZED_ID, Persona, Profile, Rating

# Bump this whenever the tables below change. Existing databases are not migrated.
SCHEMA_VERSION = 1

BUSY_TIMEOUT_SECONDS = 30

SCHEMA = (
    """
    CREATE TABLE profiles (
        user_id   TEXT PRIMARY KEY,
        name      TEXT NOT NULL,
        age       INTEGER NOT NULL,
        height_cm INTEGER NOT NULL,
        religion  TEXT NOT NULL,
        school    TEXT NOT NULL,
        job       TEXT NOT NULL,
        interests TEXT NOT NULL
    )
    """,
    # rater_id and target_id are NULL only on anonymized ratings.
    """
    CREATE TABLE ratings (
        rating_id INTEGER PRIMARY KEY,
        rater_id  TEXT REFERENCES profiles (user_id) ON DELETE CASCADE,
        target_id TEXT REFERENCES profiles (user_id) ON DELETE CASCADE,
        score     INTEGER NOT NULL CHECK (score BETWEEN 1 AND 10),
        timestamp TEXT NOT NULL,
        run_id    TEXT NOT NULL,
        CHECK (rater_id <> target_id),
        UNIQUE (run_id, rater_id, target_id)
    )
    """,
    "CREATE INDEX idx_ratings_rater ON ratings (rater_id, run_id)",
    "CREATE INDEX idx_ratings_target ON ratings (target_id, run_id)",
    """
    CREATE TABLE personas (
        user_id TEXT PRIMARY KEY REFERENCES profiles (user_id) ON DELETE CASCADE,
        data    TEXT NOT NULL
    )
    """,
)

PROFILE_COLUMNS = tuple(Profile.model_fields)
EDITABLE_PROFILE_COLUMNS = tuple(c for c in PROFILE_COLUMNS if c != "user_id")

_INSERT_PROFILE = (
    f"INSERT INTO profiles ({', '.join(PROFILE_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in PROFILE_COLUMNS)})"
)
_UPDATE_PROFILE = (
    f"UPDATE profiles SET {', '.join(f'{c} = ?' for c in EDITABLE_PROFILE_COLUMNS)} "
    "WHERE user_id = ?"
)


class ProfileNotFoundError(LookupError):
    """The requested user_id has no profile."""


class SchemaVersionError(RuntimeError):
    """The database file was created with a different SCHEMA_VERSION."""


class MutualPair(NamedTuple):
    """Two users who rated each other at or above a threshold. user_a < user_b."""

    user_a: str
    user_b: str
    score_a_to_b: int
    score_b_to_a: int


def _profile_to_row(profile: Profile) -> dict[str, object]:
    row = profile.model_dump()
    row["interests"] = json.dumps(profile.interests)
    return row


def _row_to_profile(row: sqlite3.Row) -> Profile:
    data = dict(row)
    data["interests"] = json.loads(data["interests"])
    return Profile.model_validate(data)


def _row_to_rating(row: sqlite3.Row) -> Rating:
    return Rating(
        rater_id=row["rater_id"] if row["rater_id"] is not None else ANONYMIZED_ID,
        target_id=row["target_id"] if row["target_id"] is not None else ANONYMIZED_ID,
        score=row["score"],
        timestamp=row["timestamp"],
        run_id=row["run_id"],
    )


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path in ("", ":memory:"):
            # Every connection would get its own separate, empty database.
            raise ValueError("Store needs a file path, not an in-memory database")
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.db_path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")  # Off by default per connection.
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Take the write lock up front, so a read-then-write can't fail midway."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _enable_wal(self) -> None:
        """Switch to WAL, which persists in the database file.

        When several processes open a new database at once, this can fail with
        SQLITE_BUSY immediately rather than waiting for the timeout, so retry here.
        """
        deadline = time.monotonic() + BUSY_TIMEOUT_SECONDS
        while True:
            try:
                with self._connect() as conn:
                    conn.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as error:
                busy = error.sqlite_errorcode == sqlite3.SQLITE_BUSY
                if not busy or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def _init_db(self) -> None:
        self._enable_wal()
        with self._transaction() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                for statement in SCHEMA:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            elif version != SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{self.db_path} has schema version {version}, but the code "
                    f"expects {SCHEMA_VERSION}. Delete the file or migrate it."
                )

    # Profiles

    def insert_profile(self, profile: Profile) -> None:
        row = _profile_to_row(profile)
        with self._transaction() as conn:
            conn.execute(_INSERT_PROFILE, [row[c] for c in PROFILE_COLUMNS])

    def get_profile(self, user_id: str) -> Profile:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            raise ProfileNotFoundError(f"no profile with user_id {user_id!r}")
        return _row_to_profile(row)

    def list_profiles(self) -> list[Profile]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM profiles ORDER BY rowid").fetchall()
        return [_row_to_profile(row) for row in rows]

    def profile_exists(self, user_id: str) -> bool:
        with self._connect() as conn:
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
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                raise ProfileNotFoundError(f"no profile with user_id {user_id!r}")
            updated = apply(_row_to_profile(row))
            if updated.user_id != user_id:
                raise ValueError("user_id is immutable")
            new_row = _profile_to_row(updated)
            conn.execute(
                _UPDATE_PROFILE,
                [new_row[c] for c in EDITABLE_PROFILE_COLUMNS] + [user_id],
            )
        return updated

    def delete_profile(self, user_id: str, anonymize_ratings: bool = False) -> None:
        """Delete a profile and its persona, across all runs.

        The user's ratings are deleted too, unless anonymize_ratings is True. In
        that case their ID on each rating becomes NULL and the score is kept.
        Ratings with both sides anonymized are deleted, since no query can reach them.
        """
        with self._transaction() as conn:
            if anonymize_ratings:
                conn.execute(
                    "UPDATE ratings SET rater_id = NULL WHERE rater_id = ?", (user_id,)
                )
                conn.execute(
                    "UPDATE ratings SET target_id = NULL WHERE target_id = ?",
                    (user_id,),
                )
                conn.execute(
                    "DELETE FROM ratings WHERE rater_id IS NULL AND target_id IS NULL"
                )
            # ON DELETE CASCADE removes the persona and any remaining ratings.
            cursor = conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
            if cursor.rowcount == 0:
                raise ProfileNotFoundError(f"no profile with user_id {user_id!r}")

    # Ratings

    def upsert_rating(self, rating: Rating) -> None:
        """Store a rating, replacing any earlier one for the same run, rater, and target."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO ratings (rater_id, target_id, score, timestamp, run_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (run_id, rater_id, target_id) DO UPDATE SET
                    score = excluded.score,
                    timestamp = excluded.timestamp
                """,
                (
                    rating.rater_id,
                    rating.target_id,
                    rating.score,
                    # Fixed precision, so timestamps sort correctly as text.
                    rating.timestamp.isoformat(timespec="microseconds"),
                    rating.run_id,
                ),
            )

    def ratings_by(self, rater_id: str, run_id: str | None = None) -> list[Rating]:
        """Ratings this user gave, oldest first, optionally within one run."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ratings
                WHERE rater_id = ? AND (? IS NULL OR run_id = ?)
                ORDER BY timestamp, rating_id
                """,
                (rater_id, run_id, run_id),
            ).fetchall()
        return [_row_to_rating(row) for row in rows]

    def ratings_of(self, target_id: str, run_id: str | None = None) -> list[Rating]:
        """Ratings this user received, oldest first, optionally within one run."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ratings
                WHERE target_id = ? AND (? IS NULL OR run_id = ?)
                ORDER BY timestamp, rating_id
                """,
                (target_id, run_id, run_id),
            ).fetchall()
        return [_row_to_rating(row) for row in rows]

    def mutual_pairs(self, run_id: str, min_score: int) -> list[MutualPair]:
        """Pairs in this run who each rated the other at min_score or higher."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.rater_id, a.target_id, a.score, b.score
                FROM ratings AS a
                JOIN ratings AS b
                    ON a.run_id = b.run_id
                    AND a.rater_id = b.target_id
                    AND a.target_id = b.rater_id
                WHERE a.run_id = ?
                    AND a.rater_id < a.target_id
                    AND a.score >= ?
                    AND b.score >= ?
                ORDER BY a.rater_id, a.target_id
                """,
                (run_id, min_score, min_score),
            ).fetchall()
        return [MutualPair(*row) for row in rows]

    # Personas

    def save_persona(self, persona: Persona) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO personas (user_id, data) VALUES (?, ?)
                ON CONFLICT (user_id) DO UPDATE SET data = excluded.data
                """,
                (persona.user_id, persona.model_dump_json()),
            )

    def get_persona(self, user_id: str) -> Persona | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM personas WHERE user_id = ?", (user_id,)
            ).fetchone()
        return None if row is None else Persona.model_validate_json(row["data"])

    # Export

    def export_profiles_json(self, path: str | Path) -> None:
        """Write all profiles to a JSON file for inspection. Personas are excluded."""
        profiles = [profile.model_dump(mode="json") for profile in self.list_profiles()]
        Path(path).write_text(json.dumps(profiles, indent=2), encoding="utf-8")
