"""SQLite connection helpers shared by store.py and personas.py.

Each operation opens its own short-lived connection, so the same database file
can be used from multiple threads and processes.
"""

import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

BUSY_TIMEOUT_SECONDS = 60


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # Off by default per connection.
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Take the write lock up front, so a read-then-write can't fail midway."""
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")


def _enable_wal(db_path: str | Path) -> None:
    """Switch to WAL, which persists in the database file.

    When several processes open a new database at once, this can fail with
    SQLITE_BUSY immediately rather than waiting for the timeout, so retry here.
    """
    deadline = time.monotonic() + BUSY_TIMEOUT_SECONDS
    while True:
        try:
            with connect(db_path) as conn:
                conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as error:
            busy = error.sqlite_errorcode == sqlite3.SQLITE_BUSY
            if not busy or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def initialize(db_path: str | Path, schema: Iterable[str]) -> None:
    """Enable WAL and create any missing tables."""
    if str(db_path) in ("", ":memory:"):
        # Every connection would get its own separate, empty database.
        raise ValueError("a file path is required, not an in-memory database")
    _enable_wal(db_path)
    with transaction(db_path) as conn:
        for statement in schema:
            conn.execute(statement)
