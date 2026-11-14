"""Copies of the live database, made with SQLite's backup API, so each one is consistent
even while people are rating.

The running app makes one every BACKUP_EVERY_HOURS into a backups folder next to the
database, keeping the newest BACKUP_KEEP. Those sit on the same disk, so they protect
against mistakes, not against losing the disk: admins should also download a copy now
and then (Admin → Overview → Download a full backup).
"""

import logging
import sqlite3
from contextlib import closing
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def backup_to(source: Path, target: Path) -> Path:
    """Copy the database at source into a new file at target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # closing() matters: sqlite3's own "with" commits but leaves the connection open.
    with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as src, closing(sqlite3.connect(target)) as dst:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")  # One self-contained file, with no -wal or -shm.
        if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"The backup at {target} failed its integrity check")
    return target


def backup_now(source: Path, folder: Path, keep: int) -> Path:
    """Make a dated backup in folder, and delete all but the newest keep."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = backup_to(source, folder / f"main-{stamp}.db")
    for old in sorted(folder.glob("main-*.db"))[:-keep]:
        old.unlink()
    return path


def start_backups(source: Path, folder: Path, every_hours: int, keep: int) -> threading.Thread:
    """Back up now, then every every_hours, on a background thread."""
    stop = threading.Event()

    def loop() -> None:
        while True:
            try:
                path = backup_now(source, folder, keep)
                logger.info("Database backed up to %s", path.name)
            except Exception:
                logger.exception("The database backup failed")
            if stop.wait(every_hours * 3600):
                return

    thread = threading.Thread(target=loop, name="database-backups", daemon=True)
    thread.stop = stop  # type: ignore[attr-defined]
    thread.start()
    return thread
