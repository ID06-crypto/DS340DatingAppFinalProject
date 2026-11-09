"""Move interested_in off profiles into the private preferences table, and add ratings.matcher.

    python scripts/migrate_preferences.py              # data/main.db and the root main.db
    python scripts/migrate_preferences.py PATH ...     # specific databases

Each database is backed up to backups/<timestamp>-pre-migration/ first. Old values map
"men" to man and "women" to woman. Ratings, runs, and personas are kept unchanged.
Running it again on a migrated database does nothing.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paths import MAIN_DB, SOURCE_MAIN_DB  # noqa: E402
from store import Store  # noqa: E402

BACKUP_DIR = ROOT / "backups"
OLD_TO_GENDER = {"men": "man", "women": "woman"}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def needs_migration(db_path: Path) -> bool:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return "interested_in" in _columns(conn, "profiles") or "matcher" not in _columns(conn, "ratings")


def backup(db_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / f"{db_path.parent.name}_{db_path.name}"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as source:
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
    return destination


def migrate(db_path: Path) -> dict[str, int]:
    """Migrate one database in a single transaction, and return counts to check it by."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        before = {
            "profiles": conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0],
            "ratings": conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0],
        }
        conn.execute("BEGIN IMMEDIATE")
        try:
            if "interested_in" in _columns(conn, "profiles"):
                unknown = conn.execute(
                    "SELECT COUNT(*) FROM profiles WHERE interested_in NOT IN ('men', 'women')"
                ).fetchone()[0]
                if unknown:
                    raise ValueError(f"{unknown} profiles have an unexpected interested_in value")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS preferences (
                        user_id TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
                        gender  TEXT NOT NULL CHECK (gender IN ('man', 'woman', 'nonbinary')),
                        PRIMARY KEY (user_id, gender)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO preferences (user_id, gender)
                    SELECT user_id, CASE interested_in WHEN 'men' THEN 'man' ELSE 'woman' END
                    FROM profiles
                    """
                )
                conn.execute("ALTER TABLE profiles DROP COLUMN interested_in")
            if "matcher" not in _columns(conn, "ratings"):
                conn.execute("ALTER TABLE ratings ADD COLUMN matcher TEXT")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()

    store = Store(db_path)  # Adds the app's tables, and fails if anything is still outdated.
    after = {
        "profiles": store.count_profiles(),
        "ratings": store.count_ratings(),
        "users_with_preferences": len(store.all_preferences()),
    }
    if after["profiles"] != before["profiles"] or after["ratings"] != before["ratings"]:
        raise RuntimeError(f"{db_path}: counts changed from {before} to {after}")
    return after


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [MAIN_DB, SOURCE_MAIN_DB]
    backup_dir = BACKUP_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}-pre-migration"
    for db_path in paths:
        if not db_path.exists():
            print(f"{db_path}: not found, skipped")
            continue
        if not needs_migration(db_path):
            print(f"{db_path}: already up to date")
            continue
        print(f"{db_path}: backed up to {backup(db_path, backup_dir)}")
        print(f"{db_path}: migrated, {migrate(db_path)}")


if __name__ == "__main__":
    main()
