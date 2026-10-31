"""Back up the current dataset, and export a run's ratings to a compressed CSV.

    python scripts/backup_data.py
    python scripts/backup_data.py --run-id random-baseline-100k

Backups go in backups/<timestamp>/ and are safe to take during a run. They're excluded from
Git because personas are hidden data, so copy them somewhere off this computer as well.

Exports go in exports/. They hold only ratings, which are public data, so they're small
enough and safe enough to commit.
"""

import argparse
import csv
import gzip
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import Rating  # noqa: E402
from paths import MAIN_DB, PERSONA_DB  # noqa: E402
from store import Store  # noqa: E402

BACKUP_DIR = ROOT / "backups"
EXPORT_DIR = ROOT / "exports"


def backup_databases(backup_dir: Path = BACKUP_DIR) -> Path:
    destination = backup_dir / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    destination.mkdir(parents=True)
    for source in (MAIN_DB, PERSONA_DB):
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as source_conn:
            target_conn = sqlite3.connect(destination / source.name)
            try:
                source_conn.backup(target_conn)
            finally:
                target_conn.close()
    return destination


def export_ratings(store: Store, run_id: str, export_dir: Path = EXPORT_DIR) -> Path:
    """Write one run's ratings, in step order. Raises RunNotFoundError for an unknown run."""
    store.get_run(run_id)
    export_dir.mkdir(exist_ok=True)
    path = export_dir / f"{run_id}_ratings.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(Rating.model_fields))
        writer.writeheader()
        for rating in store.ratings_in_run(run_id):
            writer.writerow(rating.model_dump(mode="json"))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="also export this run's ratings")
    args = parser.parse_args()
    print("Databases backed up to:", backup_databases())
    if args.run_id:
        print("Ratings exported to:", export_ratings(Store(MAIN_DB), args.run_id))


if __name__ == "__main__":
    main()
