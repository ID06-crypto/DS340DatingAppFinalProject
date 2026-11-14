"""Save a snapshot of the live study's ground truth: who rated whom, and how much.

    python scripts/export_truth.py              # Into data/live/exports/truth-<UTC time>/
    python scripts/export_truth.py some/folder  # Into a new folder of your choice

It uses the same settings as the server (APP_MAIN_DB, EXPORT_SALT, POSITIVE_THRESHOLD),
and needs EXPORT_SALT, so that participant_ids match the admin downloads. It writes
participants.csv, scores.csv, matrix.csv, and a README.txt describing every column. It
only reads the database, and never overwrites an existing folder.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import insights  # noqa: E402
from app.analytics import to_unit  # noqa: E402
from app.config import Settings  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    if not settings.export_salt:
        sys.exit("Set EXPORT_SALT first, the same value the server uses, so ids match the admin downloads.")
    if not settings.main_db.exists():
        sys.exit(f"No database at {settings.main_db}. Set APP_MAIN_DB to the live database.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.main_db.parent / "exports" / f"truth-{stamp}"
    data = insights.load(settings.main_db, to_unit(settings.positive_threshold), settings.export_salt)
    for path in insights.write_truth(folder, data):
        print(path)
    print(f"{len(data.people)} participants, {len(data.ratings)} ratings.")


if __name__ == "__main__":
    main()
