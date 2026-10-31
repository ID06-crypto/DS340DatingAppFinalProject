"""Run a long baseline with Jev on the current dataset, with backups before and after.

    python scripts/run_experiment.py --run-id random-baseline-100k --steps 100000

Every rating is saved as it happens. If the run stops for any reason, run the same command
again to continue it. Ratings run at once in batches, which is fine for this random baseline
but not for a matcher that learns from every rating. Progress is logged every 1,000 steps
to data/<run_id>.log.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from backup_data import backup_databases, export_ratings  # noqa: E402
from manager import ProfileManager  # noqa: E402
from matching import CompatibleRandomMatcher  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from paths import DATA_DIR, MAIN_DB, PERSONA_DB  # noqa: E402
from raters import JevRater  # noqa: E402
from simulation import run_simulation  # noqa: E402

SEED = 42
DEFAULT_WORKERS = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set the TYPESAFE_API_KEY environment variable first.")

    logging.basicConfig(level=logging.WARNING)
    log_handler = logging.FileHandler(DATA_DIR / f"{args.run_id}.log")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    simulation_logger = logging.getLogger("simulation")
    simulation_logger.setLevel(logging.INFO)
    simulation_logger.addHandler(log_handler)

    print("Databases backed up to:", backup_databases(), flush=True)
    manager = ProfileManager(MAIN_DB, PERSONA_DB)
    # Includes the progress so far, so that a resumed run doesn't repeat the same choices.
    matcher_seed = f"{SEED}:{manager.count_ratings(args.run_id)}"
    summary = run_simulation(
        manager,
        CompatibleRandomMatcher(random.Random(matcher_seed)),
        JevRater(),
        args.run_id,
        args.steps,
        SEED,
        resume=True,
        n_workers=args.workers,
    )
    print(summary)
    print(compute_metrics(manager.store, args.run_id))
    print("Databases backed up to:", backup_databases())
    print("Ratings exported to:", export_ratings(manager.store, args.run_id))


if __name__ == "__main__":
    main()
