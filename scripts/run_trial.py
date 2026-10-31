"""Try a rater on a small sample and report its speed, scores, and failures.

    python scripts/run_trial.py --rater fake
    python scripts/run_trial.py --rater jev --run-id jev-trial-1

The fake rater runs on a temporary copy of the databases, so nothing is saved. The Jev
rater saves to data/main.db and data/personas.db, because its ratings cost money to produce.
"""

import argparse
import json
import logging
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from manager import ProfileManager  # noqa: E402
from matching import CompatibleRandomMatcher  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from paths import MAIN_DB, PERSONA_DB  # noqa: E402
from raters import JEV_PRICE_PER_MTOK, FakeRater, JevRater, Rater  # noqa: E402
from simulation import run_simulation  # noqa: E402

SEED = 42
N_EXAMPLES = 3


def copy_database(source: Path, destination: Path) -> None:
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as source_conn:
        with sqlite3.connect(destination) as destination_conn:
            source_conn.backup(destination_conn)


def run_trial(main_db: Path, persona_db: Path, rater: Rater, run_id: str, n_steps: int) -> None:
    manager = ProfileManager(main_db, persona_db)
    start = time.perf_counter()
    summary = run_simulation(
        manager, CompatibleRandomMatcher(random.Random(SEED)), rater, run_id, n_steps, SEED
    )
    seconds_per_step = (time.perf_counter() - start) / n_steps

    print(summary)
    print(f"Seconds per step: {seconds_per_step:.2f}")
    print(compute_metrics(manager.store, run_id))

    scores = [rating.score for rating in manager.ratings_in_run(run_id)]
    if len(scores) >= 2:
        print(f"Scores: min {min(scores):.2f}, max {max(scores):.2f}, SD {statistics.stdev(scores):.2f}")

    transcripts = manager.transcripts_in_run(run_id)
    tokens = [json.loads(t.raw_output).get("input_tokens", 0) for t in transcripts]
    if sum(tokens):
        mean_tokens = statistics.fmean(tokens)
        cost_per_100k = mean_tokens * 100_000 / 1_000_000 * JEV_PRICE_PER_MTOK
        print(f"Input tokens per rating: {mean_tokens:.0f}, so 100,000 ratings cost about ${cost_per_100k:.2f}")
    for transcript in transcripts[:N_EXAMPLES]:
        print("Rater replied:", transcript.raw_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater", choices=["fake", "jev"], default="fake")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--run-id", default="trial", help="must be new for each saved run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.rater == "jev":
        if not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("Set the TYPESAFE_API_KEY environment variable first.")
        run_trial(MAIN_DB, PERSONA_DB, JevRater(), args.run_id, args.steps)
        return

    with tempfile.TemporaryDirectory() as tmp:
        main_db, persona_db = Path(tmp) / "main.db", Path(tmp) / "personas.db"
        copy_database(MAIN_DB, main_db)
        copy_database(PERSONA_DB, persona_db)
        rater = FakeRater(random.Random(SEED))
        run_trial(main_db, persona_db, rater, args.run_id, args.steps)


if __name__ == "__main__":
    main()
