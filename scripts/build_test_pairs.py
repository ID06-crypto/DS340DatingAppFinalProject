"""Rate random pairs of test users in both directions, to make a proper test set.

    python scripts/build_test_pairs.py --pairs 5000

This calls Jev twice per pair and saves to data/main.db under the run "eval-test-pairs". If
it stops, the same command continues it. A random baseline rarely has two people rate each
other, so this is what makes "do they like each other?" measurable with enough examples.
"""

import argparse
import itertools
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import load_or_create_splits  # noqa: E402
from manager import ProfileManager  # noqa: E402
from models import mutually_interested  # noqa: E402
from paths import MAIN_DB, PERSONA_DB, ROOT as PROJECT_ROOT  # noqa: E402
from raters import JevRater, Rater  # noqa: E402
from store import RunNotFoundError  # noqa: E402

RUN_ID = "eval-test-pairs"
SPLITS_PATH = PROJECT_ROOT / "exports" / "user_splits.json"
SEED = 42
N_WORKERS = 8


def sample_pairs(
    manager: ProfileManager, splits: dict[str, str], n_pairs: int, seed: int = SEED
) -> list[tuple[str, str]]:
    """Choose random compatible pairs of test users. The same inputs give the same pairs."""
    profiles = sorted(
        (p for p in manager.list_profiles() if splits[p.user_id] == "test"),
        key=lambda p: p.user_id,
    )
    interested_in = manager.store.all_preferences()
    compatible = [
        (a.user_id, b.user_id)
        for a, b in itertools.combinations(profiles, 2)
        if a.user_id in interested_in
        and b.user_id in interested_in
        and mutually_interested(a.gender, interested_in[a.user_id], b.gender, interested_in[b.user_id])
    ]
    return random.Random(seed).sample(compatible, min(n_pairs, len(compatible)))


def rate_pairs(
    manager: ProfileManager, rater: Rater, run_id: str, pairs: list[tuple[str, str]]
) -> int:
    """Rate both directions of each pair, skipping saved ratings. Returns the number made."""
    try:
        manager.get_run(run_id)
    except RunNotFoundError:
        manager.start_run(run_id, "test-pairs", SEED, {"n_pairs": len(pairs), "rater": type(rater).__name__})
    done = {(r.rater_id, r.target_id) for r in manager.ratings_in_run(run_id)}
    directions = [d for a, b in pairs for d in ((a, b), (b, a)) if d not in done]

    def rate(direction: tuple[str, str]) -> tuple[str, str, object]:
        user, target = direction
        result = rater.rate(manager.get_profile(user), manager.get_persona(user), manager.get_profile(target))
        return user, target, result

    n_saved = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        for user, target, result in executor.map(rate, directions):
            manager.add_rating(
                user,
                target,
                min(1.0, max(0.0, result.score)),
                run_id,
                model_version=result.model_version,
                raw_output=result.raw_output,
                input_snapshot=result.input_snapshot,
            )
            n_saved += 1
    return n_saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=5000)
    args = parser.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set the TYPESAFE_API_KEY environment variable first.")

    manager = ProfileManager(MAIN_DB, PERSONA_DB)
    splits = load_or_create_splits(SPLITS_PATH, (p.user_id for p in manager.list_profiles()))
    pairs = sample_pairs(manager, splits, args.pairs)
    print(f"{len(pairs)} pairs, ratings made: {rate_pairs(manager, JevRater(), RUN_ID, pairs)}")
    print("Ratings in this run:", manager.count_ratings(RUN_ID))


if __name__ == "__main__":
    main()
