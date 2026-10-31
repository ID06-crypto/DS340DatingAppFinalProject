"""Generate synthetic agent profiles and add them to main.db.

The seed fixes the user_ids too, so running this again with the same seed raises
store.DuplicateProfileError rather than adding the same users twice. Change SEED to
add a different batch.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "Data Generation")]

from manager import ProfileManager  # noqa: E402
from synthetic_data import generate_profile  # noqa: E402

SEED = 42
N_PROFILES = 10_000


def main() -> None:
    rng = random.Random(SEED)
    manager = ProfileManager(ROOT / "main.db", ROOT / "personas.db", rng=rng)
    for _ in range(N_PROFILES):
        manager.add_profile(generate_profile(rng=rng))
    print("Profiles stored:", manager.count_profiles())


if __name__ == "__main__":
    main()
