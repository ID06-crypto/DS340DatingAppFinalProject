"""Run the random matcher with the fake rater on a few fake users, in a temporary database."""

import logging
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manager import ProfileManager  # noqa: E402
from matching import RandomMatcher  # noqa: E402
from raters import FakeRater  # noqa: E402
from simulation import run_simulation  # noqa: E402

SEED = 42
N_STEPS = 15

USERS = [
    ("Ana", 29, 168, "none", "BU", "nurse", ["hiking", "chess"], "long-term", "Values kindness."),
    ("Ben", 31, 180, "catholic", "MIT", "teacher", ["chess"], "just a date", "Wants to laugh a lot."),
    ("Cam", 26, 175, "jewish", "NYU", "designer", ["art"], "short-term", "Loves ambition."),
    ("Dee", 34, 160, "muslim", "UCLA", "lawyer", ["running"], "long-term", "Wants stability."),
    ("Eli", 28, 185, "hindu", "Yale", "chef", ["cooking"], "short-term", "Seeks adventure."),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp:
        manager = ProfileManager(
            Path(tmp) / "main.db", Path(tmp) / "personas.db", rng=random.Random(SEED)
        )
        for name, age, height_cm, religion, school, job, interests, looking_for, persona in USERS:
            profile = manager.add_profile({
                "name": name,
                "age": age,
                "height_cm": height_cm,
                "religion": religion,
                "school": school,
                "job": job,
                "interests": interests,
                "looking_for": looking_for,
                "agent": True,
            })
            manager.add_persona(profile.user_id, persona)

        summary = run_simulation(
            manager,
            RandomMatcher(random.Random(SEED)),
            FakeRater(random.Random(SEED)),
            "baseline",
            N_STEPS,
            SEED,
        )
        print(summary)
        print("Ratings stored:", manager.count_ratings("baseline"))


if __name__ == "__main__":
    main()
