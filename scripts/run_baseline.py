"""Run the random matcher with the fake rater on six fake users, in a temporary database."""

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

EVERYONE = ["man", "woman", "nonbinary"]
USERS = [
    ("Ana", 29, 168, "none", "BU", "nursing", "nurse", ["hiking", "chess"], ["outdoors club"], "woman", ["man"], "long-term", "Values kindness."),
    ("Ben", 31, 180, "catholic", "MIT", "physics", "teacher", ["chess"], ["chess club"], "man", ["woman"], "just a date", "Wants to laugh a lot."),
    ("Cam", 26, 175, "jewish", "NYU", "fine arts", "designer", ["art"], [], "nonbinary", EVERYONE, "short-term", "Loves ambition."),
    ("Dee", 34, 160, "muslim", "UCLA", "political science", "lawyer", ["running"], ["debate", "running club"], "woman", ["man", "nonbinary"], "long-term", "Wants stability."),
    ("Eli", 28, 185, "hindu", "Yale", "history", "chef", ["cooking"], [], "man", ["woman", "nonbinary"], "short-term", "Seeks adventure."),
    ("Fay", 27, 170, "none", "BU", "biology", "researcher", ["climbing"], [], "woman", ["man"], "short-term", "Loves the outdoors."),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp:
        manager = ProfileManager(
            Path(tmp) / "main.db", Path(tmp) / "personas.db", rng=random.Random(SEED)
        )
        for name, age, height_cm, religion, school, major, job, interests, clubs, gender, interested_in, looking_for, persona in USERS:
            profile = manager.add_profile({
                "name": name,
                "age": age,
                "height_cm": height_cm,
                "religion": religion,
                "school": school,
                "major": major,
                "job": job,
                "interests": interests,
                "clubs": clubs,
                "gender": gender,
                "looking_for": looking_for,
                "agent": True,
            })
            manager.set_preferences(profile.user_id, interested_in)  # Private, unlike the profile.
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
