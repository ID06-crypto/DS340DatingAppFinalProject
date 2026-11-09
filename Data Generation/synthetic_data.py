"""Generate random profiles for simulated users."""

import csv
import random
from pathlib import Path
from typing import Any

from bu_clubs import BU_CLUBS, MEN_ONLY_CLUBS, WOMEN_ONLY_CLUBS
from profile_lists import MAJORS, MEN_NAMES, WOMEN_NAMES

HOBBIES_PATH = Path(__file__).resolve().parent / "hobbylist.csv"

SCHOOL = "BU"
GAY_RATE = 0.03  # The share of users interested in their own gender.
NAMES_BY_GENDER = {"man": MEN_NAMES, "woman": WOMEN_NAMES}
SAME_GENDER = {"man": "man", "woman": "woman"}
OPPOSITE_GENDER = {"man": "woman", "woman": "man"}
CLUBS_OPEN_TO = {
    "man": [club for club in BU_CLUBS if club not in WOMEN_ONLY_CLUBS],
    "woman": [club for club in BU_CLUBS if club not in MEN_ONLY_CLUBS],
}
LOOKING_FOR = ("short-term", "long-term", "just a date")
JOBS = (
    "tutor",
    "intern",
    "research assistant",
    "server",
    "teaching assistant",
    "bartender",
    "front desk",
    "on-campus work",
)
RELIGIONS = (
    "catholic",
    "christian",
    "hindu",
    "buddhist",
    "muslim",
    "jewish",
    "mormon",
    "atheist",
    "agnostic",
    "spiritual",
)
MIN_AGE = 18
MAX_AGE = 24
MIN_HEIGHT_CM = 150
MAX_HEIGHT_CM = 200
MAX_INTERESTS = 3
MAX_CLUBS = 3


def load_hobbies(path: str | Path = HOBBIES_PATH) -> list[str]:
    """Return the hobby names from the CSV, lowercased and deduplicated.

    Profile normalizes interests the same way, so a duplicate here could otherwise
    collapse and leave a profile with fewer interests than were picked.
    """
    with open(path, encoding="utf-8", newline="") as file:
        return list(dict.fromkeys(row["Hobby-name"].strip().lower() for row in csv.DictReader(file)))


HOBBIES = load_hobbies()


def generate_profile(
    *,
    gay_rate: float = GAY_RATE,
    rng: random.Random | None = None,
    blank_religion_rng: random.Random | None = None,
    blank_religion_rate: float = 0.0,
) -> dict[str, Any]:
    """Return one random agent profile, plus "interested_in", which is private.

    Pop "interested_in" and pass it to ProfileManager.set_preferences, and pass the rest
    to ProfileManager.add_profile, which generates the user_id. A seeded rng makes the
    profile reproducible, and the draws are in the same order as before preferences
    moved off profiles. Names and clubs are limited to those fitting the user's gender.

    Religion is optional. To leave it blank for some users, pass blank_religion_rate and
    a separate blank_religion_rng. Using a separate generator keeps every other field the
    same as before, so existing datasets still reproduce exactly.
    """
    rng = rng if rng is not None else random.Random()
    gender = rng.choice(tuple(SAME_GENDER))
    profile = {
        "name": rng.choice(NAMES_BY_GENDER[gender]),
        "age": rng.randint(MIN_AGE, MAX_AGE),
        "height_cm": rng.randint(MIN_HEIGHT_CM, MAX_HEIGHT_CM),
        "religion": rng.choice(RELIGIONS),
        "school": SCHOOL,
        "major": rng.choice(MAJORS),
        "job": rng.choice(JOBS),
        "interests": rng.sample(HOBBIES, rng.randint(1, MAX_INTERESTS)),
        "clubs": rng.sample(CLUBS_OPEN_TO[gender], rng.randint(0, MAX_CLUBS)),
        "gender": gender,
        "interested_in": [SAME_GENDER[gender] if rng.random() < gay_rate else OPPOSITE_GENDER[gender]],
        "looking_for": rng.choice(LOOKING_FOR),
        "agent": True,
    }
    if blank_religion_rate and (blank_religion_rng or random.Random()).random() < blank_religion_rate:
        profile["religion"] = None
    return profile
