"""SUPERSEDED by written_personas.py, because Jev barely responds to trait scores.

This is kept so the original dataset and the persona test notebooks can be reproduced.

Generate a random persona for each simulated user and add it to personas.db.

A persona's scores are stored as JSON in its description, because that's the part of a
persona that raters are shown. Read them back with json.loads(persona.description).

Each user's scores depend only on SEED and their user_id. Users who already have a
persona are skipped, so running this again finishes an interrupted run, or covers
profiles added since, without changing any existing persona.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from manager import ProfileManager  # noqa: E402

SEED = 42
SCORE_MEAN = 0.5
SCORE_SD = 0.2

TRAITS = {
    "core_personality": (
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "honesty_humility",
    ),
    "thinking_style": (
        "analytical_thinking",
        "need_for_cognition",
        "abstract_thinking",
        "attention_to_detail",
        "cognitive_flexibility",
        "tolerance_for_ambiguity",
    ),
    "decision_making": (
        "risk_tolerance",
        "impulsivity",
        "future_orientation",
    ),
    "emotional_style": (
        "rumination",
        "optimism",
        "emotional_expressiveness",
    ),
    "social_style": (
        "empathy",
        "trust_in_others",
        "conformity",
        "assertiveness",
    ),
    "motivation_and_values": (
        "achievement_drive",
        "autonomy_need",
        "traditionalism",
    ),
}


def sample_score(rng: random.Random) -> float:
    """Sample a normal distribution limited to 0-1, rounded to two decimal places.

    Out-of-range samples are redrawn, because clipping them would pile users up at
    exactly 0 and 1.
    """
    while True:
        value = rng.gauss(SCORE_MEAN, SCORE_SD)
        if 0 <= value <= 1:
            return round(value, 2)


def generate_scores(rng: random.Random) -> dict[str, dict[str, float]]:
    """Return one score per trait, grouped by category."""
    return {
        category: {trait: sample_score(rng) for trait in traits}
        for category, traits in TRAITS.items()
    }


def main() -> None:
    manager = ProfileManager(ROOT / "main.db", ROOT / "personas.db")
    existing = {persona.user_id for persona in manager.personas.list_personas()}
    n_added = 0
    for profile in manager.list_profiles():
        if not profile.agent or profile.user_id in existing:
            continue
        scores = generate_scores(random.Random(f"{SEED}:{profile.user_id}"))
        manager.add_persona(
            profile.user_id,
            json.dumps({"scores": scores}),
            {"seed": SEED, "score_mean": SCORE_MEAN, "score_sd": SCORE_SD},
        )
        n_added += 1
    print("Personas added:", n_added)
    print("Personas stored:", len(manager.personas.list_personas()))


if __name__ == "__main__":
    main()
