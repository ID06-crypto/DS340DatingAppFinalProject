"""Build the dataset of users with written personas, in data/main.db and data/personas.db.

Samples users from the original profiles, then has Claude write each one a persona from
five random traits and their profile. Users who already have a persona are skipped, so
running this again finishes an interrupted run without changing any existing persona.

Traits and the sample of users repeat for the same SEED, but Claude's wording does not, so
deleting a persona and generating it again gives different text.
"""

import argparse
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from manager import ProfileManager  # noqa: E402
from models import Profile  # noqa: E402
from paths import DATA_DIR, MAIN_DB, PERSONA_DB, SOURCE_MAIN_DB, SOURCE_PERSONA_DB  # noqa: E402

logger = logging.getLogger(__name__)

SEED = 42
N_USERS = 2000
N_WORKERS = 8
CLAUDE_MODEL = "claude-haiku-5-5"
INPUT_PRICE_PER_MTOK = 0.10
OUTPUT_PRICE_PER_MTOK = 0.50
PERSONA_TRAITS = ("extraversion", "openness", "conscientiousness", "traditionalism", "achievement_drive")

SYSTEM_PROMPT = (
    "You write short fictional personas for a research simulation of a dating app. "
    "Everyone in it is a simulated adult college student."
)
PROMPT = """Here is a simulated student's dating profile:
{profile}

Here are their trait scores, each from 0 to 1, where 0.5 is about average:
{traits}

Write 2 to 3 sentences in the third person, using "they". First say what they're like, in a way that fits both the traits and the profile. Then say specifically what they want and don't want in a partner, using only things a dating profile shows: age, height, religion, major, job, interests, clubs, and relationship goals. Make the preferences clear enough that they'd like some profiles and pass on others.

Don't mention the trait names or scores, their name, or gender, and don't list their profile back. Keep it non-sexual. Reply with only the persona."""

# Worth logging and leaving for the next run. Anything else, like a bad key, stops the run.
TEMPORARY_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


class PersonaWritingError(Exception):
    """Claude didn't finish writing a persona, for example because it declined."""


def sample_traits(user_id: str, seed: int = SEED) -> dict[str, float]:
    """Draw each trait evenly from 0 to 1. The result depends only on the seed and user_id."""
    rng = random.Random(f"{seed}:{user_id}")
    return {trait: round(rng.uniform(0, 1), 2) for trait in PERSONA_TRAITS}


def build_prompt(profile: Profile, traits: dict[str, float]) -> str:
    """Fill in the prompt. The name and IDs don't help, and agent must stay out of personas."""
    shown = profile.model_dump(exclude={"user_id", "name", "agent"})
    return PROMPT.format(profile=json.dumps(shown), traits=json.dumps(traits))


def write_persona(
    client: anthropic.Anthropic, profile: Profile, traits: dict[str, float]
) -> tuple[str, int, int]:
    """Return (persona, input_tokens, output_tokens)."""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(profile, traits)}],
    )
    if response.stop_reason != "end_turn":
        raise PersonaWritingError(f"Claude stopped early: {response.stop_reason}")
    text = next((block.text for block in response.content if block.type == "text"), "").strip()
    if not text:
        raise PersonaWritingError("Claude returned no text")
    return text, response.usage.input_tokens, response.usage.output_tokens


def copy_sampled_users(
    source: ProfileManager, manager: ProfileManager, n_users: int = N_USERS, seed: int = SEED
) -> int:
    """Copy a random sample of profiles and preferences, keeping their IDs.

    Does nothing if users already exist.
    """
    if manager.count_profiles() > 0:
        return 0
    profiles = source.list_profiles()
    if len(profiles) < n_users:
        raise ValueError(f"need {n_users} source profiles, found {len(profiles)}")
    for profile in random.Random(seed).sample(profiles, n_users):
        manager.add_profile(profile.model_dump())
        preferences = source.get_preferences(profile.user_id)
        if preferences is not None:
            manager.set_preferences(profile.user_id, preferences.interested_in)
    return n_users


def generate_personas(
    manager: ProfileManager, client: anthropic.Anthropic, limit: int | None = None
) -> dict[str, int]:
    """Write and save a persona for each user without one, up to limit users."""
    existing = {persona.user_id for persona in manager.personas.list_personas()}
    missing = [p for p in manager.list_profiles() if p.user_id not in existing]
    if limit is not None:
        missing = missing[:limit]

    def write_and_save(profile: Profile) -> tuple[int, int] | None:
        traits = sample_traits(profile.user_id)
        try:
            text, input_tokens, output_tokens = write_persona(client, profile, traits)
        except (PersonaWritingError, *TEMPORARY_ERRORS):
            logger.exception("Skipping user %s", profile.user_id)
            return None
        generation_seeds: dict[str, Any] = {"seed": SEED, "model": CLAUDE_MODEL, "traits": traits}
        manager.add_persona(profile.user_id, text, generation_seeds)
        return input_tokens, output_tokens

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(write_and_save, missing))

    written = [result for result in results if result is not None]
    return {
        "written": len(written),
        "failed": len(results) - len(written),
        "input_tokens": sum(result[0] for result in written),
        "output_tokens": sum(result[1] for result in written),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="write at most this many personas")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    DATA_DIR.mkdir(exist_ok=True)
    manager = ProfileManager(MAIN_DB, PERSONA_DB)
    source = ProfileManager(SOURCE_MAIN_DB, SOURCE_PERSONA_DB)
    print("Users copied:", copy_sampled_users(source, manager))

    stats = generate_personas(manager, anthropic.Anthropic(max_retries=5), args.limit)
    cost = (
        stats["input_tokens"] * INPUT_PRICE_PER_MTOK + stats["output_tokens"] * OUTPUT_PRICE_PER_MTOK
    ) / 1e6
    print(stats, f"cost ${cost:.3f}")
    print(f"Users: {manager.count_profiles()}, personas: {len(manager.personas.list_personas())}")
    if stats["failed"]:
        print("Some personas failed. Run this again to finish them.")


if __name__ == "__main__":
    main()
