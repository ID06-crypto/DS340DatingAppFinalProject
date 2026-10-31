"""Runs matching experiments: a user is shown a profile, rates it, and it's saved."""

import logging
import random
import statistics

from pydantic import BaseModel, ConfigDict

from manager import ProfileManager
from matching import Matcher, PublicView
from models import Rating
from raters import Rater, RatingResult

logger = logging.getLogger(__name__)

MIN_SCORE = 1.0
MAX_SCORE = 10.0


class NoEligibleUsersError(ValueError):
    """No profile has a persona, so nobody can rate."""


class RunSummary(BaseModel):
    """The result of a run. users_exhausted only counts users found to be out of profiles."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    steps_completed: int
    failures: int
    mean_score: float | None
    users_exhausted: int


def _eligible_users(manager: ProfileManager) -> list[str]:
    """Return the IDs of profiles with a persona, sorted so that seeded runs repeat."""
    eligible = []
    for profile in manager.list_profiles():
        if manager.get_persona(profile.user_id) is None:
            logger.warning("Skipping user %s: no persona", profile.user_id)
        else:
            eligible.append(profile.user_id)
    if not eligible:
        raise NoEligibleUsersError("no profiles have a persona, so nobody can rate")
    return sorted(eligible)


def _clamp_score(score: float) -> float:
    clamped = min(MAX_SCORE, max(MIN_SCORE, score))
    if clamped != score:
        logger.warning("Clamped score %s to %s", score, clamped)
    return clamped


def _rate_and_save(
    manager: ProfileManager,
    rater: Rater,
    view: PublicView,
    user_id: str,
    target_id: str,
    step: int,
) -> Rating:
    # Saving a repeat would silently overwrite the user's earlier rating in this run.
    if target_id in view.already_rated(user_id):
        raise ValueError(f"user {user_id} has already rated {target_id} in this run")
    persona = manager.get_persona(user_id)
    if persona is None:
        raise ValueError(f"user {user_id} no longer has a persona")

    result = RatingResult.model_validate(
        rater.rate(manager.get_profile(user_id), persona, manager.get_profile(target_id))
    )
    return manager.add_rating(
        user_id,
        target_id,
        _clamp_score(result.score),
        view.run_id,
        model_version=result.model_version,
        raw_output=result.raw_output,
        input_snapshot=result.input_snapshot,
        step=step,
    )


def run_simulation(
    manager: ProfileManager,
    matcher: Matcher,
    rater: Rater,
    run_id: str,
    n_steps: int,
    seed: int | None,
) -> RunSummary:
    """Run up to n_steps ratings and return a summary.

    A failed rating uses up its step, so step numbers have a gap where it happened. A
    user with nothing left to see doesn't use a step. An exception from the matcher
    itself is a bug in the algorithm, so it propagates and the run stays unfinished.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")

    pool = _eligible_users(manager)
    manager.start_run(
        run_id,
        matcher.name,
        seed,
        {
            "n_steps": n_steps,
            "rater": type(rater).__name__,
            "eligible_users": len(pool),
        },
    )

    rng = random.Random(seed)
    view = PublicView(manager.store, run_id)
    scores: list[float] = []
    failures = 0
    users_exhausted = 0
    step = 0

    while step < n_steps and pool:
        user_id = rng.choice(pool)
        target_id = matcher.next_profile(user_id, view)
        if target_id is None:
            pool.remove(user_id)
            users_exhausted += 1
            continue

        step += 1
        try:
            rating = _rate_and_save(manager, rater, view, user_id, target_id, step)
        except Exception:
            logger.exception(
                "Rating failed at step %d: %s rating %s", step, user_id, target_id
            )
            failures += 1
            continue

        scores.append(rating.score)
        matcher.observe(rating)

    manager.finish_run(run_id)
    return RunSummary(
        run_id=run_id,
        steps_completed=len(scores),
        failures=failures,
        mean_score=statistics.fmean(scores) if scores else None,
        users_exhausted=users_exhausted,
    )
