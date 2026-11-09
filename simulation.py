"""Runs matching experiments: a user is shown a profile, rates it, and it's saved."""

import logging
import random
import statistics
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, ConfigDict

from manager import ProfileManager
from matching import Matcher, PublicView
from models import LIVE_RUN_ID, Rating
from raters import Rater, RatingResult
from store import RunNotFoundError

logger = logging.getLogger(__name__)

MIN_SCORE = 0.0
MAX_SCORE = 1.0
PROGRESS_EVERY = 1000
MAX_CONSECUTIVE_FAILURES = 50


class NoEligibleUsersError(ValueError):
    """No profile has a persona, so nobody can rate."""


class RunAbortedError(RuntimeError):
    """Too many ratings failed in a row, which suggests an outage rather than bad luck."""


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


def _rate(
    manager: ProfileManager, rater: Rater, view: PublicView, user_id: str, target_id: str
) -> RatingResult | Exception:
    """Get one rating without saving it. Returns an exception rather than raising one.

    This can run on a worker thread, where one failed rating mustn't affect the others.
    """
    try:
        # Also rules out repeats, which would silently overwrite an earlier rating.
        if target_id not in view.eligible_candidates(user_id):
            raise ValueError(f"matcher picked {target_id}, outside {user_id}'s eligible pool")
        persona = manager.get_persona(user_id)
        if persona is None:
            raise ValueError(f"user {user_id} no longer has a persona")
        return RatingResult.model_validate(
            rater.rate(manager.get_profile(user_id), persona, manager.get_profile(target_id))
        )
    except Exception as error:
        return error


def _choose_batch(
    matcher: Matcher,
    view: PublicView,
    rng: random.Random,
    pool: list[str],
    size: int,
) -> tuple[list[tuple[str, str]], int]:
    """Choose up to size (user, target) pairs. Also returns how many users ran out of profiles.

    Ratings in a batch aren't saved yet, so the matcher could choose the same pair twice.
    Ending the batch there lets the first one be saved before the matcher chooses again.
    """
    pairs: list[tuple[str, str]] = []
    users_exhausted = 0
    while len(pairs) < size and pool:
        user_id = rng.choice(pool)
        target_id = matcher.next_profile(user_id, view)
        if target_id is None:
            pool.remove(user_id)
            users_exhausted += 1
        elif (user_id, target_id) in pairs:
            break
        else:
            pairs.append((user_id, target_id))
    return pairs, users_exhausted


def _saved_ratings(
    manager: ProfileManager, run_id: str, matcher: Matcher, seed: int | None
) -> list[Rating] | None:
    """Return an unfinished run's ratings, or None if the run doesn't exist yet."""
    try:
        run = manager.get_run(run_id)
    except RunNotFoundError:
        return None
    if run.finished_at is not None:
        raise ValueError(f"run {run_id!r} has already finished")
    if run.matcher_name != matcher.name or run.seed != seed:
        raise ValueError(
            f"run {run_id!r} used matcher {run.matcher_name!r} and seed {run.seed}, "
            f"not {matcher.name!r} and {seed}"
        )
    return manager.ratings_in_run(run_id)


def run_simulation(
    manager: ProfileManager,
    matcher: Matcher,
    rater: Rater,
    run_id: str,
    n_steps: int,
    seed: int | None,
    resume: bool = False,
    n_workers: int = 1,
) -> RunSummary:
    """Run up to n_steps ratings and return a summary.

    A failed rating uses up its step, so step numbers have a gap where it happened. A
    user with nothing left to see doesn't use a step. An exception from the matcher
    itself is a bug in the algorithm, so it propagates and the run stays unfinished. So
    does RunAbortedError, raised after MAX_CONSECUTIVE_FAILURES failures in a row.

    With resume=True, an unfinished run with this run_id continues from its last saved
    rating, and n_steps is still the total for the whole run. Saved ratings are replayed
    to matcher.observe. A resumed run won't make the same choices as an uninterrupted one.

    With n_workers above 1, ratings run at once in batches, and the rater must be
    thread-safe. The matcher doesn't see a batch's results until the whole batch is done,
    which is fine for a random matcher. A matcher that learns from each rating should use
    1 or a small number. Exact seeded repeats also need 1, because a seeded rater's numbers
    could reach different ratings in a different order.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}")
    if run_id == LIVE_RUN_ID:
        raise ValueError(f"run_id {LIVE_RUN_ID!r} is reserved for real people's ratings")

    pool = _eligible_users(manager)
    saved = _saved_ratings(manager, run_id, matcher, seed) if resume else None
    if saved is None:
        saved = []
        manager.start_run(
            run_id,
            matcher.name,
            seed,
            {
                "n_steps": n_steps,
                "rater": type(rater).__name__,
                "eligible_users": len(pool),
                "n_workers": n_workers,
            },
        )

    step = max((rating.step or 0 for rating in saved), default=0)
    if saved:
        logger.info("Resuming %s after step %d", run_id, step)
    rng = random.Random(f"{seed}:{step}" if step else seed)
    view = PublicView(manager.store, run_id)
    scores = [rating.score for rating in saved]
    for rating in saved:
        matcher.observe(rating)
    users_exhausted = 0
    consecutive_failures = 0

    def record_failure(error: Exception, step: int, user_id: str, target_id: str) -> None:
        nonlocal consecutive_failures
        logger.error(
            "Rating failed at step %d: %s rating %s", step, user_id, target_id, exc_info=error
        )
        consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise RunAbortedError(
                f"{consecutive_failures} ratings failed in a row at step {step}. "
                "Fix the problem, then resume this run."
            )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        while step < n_steps and pool:
            size = min(n_workers, n_steps - step)
            pairs, n_exhausted = _choose_batch(matcher, view, rng, pool, size)
            users_exhausted += n_exhausted

            def rate_pair(pair: tuple[str, str]) -> RatingResult | Exception:
                return _rate(manager, rater, view, *pair)

            # One worker stays on this thread, so simple raters don't have to be thread-safe.
            outcomes = list(map(rate_pair, pairs) if n_workers == 1 else executor.map(rate_pair, pairs))

            # Saved in order, here, so steps and matcher.observe stay in step order.
            for (user_id, target_id), outcome in zip(pairs, outcomes):
                step += 1
                if isinstance(outcome, Exception):
                    record_failure(outcome, step, user_id, target_id)
                    continue
                try:
                    rating = manager.add_rating(
                        user_id,
                        target_id,
                        _clamp_score(outcome.score),
                        run_id,
                        model_version=outcome.model_version,
                        raw_output=outcome.raw_output,
                        input_snapshot=outcome.input_snapshot,
                        step=step,
                        matcher=matcher.name,
                    )
                except Exception as error:
                    record_failure(error, step, user_id, target_id)
                    continue

                consecutive_failures = 0
                scores.append(rating.score)
                matcher.observe(rating)
                if step % PROGRESS_EVERY == 0:
                    logger.info("Step %d of %d, %d ratings saved", step, n_steps, len(scores))

    manager.finish_run(run_id)
    return RunSummary(
        run_id=run_id,
        steps_completed=len(scores),
        failures=step - len(scores),
        mean_score=statistics.fmean(scores) if scores else None,
        users_exhausted=users_exhausted,
    )
