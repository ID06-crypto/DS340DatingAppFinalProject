"""Measures how a run went, using only public data from main.db."""

import statistics

from pydantic import BaseModel, ConfigDict

from matching import is_compatible
from store import Store

DEFAULT_WINDOWS = 10


class RunMetrics(BaseModel):
    """Results for one run. Scores are probabilities that the rater likes the target.

    A pair's chance of matching is the product of their two scores, which assumes each
    person's like is independent of the other's. expected_matches adds that up over the
    pairs who rated each other, and match_rate is its average per pair.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    n_ratings: int
    mean_score: float | None
    compatible_share: float | None
    pairs_rated_both_ways: int
    expected_matches: float
    match_rate: float | None
    learning_curve: list[float]


def learning_curve(scores: list[float], n_windows: int = DEFAULT_WINDOWS) -> list[float]:
    """Return the mean score in each of up to n_windows equal parts of the run, in order."""
    n_windows = min(n_windows, len(scores))
    if n_windows == 0:
        return []
    bounds = [len(scores) * i // n_windows for i in range(n_windows + 1)]
    return [statistics.fmean(scores[a:b]) for a, b in zip(bounds, bounds[1:])]


def compute_metrics(store: Store, run_id: str, n_windows: int = DEFAULT_WINDOWS) -> RunMetrics:
    ratings = store.ratings_in_run(run_id)
    scores = [rating.score for rating in ratings]
    profiles = {profile.user_id: profile for profile in store.list_all_profiles()}

    n_compatible = sum(
        is_compatible(profiles[rating.rater_id], profiles[rating.target_id])
        for rating in ratings
    )
    score_of = {(rating.rater_id, rating.target_id): rating.score for rating in ratings}
    # Comparing IDs counts each pair once, not once per direction.
    match_chances = [
        score * score_of[(b, a)] for (a, b), score in score_of.items() if a < b and (b, a) in score_of
    ]

    return RunMetrics(
        run_id=run_id,
        n_ratings=len(ratings),
        mean_score=statistics.fmean(scores) if scores else None,
        compatible_share=n_compatible / len(ratings) if ratings else None,
        pairs_rated_both_ways=len(match_chances),
        expected_matches=sum(match_chances),
        match_rate=statistics.fmean(match_chances) if match_chances else None,
        learning_curve=learning_curve(scores, n_windows),
    )
