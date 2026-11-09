"""Measures how a simulation run went, using only public data from main.db."""

import statistics

from pydantic import BaseModel, ConfigDict

from metrics import DEFAULT_WINDOWS, learning_curve
from models import mutually_interested
from store import Store



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


def compute_metrics(store: Store, run_id: str, n_windows: int = DEFAULT_WINDOWS) -> RunMetrics:
    ratings = store.ratings_in_run(run_id)
    scores = [rating.score for rating in ratings]
    gender = {profile.user_id: profile.gender for profile in store.list_all_profiles()}
    interested_in = store.all_preferences()

    def compatible(a: str, b: str) -> bool:
        if a not in interested_in or b not in interested_in:
            return False
        return mutually_interested(gender[a], interested_in[a], gender[b], interested_in[b])

    n_compatible = sum(compatible(rating.rater_id, rating.target_id) for rating in ratings)
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
