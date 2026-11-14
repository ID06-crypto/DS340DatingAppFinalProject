"""Pure metric functions for comparing matchers. No web or database imports.

The functions don't depend on the rating scale. The live app passes scores on the 0 to 1
scale they're stored on, with thresholds converted to match. A rating counts as positive
at a threshold or above, and a prediction counts as positive at PREDICTION_THRESHOLD or
above. Matchers should give predictions on a 0 to 1 scale, as probabilities of a
positive rating.

Functions that can't be computed, for example precision with no predicted positives,
return None rather than a misleading number.
"""

import random
import statistics
from collections.abc import Callable, Sequence
from typing import NamedTuple, TypeVar

POSITIVE_THRESHOLD = 7.0
PREDICTION_THRESHOLD = 0.5
DEFAULT_WINDOWS = 10

Unit = TypeVar("Unit")


class ConfusionMatrix(NamedTuple):
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


class Estimate(NamedTuple):
    """A statistic with its bootstrap 95% interval and how many users it rests on."""

    value: float | None
    low: float | None
    high: float | None
    n_users: int


def _ratio(top: float, bottom: float) -> float | None:
    return top / bottom if bottom else None


def confusion_matrix(actual: Sequence[bool], predicted: Sequence[bool]) -> ConfusionMatrix:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must be the same length")
    pairs = list(zip(actual, predicted))
    return ConfusionMatrix(
        tp=sum(a and p for a, p in pairs),
        fp=sum(not a and p for a, p in pairs),
        tn=sum(not a and not p for a, p in pairs),
        fn=sum(a and not p for a, p in pairs),
    )


def accuracy(cm: ConfusionMatrix) -> float | None:
    return _ratio(cm.tp + cm.tn, cm.n)


def precision(cm: ConfusionMatrix) -> float | None:
    """Of those predicted positive, the share that really were."""
    return _ratio(cm.tp, cm.tp + cm.fp)


def recall(cm: ConfusionMatrix) -> float | None:
    """Of the real positives, the share that were predicted."""
    return _ratio(cm.tp, cm.tp + cm.fn)


def f1(cm: ConfusionMatrix) -> float | None:
    return _ratio(2 * cm.tp, 2 * cm.tp + cm.fp + cm.fn)


def roc_auc(actual: Sequence[bool], scores: Sequence[float]) -> float | None:
    """The chance a random positive is scored above a random negative, ties counting half.

    Computed from ranks (the Mann-Whitney U statistic). None without both classes.
    """
    n_positive = sum(actual)
    n_negative = len(actual) - n_positive
    if n_positive == 0 or n_negative == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        average_rank = (start + end) / 2 + 1  # Ranks start at 1, and ties share their average.
        for i in order[start : end + 1]:
            ranks[i] = average_rank
        start = end + 1
    positive_rank_sum = sum(rank for rank, is_positive in zip(ranks, actual) if is_positive)
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def precision_at_k(per_user: Sequence[Sequence[bool]], k: int) -> float | None:
    """The share of positives among each user's first k items, averaged over users.

    Each inner sequence is one user's outcomes in the order they were shown. Users with
    fewer than k items count with what they have, and users with none are left out.
    """
    shares = [sum(items[:k]) / len(items[:k]) for items in per_user if items]
    return statistics.fmean(shares) if shares else None


def mean_rating_first_k(per_user: Sequence[Sequence[float]], k: int) -> float | None:
    """The mean of every user's first k ratings, pooled."""
    ratings = [rating for items in per_user for rating in items[:k]]
    return statistics.fmean(ratings) if ratings else None


def mean_rating_at_position(
    positions: Sequence[int], ratings: Sequence[float], max_position: int
) -> list[tuple[int, float, int]]:
    """For positions 1 to max_position: (position, mean rating, number of ratings).

    Positions nobody has reached are left out.
    """
    by_position: dict[int, list[float]] = {}
    for position, rating in zip(positions, ratings):
        if 1 <= position <= max_position:
            by_position.setdefault(position, []).append(rating)
    return [(p, statistics.fmean(by_position[p]), len(by_position[p])) for p in sorted(by_position)]


def mutual_match_rate(produced_match: Sequence[bool]) -> float | None:
    """The share of ratings that ended in a match."""
    return _ratio(sum(produced_match), len(produced_match))


def bootstrap_ci(
    units: Sequence[Unit],
    statistic: Callable[[Sequence[Unit]], float | None],
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Estimate:
    """A statistic and its percentile bootstrap interval, resampling whole units.

    Pass one unit per user, holding all of that user's data. One person's ratings are
    correlated, so resampling individual ratings would make intervals too narrow.
    Resamples where the statistic can't be computed are skipped.
    """
    if not units:
        return Estimate(None, None, None, 0)
    value = statistic(units)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        result = statistic([units[rng.randrange(len(units))] for _ in units])
        if result is not None:
            samples.append(result)
    if not samples:
        return Estimate(value, None, None, len(units))
    samples.sort()
    tail = (1 - confidence) / 2
    low = samples[int(tail * (len(samples) - 1))]
    high = samples[int(round((1 - tail) * (len(samples) - 1)))]
    return Estimate(value, low, high, len(units))


def suppressed_groups(group_sizes: dict[str, int], min_size: int) -> set[str]:
    """Groups to hide because they have too few users.

    Every group under min_size is hidden. If that hides exactly one group, the next
    smallest is hidden too, so its number can't be worked out from the total.
    """
    hidden = {group for group, size in group_sizes.items() if size < min_size}
    if len(hidden) == 1:
        visible = sorted((size, group) for group, size in group_sizes.items() if group not in hidden)
        if visible:
            hidden.add(visible[0][1])
    return hidden


def learning_curve(scores: list[float], n_windows: int = DEFAULT_WINDOWS) -> list[float]:
    """Return the mean score in each of up to n_windows equal parts of the run, in order."""
    n_windows = min(n_windows, len(scores))
    if n_windows == 0:
        return []
    bounds = [len(scores) * i // n_windows for i in range(n_windows + 1)]
    return [statistics.fmean(scores[a:b]) for a, b in zip(bounds, bounds[1:])]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """The Pearson correlation of two equal-length sequences, or None if either is constant
    or there are fewer than 3 pairs."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length")
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """The least-squares line y = intercept + slope * x, as (slope, intercept).

    slope = sum((x - mean x)(y - mean y)) / sum((x - mean x)^2). None if x doesn't vary or
    there are fewer than 3 pairs. Its R^2 equals pearson(xs, ys) squared.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length")
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def correlation_ratio(groups: dict[str, Sequence[float]]) -> float | None:
    """Eta squared: the share of the variance in values explained by which group they're in.

    0 means the groups have identical means, and 1 means every value equals its group's mean.
    It's the categorical counterpart of a squared correlation. None if there's no variance,
    or fewer than two non-empty groups.
    """
    filled = [list(values) for values in groups.values() if values]
    if len(filled) < 2:
        return None
    every = [v for values in filled for v in values]
    grand = statistics.fmean(every)
    total = sum((v - grand) ** 2 for v in every)
    if total == 0:
        return None
    between = sum(len(values) * (statistics.fmean(values) - grand) ** 2 for values in filled)
    return between / total


def histogram(values: Sequence[float], bins: int = 10, low: float = 0.0, high: float = 1.0) -> list[int]:
    """Counts in equal-width bins from low to high. The top edge belongs to the last bin."""
    counts = [0] * bins
    width = (high - low) / bins
    for value in values:
        if low <= value <= high:
            counts[min(int((value - low) / width), bins - 1)] += 1
    return counts
