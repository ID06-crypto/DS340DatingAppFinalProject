import ast
from pathlib import Path

import pytest

import metrics
from metrics import (
    ConfusionMatrix,
    accuracy,
    bootstrap_ci,
    confusion_matrix,
    f1,
    mean_rating_at_position,
    mean_rating_first_k,
    mutual_match_rate,
    precision,
    precision_at_k,
    recall,
    roc_auc,
    suppressed_groups,
)

ACTUAL = [True, True, False, False, False]
PREDICTED = [True, False, True, False, False]


def test_metrics_module_has_no_web_or_database_imports() -> None:
    tree = ast.parse(Path(metrics.__file__).read_text())
    imported = {
        (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported <= {"random", "statistics", "collections", "typing"}


def test_confusion_matrix_by_hand() -> None:
    assert confusion_matrix(ACTUAL, PREDICTED) == ConfusionMatrix(tp=1, fp=1, tn=2, fn=1)
    with pytest.raises(ValueError):
        confusion_matrix([True], [True, False])


def test_accuracy_precision_recall_f1_by_hand() -> None:
    cm = confusion_matrix(ACTUAL, PREDICTED)
    assert accuracy(cm) == pytest.approx(3 / 5)
    assert precision(cm) == pytest.approx(1 / 2)
    assert recall(cm) == pytest.approx(1 / 2)
    assert f1(cm) == pytest.approx(1 / 2)


def test_undefined_metrics_are_none_not_zero() -> None:
    no_predictions = confusion_matrix([True, False], [False, False])
    assert precision(no_predictions) is None
    assert recall(no_predictions) == 0
    assert accuracy(ConfusionMatrix(0, 0, 0, 0)) is None


def test_auc_by_hand_with_ties() -> None:
    # Positives 0.9 and 0.4, negatives 0.6, 0.4, 0.1. Pairs: (0.9 beats all 3), (0.4 beats 0.1,
    # ties 0.4, loses to 0.6) = 3 + 1.5 = 4.5 of 6.
    assert roc_auc(ACTUAL, [0.9, 0.4, 0.6, 0.4, 0.1]) == pytest.approx(4.5 / 6)
    assert roc_auc([True, False], [1.0, 0.0]) == 1.0
    assert roc_auc([True, False], [0.0, 1.0]) == 0.0
    assert roc_auc([True, True], [0.2, 0.3]) is None


def test_precision_at_k_averages_users() -> None:
    per_user = [[True, False, True, True], [False, False], [True], []]
    # First 3: 2/3, first 2 of a short list: 0/2, the single item: 1/1. The empty user is left out.
    assert precision_at_k(per_user, 3) == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert precision_at_k([[]], 3) is None


def test_mean_rating_first_k_and_by_position() -> None:
    assert mean_rating_first_k([[8.0, 2.0, 9.0], [4.0]], 2) == pytest.approx((8 + 2 + 4) / 3)
    rows = mean_rating_at_position([1, 1, 2, 3, 60], [8.0, 6.0, 5.0, 9.0, 1.0], max_position=50)
    assert rows == [(1, 7.0, 2), (2, 5.0, 1), (3, 9.0, 1)]


def test_mutual_match_rate() -> None:
    assert mutual_match_rate([True, False, False, True]) == 0.5
    assert mutual_match_rate([]) is None


def test_cluster_bootstrap_resamples_whole_users() -> None:
    # Two users with very different, internally identical ratings. Resampling users means
    # every resample is all-user-A, all-user-B, or a mix, never a reshuffle of ratings.
    users = [[1.0] * 50, [9.0] * 50]
    seen = []

    def mean_of(units):
        seen.append(sorted(len(u) for u in units))
        values = [v for u in units for v in u]
        return sum(values) / len(values)

    estimate = bootstrap_ci(users, mean_of, n_resamples=200, seed=3)
    assert estimate.value == 5.0
    assert estimate.n_users == 2
    assert all(sizes == [50, 50] for sizes in seen)
    assert estimate.low == 1.0 and estimate.high == 9.0  # Only 1, 5, or 9 are possible.


def test_cluster_bootstrap_is_seeded_and_contains_the_estimate() -> None:
    users = [[float(i % 7)] * (i % 5 + 1) for i in range(40)]

    def mean_of(units):
        values = [v for u in units for v in u]
        return sum(values) / len(values)

    first = bootstrap_ci(users, mean_of, 300, seed=1)
    assert first == bootstrap_ci(users, mean_of, 300, seed=1)
    assert first.low <= first.value <= first.high
    assert bootstrap_ci([], mean_of) == metrics.Estimate(None, None, None, 0)


def test_bootstrap_skips_resamples_that_cannot_be_computed() -> None:
    estimate = bootstrap_ci([[1]], lambda units: None, 20)
    assert (estimate.value, estimate.low, estimate.high, estimate.n_users) == (None, None, None, 1)


def test_group_suppression() -> None:
    assert suppressed_groups({"a": 50, "b": 30, "c": 12}, 10) == set()
    # Two small groups: each hidden, and that's enough.
    assert suppressed_groups({"a": 50, "b": 4, "c": 3}, 10) == {"b", "c"}


def test_complementary_suppression_hides_the_next_smallest() -> None:
    # Only "c" is small, so "b", the next smallest, is hidden too, or c = total - a - b.
    assert suppressed_groups({"a": 50, "b": 20, "c": 4}, 10) == {"b", "c"}
    assert suppressed_groups({"only": 3}, 10) == {"only"}
