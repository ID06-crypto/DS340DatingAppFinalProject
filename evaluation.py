"""Measures how well a model predicts whether people like each other.

Truth comes from saved Jev ratings: a person "likes" a target if the score reaches
LIKE_THRESHOLD, and a pair "mutually likes" if both directions do. Everything here reads
public data only, and profile tables leave out the agent flag and never touch personas.

Splits are by user rather than by rating. A test pair's users are never seen in training,
which is the situation with new people, and it stops a model memorizing a user's tastes.
"""

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)

from matching import PublicView
from store import Store

LIKE_THRESHOLD = 0.5
SPLIT_SEED = 2026
TRAIN_FRACTION = 0.64
VALIDATION_FRACTION = 0.16  # The remaining 20% is the test set.
N_BOOTSTRAP = 500
SMALL_GROUP = 30


def assign_user_splits(user_ids: Iterable[str], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Assign each user to train, validation, or test. Input order doesn't matter."""
    users = sorted(set(user_ids))
    random.Random(seed).shuffle(users)
    n_train = int(len(users) * TRAIN_FRACTION)
    n_validation = int(len(users) * VALIDATION_FRACTION)
    splits = {user: "train" for user in users[:n_train]}
    splits.update({user: "validation" for user in users[n_train : n_train + n_validation]})
    splits.update({user: "test" for user in users[n_train + n_validation :]})
    return splits


def load_or_create_splits(path: str | Path, user_ids: Iterable[str]) -> dict[str, str]:
    """Reuse a saved split so the test users stay the same, creating it the first time."""
    path = Path(path)
    users = set(user_ids)
    if path.exists():
        splits = json.loads(path.read_text())
        if set(splits) != users:
            raise ValueError(f"{path} doesn't cover the same users as this dataset")
        return splits
    splits = assign_user_splits(users)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=0, sort_keys=True))
    return splits


def load_ratings(store: Store, run_ids: Iterable[str]) -> pd.DataFrame:
    """Load ratings from the given runs. A pair rated in two runs keeps its first rating."""
    rows = [
        {"rater_id": r.rater_id, "target_id": r.target_id, "score": r.score, "run_id": r.run_id}
        for run_id in run_ids
        for r in store.ratings_in_run(run_id)
    ]
    return pd.DataFrame(rows).drop_duplicates(["rater_id", "target_id"]).reset_index(drop=True)


def public_profiles(store: Store) -> dict[str, dict[str, Any]]:
    """Profiles as a matcher sees them, so they can't include the agent flag."""
    # PublicView needs a run_id only for ratings, which aren't read here.
    profiles = PublicView(store, "").list_profiles()
    return {p.user_id: p.model_dump(exclude={"user_id", "name"}) for p in profiles}


def directional_examples(
    ratings: pd.DataFrame,
    splits: dict[str, str],
    split: str,
    profiles: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """One row per rating whose rater and target are both in this split, with public fields."""
    in_split = ratings["rater_id"].map(splits).eq(split) & ratings["target_id"].map(splits).eq(split)
    examples = ratings[in_split].drop(columns="run_id").reset_index(drop=True)
    for prefix, column in (("rater", "rater_id"), ("target", "target_id")):
        fields = pd.DataFrame([profiles[user] for user in examples[column]])
        examples = pd.concat([examples, fields.add_prefix(f"{prefix}_")], axis=1)
    examples["like"] = (examples["score"] >= LIKE_THRESHOLD).astype(int)
    return examples


def mutual_examples(examples: pd.DataFrame, probability: np.ndarray | None = None) -> pd.DataFrame:
    """One row per pair who rated each other. With predictions, adds both directions' values."""
    frame = examples[["rater_id", "target_id", "score"]].copy()
    frame["probability"] = probability if probability is not None else np.nan
    reverse = frame.rename(
        columns={
            "rater_id": "target_id",
            "target_id": "rater_id",
            "score": "score_reverse",
            "probability": "probability_reverse",
        }
    )
    pairs = frame.merge(reverse, on=["rater_id", "target_id"])
    pairs = pairs[pairs["rater_id"] < pairs["target_id"]].reset_index(drop=True)
    pairs["mutual"] = (
        (pairs["score"] >= LIKE_THRESHOLD) & (pairs["score_reverse"] >= LIKE_THRESHOLD)
    ).astype(int)
    pairs["match_probability"] = pairs["probability"] * pairs["probability_reverse"]
    return pairs.rename(columns={"rater_id": "user_a", "target_id": "user_b"})


def classification_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Every pair-level metric, at one decision threshold.

    Type 1 rate is false positives among real negatives (saying they like each other when
    they don't). Type 2 rate is false negatives among real positives (missing a real like).
    Bias is the mean predicted probability minus the real rate, so above 0 means overpredicting.
    AUC metrics are NaN if the data has only one class.
    """
    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability, dtype=float)
    y_predicted = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_predicted, labels=[0, 1], sample_weight=sample_weight
    ).ravel()

    def ratio(top: float, bottom: float) -> float:
        return float(top / bottom) if bottom else float("nan")

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    two_classes = len(np.unique(y_true[sample_weight > 0] if sample_weight is not None else y_true)) == 2
    return {
        "n": float(tp + fp + tn + fn),
        "positive_rate": ratio(tp + fn, tp + fp + tn + fn),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": ratio(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": (recall + ratio(tn, tn + fp)) / 2,
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "type_1_rate": ratio(fp, fp + tn),
        "type_2_rate": ratio(fn, fn + tp),
        "mcc": float(matthews_corrcoef(y_true, y_predicted, sample_weight=sample_weight)),
        "roc_auc": float(roc_auc_score(y_true, y_probability, sample_weight=sample_weight)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_probability, sample_weight=sample_weight)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, y_probability, sample_weight=sample_weight)),
        "log_loss": float(log_loss(y_true, np.clip(y_probability, 1e-6, 1 - 1e-6), labels=[0, 1], sample_weight=sample_weight)),
        "mean_probability": float(np.average(y_probability, weights=sample_weight)),
        "bias": float(np.average(y_probability, weights=sample_weight) - np.average(y_true, weights=sample_weight)),
    }


def best_f1_threshold(y_true: np.ndarray, y_probability: np.ndarray) -> float:
    """The threshold with the highest F1. Choose it on validation data, never on test data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probability)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[int(np.argmax(f1))])


def bootstrap_intervals(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    users_a: np.ndarray,
    users_b: np.ndarray | None = None,
    baseline_probability: np.ndarray | None = None,
    baseline_threshold: float | None = None,
    n_samples: int = N_BOOTSTRAP,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """95% intervals from resampling users, since one user's ratings aren't independent.

    A row's weight is how many times its user was drawn. Pass users_b for pairs, which then
    count as both users were drawn. With baseline_probability, also gives intervals for
    the metric minus the baseline's on the same resamples, at the baseline's own threshold
    if one is given.
    """
    users = pd.unique(np.concatenate([users_a, users_b]) if users_b is not None else users_a)
    index = pd.Series(np.arange(len(users)), index=users)
    a = index.loc[users_a].to_numpy()
    b = index.loc[users_b].to_numpy() if users_b is not None else None
    rng = np.random.default_rng(seed)
    samples, differences = [], []
    for _ in range(n_samples):
        counts = np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users))
        weights = counts[a] * (counts[b] if b is not None else 1)
        if weights.sum() == 0:
            continue
        kept = weights > 0
        result = classification_metrics(y_true[kept], y_probability[kept], threshold, weights[kept])
        samples.append(result)
        if baseline_probability is not None:
            base_threshold = threshold if baseline_threshold is None else baseline_threshold
            base = classification_metrics(y_true[kept], baseline_probability[kept], base_threshold, weights[kept])
            differences.append({key: result[key] - base[key] for key in result})

    def interval(rows: list[dict[str, float]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        return pd.DataFrame({"low": frame.quantile(0.025), "high": frame.quantile(0.975)})

    intervals = interval(samples)
    if differences:
        difference = interval(differences)
        intervals["difference_low"] = difference["low"]
        intervals["difference_high"] = difference["high"]
    return intervals


def calibration_table(y_true: np.ndarray, y_probability: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Compare predicted and real rates within equal-sized groups of predictions."""
    frame = pd.DataFrame({"y": y_true, "p": y_probability})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    table = frame.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return table.reset_index(drop=True)


def expected_calibration_error(y_true: np.ndarray, y_probability: np.ndarray, n_bins: int = 10) -> float:
    """The average gap between predicted and real rates, weighted by group size."""
    table = calibration_table(y_true, y_probability, n_bins)
    return float(np.average((table["predicted"] - table["observed"]).abs(), weights=table["n"]))


def subgroup_table(
    groups: pd.Series, y_true: np.ndarray, y_probability: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Metrics per group, to check that errors aren't concentrated on some people."""
    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)
    rows = []
    for name in sorted(groups.dropna().unique(), key=str):
        mask = (groups == name).to_numpy()
        metrics = classification_metrics(y_true[mask], y_probability[mask], threshold)
        rows.append({
            "group": name,
            "n": int(metrics["n"]),
            "positive_rate": metrics["positive_rate"],
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "type_1_rate": metrics["type_1_rate"],
            "type_2_rate": metrics["type_2_rate"],
            "bias": metrics["bias"],
            "small_sample": metrics["n"] < SMALL_GROUP,
        })
    return pd.DataFrame(rows).set_index("group")


def ranking_metrics(raters: pd.Series, y_true: np.ndarray, y_probability: np.ndarray, k: int = 3) -> dict[str, float]:
    """How well each person's own candidates are ordered, averaged over people.

    Overall AUC can look good just by learning that some people like everyone. Within-rater
    AUC only compares candidates for the same person, which is what a matcher has to do.
    """
    frame = pd.DataFrame({"rater": np.asarray(raters), "y": y_true, "p": y_probability})
    aucs, precisions, lifts = [], [], []
    for _, group in frame.groupby("rater"):
        if group["y"].nunique() == 2:
            aucs.append(roc_auc_score(group["y"], group["p"]))
        if len(group) > k and group["y"].sum() > 0:
            top = group.sort_values("p", ascending=False, kind="stable").head(k)["y"].mean()
            precisions.append(top)
            lifts.append(top - group["y"].mean())
    return {
        "within_rater_auc": float(np.mean(aucs)) if aucs else float("nan"),
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else float("nan"),
        f"lift_at_{k}": float(np.mean(lifts)) if lifts else float("nan"),
        "raters_counted": float(len(precisions)),
    }


def exposure_gini(counts: Iterable[int]) -> float:
    """How unevenly users are shown: 0 means everyone equally, near 1 means a few get it all."""
    values = np.sort(np.asarray(list(counts), dtype=float))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    ranks = np.arange(1, len(values) + 1)
    return float((2 * ranks - len(values) - 1).dot(values) / (len(values) * values.sum()))
