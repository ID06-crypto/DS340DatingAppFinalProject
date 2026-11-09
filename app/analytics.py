"""Loads live experiment data and computes the comparison between matcher arms.

Only impressions where neither person opted out of the experiment are used. Ratings are
shown on the 1 to 10 scale people used. The calculations themselves are in metrics.py.

Each impression's matcher is chosen at random, so one person sees several arms. Survey
answers are credited to the arm that served the person's most recent impression before
they answered.
"""

import csv
import hashlib
import io
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import metrics
from db import connect

AGE_BANDS = (("18-20", 18, 20), ("21-23", 21, 23), ("24+", 24, 200))
MAX_CHART_POSITION = 50
N_RESAMPLES = 500


def age_band(age: int) -> str:
    return next(name for name, low, high in AGE_BANDS if low <= age <= high)


@dataclass
class Row:
    """One impression of a non-opted-out pair."""

    impression_id: str
    viewer: str
    target: str
    matcher: str
    position: int
    pool_size: int
    predicted: float | None
    rating: float | None  # 1 to 10, or None if not rated yet.
    ms_to_rate: int | None
    matched: bool
    viewer_age: int
    viewer_gender: str
    target_age: int
    target_gender: str


_ROWS = """
    SELECT i.impression_id, i.viewer_id, i.target_id, i.matcher, i.position, i.pool_size,
           i.predicted_score, i.shown_at, i.rated_at, r.score,
           pv.age AS viewer_age, pv.gender AS viewer_gender,
           pt.age AS target_age, pt.gender AS target_gender,
           EXISTS (
               SELECT 1 FROM matches m
               WHERE m.user_a = MIN(i.viewer_id, i.target_id) AND m.user_b = MAX(i.viewer_id, i.target_id)
           ) AS matched
    FROM impressions i
    JOIN profiles pv ON pv.user_id = i.viewer_id
    JOIN profiles pt ON pt.user_id = i.target_id
    JOIN accounts av ON av.user_id = i.viewer_id
    JOIN accounts at ON at.user_id = i.target_id
    LEFT JOIN ratings r ON r.rater_id = i.viewer_id AND r.target_id = i.target_id AND r.run_id = 'live'
    WHERE av.experiment_opt_out = 0 AND at.experiment_opt_out = 0
    ORDER BY i.viewer_id, i.position
"""


def load_rows(db_path: str | Path) -> list[Row]:
    with connect(db_path) as conn:
        records = conn.execute(_ROWS).fetchall()
    rows = []
    for r in records:
        rated = r["rated_at"] is not None and r["score"] is not None
        ms = None
        if rated:
            ms = int((datetime.fromisoformat(r["rated_at"]) - datetime.fromisoformat(r["shown_at"])).total_seconds() * 1000)
        rows.append(Row(
            impression_id=r["impression_id"],
            viewer=r["viewer_id"],
            target=r["target_id"],
            matcher=r["matcher"],
            position=r["position"] or 0,
            pool_size=r["pool_size"],
            predicted=r["predicted_score"],
            rating=1 + 9 * r["score"] if rated else None,
            ms_to_rate=ms,
            matched=bool(r["matched"]),
            viewer_age=r["viewer_age"],
            viewer_gender=r["viewer_gender"],
            target_age=r["target_age"],
            target_gender=r["target_gender"],
        ))
    return rows


def load_survey_scores_by_arm(db_path: str | Path) -> dict[str, dict[str, list[int]]]:
    """Survey answers of non-opted-out users, by arm, then by user."""
    with connect(db_path) as conn:
        records = conn.execute(
            """
            SELECT s.user_id, s.score,
                   (SELECT i.matcher FROM impressions i
                    WHERE i.viewer_id = s.user_id AND i.shown_at <= s.created_at
                    ORDER BY i.shown_at DESC LIMIT 1) AS matcher
            FROM surveys s JOIN accounts a ON a.user_id = s.user_id
            WHERE a.experiment_opt_out = 0
            """
        ).fetchall()
    result: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["matcher"] is not None:
            result[r["matcher"]][r["user_id"]].append(r["score"])
    return result


def _fmt(estimate: metrics.Estimate, n_ratings: int, digits: int = 2) -> dict[str, Any]:
    def round_or_none(x: float | None) -> float | None:
        return None if x is None else round(x, digits)

    return {
        "value": round_or_none(estimate.value),
        "low": round_or_none(estimate.low),
        "high": round_or_none(estimate.high),
        "n_users": estimate.n_users,
        "n_ratings": n_ratings,
    }


@dataclass
class ArmReport:
    matcher: str
    users: int
    impressions: int
    ratings: int
    headline: dict[str, dict[str, Any]] = field(default_factory=dict)
    classification: dict[str, Any] | None = None
    positions: list[tuple[int, float, int]] = field(default_factory=list)
    histogram: list[int] = field(default_factory=list)
    rapid_share: float | None = None
    groups: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _by_user(rows: list[Row]) -> list[list[Row]]:
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[row.viewer].append(row)
    return [sorted(items, key=lambda r: r.position) for _, items in sorted(grouped.items())]


def _pooled(units: list[list[Row]]) -> list[Row]:
    return [row for unit in units for row in unit]


def arm_report(
    matcher: str,
    rows: list[Row],
    surveys: dict[str, list[int]],
    k: int,
    positive_threshold: float,
    prediction_threshold: float,
    rapid_ms: int,
    min_group_size: int,
    seed: int = 0,
) -> ArmReport:
    rated = [r for r in rows if r.rating is not None]
    units = _by_user(rated)
    n = len(rated)
    report = ArmReport(matcher, users=len({r.viewer for r in rows}), impressions=len(rows), ratings=n)

    def boot(statistic: Any, unit_list: list[Any] | None = None, count: int = n) -> dict[str, Any]:
        return _fmt(metrics.bootstrap_ci(unit_list if unit_list is not None else units, statistic, N_RESAMPLES, seed), count)

    def mean_rating(us: list[list[Row]]) -> float | None:
        values = [r.rating for r in _pooled(us)]
        return statistics.fmean(values) if values else None

    def share_positive(us: list[list[Row]]) -> float | None:
        values = [r.rating >= positive_threshold for r in _pooled(us)]
        return sum(values) / len(values) if values else None

    def first_k(us: list[list[Row]]) -> float | None:
        return metrics.mean_rating_first_k([[r.rating for r in u if r.position <= k] for u in us], k)

    def precision_k(us: list[list[Row]]) -> float | None:
        return metrics.precision_at_k([[r.rating >= positive_threshold for r in u if r.position <= k] for u in us], k)

    def match_rate(us: list[list[Row]]) -> float | None:
        return metrics.mutual_match_rate([r.matched for r in _pooled(us)])

    def median_ms(us: list[list[Row]]) -> float | None:
        values = [r.ms_to_rate for r in _pooled(us) if r.ms_to_rate is not None]
        return statistics.median(values) if values else None

    survey_units = [scores for _, scores in sorted(surveys.items())]

    def mean_survey(us: list[list[int]]) -> float | None:
        values = [score for unit in us for score in unit]
        return statistics.fmean(values) if values else None

    first_k_count = sum(1 for r in rated if r.position <= k)
    report.headline = {
        "Mean rating": boot(mean_rating),
        "Share positive": boot(share_positive),
        f"Mean rating, first {k} impressions": boot(first_k, count=first_k_count),
        f"Precision at {k}": boot(precision_k, count=first_k_count),
        "Mutual-match rate": boot(match_rate),
        "Mean survey score": _fmt(metrics.bootstrap_ci(survey_units, mean_survey, N_RESAMPLES, seed), sum(map(len, survey_units))),
        "Median time to rate (ms)": boot(median_ms, count=sum(r.ms_to_rate is not None for r in rated)),
    }
    report.positions = metrics.mean_rating_at_position([r.position for r in rated], [r.rating for r in rated], MAX_CHART_POSITION)
    report.histogram = [sum(1 for r in rated if min(int(r.rating), 10) == bucket) for bucket in range(1, 11)]
    timed = [r for r in rated if r.ms_to_rate is not None]
    report.rapid_share = sum(r.ms_to_rate < rapid_ms for r in timed) / len(timed) if timed else None

    predicted = [r for r in rated if r.predicted is not None]
    if predicted:
        actual = [r.rating >= positive_threshold for r in predicted]
        guessed = [r.predicted >= prediction_threshold for r in predicted]
        cm = metrics.confusion_matrix(actual, guessed)
        p_units = _by_user(predicted)

        def stat(fn: Any) -> Any:
            def inner(us: list[list[Row]]) -> float | None:
                pool = _pooled(us)
                return fn([r.rating >= positive_threshold for r in pool], [r.predicted for r in pool])
            return inner

        def f1_of(a: list[bool], p: list[float]) -> float | None:
            return metrics.f1(metrics.confusion_matrix(a, [x >= prediction_threshold for x in p]))

        report.classification = {
            "matrix": cm._asdict(),
            "accuracy": metrics.accuracy(cm),
            "precision": metrics.precision(cm),
            "recall": metrics.recall(cm),
            "f1": _fmt(metrics.bootstrap_ci(p_units, stat(f1_of), N_RESAMPLES, seed), len(predicted)),
            "auc": _fmt(metrics.bootstrap_ci(p_units, stat(metrics.roc_auc), N_RESAMPLES, seed), len(predicted)),
        }

    for title, key in (("Age band", lambda r: age_band(r.viewer_age)), ("Gender", lambda r: r.viewer_gender)):
        members: dict[str, set[str]] = defaultdict(set)
        for row in rated:
            members[key(row)].add(row.viewer)
        hidden = metrics.suppressed_groups({g: len(u) for g, u in members.items()}, min_group_size)
        entries = []
        for group in sorted(members):
            if group in hidden:
                entries.append({"group": group, "suppressed": True})
                continue
            group_units = [u for u in units if u and key(u[0]) == group]
            entries.append({
                "group": group,
                "suppressed": False,
                "mean_rating": boot(mean_rating, group_units, sum(map(len, group_units))),
                "share_positive": boot(share_positive, group_units, sum(map(len, group_units))),
            })
        report.groups[title] = entries
    return report


def compare(db_path: str | Path, settings: Any, k: int) -> list[ArmReport]:
    rows = load_rows(db_path)
    surveys = load_survey_scores_by_arm(db_path)
    by_arm: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_arm[row.matcher].append(row)
    return [
        arm_report(
            arm, arm_rows, surveys.get(arm, {}), k,
            settings.positive_threshold, settings.prediction_threshold,
            settings.rapid_rating_ms, settings.min_group_size,
        )
        for arm, arm_rows in sorted(by_arm.items())
    ]


def overview(db_path: str | Path) -> dict[str, Any]:
    """Counts for the admin overview. Opted-out users are left out of per-matcher numbers."""
    rows = load_rows(db_path)
    with connect(db_path) as conn:
        approved = conn.execute("SELECT COUNT(*) FROM accounts WHERE status = 'approved' AND role = 'user'").fetchone()[0]
        opted_out = conn.execute("SELECT COUNT(*) FROM accounts WHERE experiment_opt_out = 1").fetchone()[0]
        matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    arms: dict[str, dict[str, Any]] = {}
    for row in rows:
        arm = arms.setdefault(row.matcher, {"impressions": 0, "ratings": 0, "rated_into_match": 0, "pool_sizes": []})
        arm["impressions"] += 1
        arm["pool_sizes"].append(row.pool_size)
        if row.rating is not None:
            arm["ratings"] += 1
            arm["rated_into_match"] += row.matched
    for arm in arms.values():
        sizes = arm.pop("pool_sizes")
        arm["mean_pool_size"] = round(statistics.fmean(sizes), 1)
    return {"approved_users": approved, "opted_out": opted_out, "matches": matches, "arms": dict(sorted(arms.items()))}


EXPORT_COLUMNS = (
    "viewer", "target", "matcher", "position", "predicted_score", "rating", "time_to_rate_ms",
    "pool_size", "viewer_age_band", "viewer_gender", "target_age_band", "target_gender", "produced_match",
)


def pseudonym(user_id: str, salt: str) -> str:
    """A stable salted hash, so the same person has the same code in every export."""
    return hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:16]


def export_rows(db_path: str | Path, salt: str) -> list[dict[str, Any]]:
    """One row per impression, pseudonymized. No emails, names, free text, photos,
    religion, or exact ages. Opted-out users are already left out by load_rows."""
    return [
        {
            "viewer": pseudonym(r.viewer, salt),
            "target": pseudonym(r.target, salt),
            "matcher": r.matcher,
            "position": r.position,
            "predicted_score": r.predicted,
            "rating": round(r.rating, 2) if r.rating is not None else None,
            "time_to_rate_ms": r.ms_to_rate,
            "pool_size": r.pool_size,
            "viewer_age_band": age_band(r.viewer_age),
            "viewer_gender": r.viewer_gender,
            "target_age_band": age_band(r.target_age),
            "target_gender": r.target_gender,
            "produced_match": r.matched if r.rating is not None else False,
        }
        for r in load_rows(db_path)
    ]


def to_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=EXPORT_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "" if v is None else v for k, v in row.items()})
    return output.getvalue()
