"""Who rated whom, and what goes with interest: the data behind the admin Users page.

Everything is on the 0 to 1 scale scores are stored on, like the simulated agents'
ratings. Only approved real participants who haven't opted out of the experiment are
included, and only live ratings between two of them.

People appear as codes, never names: W for women, M for men, N for nonbinary people,
numbered in the order they joined. Codes change when people join or leave, so they're for
reading the page. For keeping and joining data, each person also has a participant_id: a
salted hash of their user id (see analytics.pseudonym), stable for as long as EXPORT_SALT
stays the same, and identical to the ids in the impressions export.

truth_tables() turns the same data into the ground truth for comparing models with: who
rated whom and how much, by participant_id. See "Ground truth" in the README.

These are descriptive numbers for exploring the data. Ratings from the same person are
correlated, so correlations here aren't tests of anything.
"""

import csv
import io
import json
import statistics
import zipfile
from datetime import datetime, timezone
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import metrics
from app.analytics import age_band, pseudonym
from db import connect
from models import LIVE_RUN_ID

GENDER_ORDER = ("woman", "man", "nonbinary")
GENDER_CODE = {"woman": "W", "man": "M", "nonbinary": "N"}
GENDER_PLURAL = {"woman": "Women", "man": "Men", "nonbinary": "Nonbinary people"}
MIN_CATEGORY_RATINGS = 5  # Categories with fewer ratings are pooled into "other".
MAX_CATEGORIES = 12
MAX_INTERESTS = 15
HEIGHT_BINS = (("under 160", 0, 159), ("160–169", 160, 169), ("170–179", 170, 179), ("180–189", 180, 189), ("190+", 190, 999))
AGE_BINS = (("18–19", 18, 19), ("20–21", 20, 21), ("22–23", 22, 23), ("24–26", 24, 26), ("27+", 27, 999))
AGE_GAP_BINS = (("same age", 0, 0), ("1–2 years", 1, 2), ("3–4 years", 3, 4), ("5+ years", 5, 999))
HEIGHT_GAP_BINS = (
    ("10+ cm shorter", -999, -10), ("1–9 cm shorter", -9, -1), ("0–9 cm taller", 0, 9), ("10+ cm taller", 10, 999)
)


@dataclass
class Person:
    user_id: str
    code: str
    gender: str
    age: int
    height_cm: int
    school: str
    major: str
    job: str
    looking_for: str
    religion: str | None
    interests: list[str]
    participant_id: str | None = None  # Set when a salt is given.


@dataclass
class Rated:
    rater: Person
    target: Person
    score: float
    rated_at: str | None = None


@dataclass
class Effect:
    """Mean score for each value of something, as a chart and a summary statistic."""

    title: str
    question: str
    labels: list[str]
    series: dict[str, list[float | None]]  # Name of a line or set of bars, to one mean per label.
    counts: list[int]  # Ratings behind each label, over all series.
    statistic: str | None = None  # For example "r = 0.12" or "η² = 0.03".
    note: str | None = None


@dataclass
class Insights:
    people: list[Person]
    ratings: list[Rated]
    positive_threshold: float
    summary: dict[str, Any] = field(default_factory=dict)
    interest_map: dict[str, Any] = field(default_factory=dict)
    reciprocal_map: dict[str, Any] = field(default_factory=dict)
    gender_given: dict[str, Any] = field(default_factory=dict)
    gender_received: dict[str, Any] = field(default_factory=dict)
    liking: list[dict[str, Any]] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)


_PEOPLE = """
    SELECT p.user_id, p.gender, p.age, p.height_cm, p.school, p.major, p.job, p.looking_for,
           p.religion, p.interests
    FROM profiles p
    JOIN accounts a ON a.user_id = p.user_id
    WHERE a.status = 'approved' AND a.role = 'user' AND a.experiment_opt_out = 0 AND p.agent = 0
    ORDER BY a.created_at, p.user_id
"""


def load(db_path: str | Path, positive_threshold: float, salt: str = "") -> Insights:
    """positive_threshold is on the 0 to 1 scale. With a salt, people get participant_ids."""
    with connect(db_path) as conn:
        rows = conn.execute(_PEOPLE).fetchall()
        raw = conn.execute(
            "SELECT rater_id, target_id, score, timestamp FROM ratings WHERE run_id = ?", (LIVE_RUN_ID,)
        ).fetchall()
    counters: dict[str, int] = defaultdict(int)
    people = []
    for r in rows:
        gender = r["gender"]
        counters[gender] += 1
        people.append(Person(
            user_id=r["user_id"],
            code=f"{GENDER_CODE.get(gender, '?')}{counters[gender]:02d}",
            gender=gender,
            age=r["age"],
            height_cm=r["height_cm"],
            school=r["school"],
            major=r["major"],
            job=r["job"],
            looking_for=r["looking_for"],
            religion=r["religion"],
            interests=json.loads(r["interests"]),
            participant_id=pseudonym(r["user_id"], salt) if salt else None,
        ))
    by_id = {p.user_id: p for p in people}
    ratings = [
        Rated(by_id[r["rater_id"]], by_id[r["target_id"]], r["score"], r["timestamp"])
        for r in raw
        if r["rater_id"] in by_id and r["target_id"] in by_id
    ]
    return compute(people, ratings, positive_threshold)


def compute(people: list[Person], ratings: list[Rated], positive_threshold: float) -> Insights:
    result = Insights(people, ratings, positive_threshold)
    result.summary = _summary(people, ratings, positive_threshold)
    result.interest_map, result.reciprocal_map = _maps(people, ratings)
    result.gender_given = _distribution(ratings, lambda r: r.rater.gender)
    result.gender_received = _distribution(ratings, lambda r: r.target.gender)
    result.liking = _liking(ratings, positive_threshold)
    result.effects = _effects(ratings)
    return result


def _summary(people: list[Person], ratings: list[Rated], threshold: float) -> dict[str, Any]:
    scores = {(r.rater.user_id, r.target.user_id): r.score for r in ratings}
    both = [(a, b) for (a, b) in scores if a < b and (b, a) in scores]
    mutual = [pair for pair in both if scores[pair] >= threshold and scores[pair[::-1]] >= threshold]
    possible = len(people) * (len(people) - 1)
    return {
        "people": len(people),
        "by_gender": {GENDER_PLURAL[g]: sum(p.gender == g for p in people) for g in GENDER_ORDER if any(p.gender == g for p in people)},
        "ratings": len(ratings),
        "possible": possible,
        "coverage": len(ratings) / possible if possible else None,
        "pairs_both_rated": len(both),
        "pairs_both_positive": len(mutual),
        "mean_score": statistics.fmean(scores.values()) if scores else None,
    }


def _maps(people: list[Person], ratings: list[Rated]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two square grids over the same people, ordered by gender and then by mean score
    received, so the most liked people in each group come first.

    interest[i][j] is how person i rated person j, from 0 to 1. reciprocal[i][j] is the sum
    of both directions, from 0 to 2, and only where both have rated each other. Missing
    cells are None.
    """
    scores = {(r.rater.user_id, r.target.user_id): r.score for r in ratings}
    received: dict[str, list[float]] = defaultdict(list)
    for r in ratings:
        received[r.target.user_id].append(r.score)

    def order(p: Person) -> tuple[int, float, str]:
        mean = statistics.fmean(received[p.user_id]) if received[p.user_id] else -1.0
        rank = GENDER_ORDER.index(p.gender) if p.gender in GENDER_ORDER else len(GENDER_ORDER)
        return (rank, -mean, p.code)

    ordered = sorted(people, key=order)
    ids = [p.user_id for p in ordered]
    interest = [[_round(scores.get((a, b))) for b in ids] for a in ids]
    reciprocal = [
        [_round(scores[(a, b)] + scores[(b, a)]) if (a, b) in scores and (b, a) in scores else None for b in ids]
        for a in ids
    ]
    axes = {
        "labels": [p.code for p in ordered],
        "ids": [p.participant_id for p in ordered],
        "groups": [GENDER_CODE.get(p.gender, "?") for p in ordered],
    }
    return ({**axes, "values": interest, "max": 1}, {**axes, "values": reciprocal, "max": 2})


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _distribution(ratings: list[Rated], gender_of: Callable[[Rated], str]) -> dict[str, Any]:
    """For each gender, the share of its scores in each tenth of 0 to 1, with the mean."""
    groups: dict[str, list[float]] = defaultdict(list)
    for r in ratings:
        groups[gender_of(r)].append(r.score)
    series = {}
    stats = []
    for gender in GENDER_ORDER:
        values = groups.get(gender)
        if not values:
            continue
        counts = metrics.histogram(values)
        series[GENDER_PLURAL[gender]] = [round(c / len(values), 4) for c in counts]
        stats.append({"group": GENDER_PLURAL[gender], "n": len(values), "mean": round(statistics.fmean(values), 3),
                      "median": round(statistics.median(values), 3)})
    labels = [f"{b / 10:.1f}–{(b + 1) / 10:.1f}" for b in range(10)]
    return {"labels": labels, "series": series, "stats": stats}


def _liking(ratings: list[Rated], threshold: float) -> list[dict[str, Any]]:
    """For each pair of rater gender and target gender: how often the rating was positive."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in ratings:
        cells[(r.rater.gender, r.target.gender)].append(r.score)
    present = [g for g in GENDER_ORDER if any(g in pair for pair in cells)]
    table = []
    for rater in present:
        row = {"rater": GENDER_PLURAL[rater], "cells": []}
        for target in present:
            values = cells.get((rater, target), [])
            row["cells"].append({
                "target": GENDER_PLURAL[target],
                "n": len(values),
                "share_positive": round(sum(v >= threshold for v in values) / len(values), 3) if values else None,
                "mean": round(statistics.fmean(values), 3) if values else None,
            })
        table.append(row)
    return table


def _binned(value: int, bins: tuple[tuple[str, int, int], ...]) -> str:
    return next(name for name, low, high in bins if low <= value <= high)


def _means_by(
    ratings: Iterable[Rated], key: Callable[[Rated], str | None], labels: list[str]
) -> tuple[list[float | None], list[int]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in ratings:
        k = key(r)
        if k is not None:
            groups[k].append(r.score)
    return (
        [round(statistics.fmean(groups[l]), 3) if groups.get(l) else None for l in labels],
        [len(groups.get(l, [])) for l in labels],
    )


def _numeric_effect(
    ratings: list[Rated], title: str, question: str, value: Callable[[Rated], int],
    bins: tuple[tuple[str, int, int], ...], unit: str, note: str | None = None,
) -> Effect:
    """Mean score for each bin of a number, as one line per rater gender, plus Pearson r
    and the least-squares slope, both from the raw numbers rather than the bins."""
    labels = [name for name, _, _ in bins]
    series = {}
    for gender in GENDER_ORDER:
        subset = [r for r in ratings if r.rater.gender == gender]
        if subset:
            means, _ = _means_by(subset, lambda r: _binned(value(r), bins), labels)
            series[f"Rated by {GENDER_PLURAL[gender].lower()}"] = means
    _, counts = _means_by(ratings, lambda r: _binned(value(r), bins), labels)
    xs, ys = [float(value(r)) for r in ratings], [r.score for r in ratings]
    r_value = metrics.pearson(xs, ys) if ratings else None
    fit = metrics.linear_fit(xs, ys) if ratings else None
    parts = []
    if r_value is not None:
        parts.append(f"r = {r_value:+.2f}")
    if fit is not None:
        parts.append(f"slope {fit[0]:+.4f} per {unit}")
    return Effect(title, question, labels, series, counts, " · ".join(parts) or None, note)


def _category_effect(ratings: list[Rated], title: str, question: str, value: Callable[[Rated], str]) -> Effect:
    """Mean score for each category, largest first. Rare ones are pooled into "other"."""
    sizes: dict[str, int] = defaultdict(int)
    for r in ratings:
        sizes[value(r)] += 1
    kept = [c for c, n in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])) if n >= MIN_CATEGORY_RATINGS][:MAX_CATEGORIES]

    def key(r: Rated) -> str:
        return value(r) if value(r) in kept else "other"

    labels = kept + (["other"] if any(c not in kept for c in sizes) else [])
    means, counts = _means_by(ratings, key, labels)
    groups: dict[str, list[float]] = defaultdict(list)
    for r in ratings:
        groups[value(r)].append(r.score)
    eta = metrics.correlation_ratio(groups)
    return Effect(title, question, labels, {"Mean score": means}, counts, None if eta is None else f"η² = {eta:.3f}",
                  f"Categories with fewer than {MIN_CATEGORY_RATINGS} ratings are pooled into \"other\".")


def _effects(ratings: list[Rated]) -> list[Effect]:
    if not ratings:
        return []
    effects = [
        _numeric_effect(ratings, "Height", "Mean score received, by the height of the person rated", lambda r: r.target.height_cm, HEIGHT_BINS, "cm"),
        _numeric_effect(ratings, "Age", "Mean score received, by the age of the person rated", lambda r: r.target.age, AGE_BINS, "year"),
        _numeric_effect(
            ratings, "Height difference", "Mean score, by how much taller the person rated is than the rater",
            lambda r: r.target.height_cm - r.rater.height_cm, HEIGHT_GAP_BINS, "cm",
            "r and the slope use the height difference in cm, positive when the person rated is taller.",
        ),
        _numeric_effect(
            ratings, "Age gap", "Mean score, by the age gap between rater and the person rated",
            lambda r: abs(r.target.age - r.rater.age), AGE_GAP_BINS, "year",
        ),
        _category_effect(ratings, "Looking for", "Mean score received, by what the person rated is looking for", lambda r: r.target.looking_for),
        _category_effect(ratings, "Major", "Mean score received, by major", lambda r: r.target.major.strip().lower()),
        _category_effect(ratings, "Job", "Mean score received, by job", lambda r: r.target.job.strip().lower()),
        _category_effect(ratings, "School", "Mean score received, by school", lambda r: r.target.school.strip()),
        _category_effect(ratings, "Religion", "Mean score received, by religion", lambda r: (r.target.religion or "not given").strip().lower()),
        _category_effect(
            ratings, "Same looking for", "Mean score, when rater and rated want the same thing or not",
            lambda r: "same" if r.rater.looking_for == r.target.looking_for else "different",
        ),
        _category_effect(
            ratings, "Shared interests", "Mean score, by how many interests the two have in common",
            lambda r: _shared_label(len(set(r.rater.interests) & set(r.target.interests))),
        ),
        _interest_effect(ratings),
    ]
    return effects


def _shared_label(n: int) -> str:
    return "none" if n == 0 else "1" if n == 1 else "2 or more"


def _interest_effect(ratings: list[Rated]) -> Effect:
    """Mean score received by people who list each interest. People list several, so these
    groups overlap, and the statistic is each interest's difference from the overall mean."""
    groups: dict[str, list[float]] = defaultdict(list)
    for r in ratings:
        for interest in set(r.target.interests):
            groups[interest].append(r.score)
    overall = statistics.fmean(r.score for r in ratings)
    kept = [i for i, v in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])) if len(v) >= MIN_CATEGORY_RATINGS][:MAX_INTERESTS]
    means = [round(statistics.fmean(groups[i]), 3) for i in kept]
    return Effect(
        "Interests", "Mean score received by people who list each interest", kept, {"Mean score": means},
        [len(groups[i]) for i in kept], f"overall mean = {overall:.3f}",
        f"The {MAX_INTERESTS} most common interests with at least {MIN_CATEGORY_RATINGS} ratings. People list several, so groups overlap.",
    )


# Ground truth

TRUTH_FILES = ("participants.csv", "scores.csv", "matrix.csv")


def truth_tables(data: Insights) -> dict[str, list[list[Any]]]:
    """The ground truth, as rows for three CSV files, keyed by participant_id.

    participants.csv: one row per participant, with gender and age band only.
    scores.csv: one row per rating: rater_id, target_id, score from 0 to 1, and when.
    matrix.csv: one row per rater and one column per person rated, both in participant_id
    order. A blank cell means not rated yet, and the diagonal is always blank.
    """
    if any(p.participant_id is None for p in data.people):
        raise ValueError("load() needs a salt to give people participant_ids")
    people = sorted(data.people, key=lambda p: p.participant_id)
    given: dict[str, int] = defaultdict(int)
    received: dict[str, int] = defaultdict(int)
    scores = {}
    for r in data.ratings:
        given[r.rater.user_id] += 1
        received[r.target.user_id] += 1
        scores[(r.rater.participant_id, r.target.participant_id)] = r.score
    participants = [["participant_id", "gender", "age_band", "ratings_given", "ratings_received"]] + [
        [p.participant_id, p.gender, age_band(p.age), given[p.user_id], received[p.user_id]] for p in people
    ]
    ordered = sorted(data.ratings, key=lambda r: (r.rater.participant_id, r.target.participant_id))
    score_rows = [["rater_id", "target_id", "score", "rated_at"]] + [
        [r.rater.participant_id, r.target.participant_id, f"{r.score:.6f}", r.rated_at or ""] for r in ordered
    ]
    ids = [p.participant_id for p in people]
    matrix = [["rater_id"] + ids] + [
        [a] + ["" if (a, b) not in scores else f"{scores[(a, b)]:.6f}" for b in ids] for a in ids
    ]
    return {"participants.csv": participants, "scores.csv": score_rows, "matrix.csv": matrix}


def to_csv(rows: list[list[Any]]) -> str:
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()


TRUTH_README = """Ground truth from the live study: who rated whom, and how much.

Saved {saved_at} UTC. {people} participants, {ratings} ratings.

Who's included: approved real participants who haven't opted out of the experiment, and only
ratings between two of them. Simulated agents are never included. Someone who deletes their
account is removed, with all their ratings, as the consent form promises.

participant_id: a salted SHA-256 hash of the person's internal user id, first 16 hex
characters. It's stable for as long as EXPORT_SALT stays the same, so files saved on
different days, and the impressions export, use the same ids. Keep EXPORT_SALT secret:
with it and database access, ids can be traced back to accounts.

score: from 0 to 1, the scale the simulated agents' ratings use. People rate from 1 to 10,
and score = (rating - 1) / 9. Scores aren't normalized per rater.

participants.csv  participant_id, gender, age_band, ratings_given, ratings_received
scores.csv        rater_id, target_id, score, rated_at (UTC, ISO 8601). One row per rating.
matrix.csv        Rows are raters and columns are the people they rated, both by
                  participant_id. A blank cell means not rated yet. The diagonal is blank.

To compare a model with these: predict a score from 0 to 1 for each (rater_id, target_id)
in scores.csv, and compare, for example with mean absolute error, rank correlation per
rater, or precision at k. A pair is "liked" at a score of {threshold:.4f} or higher
(7.0 on the 1 to 10 scale), and "mutual" when both directions are.
"""


def truth_files(data: Insights) -> dict[str, str]:
    """Every ground-truth file, by name, as text: the three CSVs and a README."""
    files = {name: to_csv(rows) for name, rows in truth_tables(data).items()}
    files["README.txt"] = TRUTH_README.format(
        saved_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        people=len(data.people),
        ratings=len(data.ratings),
        threshold=data.positive_threshold,
    )
    return files


def truth_zip(data: Insights) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in truth_files(data).items():
            archive.writestr(f"ground-truth/{name}", text)
    return buffer.getvalue()


def write_truth(folder: Path, data: Insights) -> list[Path]:
    """Save the ground-truth files into a new folder, which must not exist yet."""
    folder.mkdir(parents=True, exist_ok=False)
    paths = []
    for name, text in truth_files(data).items():
        path = folder / name
        path.write_text(text)
        paths.append(path)
    return paths
