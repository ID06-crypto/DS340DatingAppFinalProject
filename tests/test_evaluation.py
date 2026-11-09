import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import evaluation
from evaluation import (
    assign_user_splits,
    best_f1_threshold,
    bootstrap_intervals,
    calibration_table,
    classification_metrics,
    directional_examples,
    expected_calibration_error,
    exposure_gini,
    load_or_create_splits,
    load_ratings,
    mutual_examples,
    public_profiles,
    ranking_metrics,
    subgroup_table,
)
from manager import ProfileManager

Y = np.array([1, 1, 0, 0, 0])
P = np.array([0.9, 0.4, 0.6, 0.2, 0.1])


def profile_data(**overrides: Any) -> dict[str, Any]:
    data = {
        "name": "SECRET_NAME",
        "age": 22,
        "height_cm": 175,
        "religion": "catholic",
        "school": "BU",
        "major": "biology",
        "job": "tutor",
        "interests": ["hiking"],
        "clubs": [],
        "gender": "man",
        "looking_for": "long-term",
        "agent": True,
    }
    return {**data, **overrides}


def test_metrics_match_a_hand_calculation() -> None:
    m = classification_metrics(Y, P, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 2, 1)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["type_1_rate"] == pytest.approx(1 / 3)
    assert m["type_2_rate"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert m["brier"] == pytest.approx(0.156)
    assert m["mean_probability"] == pytest.approx(0.44)
    assert m["bias"] == pytest.approx(0.04)
    assert m["positive_rate"] == pytest.approx(0.4)
    assert m["roc_auc"] == pytest.approx(5 / 6)


def test_accuracy_can_look_good_while_f1_exposes_a_useless_model() -> None:
    y = np.array([1] + [0] * 9)
    m = classification_metrics(y, np.zeros(10), 0.5)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["f1"] == 0
    assert m["recall"] == 0
    assert m["type_2_rate"] == 1
    assert np.isnan(m["precision"])


def test_weights_match_repeating_rows() -> None:
    weights = np.array([2, 1, 3, 1, 1])
    repeated = classification_metrics(np.repeat(Y, weights), np.repeat(P, weights), 0.5)
    weighted = classification_metrics(Y, P, 0.5, weights)
    for key in repeated:
        assert weighted[key] == pytest.approx(repeated[key]), key


def test_one_class_gives_nan_auc_rather_than_an_error() -> None:
    m = classification_metrics(np.array([0, 0]), np.array([0.2, 0.8]), 0.5)
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["type_2_rate"])
    assert m["type_1_rate"] == 0.5


def test_best_threshold_separates_clean_data() -> None:
    y = np.array([0, 0, 1, 1])
    assert best_f1_threshold(y, np.array([0.1, 0.2, 0.7, 0.8])) == pytest.approx(0.7)


def test_splits_are_stable_complete_and_sized() -> None:
    users = [f"user-{i}" for i in range(1000)]
    splits = assign_user_splits(users)
    assert assign_user_splits(reversed(users)) == splits
    assert set(splits) == set(users)
    assert {name: list(splits.values()).count(name) for name in ("train", "validation", "test")} == {
        "train": 640,
        "validation": 160,
        "test": 200,
    }


def test_saved_splits_are_reused_and_checked(tmp_path: Path) -> None:
    path = tmp_path / "exports" / "splits.json"
    first = load_or_create_splits(path, ["a", "b", "c"])
    path.write_text(json.dumps({"a": "test", "b": "train", "c": "train"}))
    assert load_or_create_splits(path, ["a", "b", "c"])["a"] == "test"
    assert set(first) == {"a", "b", "c"}
    with pytest.raises(ValueError, match="same users"):
        load_or_create_splits(path, ["a", "b", "z"])


@pytest.fixture
def people(tmp_path: Path) -> tuple[ProfileManager, list[str]]:
    manager = ProfileManager(tmp_path / "main.db", tmp_path / "personas.db")
    users = [manager.add_profile(profile_data()).user_id for _ in range(4)]
    manager.add_persona(users[0], "SECRET_PERSONA")
    for run in ("run-1", "run-2"):
        manager.start_run(run, "test", 1, {})
    return manager, users


def test_public_profiles_have_no_agent_flag_name_or_persona(people: Any) -> None:
    manager, users = people
    fields = public_profiles(manager.store)[users[0]]
    assert "agent" not in fields
    assert "name" not in fields
    assert "SECRET" not in str(fields)
    assert fields["major"] == "biology"


def test_examples_only_include_pairs_inside_one_split(people: Any) -> None:
    manager, (a, b, c, d) = people
    manager.add_rating(a, b, 0.8, "run-1")
    manager.add_rating(b, a, 0.4, "run-1")
    manager.add_rating(a, c, 0.9, "run-1")
    manager.add_rating(c, d, 0.5, "run-1")
    splits = {a: "test", b: "test", c: "train", d: "train"}
    ratings = load_ratings(manager.store, ["run-1"])
    profiles = public_profiles(manager.store)

    test = directional_examples(ratings, splits, "test", profiles)
    train = directional_examples(ratings, splits, "train", profiles)

    assert sorted(test["score"]) == [0.4, 0.8]
    assert list(train["score"]) == [0.5]
    assert list(train["like"]) == [1]
    assert "rater_major" in test
    assert "target_looking_for" in test
    assert not [c for c in test.columns if "agent" in c or "name" in c]


def test_a_pair_rated_in_two_runs_is_kept_once(people: Any) -> None:
    manager, (a, b, *_) = people
    manager.add_rating(a, b, 0.8, "run-1")
    manager.add_rating(a, b, 0.1, "run-2")
    ratings = load_ratings(manager.store, ["run-1", "run-2"])
    assert list(ratings["score"]) == [0.8]


def test_mutual_examples_need_both_people_to_like(people: Any) -> None:
    manager, (a, b, c, d) = people
    for rater, target, score in ((a, b, 0.8), (b, a, 0.7), (a, c, 0.9), (c, a, 0.3), (a, d, 0.9)):
        manager.add_rating(rater, target, score, "run-1")
    splits = dict.fromkeys((a, b, c, d), "test")
    examples = directional_examples(load_ratings(manager.store, ["run-1"]), splits, "test", public_profiles(manager.store))
    probability = examples["score"].to_numpy() / 2

    pairs = mutual_examples(examples, probability)

    assert len(pairs) == 2
    assert sorted(pairs["mutual"]) == [0, 1]
    row = pairs[pairs["mutual"] == 1].iloc[0]
    assert row["match_probability"] == pytest.approx(0.4 * 0.35)
    assert (pairs["user_a"] < pairs["user_b"]).all()


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    users = np.repeat(np.arange(60), 5)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    first = bootstrap_intervals(y, p, 0.5, users, n_samples=100)
    second = bootstrap_intervals(y, p, 0.5, users, n_samples=100)
    pd.testing.assert_frame_equal(first, second)
    f1 = classification_metrics(y, p, 0.5)["f1"]
    assert first.loc["f1", "low"] <= f1 <= first.loc["f1", "high"]


def test_bootstrap_difference_is_zero_against_itself() -> None:
    rng = np.random.default_rng(2)
    users = np.repeat(np.arange(40), 5)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    intervals = bootstrap_intervals(y, p, 0.5, users, baseline_probability=p, n_samples=30)
    assert intervals.loc["f1", "difference_low"] == 0
    assert intervals.loc["f1", "difference_high"] == 0


def test_bootstrap_difference_uses_the_baselines_own_threshold() -> None:
    rng = np.random.default_rng(4)
    users = np.repeat(np.arange(40), 5)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    constant = np.full(200, 0.3)
    # At its own threshold of 0.3 the baseline says yes to everyone, so its recall is 1.
    intervals = bootstrap_intervals(
        y, p, 0.5, users, baseline_probability=constant, baseline_threshold=0.3, n_samples=30
    )
    assert intervals.loc["recall", "difference_high"] < 0
    assert intervals.loc["type_1_rate", "difference_high"] < 0


def test_pair_bootstrap_uses_both_users() -> None:
    a = np.array(["u1", "u2", "u3", "u4"] * 10)
    b = np.array(["u5", "u6", "u7", "u8"] * 10)
    y = np.array([1, 0, 1, 0] * 10)
    p = np.array([0.9, 0.1, 0.4, 0.6] * 10)
    intervals = bootstrap_intervals(y, p, 0.5, a, b, n_samples=50)
    assert intervals.loc["accuracy", "low"] <= 0.5 <= intervals.loc["accuracy", "high"]


def test_calibration_of_a_perfectly_calibrated_model_is_close_to_zero() -> None:
    rng = np.random.default_rng(3)
    p = rng.random(50_000)
    y = (rng.random(50_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert len(calibration_table(y, p, 5)) == 5
    assert expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_subgroups_show_where_errors_fall_and_flag_small_groups() -> None:
    groups = pd.Series(["a", "a", "a", "b", "b"])
    table = subgroup_table(groups, Y, P, 0.5)
    assert table.loc["a", "n"] == 3
    assert table.loc["b", "type_1_rate"] == 0
    assert table["small_sample"].all()


def test_ranking_rewards_ordering_within_each_person() -> None:
    raters = pd.Series(["a"] * 5 + ["b"] * 5)
    y = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    good = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.2, 0.01, 0.01, 0.01, 0.01])
    bad = good[::-1].copy()
    assert ranking_metrics(raters, y, good, k=1)["within_rater_auc"] == 1
    assert ranking_metrics(raters, y, good, k=1)["precision_at_1"] == 1
    assert ranking_metrics(raters, y, good, k=1)["lift_at_1"] == pytest.approx(0.8)
    assert ranking_metrics(raters, y, bad, k=1)["within_rater_auc"] < 0.5


def test_exposure_gini() -> None:
    assert exposure_gini([5, 5, 5, 5]) == 0
    assert exposure_gini([0, 0, 0, 20]) == pytest.approx(0.75)
    assert exposure_gini([]) == 0


def test_test_pairs_script_rates_both_directions_and_can_continue(people: Any, tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_test_pairs

    from raters import FakeRater

    manager = ProfileManager(tmp_path / "t.db", tmp_path / "t_p.db")
    men = [manager.add_profile(profile_data()).user_id for _ in range(4)]
    women = [manager.add_profile(profile_data(gender="woman")).user_id for _ in range(4)]
    for user in men:
        manager.set_preferences(user, ["woman"])
    for user in women:
        manager.set_preferences(user, ["man"])
    for user in men + women:
        manager.add_persona(user, "persona")
    splits = dict.fromkeys(men + women, "test")
    pairs = build_test_pairs.sample_pairs(manager, splits, n_pairs=5, seed=1)
    assert len(pairs) == 5 == len(set(pairs))
    assert all(a in men and b in women or a in women and b in men for a, b in pairs)

    rater = FakeRater(random.Random(1))
    assert build_test_pairs.rate_pairs(manager, rater, "pairs", pairs[:2]) == 4
    assert build_test_pairs.rate_pairs(manager, rater, "pairs", pairs) == 6
    ratings = manager.ratings_in_run("pairs")
    assert len(ratings) == 10
    assert {(r.rater_id, r.target_id) for r in ratings} == {
        direction for a, b in pairs for direction in ((a, b), (b, a))
    }


def test_module_constants_describe_a_real_split() -> None:
    assert 0 < evaluation.TRAIN_FRACTION + evaluation.VALIDATION_FRACTION < 1
