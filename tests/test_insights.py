import pytest

from app import insights
from app.insights import Person, Rated


def person(code: str, gender: str, **overrides) -> Person:
    fields = dict(user_id=code.lower(), code=code, gender=gender, age=21, height_cm=170, school="BU",
                  major="biology", job="tutor", looking_for="long-term", religion=None, interests=["chess"])
    return Person(**{**fields, **overrides})


@pytest.fixture
def people() -> list[Person]:
    return [person("W01", "woman"), person("W02", "woman", height_cm=160), person("M01", "man", height_cm=185), person("N01", "nonbinary")]


def test_maps_are_grouped_by_gender_and_reciprocal_is_the_symmetric_sum(people: list[Person]) -> None:
    w1, w2, m1, n1 = people
    ratings = [Rated(w1, m1, 0.8), Rated(m1, w1, 0.6), Rated(w1, w2, 0.2), Rated(w2, m1, 1.0)]
    data = insights.compute(people, ratings, positive_threshold=0.6)
    interest, reciprocal = data.interest_map, data.reciprocal_map
    assert interest["groups"] == ["W", "W", "M", "N"]
    # W01 received 0.6 and W02 0.2, so W01 comes first among women.
    assert interest["labels"] == ["W01", "W02", "M01", "N01"]
    index = {code: i for i, code in enumerate(interest["labels"])}
    assert interest["values"][index["W01"]][index["M01"]] == 0.8
    assert interest["values"][index["M01"]][index["W01"]] == 0.6
    assert interest["values"][index["W01"]][index["W01"]] is None
    assert interest["values"][index["N01"]][index["W01"]] is None
    assert reciprocal["values"][index["W01"]][index["M01"]] == reciprocal["values"][index["M01"]][index["W01"]] == 1.4
    assert reciprocal["values"][index["W01"]][index["W02"]] is None  # Only one direction rated.
    assert (interest["max"], reciprocal["max"]) == (1, 2)
    summary = data.summary
    assert (summary["people"], summary["ratings"], summary["possible"]) == (4, 4, 12)
    assert (summary["pairs_both_rated"], summary["pairs_both_positive"]) == (1, 1)


def test_given_received_and_liking_by_gender(people: list[Person]) -> None:
    w1, w2, m1, n1 = people
    ratings = [Rated(w1, m1, 0.9), Rated(w2, m1, 0.3), Rated(m1, w1, 0.65), Rated(n1, w1, 0.05)]
    data = insights.compute(people, ratings, positive_threshold=0.6)
    given = {row["group"]: row for row in data.gender_given["stats"]}
    received = {row["group"]: row for row in data.gender_received["stats"]}
    assert given["Women"]["n"] == 2 and given["Women"]["mean"] == 0.6
    assert received["Men"]["n"] == 2 and received["Women"]["n"] == 2
    assert sum(data.gender_given["series"]["Women"]) == pytest.approx(1.0)
    cells = {(row["rater"], c["target"]): c for row in data.liking for c in row["cells"]}
    assert cells[("Women", "Men")]["share_positive"] == 0.5
    assert cells[("Men", "Women")]["share_positive"] == 1.0
    assert cells[("Women", "Women")]["n"] == 0


def test_effects_report_height_correlation_and_pool_rare_categories(people: list[Person]) -> None:
    w1, w2, m1, n1 = people
    tall, short = m1, w2
    ratings = [Rated(w1, tall, 0.9), Rated(n1, tall, 0.8), Rated(w1, short, 0.2), Rated(n1, short, 0.1)]
    effects = {e.title: e for e in insights.compute(people, ratings, 0.6).effects}
    height = effects["Height"]
    # Taller people got the higher scores: 25 cm taller, 0.7 higher on average.
    assert height.statistic == "r = +0.99 · slope +0.0280 per cm"
    assert height.counts[height.labels.index("180–189")] == 2
    assert set(height.series) == {"Rated by women", "Rated by nonbinary people"}
    major = effects["Major"]
    assert major.labels == ["other"]  # Fewer than 5 ratings in every category.
    assert insights.compute(people, [], 0.6).effects == []
