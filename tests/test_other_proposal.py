"""Tests for proposal 3 (matching_algorithms/other_proposal.py) and its simulator."""

import ast
import gc
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from matching import PublicView
from matching_algorithms import other_proposal as op
from matching_algorithms.other_proposal_sim import (
    ExperimentConfig,
    Simulator,
    SimulatorConfig,
    gini,
    ndcg_at_k,
    run_experiment,
)
from models import Profile, PublicProfile, Rating, generate_user_id
from store import Store

ALGORITHM_FILE = Path(op.__file__)


def _profile(rng: random.Random, **overrides) -> PublicProfile:
    gender = rng.choice(["man", "woman"])
    data = dict(
        user_id=generate_user_id(rng),
        name="Sam",
        age=rng.randint(18, 24),
        height_cm=rng.randint(150, 200),
        religion=rng.choice(["catholic", "jewish", "atheist", "hindu"]),
        school=rng.choice(["BU", "MIT", "Northeastern"]),
        major=rng.choice(["biology", "finance", "history", "psychology"]),
        job=rng.choice(["tutor", "server", "intern"]),
        interests=rng.sample(["chess", "hiking", "cooking", "running", "film"], k=rng.randint(1, 3)),
        clubs=rng.sample(["debate", "a cappella", "outing club"], k=rng.randint(0, 2)),
        gender=gender,
        interested_in=rng.choice(["men", "women"]),
        looking_for=rng.choice(["short-term", "long-term", "just a date"]),
    )
    data.update(overrides)
    return PublicProfile(**data)


@pytest.fixture(scope="module")
def sim() -> Simulator:
    return Simulator(SimulatorConfig(n_users=80, seed=3))


# Features


def test_feature_order_and_length(sim):
    builder = op.FeatureBuilder.fit(sim.profiles)
    assert builder.names[: len(op.PAIR_FEATURES)] == op.PAIR_FEATURES
    assert len(set(builder.names)) == builder.dim
    phi = builder.build_features(sim.profiles[0], sim.profiles[1])
    assert phi.shape == (builder.dim,)
    assert phi[0] == 1.0  # Intercept.
    # One-hot blocks each have exactly one 1.
    for prefix in ("cand_religion=", "cand_major_group=", "cand_job_group=", "cand_school="):
        block = [i for i, n in enumerate(builder.names) if n.startswith(prefix)]
        assert phi[block].sum() == 1.0
    # Same vocabularies give the same order, whatever order the profiles come in.
    assert op.FeatureBuilder.fit(list(reversed(sim.profiles))).names == builder.names


def test_matrix_matches_single_builds(sim):
    builder = op.FeatureBuilder.fit(sim.profiles)
    viewer, cands = sim.profiles[0], sim.profiles[1:9]
    expected = np.stack([builder.build_features(viewer, c) for c in cands])
    np.testing.assert_allclose(builder.build_matrix(viewer, cands), expected)


def test_unseen_categories_go_to_other():
    rng = random.Random(0)
    builder = op.FeatureBuilder.fit([_profile(rng) for _ in range(20)])
    stranger = _profile(rng, religion="zoroastrian", school="Tufts", major="astrology", job="astronaut")
    phi = builder.build_features(_profile(rng), stranger)
    names = builder.names
    for field in ("religion", "major_group", "job_group", "school"):
        assert phi[names.index(f"cand_{field}=other")] == 1.0
    assert np.isfinite(phi).all()


def test_symmetric_and_antisymmetric_features(sim):
    builder = op.FeatureBuilder.fit(sim.profiles)
    names = builder.names
    for a, b in [(sim.profiles[i], sim.profiles[i + 1]) for i in range(0, 20, 2)]:
        ab, ba = builder.build_features(a, b), builder.build_features(b, a)
        for n in op.SYMMETRIC_FEATURES:
            assert ab[names.index(n)] == pytest.approx(ba[names.index(n)])
        for n in op.ANTISYMMETRIC_FEATURES:
            assert ab[names.index(n)] == pytest.approx(-ba[names.index(n)])


def test_jaccard_of_two_empty_lists_is_zero():
    assert op._jaccard(frozenset(), frozenset()) == 0.0
    assert op._jaccard(frozenset("ab"), frozenset("bc")) == pytest.approx(1 / 3)


# Bayesian models


def test_sequential_update_matches_closed_form_batch():
    rng = np.random.default_rng(0)
    d, n, sigma2 = 6, 40, 0.05
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + rng.normal(0, np.sqrt(sigma2), size=n)
    m0 = rng.normal(size=d)
    L0 = np.diag(rng.uniform(0.5, 3.0, size=d))

    model = op.BayesianUserModel(m0, L0, sigma2)
    for x, r in zip(X, y):
        model.update(x, r)

    P = L0 + X.T @ X / sigma2
    mean = np.linalg.solve(P, L0 @ m0 + X.T @ y / sigma2)
    np.testing.assert_allclose(model.mean, mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(model.covariance, np.linalg.inv(P), rtol=1e-8, atol=1e-12)
    batch = op.BayesianUserModel.from_data(m0, L0, sigma2, X.T @ X, X.T @ y, n)
    np.testing.assert_allclose(batch.mean, mean, rtol=1e-9, atol=1e-10)

    Phi = rng.normal(size=(5, d))
    std = np.sqrt(np.einsum("ij,jk,ik->i", Phi, np.linalg.inv(P), Phi))
    np.testing.assert_allclose(model.predictive_std(Phi), std, rtol=1e-8)
    np.testing.assert_allclose(model.ucb_scores(Phi, 2.0), Phi @ mean + 2 * std, rtol=1e-8)
    np.testing.assert_allclose(model.conservative_scores(Phi, 1.0), Phi @ mean - std, rtol=1e-8)


def test_thompson_samples_have_the_posterior_moments():
    rng = np.random.default_rng(1)
    model = op.BayesianUserModel(np.zeros(3), np.eye(3), 0.1)
    for _ in range(10):
        model.update(rng.normal(size=3), rng.normal())
    draws = model.sample_weights(np.random.default_rng(2), 40_000)
    np.testing.assert_allclose(draws.mean(0), model.mean, atol=0.01)
    np.testing.assert_allclose(np.cov(draws.T), model.covariance, atol=0.01)


def test_pairwise_model_ignores_rating_scale():
    rng = np.random.default_rng(3)
    Phi = rng.normal(size=(25, 5))
    r = rng.uniform(0.2, 0.6, size=25)
    a = op.PairwiseUserModel(np.zeros(5), np.eye(5)).fit(Phi, r, margin=0.0)
    b = op.PairwiseUserModel(np.zeros(5), np.eye(5)).fit(Phi, 1.5 * r + 0.1, margin=0.0)
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-8)


def test_shrunk_intercept_is_a_bayesian_average():
    cal = op.ShrunkInterceptCalibrator(k=5)
    cal.pop_mean = 0.4
    few = [("x", 0.9)]
    many = [(str(i), 0.9) for i in range(95)]
    # Offset = n / (n + k) * (user mean - population mean).
    assert cal.affine(few) == pytest.approx((1.0, -(1 / 6) * 0.5))
    assert cal.affine(many) == pytest.approx((1.0, -(95 / 100) * 0.5))
    assert cal.affine([]) == (1.0, 0.0)


# Storage


def test_user_state_rebuilt_from_ratings_equals_cache():
    result = run_experiment(op.ProposalConfig(n_onboarding=4), ExperimentConfig(n_users=60, budget=8, seed=5))
    assert result["cache_rebuild_max_abs_diff"] < 1e-9


def test_storage_round_trip(sim):
    storage = op.ProposalStorage(":memory:")
    engine = op.ReciprocalBayesEngine(sim.profiles[:30], op.ProposalConfig(), storage, "s1")
    user = engine.ids[0]
    for cand in engine.candidates(user)[:5]:
        engine.observe(user, cand, 0.5, op.Choice(cand, 0.25, "bandit", "learn"))
    engine.flush()
    logs = storage.ratings("s1")
    assert [l.score for l in logs] == [50.0] * len(logs)
    assert {l.phase for l in logs} == {"bandit"} and {l.propensity for l in logs} == {0.25}
    rebuilt = storage.rebuild_user_states("s1", engine.builder, engine.profiles)
    cached = storage.load_user_states("s1", engine.dim)
    np.testing.assert_allclose(rebuilt[user].P, cached[user].P)
    np.testing.assert_allclose(rebuilt[user].b, cached[user].b)
    assert cached[user].n_ratings == len(logs)
    engine.retrain()
    assert storage.latest_population_model("s1").version == 1


# Hard filters


def test_hard_filters_never_return_ineligible_candidates():
    rng = random.Random(7)
    profiles = [_profile(rng) for _ in range(120)]
    engine = op.ReciprocalBayesEngine(profiles, op.ProposalConfig(n_onboarding=3))
    by_id = {p.user_id: p for p in profiles}
    for viewer in profiles:
        cands = engine.candidates(viewer.user_id)
        assert cands == op.eligible_candidates(viewer, profiles)
        for c in cands:
            other = by_id[c]
            assert c != viewer.user_id
            assert op.is_compatible(viewer, other)
            assert op.looking_for_compatible(viewer.looking_for, other.looking_for)
            assert {viewer.looking_for, other.looking_for} != {"long-term", "short-term"}
    # Picks made during onboarding and the feed stay inside the filter, and never repeat.
    engine.retrain()
    for viewer in profiles[:40]:
        for _ in range(6):
            choice = engine.pick(viewer.user_id)
            if choice is None:
                break
            assert op.is_eligible(viewer, by_id[choice.candidate_id])
            assert choice.candidate_id not in engine.rated[viewer.user_id]
            assert 0 < choice.propensity <= 1
            engine.observe(viewer.user_id, choice.candidate_id, rng.random(), choice)


def test_looking_for_rule():
    assert not op.looking_for_compatible("long-term", "short-term")
    assert not op.looking_for_compatible("short-term", "long-term")
    for a in ("short-term", "long-term", "just a date"):
        assert op.looking_for_compatible(a, a)
        assert op.looking_for_compatible(a, "just a date")


# Onboarding


def test_d_optimal_onboarding_beats_random_on_posterior_variance(sim):
    engine = op.ReciprocalBayesEngine(sim.profiles, op.ProposalConfig())
    _, precision = engine._prior()
    sigma2 = engine.noise_scale**2
    rng = np.random.default_rng(0)
    d_var, r_var, d_logdet, r_logdet = [], [], [], []
    users = [u for u in engine.ids if len(engine.candidates(u)) >= 20][:30]
    assert len(users) >= 20
    for user in users:
        Phi = engine.features(user, engine.candidates(user))
        for picks, var, logdet in (
            (op.d_optimal_onboarding(Phi, precision, sigma2, 10), d_var, d_logdet),
            (op.random_onboarding(len(Phi), 10, rng), r_var, r_logdet),
        ):
            model = op.BayesianUserModel(np.zeros(engine.dim), precision, sigma2)
            for i in picks:
                model.update(Phi[i], 0.0)
            var.append(np.mean(model.predictive_std(Phi) ** 2))
            logdet.append(np.linalg.slogdet(model.covariance)[1])
    assert np.mean(d_var) < np.mean(r_var)
    assert np.mean(d_logdet) < np.mean(r_logdet)


def test_onboarding_respects_size_and_anchors(sim):
    engine = op.ReciprocalBayesEngine(sim.profiles, op.ProposalConfig(n_onboarding=8, n_anchors=3))
    user = engine.ids[0]
    plan = engine.onboarding_plan(user)
    assert len(plan) == len(set(plan)) == 8
    assert plan[:3] == engine._anchor_ids(user)


# Policies and matching


def test_mixture_probabilities_are_a_distribution():
    match = np.array([0.1, 0.9, 0.3])
    learn = op.thompson_propensities(np.random.default_rng(0).normal(size=(3, 500)))
    probs = op.mixture_probabilities(match, learn, explore_share=0.15, uniform_floor=0.02)
    assert probs.sum() == pytest.approx(1.0)
    assert (probs > 0).all() and probs.argmax() == 1


def test_exposure_cap_drops_overexposed_profiles():
    scores = np.array([0.9, 0.5, 0.4])
    adjusted = op.exposure_adjust(scores, np.array([10.0, 1.0, 1.0]), 2.0, cap=3.0, penalty=0.05)
    assert adjusted[0] == -np.inf and adjusted.argmax() == 1
    # If everyone is over the cap, nobody is dropped.
    all_over = op.exposure_adjust(scores, np.array([10.0, 10.0, 10.0]), 2.0, cap=3.0, penalty=0.0)
    assert np.isfinite(all_over).all()


@pytest.mark.parametrize("method", op.COMBINATIONS)
def test_combinations_are_symmetric_and_bounded(method):
    x, y = np.random.default_rng(0).uniform(size=(2, 50))
    np.testing.assert_allclose(op.combine(x, y, method), op.combine(y, x, method))
    out = op.combine(x, y, method)
    assert (out <= np.maximum(x, y) + 1e-12).all() and (out >= np.minimum(x, y) - 1e-12).all()


def test_one_to_one_assignment_uses_each_user_once(sim):
    engine = op.ReciprocalBayesEngine(sim.profiles, op.ProposalConfig())
    pairs = engine.one_to_one()
    people = [p for a, b, _ in pairs for p in (a, b)]
    assert pairs and len(people) == len(set(people))
    for a, b, _ in pairs:
        assert op.is_eligible(engine.profiles[a], engine.profiles[b])


# Oracle isolation


def test_algorithm_file_cannot_reach_the_simulator():
    tree = ast.parse(ALGORITHM_FILE.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("sim" in m.split(".")[-1] for m in imported), imported
    assert not any(m.startswith(("personas", "manager", "raters")) for m in imported), imported
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert not names & {"Simulator", "oracle_score", "oracle_matrix", "like_probability"}


def test_engine_holds_no_reference_to_the_simulator(sim):
    engine = op.ReciprocalBayesEngine(sim.profiles, op.ProposalConfig(n_onboarding=2))
    for user in engine.ids[:10]:
        choice = engine.pick(user)
        engine.observe(user, choice.candidate_id, sim.rate(user, choice.candidate_id) / 100, choice)
    engine.retrain()
    seen, stack = set(), [engine]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (type, type(op))):
            continue
        seen.add(id(obj))
        assert not isinstance(obj, Simulator)
        stack.extend(gc.get_referents(obj))
    assert all(type(p) is PublicProfile for p in engine.profiles.values())


# Metrics and the whole pipeline


def test_metric_helpers():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    assert gini(np.r_[np.zeros(9), 1.0]) == pytest.approx(0.9)
    gains = np.array([3.0, 2.0, 1.0, 0.0])
    assert ndcg_at_k(gains, gains, 3) == pytest.approx(1.0)
    assert ndcg_at_k(-gains, gains, 3) < 1.0


def test_pipeline_smoke_50_users():
    exp = ExperimentConfig(n_users=50, budget=14, retrain_every=3, seed=11)
    result = run_experiment(replace(op.ProposalConfig(), n_onboarding=6), exp)
    for key in ("spearman", "ndcg10", "precision_at_k", "mutual_match_rate", "exposure_gini", "cum_regret_match"):
        assert np.isfinite(result[key]), key
    assert 0 <= result["precision_at_k"] <= result["oracle_precision_at_k"] + 1e-9
    assert len(result["regret_match_curve"]) == 8
    assert result["spearman"] > 0.1  # Learns something real about each user's taste.
    for config in (
        op.ProposalConfig(n_onboarding=6, policy="ucb", calibration="pairwise", reciprocal="min"),
        op.ProposalConfig(n_onboarding=6, scorer="popularity", onboarding="random"),
    ):
        assert np.isfinite(run_experiment(config, replace(exp, budget=9))["ndcg10"])


# Matcher adapter


def test_matcher_runs_against_the_public_view(tmp_path, sim):
    store = Store(tmp_path / "main.db")
    for p in sim.profiles[:40]:
        store.insert_profile(Profile(**p.model_dump(), agent=True))
    view = PublicView(store, "run-1")
    matcher = op.BayesianReciprocalMatcher(op.ProposalConfig(n_onboarding=3), min_ratings=30, retrain_every=40)
    users = [p.user_id for p in sim.profiles[:40]]
    shown = 0
    for step in range(1, 240):
        user = users[step % len(users)]
        target = matcher.next_profile(user, view)
        if target is None:
            continue
        assert target not in view.already_rated(user)
        rating = Rating(rater_id=user, target_id=target, score=sim.rate(user, target) / 100, run_id="run-1", step=step)
        store.upsert_rating(rating)
        matcher.observe(rating)
        shown += 1
    assert shown > 100
    assert matcher.engine.population is not None and matcher.engine.population.version >= 2
