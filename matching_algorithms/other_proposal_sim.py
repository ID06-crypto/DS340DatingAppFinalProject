"""Simulator, metrics, baselines, and ablations for proposal 3 (other_proposal.py).

The simulator invents Boston students with hidden, heterogeneous tastes and an oracle
that knows each pair's true expected rating. Only this file sees the oracle. The engine
is handed public profiles, and gets ratings one at a time through observe().

Run everything (about 15 minutes on 8 cores):

    python matching_algorithms/other_proposal_sim.py --out exports/other_proposal

A quick check:

    python matching_algorithms/other_proposal_sim.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# One BLAS thread per process. The matrices are tiny (34 x 34), so threads only add
# overhead, and with a process pool they oversubscribe the CPU badly. This must run
# before numpy is imported, and it's re-run in each spawned worker.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
from scipy.special import ndtr
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT, _ROOT / "Data Generation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bu_clubs import BU_CLUBS, MEN_ONLY_CLUBS, WOMEN_ONLY_CLUBS  # noqa: E402
from matching_algorithms.other_proposal import (  # noqa: E402
    MAJOR_GROUPS,
    ProposalConfig,
    ProposalStorage,
    ReciprocalBayesEngine,
)
from models import PublicProfile, generate_user_id  # noqa: E402
from profile_lists import MAJORS, MEN_NAMES, WOMEN_NAMES  # noqa: E402

HOBBIES_PATH = _ROOT / "Data Generation" / "hobbylist.csv"


# Simulator

AGES = np.arange(18, 25)
AGE_P = np.array([0.12, 0.17, 0.18, 0.18, 0.16, 0.11, 0.08])
HEIGHT = {"man": (177.0, 7.5), "woman": (163.5, 6.5)}  # Mean and SD in cm.
RELIGIONS = {
    "catholic": 0.20, "christian": 0.22, "hindu": 0.04, "buddhist": 0.03, "muslim": 0.04,
    "jewish": 0.07, "mormon": 0.01, "atheist": 0.12, "agnostic": 0.15, "spiritual": 0.12,
}
NONRELIGIOUS = {"atheist", "agnostic"}
SCHOOLS = {
    "BU": 0.40, "Northeastern": 0.25, "Boston College": 0.12, "MIT": 0.08, "Harvard": 0.07,
    "Emerson": 0.05, "Suffolk": 0.03,
}
JOBS = ("tutor", "intern", "research assistant", "server", "teaching assistant", "bartender", "front desk", "on-campus work")
LOOKING_FOR = {"long-term": 0.40, "short-term": 0.25, "just a date": 0.35}

# The simulator's own, hidden description of a pair. It overlaps with the algorithm's
# features on purpose, but it's built separately and isn't the same.
TRUE_FEATURES = (
    "cand_age", "cand_height_for_gender", "age_gap", "same_religion", "cand_religious",
    "same_school", "interest_overlap", "club_overlap", "same_looking_for",
    "cand_stem", "cand_arts", "cand_business", "cand_academic_job",
)
# Hand-designed taste clusters, in logit units per unit of each feature.
CLUSTER_CENTERS = np.array(
    [
        # Traditional: shared faith and goals.
        [0.0, 0.1, 0.1, 1.3, 0.4, 0.0, 0.5, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0],
        # Shared interests: what you do together.
        [0.0, 0.0, 0.0, 0.1, -0.2, 0.2, 3.5, 2.5, 0.2, -0.2, 0.5, -0.2, 0.0],
        # Ambitious: STEM, business, academic jobs, same school.
        [0.1, 0.2, 0.0, 0.0, 0.0, 0.6, 0.5, 0.3, 0.2, 0.6, -0.3, 0.6, 0.5],
        # Physical: height and age.
        [-0.3, 0.7, 0.3, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
TASTE_NOISE_SCALE = np.array([0.2, 0.25, 0.15, 0.4, 0.25, 0.25, 1.2, 0.9, 0.25, 0.25, 0.25, 0.25, 0.25])


@dataclass(frozen=True)
class SimulatorConfig:
    n_users: int = 300
    n_clusters: int = 4
    taste_noise: float = 1.0          # Multiplies TASTE_NOISE_SCALE: individual deviation from the cluster.
    attractiveness_mean: float = 0.5  # Weight on the hidden attractiveness scalar, per rater.
    attractiveness_sd: float = 0.2
    chemistry_dim: int = 2
    chemistry_scale: float = 0.6      # Hidden pairwise "chemistry", symmetric.
    offset_sd: float = 0.5            # Rater generosity, in logits.
    scale_log_sd: float = 0.3         # Rater decisiveness: how strongly tastes show in ratings.
    base_logit: float = -0.75
    rating_noise_sd: float = 8.0      # On the 0-100 scale.
    same_gender_rate: float = 0.05
    seed: int = 0


def _load_hobbies() -> list[str]:
    with open(HOBBIES_PATH, encoding="utf-8", newline="") as f:
        return list(dict.fromkeys(row["Hobby-name"].strip().lower() for row in csv.DictReader(f)))


def _zipf_weights(n: int, s: float, rng: np.random.Generator) -> np.ndarray:
    """Zipf popularity over a shuffled list, so a few items are common and most are rare."""
    w = 1.0 / np.arange(1, n + 1) ** s
    return rng.permutation(w / w.sum())


class Simulator:
    """Synthetic students with hidden tastes, and the oracle of their true ratings.

    True utility of A looking at B, in logits:
        u = w_A . psi(A, B)                     (A's taste: cluster center plus noise)
          + gamma_A * attract_B                 (hidden, never in a profile)
          + kappa * c_A . c_B                   (hidden mutual chemistry)
          - 0.6 * relu(|age gap| - 2)           (non-linear: big gaps hurt)
          - 0.8 * relu((h_A - h_B) / 5)         (non-linear: women who are taller than him)
          + 0.8 * same_religion * cand_religious (interaction)
          + w_interest * overlap * [A long-term] (interests matter twice for long-term)
    oracle(A, B) = 100 * sigmoid(base + offset_A + scale_A * u), and a rating is the
    oracle plus Gaussian noise, clipped to [0, 100]. The noise is fixed per pair, so
    every policy faces the same draws (common random numbers).
    """

    def __init__(self, config: SimulatorConfig = SimulatorConfig()) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.profiles = self._make_profiles(rng)
        self.ids = [p.user_id for p in self.profiles]
        self.index = {uid: i for i, uid in enumerate(self.ids)}
        self._make_tastes(rng)
        self._oracle = self._compute_oracle()
        n = len(self.ids)
        self._noise = rng.normal(0.0, config.rating_noise_sd, size=(n, n))

    def _make_profiles(self, rng: np.random.Generator) -> list[PublicProfile]:
        cfg = self.config
        id_rng = random.Random(cfg.seed)
        hobbies = _load_hobbies()
        hobby_w = _zipf_weights(len(hobbies), 1.0, rng)
        clubs_open = {
            "man": [c for c in BU_CLUBS if c not in WOMEN_ONLY_CLUBS],
            "woman": [c for c in BU_CLUBS if c not in MEN_ONLY_CLUBS],
        }
        club_w = {g: _zipf_weights(len(c), 1.0, rng) for g, c in clubs_open.items()}
        profiles = []
        for _ in range(cfg.n_users):
            gender = "man" if rng.random() < 0.5 else "woman"
            same = rng.random() < cfg.same_gender_rate
            interested_in = {"man": "men", "woman": "women"}[gender] if same else {"man": "women", "woman": "men"}[gender]
            mean, sd = HEIGHT[gender]
            n_int = int(rng.integers(2, 6))
            n_club = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15]))
            profiles.append(
                PublicProfile(
                    user_id=generate_user_id(id_rng),
                    name=str(rng.choice(MEN_NAMES if gender == "man" else WOMEN_NAMES)),
                    age=int(rng.choice(AGES, p=AGE_P)),
                    height_cm=int(np.clip(round(rng.normal(mean, sd)), 150, 205)),
                    religion=str(rng.choice(list(RELIGIONS), p=list(RELIGIONS.values()))),
                    school=str(rng.choice(list(SCHOOLS), p=list(SCHOOLS.values()))),
                    major=str(rng.choice(MAJORS)),
                    job=str(rng.choice(JOBS)),
                    interests=list(rng.choice(hobbies, size=n_int, replace=False, p=hobby_w)),
                    clubs=list(rng.choice(clubs_open[gender], size=n_club, replace=False, p=club_w[gender])),
                    gender=gender,
                    interested_in=interested_in,
                    looking_for=str(rng.choice(list(LOOKING_FOR), p=list(LOOKING_FOR.values()))),
                )
            )
        return profiles

    def _make_tastes(self, rng: np.random.Generator) -> None:
        cfg = self.config
        n, k = len(self.profiles), cfg.n_clusters
        centers = CLUSTER_CENTERS[: min(k, len(CLUSTER_CENTERS))]
        if k > len(centers):
            extra = rng.normal(0, 0.5, size=(k - len(centers), len(TRUE_FEATURES)))
            centers = np.vstack([centers, extra])
        self._clusters = rng.integers(0, k, size=n)
        self._weights = centers[self._clusters] + rng.normal(size=(n, len(TRUE_FEATURES))) * TASTE_NOISE_SCALE * cfg.taste_noise
        self._attract = rng.normal(size=n)
        self._gamma = rng.normal(cfg.attractiveness_mean, cfg.attractiveness_sd, size=n)
        self._chem = rng.normal(size=(n, cfg.chemistry_dim))
        self._offset = rng.normal(0, cfg.offset_sd, size=n)
        self._scale = np.exp(rng.normal(0, cfg.scale_log_sd, size=n))

    def _compute_oracle(self) -> np.ndarray:
        cfg, P = self.config, self.profiles
        age = np.array([p.age for p in P], dtype=float)
        h = np.array([p.height_cm for p in P], dtype=float)
        woman = np.array([p.gender == "woman" for p in P])
        h_rel = np.where(woman, (h - HEIGHT["woman"][0]) / HEIGHT["woman"][1], (h - HEIGHT["man"][0]) / HEIGHT["man"][1])

        def same(attr: str) -> np.ndarray:
            v = np.array([getattr(p, attr) for p in P])
            return (v[:, None] == v[None, :]).astype(float)

        def overlap(attr: str) -> np.ndarray:
            vocab = sorted({x for p in P for x in getattr(p, attr)})
            pos = {x: i for i, x in enumerate(vocab)}
            M = np.zeros((len(P), len(vocab)))
            for i, p in enumerate(P):
                M[i, [pos[x] for x in getattr(p, attr)]] = 1
            inter = M @ M.T
            union = M.sum(1)[:, None] + M.sum(1)[None, :] - inter
            return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        groups = np.array([MAJOR_GROUPS.get(p.major, "other") for p in P])
        cand = lambda x: np.broadcast_to(np.asarray(x, dtype=float)[None, :], (len(P), len(P)))  # noqa: E731
        religious = np.array([p.religion not in NONRELIGIOUS for p in P], dtype=float)
        gap = age[None, :] - age[:, None]
        same_rel = same("religion")
        interests = overlap("interests")
        psi = np.stack(
            [
                cand((age - 21.0) / 1.8), cand(h_rel), gap / 2.0, same_rel, cand(religious),
                same("school"), interests, overlap("clubs"), same("looking_for"),
                cand(groups == "stem"), cand(groups == "arts_humanities"), cand(groups == "business"),
                cand([p.job in ("tutor", "research assistant", "teaching assistant") for p in P]),
            ],
            axis=-1,
        )
        u = np.einsum("abf,af->ab", psi, self._weights)
        u += self._gamma[:, None] * self._attract[None, :]
        u += cfg.chemistry_scale * (self._chem @ self._chem.T) / math.sqrt(cfg.chemistry_dim)
        u -= 0.6 * np.maximum(0.0, np.abs(gap) - 2.0)
        u -= 0.8 * woman[:, None] * np.maximum(0.0, (h[:, None] - h[None, :]) / 5.0)
        u += 0.8 * same_rel * religious[None, :]
        long_term = np.array([p.looking_for == "long-term" for p in P], dtype=float)
        u += self._weights[:, 6:7] * interests * long_term[:, None]
        logit = cfg.base_logit + self._offset[:, None] + self._scale[:, None] * u
        oracle = 100.0 / (1.0 + np.exp(-logit))
        np.fill_diagonal(oracle, np.nan)
        return oracle

    def rate(self, rater_id: str, ratee_id: str) -> float:
        """A noisy rating, 0-100, as the user would give it."""
        i, j = self.index[rater_id], self.index[ratee_id]
        return float(np.clip(round(self._oracle[i, j] + self._noise[i, j], 1), 0.0, 100.0))

    def oracle_score(self, rater_id: str, ratee_id: str) -> float:
        """The true expected rating, 0-100. For the evaluator only."""
        return float(self._oracle[self.index[rater_id], self.index[ratee_id]])

    def oracle_matrix(self) -> np.ndarray:
        """[i, j] is oracle_score(ids[i], ids[j]). For the evaluator only."""
        return self._oracle.copy()

    def like_probability(self, threshold: float = 70.0) -> np.ndarray:
        """[i, j] is P(rating >= threshold). Clipping doesn't change it for thresholds in (0, 100)."""
        return ndtr((self._oracle - threshold) / self.config.rating_noise_sd)

    def cluster_of(self, user_id: str) -> int:
        return int(self._clusters[self.index[user_id]])


# Metrics


def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative counts: 0 is perfectly even, near 1 is one winner."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.sum(np.arange(1, n + 1) * x)) / (n * x.sum()) - (n + 1) / n)


def ndcg_at_k(pred: np.ndarray, gains: np.ndarray, k: int) -> float:
    """NDCG@k of ranking by pred, against the ideal ranking by gains."""
    k = min(k, len(gains))
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = gains[np.argsort(-pred, kind="stable")[:k]] @ discounts
    idcg = np.sort(gains)[::-1][:k] @ discounts
    return float(dcg / idcg) if idcg > 0 else float("nan")


@dataclass(frozen=True)
class ExperimentConfig:
    n_users: int = 300
    budget: int = 45           # Ratings per user: onboarding plus bandit rounds.
    holdout_fraction: float = 0.3
    retrain_every: int = 10    # Bandit rounds between population retrains.
    threshold: float = 70.0    # A true match needs both ratings at or above this.
    k: int = 10
    seed: int = 0
    use_storage: bool = True
    sim: SimulatorConfig = field(default_factory=SimulatorConfig)


def evaluate(
    engine: ReciprocalBayesEngine,
    sim: Simulator,
    heldout: dict[str, list[str]],
    oracle: np.ndarray,
    p_match: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Score the engine on each user's held-out candidates, which it has never been shown."""
    idx = sim.index
    spearman, ndcg, prec, hit, rndcg, ideal = [], [], [], [], [], []
    shown_in_topk = np.zeros(len(sim.ids))
    for u, held in heldout.items():
        if len(held) < max(5, k // 2):
            continue
        i, cols = idx[u], np.array([idx[c] for c in held])
        true = oracle[i, cols]
        pred = engine.predicted_scores(u, held)
        if np.ptp(pred) > 1e-12:
            spearman.append(spearmanr(pred, true).statistic)
        else:
            spearman.append(0.0)
        ndcg.append(ndcg_at_k(pred, true / 100.0, k))
        compat = engine.compatibility(u, held)
        top = np.argsort(-compat, kind="stable")[:k]
        pm = p_match[i, cols]
        prec.append(pm[top].mean())
        hit.append(1.0 - np.prod(1.0 - pm[top]))
        rndcg.append(ndcg_at_k(compat, pm, k))
        ideal.append(np.sort(pm)[::-1][:k].mean())
        shown_in_topk[cols[top]] += 1
    if not prec:
        raise ValueError(f"no user has at least {max(5, k // 2)} held-out candidates; use more users")
    return {
        "spearman": float(np.nanmean(spearman)),
        "ndcg10": float(np.nanmean(ndcg)),
        "precision_at_k": float(np.mean(prec)),
        "oracle_precision_at_k": float(np.mean(ideal)),
        "mutual_match_rate": float(np.mean(hit)),
        "reciprocal_ndcg10": float(np.nanmean(rndcg)),
        "topk_gini": gini(shown_in_topk),
    }


def religion_audit(engine: ReciprocalBayesEngine, sim: Simulator, heldout: dict[str, list[str]], oracle: np.ndarray) -> dict[str, Any]:
    """Predicted vs true score and exposure share by candidate religion, plus learned weights."""
    rel = {p.user_id: p.religion for p in sim.profiles}
    pred_by, true_by = {}, {}
    for u, held in heldout.items():
        if not held:
            continue
        pred = engine.predicted_scores(u, held)
        for c, p in zip(held, pred):
            pred_by.setdefault(rel[c], []).append(p)
            true_by.setdefault(rel[c], []).append(oracle[sim.index[u], sim.index[c]] / 100.0)
    exposure = {}
    for uid, count in zip(engine.ids, engine.exposure):
        exposure[rel[uid]] = exposure.get(rel[uid], 0.0) + count
    total = sum(exposure.values()) or 1.0
    pop = {r: sum(1 for p in sim.profiles if p.religion == r) / len(sim.profiles) for r in RELIGIONS}
    rows = [
        {
            "religion": r,
            "population_share": pop[r],
            "exposure_share": exposure.get(r, 0.0) / total,
            "mean_predicted": float(np.mean(pred_by[r])) if r in pred_by else float("nan"),
            "mean_true": float(np.mean(true_by[r])) if r in true_by else float("nan"),
        }
        for r in RELIGIONS
    ]
    names = engine.builder.names
    keep = [i for i, n in enumerate(names) if n == "same_religion" or n.startswith("cand_religion=")]
    pop_w = engine.population.weights if engine.population is not None else np.zeros(len(names))
    user_w = np.stack([engine.user_weights(u) for u in engine.ids])
    weights = [
        {"feature": names[i], "population_weight": float(pop_w[i]), "mean_user_weight": float(user_w[:, i].mean()),
         "sd_user_weight": float(user_w[:, i].std())}
        for i in keep
    ]
    return {"groups": rows, "weights": weights}


def run_experiment(proposal: ProposalConfig, exp: ExperimentConfig = ExperimentConfig()) -> dict[str, Any]:
    """Simulate exp.n_users users through onboarding and bandit rounds, and score the engine."""
    sim = Simulator(replace(exp.sim, n_users=exp.n_users, seed=exp.seed))
    storage = ProposalStorage(":memory:") if exp.use_storage else None
    engine = ReciprocalBayesEngine(sim.profiles, replace(proposal, seed=exp.seed), storage, f"seed-{exp.seed}")
    oracle = np.nan_to_num(sim.oracle_matrix())
    like = np.nan_to_num(sim.like_probability(exp.threshold))
    p_match = like * like.T
    rng = np.random.default_rng(exp.seed + 1)

    heldout: dict[str, list[str]] = {}
    for u in sim.ids:
        eligible = engine.eligible(u)
        n_held = int(round(len(eligible) * exp.holdout_fraction))
        heldout[u] = sorted(rng.choice(eligible, size=n_held, replace=False).tolist()) if n_held else []
        engine.exclude(u, heldout[u])

    start = time.perf_counter()
    order = list(sim.ids)
    ratings: dict[tuple[str, str], float] = {}
    ope_w, ope_wr, ope_truth = [], [], []

    def show(u: str) -> tuple[str, Any] | None:
        choice = engine.pick(u)
        if choice is None:
            return None
        score = sim.rate(u, choice.candidate_id)
        ratings[(u, choice.candidate_id)] = score
        engine.observe(u, choice.candidate_id, score / 100.0, choice)
        return choice.candidate_id, choice

    for _ in range(proposal.n_onboarding):
        rng.shuffle(order)
        for u in order:
            show(u)
        engine.flush()
    engine.retrain()
    after_onboarding = evaluate(engine, sim, heldout, oracle, p_match, exp.k)

    rounds = max(exp.budget - proposal.n_onboarding, 0)
    regret_match, regret_own = np.zeros(rounds), np.zeros(rounds)
    for t in range(rounds):
        rng.shuffle(order)
        for u in order:
            avail = engine.candidates(u)
            if not avail:
                continue
            i, cols = sim.index[u], np.array([sim.index[c] for c in avail])
            shown = show(u)
            if shown is None:
                continue
            c, choice = shown
            j = sim.index[c]
            regret_match[t] += p_match[i, cols].max() - p_match[i, j]
            regret_own[t] += (oracle[i, cols].max() - oracle[i, j]) / 100.0
            # Off-policy check: estimate a uniform-random policy's mean rating from the logs.
            ope_w.append((1.0 / len(avail)) / choice.propensity)
            ope_wr.append(ope_w[-1] * oracle[i, j] / 100.0)
            ope_truth.append(oracle[i, cols].mean() / 100.0)
        engine.flush()
        if (t + 1) % exp.retrain_every == 0 and t + 1 < rounds:
            engine.retrain()
    engine.retrain()
    final = evaluate(engine, sim, heldout, oracle, p_match, exp.k)

    realized = sum(
        1 for (a, b), s in ratings.items() if a < b and s >= exp.threshold and ratings.get((b, a), -1) >= exp.threshold
    )
    n_users = len(sim.ids)
    result: dict[str, Any] = {
        **final,
        **{f"onb_{k}": v for k, v in after_onboarding.items()},
        "cum_regret_match": float(regret_match.sum() / n_users),
        "cum_regret_own": float(regret_own.sum() / n_users),
        "realized_matches_per_user": realized / n_users,
        "exposure_gini": gini(engine.exposure),
        "mean_rating": float(np.mean(list(ratings.values()))),
        "prior_scale": engine.prior_scale,
        "noise_scale": engine.noise_scale,
        "seconds": time.perf_counter() - start,
        "regret_match_curve": (np.cumsum(regret_match) / n_users).tolist(),
        "regret_own_curve": (np.cumsum(regret_own) / n_users).tolist(),
        "audit": religion_audit(engine, sim, heldout, oracle),
    }
    if ope_w:
        w, wr = np.array(ope_w), np.array(ope_wr)
        result["ope_uniform_snips"] = float(wr.sum() / w.sum())
        result["ope_uniform_truth"] = float(np.mean(ope_truth))
    if storage is not None:
        rebuilt = storage.rebuild_user_states(engine.session_id, engine.builder, engine.profiles)
        cached = storage.load_user_states(engine.session_id, engine.dim)
        result["cache_rebuild_max_abs_diff"] = max(
            (float(np.abs(rebuilt[u].P - cached[u].P).max() + np.abs(rebuilt[u].b - cached[u].b).max()) for u in cached),
            default=0.0,
        )
        storage.close()
    return result


# Ablations

DEFAULT = ProposalConfig()
BASELINES = {
    "random": ProposalConfig(scorer="random", onboarding="random", exposure_cap=None, exposure_penalty=0.0),
    "popularity": ProposalConfig(scorer="popularity", onboarding="random", exposure_cap=None, exposure_penalty=0.0),
    "global model": ProposalConfig(scorer="global", onboarding="random", exposure_cap=None, exposure_penalty=0.0),
    "per-user, no prior": replace(DEFAULT, use_population_prior=False),
    "proposal (default)": DEFAULT,
}
ABLATIONS: dict[str, dict[str, ProposalConfig]] = {
    "baselines": BASELINES,
    "onboarding_size": {f"{n}": replace(DEFAULT, n_onboarding=n) for n in (5, 10, 15, 25)},
    "onboarding_strategy": {
        "random": replace(DEFAULT, onboarding="random"),
        "d-optimal": DEFAULT,
        "d-optimal + 4 anchors": replace(DEFAULT, n_anchors=4),
    },
    "policy": {"thompson": DEFAULT, "ucb": replace(DEFAULT, policy="ucb")},
    "population_prior": {"with prior": DEFAULT, "without prior": replace(DEFAULT, use_population_prior=False)},
    "calibration": {
        name: replace(DEFAULT, calibration=name)
        for name in ("none", "zscore", "shrunk_intercept", "shrunk_intercept_scale", "pairwise")
    },
    "reciprocal": {m: replace(DEFAULT, reciprocal=m) for m in ("harmonic", "geometric", "min")},
    "exposure_cap": {"with cap": DEFAULT, "without cap": replace(DEFAULT, exposure_cap=None, exposure_penalty=0.0)},
}
SCALAR_METRICS = (
    "spearman", "ndcg10", "precision_at_k", "oracle_precision_at_k", "mutual_match_rate", "reciprocal_ndcg10",
    "cum_regret_match", "cum_regret_own", "realized_matches_per_user", "exposure_gini", "topk_gini",
    "onb_spearman", "onb_ndcg10", "onb_precision_at_k", "mean_rating", "prior_scale", "noise_scale",
    "ope_uniform_snips", "ope_uniform_truth", "cache_rebuild_max_abs_diff", "seconds",
)


def _job(args: tuple[str, str, ProposalConfig, ExperimentConfig]) -> tuple[str, str, int, dict[str, Any]]:
    ablation, label, proposal, exp = args
    return ablation, label, exp.seed, run_experiment(proposal, exp)


def run_suite(
    ablations: Sequence[str], seeds: Sequence[int], exp: ExperimentConfig, workers: int
) -> list[tuple[str, str, int, dict[str, Any]]]:
    """Run each (ablation, label, seed) once. Identical configs are run once and shared."""
    unique: dict[tuple[ProposalConfig, int], tuple[str, str, ProposalConfig, ExperimentConfig]] = {}
    wanted = []
    for ablation in ablations:
        for label, proposal in ABLATIONS[ablation].items():
            for seed in seeds:
                wanted.append((ablation, label, proposal, seed))
                unique.setdefault((proposal, seed), (ablation, label, proposal, replace(exp, seed=seed)))
    results: dict[tuple[ProposalConfig, int], dict[str, Any]] = {}
    jobs = list(unique.items())
    print(f"{len(wanted)} result rows from {len(jobs)} distinct runs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, job): key for key, job in jobs}
        for n, (future, key) in enumerate(futures.items(), 1):
            results[key] = future.result()[3]
            print(f"  [{n}/{len(jobs)}] {key[0].policy_id} seed={key[1]} ({results[key]['seconds']:.0f}s)", flush=True)
    return [(a, l, s, results[(p, s)]) for a, l, p, s in wanted]


def summarize(rows: list[tuple[str, str, int, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Mean and SD across seeds, per ablation and label, keeping the label order."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for ablation, label, _, result in rows:
        grouped.setdefault(ablation, {}).setdefault(label, []).append(result)
    out: dict[str, list[dict[str, Any]]] = {}
    for ablation, labels in grouped.items():
        out[ablation] = []
        for label, results in labels.items():
            row: dict[str, Any] = {"label": label, "n_seeds": len(results)}
            for m in SCALAR_METRICS:
                values = [r[m] for r in results if m in r]
                if values:
                    row[m] = float(np.mean(values))
                    row[f"{m}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            out[ablation].append(row)
    return out


def write_outputs(rows: list[tuple[str, str, int, dict[str, Any]]], out: Path) -> dict[str, list[dict[str, Any]]]:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ablation", "label", "seed", *SCALAR_METRICS])
        for ablation, label, seed, r in rows:
            writer.writerow([ablation, label, seed, *[r.get(m, "") for m in SCALAR_METRICS]])
    summary = summarize(rows)
    for ablation, table in summary.items():
        with open(out / f"summary_{ablation}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    curves = {
        f"{a}|{l}|{s}": {"match": r["regret_match_curve"], "own": r["regret_own_curve"]} for a, l, s, r in rows
    }
    (out / "regret_curves.json").write_text(json.dumps(curves))
    default_audits = [r["audit"] for a, l, _, r in rows if a == "baselines" and l == "proposal (default)"]
    if default_audits:
        write_audit(default_audits, out)
    return summary


def write_audit(audits: list[dict[str, Any]], out: Path) -> None:
    """Average the religion audit over seeds, and save it as CSVs."""
    groups = {}
    for audit in audits:
        for row in audit["groups"]:
            groups.setdefault(row["religion"], []).append(row)
    with open(out / "religion_audit_groups.csv", "w", newline="") as f:
        fields = ["religion", "population_share", "exposure_share", "mean_predicted", "mean_true"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for religion, rs in groups.items():
            writer.writerow({"religion": religion, **{k: float(np.nanmean([r[k] for r in rs])) for k in fields[1:]}})
    weights = {}
    for audit in audits:
        for row in audit["weights"]:
            weights.setdefault(row["feature"], []).append(row)
    with open(out / "religion_audit_weights.csv", "w", newline="") as f:
        fields = ["feature", "population_weight", "mean_user_weight", "sd_user_weight"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for feature, rs in weights.items():
            writer.writerow({"feature": feature, **{k: float(np.mean([r[k] for r in rs])) for k in fields[1:]}})


def format_summary(summary: dict[str, list[dict[str, Any]]]) -> str:
    """Markdown tables of the headline metrics, mean ± SD across seeds."""
    cols = [
        ("spearman", "Spearman"), ("ndcg10", "NDCG@10"), ("precision_at_k", "P@10 match"),
        ("mutual_match_rate", "Match rate"), ("cum_regret_match", "Regret (match)"),
        ("realized_matches_per_user", "Matches/user"), ("exposure_gini", "Exposure Gini"),
    ]
    lines = []
    for ablation, table in summary.items():
        lines += [f"\n### {ablation}\n", "| config | " + " | ".join(c for _, c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for row in table:
            cells = [f"{row[m]:.3f} ± {row[m + '_sd']:.3f}" if m in row else "" for m, _ in cols]
            lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run proposal 3's baselines and ablations on the simulator.")
    parser.add_argument("--out", default=str(_ROOT / "exports" / "other_proposal"))
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS), choices=list(ABLATIONS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--users", type=int, default=ExperimentConfig.n_users)
    parser.add_argument("--budget", type=int, default=ExperimentConfig.budget)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--quick", action="store_true", help="60 users, 1 seed, baselines only.")
    args = parser.parse_args()
    if args.quick:
        args.users, args.seeds, args.ablations = 60, 1, ["baselines"]
    exp = ExperimentConfig(n_users=args.users, budget=args.budget)
    rows = run_suite(args.ablations, list(range(args.seeds)), exp, args.workers)
    out = Path(args.out)
    summary = write_outputs(rows, out)
    try:
        from matching_algorithms.other_proposal_plots import make_figures
    except ImportError:
        make_figures = None
    if make_figures is not None:
        make_figures(out)
    text = format_summary(summary)
    (out / "summary.md").write_text(text + "\n")
    print(text)
    print(f"\nSaved to {out}")
    print("Experiment settings:", json.dumps({**asdict(exp), "sim": asdict(exp.sim)}))


if __name__ == "__main__":
    main()
