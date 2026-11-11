"""Proposal 3: per-user Bayesian linear models with a population prior, D-optimal
onboarding, bandit exploration, and reciprocal matching.

How it differs from proposals 1 and 2 (matching_proposals.py), which train one neural
network on everybody's ratings:

* Every user gets their own small, closed-form model of their taste,
  rating(A -> B) ~ theta_A . phi(A, B), with a Gaussian posterior over theta_A. A pooled
  ridge regression on everyone's calibrated ratings gives the prior mean, so a new user
  starts at "what people like on average" and moves away with every rating.
* The posterior's uncertainty drives everything else: which profiles a new user rates
  first (greedy D-optimal design), how the feed explores (UCB or Thompson sampling), and
  how cautious the matcher is (mean minus beta * std).
* Reciprocal matching combines s(A -> B) with s(B -> A) from B's own model, using a
  harmonic, geometric, or min combination.

Sections (in the order of the spec's modules):
    1. Profiles and hard filters    5. Calibration          9. Storage (SQLite)
    2. Features phi(A, B)           6. Onboarding          10. The engine
    3. Bayesian user models         7. Policies            11. Matcher adapter for run_simulation
    4. Population model             8. Reciprocal matching

The simulator, metrics, baselines, and ablations live in other_proposal_sim.py. This
file never imports it, so the algorithm can't reach the simulator's oracle.

Ratings are on [0, 1] inside this file. The SQLite layer stores them as 0-100 to match
the spec, and run_simulation's scores are already on [0, 1].
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from matching import Matcher, PublicView, is_compatible  # noqa: E402
from models import PublicProfile, Rating  # noqa: E402

_EPS = 1e-6


# 1. Profiles and hard filters

# The only looking_for combination treated as incompatible: one person wants a
# relationship and the other explicitly wants something short. "just a date" is
# compatible with both, because a first date is how either one starts.
INCOMPATIBLE_LOOKING_FOR: frozenset[frozenset[str]] = frozenset(
    {frozenset({"long-term", "short-term"})}
)


def looking_for_compatible(a: str, b: str) -> bool:
    """Whether two looking_for values can be matched. The rule is symmetric."""
    return frozenset({a, b}) not in INCOMPATIBLE_LOOKING_FOR


def is_eligible(
    viewer: PublicProfile, candidate: PublicProfile, check_looking_for: bool = True
) -> bool:
    """The hard filter: not self, mutual orientation match, compatible looking_for.

    The filter is symmetric, so B is eligible for A exactly when A is eligible for B.
    """
    if viewer.user_id == candidate.user_id or not is_compatible(viewer, candidate):
        return False
    return not check_looking_for or looking_for_compatible(viewer.looking_for, candidate.looking_for)


def eligible_candidates(
    viewer: PublicProfile,
    profiles: Iterable[PublicProfile],
    already_rated: Iterable[str] = (),
    excluded: Iterable[str] = (),
    check_looking_for: bool = True,
) -> list[str]:
    """IDs of profiles the viewer may be shown, sorted so results don't depend on order."""
    skip = set(already_rated) | set(excluded)
    return sorted(
        p.user_id
        for p in profiles
        if p.user_id not in skip and is_eligible(viewer, p, check_looking_for)
    )


def load_profiles(main_db: str | Path) -> list[PublicProfile]:
    """Read every profile from a main.db made by store.py, without the agent flag."""
    from store import Store

    return [
        PublicProfile.model_validate(p.model_dump(exclude={"agent"}))
        for p in Store(main_db).list_all_profiles()
    ]


# 2. Features phi(A, B)

# Majors and jobs are grouped, so each group has enough profiles to learn a weight.
MAJOR_GROUPS: dict[str, str] = {
    **dict.fromkeys(
        [
            "biochemistry", "biology", "biomedical engineering", "chemistry",
            "computer engineering", "computer science", "data science",
            "electrical engineering", "environmental analysis", "mathematics",
            "mechanical engineering", "neuroscience", "physics",
        ],
        "stem",
    ),
    **dict.fromkeys(
        ["health science", "nutrition", "public health", "speech pathology"], "health"
    ),
    **dict.fromkeys(
        [
            "accounting", "business administration", "economics", "finance",
            "hospitality administration", "marketing",
        ],
        "business",
    ),
    **dict.fromkeys(
        [
            "anthropology", "archaeology", "education", "international relations",
            "linguistics", "political science", "psychology", "sociology",
        ],
        "social_science",
    ),
    **dict.fromkeys(
        [
            "advertising", "art history", "english", "film and television", "history",
            "journalism", "music", "philosophy", "public relations", "theatre arts",
            "visual arts",
        ],
        "arts_humanities",
    ),
}
MAJOR_GROUP_VOCAB = ("stem", "health", "business", "social_science", "arts_humanities")

JOB_GROUPS: dict[str, str] = {
    **dict.fromkeys(["tutor", "research assistant", "teaching assistant"], "academic"),
    **dict.fromkeys(["server", "bartender", "front desk", "on-campus work"], "service"),
    "intern": "professional",
}
JOB_GROUP_VOCAB = ("academic", "service", "professional")

OTHER = "other"

# The fixed order of the pair features that come before the one-hot blocks.
PAIR_FEATURES = (
    "intercept",
    "cand_age",          # Candidate's age, standardized.
    "cand_height",       # Candidate's height, standardized.
    "age_diff",          # (B - A) / sd. Antisymmetric.
    "abs_age_diff",      # |B - A| / sd. Symmetric.
    "height_diff",       # (B - A) / sd. Antisymmetric.
    "same_religion",     # Symmetric, as are all the matches and overlaps below.
    "same_school",
    "same_major",
    "same_looking_for",
    "interest_jaccard",
    "club_jaccard",
)
SYMMETRIC_FEATURES = (
    "intercept", "abs_age_diff", "same_religion", "same_school", "same_major",
    "same_looking_for", "interest_jaccard", "club_jaccard",
)
ANTISYMMETRIC_FEATURES = ("age_diff", "height_diff")


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|a & b| / |a | b|, and 0 when both are empty. Two people in no clubs share nothing."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass(frozen=True)
class ProfileCode:
    """A profile's precomputed parts, so pair features can be built quickly."""

    age: float
    height: float
    religion: str
    school: str
    major: str
    looking_for: str
    interests: frozenset[str]
    clubs: frozenset[str]
    block: np.ndarray  # The candidate-only features: standardized age, height, one-hots.


@dataclass(frozen=True)
class FeatureBuilder:
    """Builds phi(A, B), the features of "A looking at B", in a fixed, documented order.

    The order is PAIR_FEATURES, then one-hot candidate religion, major group, job group,
    and school. Each one-hot block ends with an "other" slot for values outside the
    vocabulary, including values never seen when the vocabulary was built.

    The vocabularies and the standardization come from the dataset passed to fit, and
    then stay fixed, so a model's weights keep their meaning.
    """

    religion_vocab: tuple[str, ...]
    school_vocab: tuple[str, ...]
    age_mean: float
    age_std: float
    height_mean: float
    height_std: float

    @classmethod
    def fit(cls, profiles: Sequence[PublicProfile], min_count: int = 1) -> FeatureBuilder:
        """Build the vocabularies and standardization from a dataset."""
        if not profiles:
            raise ValueError("need at least one profile to build vocabularies")

        def vocab(values: list[str]) -> tuple[str, ...]:
            counts: dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            return tuple(sorted(v for v, c in counts.items() if c >= min_count and v != OTHER))

        ages = np.array([p.age for p in profiles], dtype=float)
        heights = np.array([p.height_cm for p in profiles], dtype=float)
        return cls(
            religion_vocab=vocab([p.religion for p in profiles]),
            school_vocab=vocab([p.school for p in profiles]),
            age_mean=float(ages.mean()),
            age_std=float(ages.std()) or 1.0,
            height_mean=float(heights.mean()),
            height_std=float(heights.std()) or 1.0,
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Feature names, in the order build_features returns them."""
        return (
            PAIR_FEATURES
            + tuple(f"cand_religion={v}" for v in self.religion_vocab + (OTHER,))
            + tuple(f"cand_major_group={v}" for v in MAJOR_GROUP_VOCAB + (OTHER,))
            + tuple(f"cand_job_group={v}" for v in JOB_GROUP_VOCAB + (OTHER,))
            + tuple(f"cand_school={v}" for v in self.school_vocab + (OTHER,))
        )

    @property
    def dim(self) -> int:
        return len(self.names)

    def encode(self, profile: PublicProfile) -> ProfileCode:
        """Precompute the parts of a profile that don't depend on the other person."""

        def one_hot(value: str, vocab: tuple[str, ...]) -> np.ndarray:
            out = np.zeros(len(vocab) + 1)
            out[vocab.index(value) if value in vocab else len(vocab)] = 1.0
            return out

        block = np.concatenate(
            [
                [
                    (profile.age - self.age_mean) / self.age_std,
                    (profile.height_cm - self.height_mean) / self.height_std,
                ],
                one_hot(profile.religion, self.religion_vocab),
                one_hot(MAJOR_GROUPS.get(profile.major, OTHER), MAJOR_GROUP_VOCAB),
                one_hot(JOB_GROUPS.get(profile.job, OTHER), JOB_GROUP_VOCAB),
                one_hot(profile.school, self.school_vocab),
            ]
        )
        return ProfileCode(
            age=float(profile.age),
            height=float(profile.height_cm),
            religion=profile.religion,
            school=profile.school,
            major=profile.major,
            looking_for=profile.looking_for,
            interests=frozenset(profile.interests),
            clubs=frozenset(profile.clubs),
            block=block,
        )

    def pair_matrix(self, viewers: Sequence[ProfileCode], candidates: Sequence[ProfileCode]) -> np.ndarray:
        """Row i is phi(viewers[i], candidates[i]). Both lists must have the same length."""
        if len(viewers) != len(candidates):
            raise ValueError("viewers and candidates must be the same length")
        n = len(viewers)
        out = np.empty((n, self.dim))
        if n == 0:
            return out
        va = np.array([v.age for v in viewers])
        ca = np.array([c.age for c in candidates])
        vh = np.array([v.height for v in viewers])
        ch = np.array([c.height for c in candidates])
        blocks = np.stack([c.block for c in candidates])
        out[:, 0] = 1.0
        out[:, 1:3] = blocks[:, :2]
        out[:, 3] = (ca - va) / self.age_std
        out[:, 4] = np.abs(ca - va) / self.age_std
        out[:, 5] = (ch - vh) / self.height_std
        for i, (v, c) in enumerate(zip(viewers, candidates)):
            out[i, 6] = v.religion == c.religion
            out[i, 7] = v.school == c.school
            out[i, 8] = v.major == c.major
            out[i, 9] = v.looking_for == c.looking_for
            out[i, 10] = _jaccard(v.interests, c.interests)
            out[i, 11] = _jaccard(v.clubs, c.clubs)
        out[:, len(PAIR_FEATURES):] = blocks[:, 2:]
        return out

    def build_features(self, viewer: PublicProfile, candidate: PublicProfile) -> np.ndarray:
        """phi(viewer, candidate), a vector of length dim in the order of names."""
        return self.pair_matrix([self.encode(viewer)], [self.encode(candidate)])[0]

    def build_matrix(self, viewer: PublicProfile, candidates: Sequence[PublicProfile]) -> np.ndarray:
        """One row of phi(viewer, candidate) per candidate."""
        code = self.encode(viewer)
        return self.pair_matrix([code] * len(candidates), [self.encode(c) for c in candidates])


# 3. Bayesian user models


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


class GaussianPosterior(ABC):
    """A Gaussian posterior N(mean, P^-1) over a weight vector, plus the scores built on it.

    Subclasses provide the precision P and how to get the mean. The Cholesky factor
    P = L L^T is computed once per version and reused. The covariance is never formed by
    inverting P directly: variances come from ||L^-1 phi||^2, and samples from
    mean + L^-T z.
    """

    def __init__(self) -> None:
        self.version = 0
        self._cached_version = -1
        self._L: np.ndarray | None = None
        self._Linv: np.ndarray | None = None
        self._mean: np.ndarray | None = None

    @property
    @abstractmethod
    def precision(self) -> np.ndarray:
        """The posterior precision matrix P."""

    @abstractmethod
    def _solve_mean(self, L: np.ndarray) -> np.ndarray:
        """The posterior mean, given the Cholesky factor of P."""

    def _ensure(self) -> None:
        if self._cached_version != self.version:
            L = np.linalg.cholesky(self.precision)
            self._L = L
            self._Linv = solve_triangular(L, np.eye(len(L)), lower=True)
            self._mean = self._solve_mean(L)
            self._cached_version = self.version

    @property
    def mean(self) -> np.ndarray:
        self._ensure()
        return self._mean

    @property
    def chol_inv(self) -> np.ndarray:
        """L^-1, where P = L L^T. The covariance is chol_inv.T @ chol_inv."""
        self._ensure()
        return self._Linv

    @property
    def covariance(self) -> np.ndarray:
        Linv = self.chol_inv
        return Linv.T @ Linv

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        """Posterior mean score for each row of Phi."""
        return np.atleast_2d(Phi) @ self.mean

    def predictive_std(self, Phi: np.ndarray) -> np.ndarray:
        """Posterior std of phi . theta for each row (uncertainty in theta, without noise)."""
        V = self.chol_inv @ np.atleast_2d(Phi).T
        return np.sqrt(np.einsum("ij,ij->j", V, V))

    def ucb_scores(self, Phi: np.ndarray, alpha: float) -> np.ndarray:
        return self.predict(Phi) + alpha * self.predictive_std(Phi)

    def conservative_scores(self, Phi: np.ndarray, beta: float) -> np.ndarray:
        return self.predict(Phi) - beta * self.predictive_std(Phi)

    def sample_weights(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draws from the posterior, shape (size, d): mean + L^-T z."""
        self._ensure()
        z = rng.standard_normal((len(self._mean), size))
        return (self._mean[:, None] + solve_triangular(self._L.T, z, lower=False)).T

    def thompson_scores(self, Phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Scores under one posterior draw of theta."""
        return np.atleast_2d(Phi) @ self.sample_weights(rng, 1)[0]


class BayesianUserModel(GaussianPosterior):
    """Bayesian linear regression for one user: rating ~ N(theta . phi, sigma2).

    The prior is theta ~ N(prior_mean, prior_precision^-1). The posterior is kept in
    natural parameters, P (precision) and b (precision times mean), so an update is
    P += phi phi^T / sigma2 and b += phi * rating / sigma2, and the mean is P^-1 b.
    """

    def __init__(self, prior_mean: np.ndarray, prior_precision: np.ndarray, sigma2: float) -> None:
        super().__init__()
        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive")
        self.prior_mean = np.asarray(prior_mean, dtype=float).copy()
        self.prior_precision = np.asarray(prior_precision, dtype=float).copy()
        self.sigma2 = float(sigma2)
        self.P = self.prior_precision.copy()
        self.b = self.prior_precision @ self.prior_mean
        self.n_ratings = 0

    @classmethod
    def from_data(
        cls,
        prior_mean: np.ndarray,
        prior_precision: np.ndarray,
        sigma2: float,
        XtX: np.ndarray,
        Xty: np.ndarray,
        n_ratings: int,
    ) -> BayesianUserModel:
        """The posterior after ratings with sufficient statistics X^T X and X^T y."""
        model = cls(prior_mean, prior_precision, sigma2)
        model.P = model.P + XtX / model.sigma2
        model.b = model.b + Xty / model.sigma2
        model.n_ratings = n_ratings
        model.version += 1
        return model

    @property
    def precision(self) -> np.ndarray:
        return self.P

    def _solve_mean(self, L: np.ndarray) -> np.ndarray:
        return solve_triangular(L.T, solve_triangular(L, self.b, lower=True), lower=False)

    def update(self, phi: np.ndarray, rating: float) -> None:
        """Condition on one rating, on the [0, 1] scale."""
        phi = np.asarray(phi, dtype=float)
        self.P += np.outer(phi, phi) / self.sigma2
        self.b += phi * float(rating) / self.sigma2
        self.n_ratings += 1
        self.version += 1


class PairwiseUserModel(GaussianPosterior):
    """Bradley-Terry model for one user, with a Laplace approximation to its posterior.

    Each pair of the user's ratings with r_i > r_j + margin says "i was preferred to j",
    with P(i over j) = sigmoid(theta . (phi_i - phi_j)). The user's offset and scale
    cancel in the difference, which is the point of pairwise calibration. Each pair is
    weighted by n / n_pairs, so n ratings count as n observations rather than n^2.

    The posterior mode comes from Newton's method, and its precision is the Hessian there.
    Scores are utilities in logit units, not ratings. The intercept cancels in every
    difference, so its posterior stays at the prior.
    """

    def __init__(self, prior_mean: np.ndarray, prior_precision: np.ndarray) -> None:
        super().__init__()
        self.prior_mean = np.asarray(prior_mean, dtype=float).copy()
        self.prior_precision = np.asarray(prior_precision, dtype=float).copy()
        self.theta = self.prior_mean.copy()
        self.H = self.prior_precision.copy()
        self.n_ratings = 0

    @property
    def precision(self) -> np.ndarray:
        return self.H

    def _solve_mean(self, L: np.ndarray) -> np.ndarray:
        return self.theta

    def fit(self, Phi: np.ndarray, ratings: np.ndarray, margin: float) -> PairwiseUserModel:
        """Fit to all of this user's ratings, replacing any earlier fit."""
        diffs, weights = preference_pairs(Phi, ratings, margin)
        self.theta, self.H = fit_pairwise_logistic(diffs, weights, self.prior_mean, self.prior_precision)
        self.n_ratings = len(ratings)
        self.version += 1
        return self


def preference_pairs(Phi: np.ndarray, ratings: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """phi_i - phi_j for every pair with r_i > r_j + margin, and each pair's weight."""
    ratings = np.asarray(ratings, dtype=float)
    i, j = np.nonzero(ratings[:, None] > ratings[None, :] + margin)
    if len(i) == 0:
        return np.zeros((0, Phi.shape[1])), np.zeros(0)
    return Phi[i] - Phi[j], np.full(len(i), len(ratings) / len(i))


def fit_pairwise_logistic(
    diffs: np.ndarray,
    weights: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """MAP of sum_k w_k log sigmoid(theta . d_k) with a Gaussian prior, by Newton's method.

    Returns the mode and the Hessian of the negative log posterior there.
    """
    theta = np.asarray(prior_mean, dtype=float).copy()
    H = prior_precision.copy()
    for _ in range(max_iter):
        p = _sigmoid(diffs @ theta)
        grad = diffs.T @ (weights * (p - 1.0)) + prior_precision @ (theta - prior_mean)
        H = (diffs * (weights * p * (1.0 - p))[:, None]).T @ diffs + prior_precision
        step = np.linalg.solve(H, grad)
        theta = theta - step
        if np.linalg.norm(step) < tol:
            break
    p = _sigmoid(diffs @ theta)
    H = (diffs * (weights * p * (1.0 - p))[:, None]).T @ diffs + prior_precision
    return theta, H


# 4. Population model


@dataclass(frozen=True)
class PopulationModel:
    """Population-level weights, trained offline on everyone's pooled, calibrated ratings.

    For a regression calibrator, it's a ridge regression and sigma2 is its residual
    variance. For the pairwise calibrator, it's a pooled Bradley-Terry model. Every
    retrain gets a new version, which is saved with the weights.
    """

    weights: np.ndarray
    sigma2: float
    kind: Literal["regression", "pairwise"]
    version: int
    n_ratings: int
    alpha: float

    @classmethod
    def fit_regression(cls, Phi: np.ndarray, y: np.ndarray, alpha: float, version: int) -> PopulationModel:
        """Ridge regression. Column 0 of Phi is the intercept and isn't penalized."""
        ridge = Ridge(alpha=alpha, fit_intercept=True).fit(Phi[:, 1:], y)
        weights = np.concatenate([[ridge.intercept_], ridge.coef_])
        resid = y - Phi @ weights
        dof = max(len(y) - Phi.shape[1], 1)
        return cls(weights, float(resid @ resid / dof), "regression", version, len(y), alpha)

    @classmethod
    def fit_pairwise(
        cls, diffs: np.ndarray, weights: np.ndarray, dim: int, alpha: float, version: int, n_ratings: int
    ) -> PopulationModel:
        theta, _ = fit_pairwise_logistic(diffs, weights, np.zeros(dim), alpha * np.eye(dim))
        return cls(theta, 1.0, "pairwise", version, n_ratings, alpha)


def fit_noise_and_prior_scale(
    datasets: Sequence[tuple[np.ndarray, np.ndarray]],
    prior_mean: np.ndarray,
    tau_grid: Sequence[float] = tuple(np.geomspace(0.01, 1.0, 13)),
    sigma_grid: Sequence[float] = tuple(np.geomspace(0.02, 0.5, 13)),
) -> tuple[float, float]:
    """Empirical Bayes: the (tau, sigma) on a grid that maximizes the marginal likelihood.

    Each user's ratings are y_u ~ N(Phi_u m0, sigma^2 I + tau^2 Phi_u Phi_u^T), where tau
    is the prior std of how far a user's weights sit from the population's. Users with
    the same number of ratings are scored together, as a batch.
    """
    by_n: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for Phi, y in datasets:
        if len(y):
            by_n.setdefault(len(y), []).append((Phi, y))
    if not by_n:
        raise ValueError("no ratings to fit hyperparameters on")
    groups = []
    for n, items in by_n.items():
        Phis = np.stack([Phi for Phi, _ in items])
        resid = np.stack([y - Phi @ prior_mean for Phi, y in items])
        groups.append((n, np.einsum("uid,ujd->uij", Phis, Phis), resid))

    best, best_ll = (float(tau_grid[0]), float(sigma_grid[0])), -np.inf
    for tau in tau_grid:
        for sigma in sigma_grid:
            ll = 0.0
            for n, G, resid in groups:
                C = sigma**2 * np.eye(n) + tau**2 * G
                L = np.linalg.cholesky(C)
                z = np.linalg.solve(L, resid[..., None])[..., 0]
                logdet = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum()
                ll += -0.5 * (np.sum(z**2) + logdet + resid.size * math.log(2 * math.pi))
            if ll > best_ll:
                best, best_ll = (float(tau), float(sigma)), ll
    return best


# 5. Calibration

UserRatings = Sequence[tuple[str, float]]  # (ratee_id, rating on [0, 1]) for one rater.


class Calibrator(ABC):
    """Removes rating-scale differences between users.

    Regression calibrators are affine per user: calibrated = a * raw + c. Because the map
    is affine, a user's calibrated sufficient statistics come straight from the raw ones
    (X^T y_cal = a X^T y + c sum(phi)), so the cached state never has to be rebuilt when
    the calibration changes. The pairwise calibrator works on preference pairs instead.
    """

    name: str
    kind: Literal["regression", "pairwise"] = "regression"

    def __init__(self) -> None:
        self.pop_mean = 0.5
        self.within_sd = 0.2

    def fit_population(self, ratings_by_user: Mapping[str, UserRatings]) -> None:
        """Estimate population-level quantities from everyone's raw ratings."""
        scores = [r for ratings in ratings_by_user.values() for _, r in ratings]
        if scores:
            self.pop_mean = float(np.mean(scores))
        sq, dof = 0.0, 0
        for ratings in ratings_by_user.values():
            if len(ratings) >= 2:
                r = np.array([s for _, s in ratings])
                sq += float(((r - r.mean()) ** 2).sum())
                dof += len(r) - 1
        if dof:
            self.within_sd = max(math.sqrt(sq / dof), 0.01)

    @abstractmethod
    def affine(self, ratings: UserRatings) -> tuple[float, float]:
        """(a, c) for this user, from their raw ratings."""


class NoCalibration(Calibrator):
    name = "none"

    def affine(self, ratings: UserRatings) -> tuple[float, float]:
        return 1.0, 0.0


class ZScoreCalibrator(Calibrator):
    """calibrated = pop_mean + within_sd * (r - user_mean) / user_sd, with no shrinkage."""

    name = "zscore"

    def affine(self, ratings: UserRatings) -> tuple[float, float]:
        if len(ratings) < 2:
            return 1.0, 0.0
        r = np.array([s for _, s in ratings])
        sd = max(float(r.std(ddof=1)), 0.02)
        a = self.within_sd / sd
        return a, self.pop_mean - a * float(r.mean())


class ShrunkInterceptCalibrator(Calibrator):
    """Removes each user's offset, estimated as a Bayesian average (the default).

    offset = (n * user_mean + k * pop_mean) / (n + k) - pop_mean, so a user with few
    ratings keeps most of their raw scale. With shrink_scale, the user's spread is
    shrunk toward the population's in the same way and divided out.

    With anchors, the offset comes from the user's ratings of the shared anchor profiles,
    compared with everyone else's ratings of the same profiles. That doesn't suffer from
    selection bias, which the plain mean does once the feed shows people what they like.
    """

    name = "shrunk_intercept"

    def __init__(self, k: float = 5.0, shrink_scale: bool = False, anchors: Iterable[str] = ()) -> None:
        super().__init__()
        self.k = k
        self.shrink_scale = shrink_scale
        self.anchors = frozenset(anchors)
        self.anchor_means: dict[str, float] = {}
        if shrink_scale:
            self.name = "shrunk_intercept_scale"

    def fit_population(self, ratings_by_user: Mapping[str, UserRatings]) -> None:
        super().fit_population(ratings_by_user)
        if self.anchors:
            sums: dict[str, list[float]] = {}
            for ratings in ratings_by_user.values():
                for ratee, r in ratings:
                    if ratee in self.anchors:
                        sums.setdefault(ratee, []).append(r)
            self.anchor_means = {a: float(np.mean(v)) for a, v in sums.items()}

    def _offset(self, ratings: UserRatings) -> float:
        anchored = [r - self.anchor_means[t] for t, r in ratings if t in self.anchor_means]
        if anchored:
            return sum(anchored) / (len(anchored) + self.k)
        n = len(ratings)
        if n == 0:
            return 0.0
        mean = sum(r for _, r in ratings) / n
        return (n * mean + self.k * self.pop_mean) / (n + self.k) - self.pop_mean

    def affine(self, ratings: UserRatings) -> tuple[float, float]:
        offset = self._offset(ratings)
        if not self.shrink_scale or len(ratings) < 2:
            return 1.0, -offset
        r = np.array([s for _, s in ratings])
        n = len(r)
        var = ((n - 1) * r.var(ddof=1) + self.k * self.within_sd**2) / (n - 1 + self.k)
        a = self.within_sd / max(math.sqrt(var), 0.01)
        # Center on the shrunk user mean, then rescale: pop_mean + a * (r - (pop_mean + offset)).
        return a, self.pop_mean - a * (self.pop_mean + offset)


class PairwiseCalibrator(Calibrator):
    """Bradley-Terry: models only which of two profiles a user preferred."""

    name = "pairwise"
    kind = "pairwise"

    def affine(self, ratings: UserRatings) -> tuple[float, float]:
        return 1.0, 0.0  # Not used: pairwise models never see absolute ratings.


CALIBRATORS = ("none", "zscore", "shrunk_intercept", "shrunk_intercept_scale", "pairwise")


def make_calibrator(name: str, k: float = 5.0, anchors: Iterable[str] = ()) -> Calibrator:
    if name == "none":
        return NoCalibration()
    if name == "zscore":
        return ZScoreCalibrator()
    if name == "shrunk_intercept":
        return ShrunkInterceptCalibrator(k, anchors=anchors)
    if name == "shrunk_intercept_scale":
        return ShrunkInterceptCalibrator(k, shrink_scale=True, anchors=anchors)
    if name == "pairwise":
        return PairwiseCalibrator()
    raise ValueError(f"unknown calibrator {name!r}; choose from {CALIBRATORS}")


# 6. Onboarding


def d_optimal_onboarding(
    Phi: np.ndarray, prior_precision: np.ndarray, sigma2: float, n_select: int, forced: Sequence[int] = ()
) -> list[int]:
    """Greedy D-optimal design: repeatedly pick the row with the largest phi^T Sigma phi.

    After each pick, P += phi phi^T / sigma2. The ratings aren't needed to choose, so a
    user's whole onboarding list can be planned before they rate anything. Each greedy
    step maximizes the gain in log det P, which is what D-optimality asks for.
    forced rows (anchor profiles) are taken first, in order.
    """
    P = np.array(prior_precision, dtype=float)
    chosen: list[int] = []
    for i in forced:
        if len(chosen) < n_select and i not in chosen:
            chosen.append(int(i))
            P += np.outer(Phi[i], Phi[i]) / sigma2
    available = np.ones(len(Phi), dtype=bool)
    available[chosen] = False
    while len(chosen) < min(n_select, len(Phi)):
        L = np.linalg.cholesky(P)
        V = solve_triangular(L, Phi.T, lower=True)
        variance = np.where(available, np.einsum("ij,ij->j", V, V), -np.inf)
        best = int(np.argmax(variance))
        chosen.append(best)
        available[best] = False
        P += np.outer(Phi[best], Phi[best]) / sigma2
    return chosen


def random_onboarding(n_candidates: int, n_select: int, rng: np.random.Generator, forced: Sequence[int] = ()) -> list[int]:
    """Anchors first, then a uniform random sample of the rest."""
    chosen = [int(i) for i in forced][:n_select]
    rest = np.setdiff1d(np.arange(n_candidates), chosen)
    k = min(n_select - len(chosen), len(rest))
    return chosen + [int(i) for i in rng.permutation(rest)[:k]]


# 7. Policies


def exposure_adjust(
    scores: np.ndarray, exposure: np.ndarray, mean_exposure: float, cap: float | None, penalty: float
) -> np.ndarray:
    """Penalize over-exposed candidates, and drop those past the cap (-inf).

    A candidate shown more than cap times the average profile is skipped, unless every
    candidate is past it. Below the cap, each multiple of the average exposure above 1
    costs `penalty`. This keeps a few broadly attractive profiles from taking over every
    feed, which a mean-maximizing policy does on its own.
    """
    if cap is None and penalty == 0:
        return scores
    ratio = exposure / max(mean_exposure, 1.0)
    out = scores - penalty * np.maximum(0.0, ratio - 1.0)
    if cap is not None:
        over = ratio > cap
        if not over.all():
            out = np.where(over, -np.inf, out)
    return out


def _onehot_argmax(scores: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores))
    out[int(np.argmax(scores))] = 1.0
    return out


def mixture_probabilities(
    match_scores: np.ndarray, learn_probs: np.ndarray, explore_share: float, uniform_floor: float
) -> np.ndarray:
    """The feed's probability of showing each candidate.

    pi = u / n + (1 - u) * (explore_share * pi_learn + (1 - explore_share) * pi_match),
    where pi_match is the deterministic argmax of the matching-mode scores. The uniform
    floor u keeps every candidate's propensity above zero, which inverse-propensity
    estimates need.
    """
    n = len(match_scores)
    return uniform_floor / n + (1 - uniform_floor) * (
        explore_share * learn_probs + (1 - explore_share) * _onehot_argmax(match_scores)
    )


def thompson_propensities(scores_by_sample: np.ndarray) -> np.ndarray:
    """Monte Carlo estimate of P(each row is the argmax), from (n, M) sampled scores."""
    winners = np.argmax(scores_by_sample, axis=0)
    return np.bincount(winners, minlength=scores_by_sample.shape[0]) / scores_by_sample.shape[1]


# 8. Reciprocal matching

COMBINATIONS = ("harmonic", "geometric", "min")


def combine(x: np.ndarray, y: np.ndarray, method: str = "harmonic") -> np.ndarray:
    """Combine s(A -> B) and s(B -> A), both on [0, 1], into one compatibility score.

    All three are symmetric and punish one-sided interest. Harmonic sits between the
    others: it's dominated by the smaller score, but still rewards the larger one.
    """
    x = np.clip(np.asarray(x, dtype=float), _EPS, 1.0)
    y = np.clip(np.asarray(y, dtype=float), _EPS, 1.0)
    if method == "harmonic":
        return 2 * x * y / (x + y)
    if method == "geometric":
        return np.sqrt(x * y)
    if method == "min":
        return np.minimum(x, y)
    raise ValueError(f"unknown combination {method!r}; choose from {COMBINATIONS}")


def one_to_one_assignment(
    left: Sequence[str], right: Sequence[str], compat: np.ndarray, eligible: np.ndarray
) -> list[tuple[str, str, float]]:
    """Maximum-total-compatibility one-to-one matching, by the Hungarian algorithm.

    compat[i, j] is the compatibility of left[i] and right[j], and ineligible pairs are
    never matched. The graph has to be bipartite, so call it once per pair of groups
    (for example, women seeking men and men seeking women).
    """
    weights = np.where(eligible, compat, -1e6)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return [
        (left[i], right[j], float(compat[i, j])) for i, j in zip(rows, cols) if eligible[i, j]
    ]


# 9. Storage

STORAGE_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS profiles (
        user_id TEXT PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL,
        height_cm INTEGER NOT NULL, religion TEXT NOT NULL, school TEXT NOT NULL,
        major TEXT NOT NULL, job TEXT NOT NULL, interests TEXT NOT NULL,
        clubs TEXT NOT NULL, gender TEXT NOT NULL, interested_in TEXT NOT NULL,
        looking_for TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ratings (
        rater_id   TEXT NOT NULL REFERENCES profiles (user_id),
        ratee_id   TEXT NOT NULL REFERENCES profiles (user_id),
        score      REAL NOT NULL CHECK (score BETWEEN 0 AND 100),
        timestamp  TEXT NOT NULL,
        phase      TEXT NOT NULL CHECK (phase IN ('onboarding', 'bandit', 'exploit')),
        policy_id  TEXT NOT NULL,
        propensity REAL CHECK (propensity IS NULL OR (propensity > 0 AND propensity <= 1)),
        session_id TEXT NOT NULL,
        PRIMARY KEY (rater_id, ratee_id, session_id),
        CHECK (rater_id <> ratee_id)
    )
    """,
    # A cache of each user's sufficient statistics, rebuildable from ratings.
    """
    CREATE TABLE IF NOT EXISTS user_model_state (
        user_id    TEXT NOT NULL,
        session_id TEXT NOT NULL,
        P          BLOB NOT NULL,
        b          BLOB NOT NULL,
        phi_sum    BLOB NOT NULL,
        r_sum      REAL NOT NULL,
        r_sq_sum   REAL NOT NULL,
        n_ratings  INTEGER NOT NULL,
        version    INTEGER NOT NULL,
        PRIMARY KEY (user_id, session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS population_model (
        version       INTEGER NOT NULL,
        session_id    TEXT NOT NULL,
        kind          TEXT NOT NULL,
        calibrator    TEXT NOT NULL,
        weights       TEXT NOT NULL,
        feature_names TEXT NOT NULL,
        sigma2        REAL NOT NULL,
        prior_scale   REAL NOT NULL,
        alpha         REAL NOT NULL,
        n_ratings     INTEGER NOT NULL,
        trained_at    TEXT NOT NULL,
        PRIMARY KEY (version, session_id)
    )
    """,
)

Phase = Literal["onboarding", "bandit", "exploit"]


@dataclass
class UserState:
    """One user's raw sufficient statistics: the cached part of their model.

    P is X^T X and b is X^T r over the user's ratings, with r raw on [0, 1]. Neither
    includes the prior or the calibration, which are applied when the posterior is
    built. Retraining the population model or swapping the calibrator therefore never
    invalidates this cache. That's a deliberate difference from storing the posterior's
    P and b: these columns are exactly rebuildable from the ratings table.
    """

    user_id: str
    P: np.ndarray
    b: np.ndarray
    phi_sum: np.ndarray
    r_sum: float = 0.0
    r_sq_sum: float = 0.0
    n_ratings: int = 0
    version: int = 0

    @classmethod
    def empty(cls, user_id: str, dim: int) -> UserState:
        return cls(user_id, np.zeros((dim, dim)), np.zeros(dim), np.zeros(dim))

    def add(self, phi: np.ndarray, rating: float) -> None:
        self.P += np.outer(phi, phi)
        self.b += phi * rating
        self.phi_sum += phi
        self.r_sum += rating
        self.r_sq_sum += rating * rating
        self.n_ratings += 1
        self.version += 1

    def calibrated_xty(self, a: float, c: float) -> np.ndarray:
        """X^T y for calibrated ratings y = a * r + c."""
        return a * self.b + c * self.phi_sum


@dataclass(frozen=True)
class RatingLog:
    rater_id: str
    ratee_id: str
    score: float  # 0-100.
    timestamp: str
    phase: Phase
    policy_id: str
    propensity: float | None
    session_id: str


def _blob(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a, dtype=np.float64).tobytes()


def _unblob(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float64).reshape(shape).copy()


class ProposalStorage:
    """SQLite persistence for this proposal. The ratings table is the source of truth.

    It's a separate file from main.db, with the schema the spec asks for (0-100 scores,
    phase, policy, propensity, session). ":memory:" works for experiments.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            for statement in STORAGE_SCHEMA:
                self.conn.execute(statement)

    def close(self) -> None:
        self.conn.close()

    def insert_profiles(self, profiles: Iterable[PublicProfile]) -> None:
        rows = []
        for p in profiles:
            row = p.model_dump()
            row["interests"], row["clubs"] = json.dumps(p.interests), json.dumps(p.clubs)
            rows.append(row)
        if rows:
            cols = list(rows[0])
            with self.conn:
                self.conn.executemany(
                    f"INSERT OR REPLACE INTO profiles ({', '.join(cols)}) "
                    f"VALUES ({', '.join(':' + c for c in cols)})",
                    rows,
                )

    def log_ratings(self, logs: Iterable[RatingLog]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO ratings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (l.rater_id, l.ratee_id, l.score, l.timestamp, l.phase, l.policy_id, l.propensity, l.session_id)
                    for l in logs
                ],
            )

    def ratings(self, session_id: str | None = None) -> list[RatingLog]:
        """Ratings in insertion order, optionally limited to one session."""
        query, args = "SELECT * FROM ratings", ()
        if session_id is not None:
            query, args = query + " WHERE session_id = ?", (session_id,)
        return [RatingLog(**dict(row)) for row in self.conn.execute(query + " ORDER BY rowid", args)]

    def save_user_states(self, states: Iterable[UserState], session_id: str) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO user_model_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (s.user_id, session_id, _blob(s.P), _blob(s.b), _blob(s.phi_sum),
                     s.r_sum, s.r_sq_sum, s.n_ratings, s.version)
                    for s in states
                ],
            )

    def load_user_states(self, session_id: str, dim: int) -> dict[str, UserState]:
        rows = self.conn.execute("SELECT * FROM user_model_state WHERE session_id = ?", (session_id,))
        return {
            row["user_id"]: UserState(
                row["user_id"], _unblob(row["P"], (dim, dim)), _unblob(row["b"], (dim,)),
                _unblob(row["phi_sum"], (dim,)), row["r_sum"], row["r_sq_sum"],
                row["n_ratings"], row["version"],
            )
            for row in rows
        }

    def rebuild_user_states(self, session_id: str, builder: FeatureBuilder, profiles: Mapping[str, PublicProfile]) -> dict[str, UserState]:
        """Recompute every user's state from the ratings table alone."""
        codes = {uid: builder.encode(p) for uid, p in profiles.items()}
        states: dict[str, UserState] = {}
        for log in self.ratings(session_id):
            phi = builder.pair_matrix([codes[log.rater_id]], [codes[log.ratee_id]])[0]
            states.setdefault(log.rater_id, UserState.empty(log.rater_id, builder.dim)).add(phi, log.score / 100.0)
        return states

    def save_population_model(
        self, model: PopulationModel, session_id: str, calibrator: str, feature_names: Sequence[str], prior_scale: float
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO population_model VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model.version, session_id, model.kind, calibrator, json.dumps(model.weights.tolist()),
                    json.dumps(list(feature_names)), model.sigma2, prior_scale, model.alpha,
                    model.n_ratings, datetime.now(timezone.utc).isoformat(),
                ),
            )

    def latest_population_model(self, session_id: str) -> PopulationModel | None:
        row = self.conn.execute(
            "SELECT * FROM population_model WHERE session_id = ? ORDER BY version DESC LIMIT 1", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return PopulationModel(
            np.array(json.loads(row["weights"])), row["sigma2"], row["kind"], row["version"], row["n_ratings"], row["alpha"]
        )


# 10. The engine

Scorer = Literal["bayes", "global", "popularity", "random"]


@dataclass(frozen=True)
class ProposalConfig:
    """Every knob of the proposal. The defaults are the proposal as submitted.

    scorer selects the proposal ("bayes") or a baseline: "global" (the population model
    for everyone, no personalization), "popularity", or "random".
    """

    scorer: Scorer = "bayes"
    n_onboarding: int = 15
    onboarding: Literal["d-optimal", "random"] = "d-optimal"
    n_anchors: int = 0
    policy: Literal["thompson", "ucb"] = "thompson"
    ucb_alpha: float = 1.0
    conservative_beta: float = 1.0
    explore_share: float = 0.15
    uniform_floor: float = 0.02
    mc_samples: int = 100
    use_population_prior: bool = True
    calibration: str = "shrunk_intercept"
    shrinkage_k: float = 5.0
    reciprocal: str = "harmonic"
    exposure_cap: float | None = 3.0
    exposure_penalty: float = 0.05
    check_looking_for: bool = True
    ridge_alpha: float = 1.0
    prior_scale: float | None = None  # tau. None fits it by empirical Bayes at each retrain.
    noise_scale: float | None = None  # sigma. None fits it with tau.
    default_prior_scale: float = 0.1  # Used before the first retrain.
    default_noise_scale: float = 0.15
    pairwise_prior_scale: float = 0.5
    pairwise_margin: float = 0.05
    popularity_k: float = 5.0
    top_k: int = 10
    seed: int = 0
    feature_cache_limit: int = 1500  # Cache every viewer's feature rows up to this many users.

    @property
    def policy_id(self) -> str:
        if self.scorer != "bayes":
            return self.scorer
        return f"{self.policy}-{self.calibration}-{self.reciprocal}"


@dataclass
class Choice:
    """A profile picked for a user, with what to log about it."""

    candidate_id: str
    propensity: float
    phase: Phase
    mode: Literal["onboarding", "learn", "match", "uniform"]


@dataclass
class _Rows:
    """Cached phi(viewer, candidate) for all of a viewer's eligible candidates."""

    ids: list[str]
    Phi: np.ndarray
    pos: dict[str, int] = field(default_factory=dict)


class ReciprocalBayesEngine:
    """The proposal's state and decisions. It only ever sees public profiles and ratings.

    Typical use: pick(user) gives the next profile to show, observe(...) records the
    rating, and retrain() refits the population model and hyperparameters. top_k(user)
    is the ranked output, and one_to_one() the optional Hungarian assignment.
    """

    def __init__(
        self,
        profiles: Sequence[PublicProfile],
        config: ProposalConfig = ProposalConfig(),
        storage: ProposalStorage | None = None,
        session_id: str = "session",
    ) -> None:
        self.config = config
        self.ids = sorted(p.user_id for p in profiles)
        self.index = {uid: i for i, uid in enumerate(self.ids)}
        self.profiles = {p.user_id: p for p in profiles}
        self.builder = FeatureBuilder.fit([self.profiles[u] for u in self.ids])
        self.codes = [self.builder.encode(self.profiles[u]) for u in self.ids]
        self.dim = self.builder.dim
        self.rng = np.random.default_rng(config.seed)
        self.storage = storage
        self.session_id = session_id
        if storage is not None:
            storage.insert_profiles(profiles)

        self.states = {uid: UserState.empty(uid, self.dim) for uid in self.ids}
        self.user_ratings: dict[str, list[tuple[str, float]]] = {uid: [] for uid in self.ids}
        self.rated: dict[str, set[str]] = {uid: set() for uid in self.ids}
        self.excluded: dict[str, set[str]] = {uid: set() for uid in self.ids}
        self._eligible: dict[str, list[str]] = {}

        # Anchors come from one fixed, seeded order, so users with the same orientation and
        # goals share them. The calibrator gets the union of everyone's anchors.
        self.anchor_order = [self.ids[i] for i in np.random.default_rng(config.seed + 7919).permutation(len(self.ids))]
        anchors = self._all_anchors() if config.n_anchors else ()
        self.calibrator = make_calibrator(config.calibration, config.shrinkage_k, anchors)
        self.population: PopulationModel | None = None
        self.prior_scale = config.prior_scale or config.default_prior_scale
        self.noise_scale = config.noise_scale or config.default_noise_scale

        self.received_sum = np.zeros(len(self.ids))
        self.received_n = np.zeros(len(self.ids))
        self.exposure = np.zeros(len(self.ids))
        self._rows: dict[str, _Rows] = {}
        self._onboarding_plan: dict[str, list[str]] = {}
        self._models: dict[str, GaussianPosterior] = {}
        self._mean_bank = np.zeros((len(self.ids), self.dim))
        self._linv_bank = np.zeros((len(self.ids), self.dim, self.dim))
        self._dirty: set[str] = set(self.ids)
        self._log_buffer: list[RatingLog] = []
        self._state_dirty: set[str] = set()

    # Candidates and features

    def eligible(self, user_id: str) -> list[str]:
        """Everyone who passes the hard filter for this user. Fixed for the session."""
        if user_id not in self._eligible:
            viewer = self.profiles[user_id]
            self._eligible[user_id] = [
                uid for uid in self.ids if is_eligible(viewer, self.profiles[uid], self.config.check_looking_for)
            ]
        return self._eligible[user_id]

    def candidates(self, user_id: str) -> list[str]:
        """Eligible profiles this user hasn't rated, and that aren't excluded for them."""
        skip = self.rated[user_id] | self.excluded[user_id]
        return [c for c in self.eligible(user_id) if c not in skip]

    def exclude(self, user_id: str, candidate_ids: Iterable[str]) -> None:
        """Never show these to the user, for example a held-out evaluation set."""
        self.excluded[user_id].update(candidate_ids)

    def features(self, viewer_id: str, candidate_ids: Sequence[str]) -> np.ndarray:
        """Rows phi(viewer, candidate), one per candidate."""
        if len(self.ids) <= self.config.feature_cache_limit:
            rows = self._rows.get(viewer_id)
            if rows is None:
                ids = self.eligible(viewer_id)
                rows = _Rows(ids, self._pairs([viewer_id] * len(ids), ids), {c: i for i, c in enumerate(ids)})
                self._rows[viewer_id] = rows
            if all(c in rows.pos for c in candidate_ids):
                return rows.Phi[[rows.pos[c] for c in candidate_ids]]
        return self._pairs([viewer_id] * len(candidate_ids), list(candidate_ids))

    def _pairs(self, viewer_ids: Sequence[str], candidate_ids: Sequence[str]) -> np.ndarray:
        return self.builder.pair_matrix(
            [self.codes[self.index[v]] for v in viewer_ids], [self.codes[self.index[c]] for c in candidate_ids]
        )

    def reverse_features(self, user_id: str, candidate_ids: Sequence[str]) -> np.ndarray:
        """Rows phi(candidate, user): each candidate looking back at the user."""
        if len(self.ids) <= self.config.feature_cache_limit:
            return np.stack([self.features(c, [user_id])[0] for c in candidate_ids]) if candidate_ids else np.zeros((0, self.dim))
        return self._pairs(list(candidate_ids), [user_id] * len(candidate_ids))

    # Models

    def _prior(self) -> tuple[np.ndarray, np.ndarray]:
        pairwise = self.calibrator.kind == "pairwise"
        scale = self.config.pairwise_prior_scale if pairwise else self.prior_scale
        if self.config.use_population_prior and self.population is not None:
            mean = self.population.weights.copy()
        else:
            mean = np.zeros(self.dim)
            if not pairwise:
                mean[0] = 0.5  # An uninformed intercept: the middle of the rating scale.
        return mean, np.eye(self.dim) / scale**2

    def model(self, user_id: str) -> GaussianPosterior:
        """This user's current posterior, rebuilt from their state if anything changed."""
        if user_id in self._dirty or user_id not in self._models:
            mean, precision = self._prior()
            ratings = self.user_ratings[user_id]
            if self.calibrator.kind == "pairwise":
                model = PairwiseUserModel(mean, precision)
                if len(ratings) >= 2:
                    Phi = self.features(user_id, [t for t, _ in ratings])
                    model.fit(Phi, np.array([r for _, r in ratings]), self.config.pairwise_margin)
            else:
                state = self.states[user_id]
                a, c = self.calibrator.affine(ratings)
                model = BayesianUserModel.from_data(
                    mean, precision, self.noise_scale**2, state.P, state.calibrated_xty(a, c), state.n_ratings
                )
            self._models[user_id] = model
            i = self.index[user_id]
            self._mean_bank[i] = model.mean
            self._linv_bank[i] = model.chol_inv
            self._dirty.discard(user_id)
        return self._models[user_id]

    def _refresh(self, user_ids: Iterable[str]) -> None:
        for uid in user_ids:
            if uid in self._dirty:
                self.model(uid)

    def transform(self, utility: np.ndarray, clip: bool = True) -> np.ndarray:
        """Map model output to a calibrated score on [0, 1]."""
        if self.calibrator.kind == "pairwise" and self.config.scorer == "bayes":
            return _sigmoid(utility)
        return np.clip(utility, 0.0, 1.0) if clip else np.asarray(utility, dtype=float)

    def _direction(self, rater_ids: Sequence[str], Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of each rater's utility for the matching row of Phi.

        Popularity doesn't depend on the rater, so _scores handles it before this.
        """
        n = len(rater_ids)
        scorer = self.config.scorer
        if scorer == "random":
            return self.rng.uniform(size=n), np.zeros(n)
        if scorer == "global":
            weights = self.population.weights if self.population is not None else np.r_[0.5, np.zeros(self.dim - 1)]
            return Phi @ weights, np.zeros(n)
        self._refresh(set(rater_ids))
        idx = np.array([self.index[r] for r in rater_ids], dtype=int)
        mean = np.einsum("nd,nd->n", Phi, self._mean_bank[idx])
        V = np.einsum("nij,nj->ni", self._linv_bank[idx], Phi)
        return mean, np.sqrt(np.einsum("ni,ni->n", V, V))

    def _popularity(self, user_ids: Sequence[str]) -> np.ndarray:
        idx = np.array([self.index[u] for u in user_ids], dtype=int)
        k, prior = self.config.popularity_k, self.calibrator.pop_mean
        return (self.received_sum[idx] + k * prior) / (self.received_n[idx] + k)

    def _scores(self, user_id: str, candidate_ids: Sequence[str]) -> tuple[np.ndarray, ...]:
        """(forward mean, forward std, reverse mean, reverse std) in utility units."""
        n = len(candidate_ids)
        if self.config.scorer == "popularity":
            fwd = self._popularity(candidate_ids)
            rev = np.full(n, self._popularity([user_id])[0])
            return fwd, np.zeros(n), rev, np.zeros(n)
        fwd_m, fwd_s = self._direction([user_id] * n, self.features(user_id, candidate_ids))
        rev_m, rev_s = self._direction(list(candidate_ids), self.reverse_features(user_id, candidate_ids))
        return fwd_m, fwd_s, rev_m, rev_s

    def predicted_scores(self, user_id: str, candidate_ids: Sequence[str]) -> np.ndarray:
        """Posterior mean of s(user -> candidate), calibrated to [0, 1]."""
        if not candidate_ids:
            return np.zeros(0)
        return self.transform(self._scores(user_id, candidate_ids)[0])

    def compatibility(self, user_id: str, candidate_ids: Sequence[str]) -> np.ndarray:
        """Matching-mode compatibility: conservative scores in both directions, combined."""
        if not candidate_ids:
            return np.zeros(0)
        beta = self.config.conservative_beta
        fwd_m, fwd_s, rev_m, rev_s = self._scores(user_id, candidate_ids)
        return combine(self.transform(fwd_m - beta * fwd_s), self.transform(rev_m - beta * rev_s), self.config.reciprocal)

    def top_k(self, user_id: str, candidate_ids: Sequence[str] | None = None, k: int | None = None) -> list[tuple[str, float]]:
        """The ranked output: the k most compatible candidates, best first."""
        cands = self.candidates(user_id) if candidate_ids is None else list(candidate_ids)
        scores = self.compatibility(user_id, cands)
        order = np.argsort(-scores, kind="stable")[: k or self.config.top_k]
        return [(cands[i], float(scores[i])) for i in order]

    def user_weights(self, user_id: str) -> np.ndarray:
        """Posterior mean of this user's weights, in the order of builder.names."""
        if self.config.scorer != "bayes":
            return self.population.weights.copy() if self.population is not None else np.zeros(self.dim)
        return self.model(user_id).mean.copy()

    # Choosing what to show

    def _anchor_ids(self, user_id: str) -> list[str]:
        if not self.config.n_anchors:
            return []
        eligible = set(self.eligible(user_id)) - self.excluded[user_id]
        return [a for a in self.anchor_order if a in eligible][: self.config.n_anchors]

    def _all_anchors(self) -> set[str]:
        return {a for uid in self.ids for a in self._anchor_ids(uid)}

    def onboarding_plan(self, user_id: str) -> list[str]:
        """The user's onboarding profiles, in order. Planned once, without any ratings."""
        if user_id not in self._onboarding_plan:
            cands = self.candidates(user_id)
            anchors = [c for c in self._anchor_ids(user_id) if c in cands]
            forced = [cands.index(a) for a in anchors]
            n = self.config.n_onboarding
            if self.config.onboarding == "d-optimal" and self.config.scorer == "bayes":
                _, precision = self._prior()
                picks = d_optimal_onboarding(self.features(user_id, cands), precision, self.noise_scale**2, n, forced)
            else:
                picks = random_onboarding(len(cands), n, self.rng, forced)
            self._onboarding_plan[user_id] = [cands[i] for i in picks]
        return self._onboarding_plan[user_id]

    def in_onboarding(self, user_id: str) -> bool:
        return len(self.user_ratings[user_id]) < self.config.n_onboarding

    def pick(self, user_id: str) -> Choice | None:
        """The next profile to show this user, or None if nothing is left."""
        choice = self._onboarding_pick(user_id) if self.in_onboarding(user_id) else None
        if choice is None:
            choice = self._feed_pick(user_id)
        if choice is not None:
            self.exposure[self.index[choice.candidate_id]] += 1
        return choice

    def _onboarding_pick(self, user_id: str) -> Choice | None:
        remaining = set(self.candidates(user_id))
        for cand in self.onboarding_plan(user_id):
            if cand in remaining:
                deterministic = self.config.onboarding == "d-optimal" and self.config.scorer == "bayes"
                propensity = 1.0 if deterministic else 1.0 / len(remaining)
                return Choice(cand, propensity, "onboarding", "onboarding")
        return None

    def _feed_pick(self, user_id: str) -> Choice | None:
        cands = self.candidates(user_id)
        if not cands:
            return None
        cfg = self.config
        n = len(cands)
        idx = np.array([self.index[c] for c in cands])
        exposure, mean_exposure = self.exposure[idx], float(self.exposure.mean())

        def adjust(scores: np.ndarray) -> np.ndarray:
            return exposure_adjust(scores, exposure, mean_exposure, cfg.exposure_cap, cfg.exposure_penalty)

        match = adjust(self.compatibility(user_id, cands))
        mode_draw = self.rng.random()
        learn_scores: np.ndarray | None = None
        if cfg.scorer != "bayes":
            learn_probs = np.full(n, 1.0 / n)  # Baselines have no uncertainty, so they explore uniformly.
        elif cfg.policy == "ucb":
            model, Phi = self.model(user_id), self.features(user_id, cands)
            learn_scores = adjust(self.transform(model.ucb_scores(Phi, cfg.ucb_alpha), clip=False))
            learn_probs = _onehot_argmax(learn_scores)
        else:
            model, Phi = self.model(user_id), self.features(user_id, cands)
            samples = model.sample_weights(self.rng, cfg.mc_samples + 1)
            sampled = np.stack([adjust(col) for col in self.transform(Phi @ samples.T, clip=False).T], axis=1)
            learn_scores = sampled[:, 0]  # The draw actually used; the rest estimate propensities.
            learn_probs = thompson_propensities(sampled)

        probs = mixture_probabilities(match, learn_probs, cfg.explore_share, cfg.uniform_floor)
        if mode_draw < cfg.uniform_floor:
            chosen, mode = int(self.rng.integers(n)), "uniform"
        elif mode_draw < cfg.uniform_floor + (1 - cfg.uniform_floor) * cfg.explore_share:
            chosen = int(self.rng.integers(n)) if learn_scores is None else int(np.argmax(learn_scores))
            mode = "learn"
        else:
            chosen, mode = int(np.argmax(match)), "match"
        phase: Phase = "exploit" if mode == "match" else "bandit"
        return Choice(cands[chosen], float(max(probs[chosen], _EPS)), phase, mode)

    # Learning

    def observe(
        self,
        rater_id: str,
        ratee_id: str,
        rating: float,
        choice: Choice | None = None,
        mark_rated: bool = True,
    ) -> None:
        """Record a rating on [0, 1]. mark_rated=False adds data without hiding the profile."""
        if not 0.0 <= rating <= 1.0:
            raise ValueError(f"rating must be on [0, 1], got {rating}")
        phi = self.features(rater_id, [ratee_id])[0]
        self.states[rater_id].add(phi, rating)
        self.user_ratings[rater_id].append((ratee_id, rating))
        if mark_rated:
            self.rated[rater_id].add(ratee_id)
        j = self.index[ratee_id]
        self.received_sum[j] += rating
        self.received_n[j] += 1
        self._dirty.add(rater_id)
        self._state_dirty.add(rater_id)
        if self.storage is not None:
            phase: Phase = choice.phase if choice is not None else "bandit"
            self._log_buffer.append(
                RatingLog(
                    rater_id, ratee_id, round(rating * 100.0, 6), datetime.now(timezone.utc).isoformat(),
                    phase, self.config.policy_id, choice.propensity if choice else None, self.session_id,
                )
            )

    def flush(self) -> None:
        """Write buffered ratings and changed user states to storage."""
        if self.storage is None:
            return
        self.storage.log_ratings(self._log_buffer)
        self.storage.save_user_states([self.states[u] for u in sorted(self._state_dirty)], self.session_id)
        self._log_buffer.clear()
        self._state_dirty.clear()

    def retrain(self) -> PopulationModel | None:
        """Refit the calibrator, the population model, and (by default) tau and sigma."""
        cfg = self.config
        rated_users = [u for u in self.ids if self.user_ratings[u]]
        if not rated_users:
            return None
        self.calibrator.fit_population(self.user_ratings)
        version = (self.population.version + 1) if self.population is not None else 1
        datasets = []
        for u in rated_users:
            Phi = self.features(u, [t for t, _ in self.user_ratings[u]])
            raw = np.array([r for _, r in self.user_ratings[u]])
            datasets.append((Phi, raw))
        n_ratings = sum(len(y) for _, y in datasets)

        if self.calibrator.kind == "pairwise":
            pairs = [preference_pairs(Phi, raw, cfg.pairwise_margin) for Phi, raw in datasets]
            diffs = np.concatenate([d for d, _ in pairs])
            weights = np.concatenate([w for _, w in pairs])
            self.population = PopulationModel.fit_pairwise(diffs, weights, self.dim, cfg.ridge_alpha, version, n_ratings)
        else:
            calibrated = []
            for u, (Phi, raw) in zip(rated_users, datasets):
                a, c = self.calibrator.affine(self.user_ratings[u])
                calibrated.append((Phi, a * raw + c))
            Phi_all = np.concatenate([Phi for Phi, _ in calibrated])
            y_all = np.concatenate([y for _, y in calibrated])
            self.population = PopulationModel.fit_regression(Phi_all, y_all, cfg.ridge_alpha, version)
            if cfg.scorer == "bayes" and (cfg.prior_scale is None or cfg.noise_scale is None):
                m0, _ = self._prior()
                tau, sigma = fit_noise_and_prior_scale(calibrated, m0)
                self.prior_scale = cfg.prior_scale or tau
                self.noise_scale = cfg.noise_scale or sigma

        if self.storage is not None:
            self.flush()
            self.storage.save_population_model(
                self.population, self.session_id, self.calibrator.name, self.builder.names, self.prior_scale
            )
        self._dirty = set(self.ids)
        return self.population

    # Output

    def one_to_one(self) -> list[tuple[str, str, float]]:
        """Optional one-to-one assignment over everyone's unrated eligible candidates.

        Women seeking men and men seeking women form a bipartite graph, matched by the
        Hungarian algorithm. Same-gender pools aren't bipartite, so they're paired
        greedily by compatibility instead.
        """
        pairs: list[tuple[str, str, float]] = []
        groups: dict[tuple[str, str], list[str]] = {}
        for uid in self.ids:
            p = self.profiles[uid]
            groups.setdefault((p.gender, p.interested_in), []).append(uid)
        left, right = groups.get(("woman", "men"), []), groups.get(("man", "women"), [])
        if left and right:
            compat = np.zeros((len(left), len(right)))
            ok = np.zeros_like(compat, dtype=bool)
            pos = {c: j for j, c in enumerate(right)}
            for i, a in enumerate(left):
                cands = [c for c in self.candidates(a) if c in pos]
                if cands:
                    cols = [pos[c] for c in cands]
                    compat[i, cols] = self.compatibility(a, cands)
                    ok[i, cols] = True
            pairs += one_to_one_assignment(left, right, compat, ok)
        for key in (("man", "men"), ("woman", "women")):
            members, used = groups.get(key, []), set()
            edges = []
            for a in members:
                cands = [c for c in self.candidates(a) if c in set(members) and a < c]
                for c, s in zip(cands, self.compatibility(a, cands)):
                    edges.append((float(s), a, c))
            for s, a, c in sorted(edges, reverse=True):
                if a not in used and c not in used:
                    used |= {a, c}
                    pairs.append((a, c, s))
        return pairs


# 11. Matcher adapter for run_simulation


class BayesianReciprocalMatcher(Matcher):
    """Proposal 3 as a Matcher, so it runs in run_simulation with any Rater (Jev included).

    pretrain_ratings are optional ratings from earlier runs, such as the random baseline.
    They train the models but don't hide profiles in this run. The population model is
    first trained once min_ratings ratings are known, then every retrain_every ratings.
    Use n_workers=1, or a small number, so it learns from each rating soon after.
    """

    name = "bayes-reciprocal"

    def __init__(
        self,
        config: ProposalConfig = ProposalConfig(),
        pretrain_ratings: Iterable[Rating | tuple[str, str, float]] = (),
        min_ratings: int = 200,
        retrain_every: int = 1000,
        storage: ProposalStorage | None = None,
    ) -> None:
        self.config = config
        self._pretrain = [(r.rater_id, r.target_id, r.score) if isinstance(r, Rating) else r for r in pretrain_ratings]
        self._min_ratings = min_ratings
        self._retrain_every = retrain_every
        self._storage = storage
        self.engine: ReciprocalBayesEngine | None = None
        self._pending: dict[tuple[str, str], Choice] = {}
        self._n_seen = 0
        self._since_retrain = 0

    def _setup(self, view: PublicView) -> None:
        self.engine = ReciprocalBayesEngine(view.list_profiles(), self.config, self._storage, view.run_id)
        known = self.engine.index
        for rater, target, score in self._pretrain:
            if rater in known and target in known and rater != target:
                self.engine.observe(rater, target, float(score), mark_rated=False)
                self._n_seen += 1
        if self._n_seen >= self._min_ratings:
            self.engine.retrain()

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        if self.engine is None:
            self._setup(view)
        engine = self.engine
        if user_id not in engine.index:
            return None  # Added after the run started; this matcher doesn't know them.
        engine.rated[user_id] |= view.already_rated(user_id) & set(engine.index)
        choice = engine.pick(user_id)
        if choice is None:
            return None
        self._pending[(user_id, choice.candidate_id)] = choice
        return choice.candidate_id

    def observe(self, rating: Rating) -> None:
        engine = self.engine
        if engine is None or rating.rater_id not in engine.index or rating.target_id not in engine.index:
            return
        choice = self._pending.pop((rating.rater_id, rating.target_id), None)
        engine.observe(rating.rater_id, rating.target_id, rating.score, choice)
        self._n_seen += 1
        self._since_retrain += 1
        first = engine.population is None and self._n_seen >= self._min_ratings
        if first or (engine.population is not None and self._since_retrain >= self._retrain_every):
            engine.retrain()
            self._since_retrain = 0
