"""Proposed matchers, to be compared against CompatibleRandomMatcher.

Both proposals score a directed pair (u -> v) as "how much u would like v", and rank a
user's candidates by the chance of a mutual match, P(u -> v) * P(v -> u).

1. TwoTowerMatcher: the two-tower asymmetric scorer.
   A shared encoder f(features_u) gives each user two vectors:
     q_u, the "seeker" vector: what u wants.
     k_u, the "target" vector: what u is like.
   logit(u -> v) = q_u . k_v / sqrt(d) + b_u + c_v + g
   b_u is how generous u is with likes, c_v is how appealing v is to everyone, and g is
   a global offset. These terms are the same as adding constant dimensions to q and k,
   so the score is still a query-key dot product. It isn't symmetric, so P(u -> v) and
   P(v -> u) can differ. Training uses binary cross-entropy on observed ratings only, with
   each rating's score in [0, 1] as a soft label.

2. TwoTowerGNNMatcher: the same, plus a GNN layer over the directed rating graph.
   Each observed rating w -> u (score s) is an edge. One layer of message passing updates:
     q_u from the k_v of the profiles u has rated   ("taste": what u actually liked)
     k_u from the q_w of the users who rated u      ("appeal": who actually liked u)
   Each message is weighted by how far the score is from the rater's own average, so a
   like from a picky rater counts for more than one from someone who likes everyone.
   The output layers start at zero, so before training the GNN model is the two-tower
   model exactly. Without any ratings, the GNN model is the two-tower model too.
   During training, each epoch splits the training ratings into message edges and
   supervision edges, so a rating is never used to predict itself. When scoring, every
   known rating is a message edge. The graph can be refreshed with new ratings without
   retraining, which is cheap.

Testing offline, on a ratings export:

    python matching_algorithms/matching_proposals.py \
        --ratings exports/random-baseline-100k_ratings.csv.gz --db data/main.db

Testing online, as matchers in run_simulation:

    from matching_algorithms.matching_proposals import TwoTowerMatcher, TwoTowerGNNMatcher
    matcher = TwoTowerGNNMatcher(random.Random(1), pretrain_ratings=old_ratings)
    run_simulation(manager, matcher, rater, run_id="gnn-1", n_steps=10_000, seed=1, n_workers=1)
    compute_metrics(manager.store, "gnn-1")   # Compare expected_matches with the baseline.

Requires torch and numpy.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from matching import Matcher, PublicView, is_compatible  # noqa: E402
from models import PublicProfile, Rating  # noqa: E402

logger = logging.getLogger(__name__)

Triple = tuple[str, str, float]  # (rater_id, target_id, score)

_EPS = 1e-6


# Features


_NUMERIC = ("age", "height_cm")
_CATEGORICAL = ("gender", "interested_in", "looking_for", "religion", "school", "major", "job")
_MULTI = ("interests", "clubs")


class ProfileFeaturizer:
    """Turns public profiles into a dense feature matrix. It never sees user IDs.

    Numeric fields are standardized. Categorical fields are one-hot, and list fields are
    multi-hot divided by the list's length. Values seen fewer than min_count times share
    one "other" slot per field, so rare values don't each get their own weights.
    """

    def __init__(self, profiles: list[PublicProfile], min_count: int = 2) -> None:
        self.vocab: dict[str, dict[str, int]] = {}
        for field in _CATEGORICAL + _MULTI:
            counts: Counter[str] = Counter()
            for profile in profiles:
                counts.update(_values(profile, field))
            kept = sorted(value for value, count in counts.items() if count >= min_count)
            self.vocab[field] = {value: i for i, value in enumerate(kept)}

        numeric = np.array([[getattr(p, f) for f in _NUMERIC] for p in profiles], dtype=np.float64)
        self.mean = numeric.mean(axis=0)
        self.std = numeric.std(axis=0) + _EPS
        # One extra slot per field for "other".
        self.dim = len(_NUMERIC) + sum(len(v) + 1 for v in self.vocab.values())

    def transform(self, profiles: list[PublicProfile]) -> np.ndarray:
        x = np.zeros((len(profiles), self.dim), dtype=np.float32)
        for row, profile in enumerate(profiles):
            numeric = np.array([getattr(profile, f) for f in _NUMERIC], dtype=np.float64)
            x[row, : len(_NUMERIC)] = (numeric - self.mean) / self.std
            offset = len(_NUMERIC)
            for field in _CATEGORICAL + _MULTI:
                vocab = self.vocab[field]
                values = _values(profile, field)
                for value in values:
                    x[row, offset + vocab.get(value, len(vocab))] += 1.0 / len(values)
                offset += len(vocab) + 1
        return x


def _values(profile: PublicProfile, field: str) -> list[str]:
    value = getattr(profile, field)
    return list(value) if isinstance(value, list) else [value]


# The rating graph


@dataclass(frozen=True)
class RatingGraph:
    """Directed rating edges, src rated dst. Used only by the GNN layer.

    weight is the score minus the rater's own mean score, divided by the overall
    spread: positive means the rater liked this one more than they usually do.
    """

    src: torch.Tensor
    dst: torch.Tensor
    score: torch.Tensor
    weight: torch.Tensor
    n_nodes: int

    @classmethod
    def build(cls, src: torch.Tensor, dst: torch.Tensor, score: torch.Tensor, n_nodes: int) -> RatingGraph:
        deg = torch.zeros(n_nodes).index_add_(0, src, torch.ones_like(score))
        total = torch.zeros(n_nodes).index_add_(0, src, score)
        rater_mean = total / deg.clamp(min=1)
        centered = score - rater_mean[src]
        weight = centered / (centered.std() + _EPS) if len(centered) > 1 else centered
        return cls(src, dst, score, weight, n_nodes)


def _aggregate(messages: torch.Tensor, graph: RatingGraph, index: torch.Tensor) -> torch.Tensor:
    """Pool messages onto the nodes in index: [mean, weighted mean, mean weight, mean score].

    Sums are divided by degree + 1 rather than degree, which pulls users with very few
    ratings toward zero, so a single rating can't swing their embedding far.
    """
    n, d = graph.n_nodes, messages.shape[1]
    denom = torch.zeros(n).index_add_(0, index, torch.ones_like(graph.weight)) + 1.0
    mean = torch.zeros(n, d).index_add_(0, index, messages) / denom[:, None]
    weighted = torch.zeros(n, d).index_add_(0, index, messages * graph.weight[:, None]) / denom[:, None]
    mean_weight = torch.zeros(n).index_add_(0, index, graph.weight) / denom
    mean_score = torch.zeros(n).index_add_(0, index, graph.score) / denom
    return torch.cat([mean, weighted, mean_weight[:, None], mean_score[:, None]], dim=1)


class DirectedRatingConv(nn.Module):
    """One GNN layer with two relations: outgoing ratings update q, incoming ones update k."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.taste = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * dim + 2, dim + 1))
        self.appeal = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * dim + 2, dim + 1))
        # Zero init makes the layer a no-op at the start, so the GNN model begins as the
        # two-tower model and only moves away from it if the graph helps.
        for head in (self.taste[1], self.appeal[1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, q, k, b, c, graph: RatingGraph):
        taste = _aggregate(k[graph.dst], graph, graph.src)  # What u rated, and how.
        appeal = _aggregate(q[graph.src], graph, graph.dst)  # Who rated u, and how.
        dq, db = self.taste(taste).split([self.dim, 1], dim=1)
        dk, dc = self.appeal(appeal).split([self.dim, 1], dim=1)
        return q + dq, k + dk, b + db.squeeze(1), c + dc.squeeze(1)


# The model


class TwoTowerNet(nn.Module):
    """A shared encoder with seeker and target heads, plus optional GNN layers."""

    def __init__(self, in_dim: int, dim: int, hidden: int, dropout: float, n_gnn_layers: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.seeker_head = nn.Linear(hidden, dim)  # q_u: what u wants.
        self.target_head = nn.Linear(hidden, dim)  # k_u: what u is like.
        self.bias_head = nn.Linear(hidden, 2)  # b_u (generosity), c_u (appeal).
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.convs = nn.ModuleList(DirectedRatingConv(dim, dropout) for _ in range(n_gnn_layers))
        self.scale = dim**-0.5

    def embed(self, x: torch.Tensor, graph: RatingGraph | None = None):
        h = self.trunk(x)
        q, k = self.seeker_head(h), self.target_head(h)
        b, c = self.bias_head(h).unbind(dim=1)
        if graph is not None and len(graph.src) > 0:
            for conv in self.convs:
                q, k, b, c = conv(q, k, b, c, graph)
        return q, k, b, c

    def pair_logits(self, emb, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        q, k, b, c = emb
        return (q[src] * k[dst]).sum(dim=1) * self.scale + b[src] + c[dst] + self.global_bias

    def logit_matrix(self, emb) -> torch.Tensor:
        """Entry [u, v] is logit(u -> v). It's fine at this size: 2,000 users is 4M entries."""
        q, k, b, c = emb
        return q @ k.T * self.scale + b[:, None] + c[None, :] + self.global_bias


@dataclass(frozen=True)
class TrainConfig:
    dim: int = 32
    hidden: int = 128
    dropout: float = 0.1
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 200  # Upper bound. Early stopping usually ends sooner.
    batch_size: int = 4096
    val_fraction: float = 0.1  # Held out for early stopping. 0 trains for all epochs.
    patience: int = 8
    message_fraction: float = 0.7  # GNN only: share of training edges used as messages each epoch.
    n_gnn_layers: int = 1
    min_count: int = 2
    seed: int = 0


class AsymmetricScorer:
    """Trains a TwoTowerNet on ratings and predicts P(u -> v) for any pair of users.

    use_gnn=False is proposal 1, the two-tower model. use_gnn=True is proposal 2.
    """

    def __init__(self, profiles: list[PublicProfile], config: TrainConfig = TrainConfig(), use_gnn: bool = False) -> None:
        self.config = config
        self.use_gnn = use_gnn
        self.ids = sorted(p.user_id for p in profiles)
        self.index = {user_id: i for i, user_id in enumerate(self.ids)}
        by_id = {p.user_id: p for p in profiles}
        ordered = [by_id[user_id] for user_id in self.ids]
        self.featurizer = ProfileFeaturizer(ordered, config.min_count)
        self.x = torch.from_numpy(self.featurizer.transform(ordered))
        self.model: TwoTowerNet | None = None
        self.graph: RatingGraph | None = None
        self.history: list[dict[str, float]] = []

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def _encode(self, ratings: Iterable[Rating | Triple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert ratings to index tensors. A repeated pair keeps its last score."""
        pairs: dict[tuple[int, int], float] = {}
        skipped = 0
        for rater_id, target_id, score in map(_as_triple, ratings):
            if rater_id in self.index and target_id in self.index:
                pairs[(self.index[rater_id], self.index[target_id])] = score
            else:
                skipped += 1
        if skipped:
            logger.warning("Skipped %d ratings of unknown users", skipped)
        if not pairs:
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty, torch.zeros(0)
        keys = torch.tensor(list(pairs.keys()), dtype=torch.long)
        return keys[:, 0], keys[:, 1], torch.tensor(list(pairs.values()), dtype=torch.float32)

    def _graph(self, src, dst, score) -> RatingGraph | None:
        return RatingGraph.build(src, dst, score, len(self.ids)) if self.use_gnn else None

    def fit(self, ratings: Iterable[Rating | Triple], epochs: int | None = None) -> AsymmetricScorer:
        """Train from scratch on these ratings, then use all of them as the scoring graph."""
        cfg = self.config
        torch.manual_seed(cfg.seed)
        gen = torch.Generator().manual_seed(cfg.seed)
        src, dst, y = self._encode(ratings)
        if len(y) == 0:
            raise ValueError("no ratings to train on")

        perm = torch.randperm(len(y), generator=gen)
        n_val = int(len(y) * cfg.val_fraction) if cfg.val_fraction > 0 and len(y) >= 50 else 0
        val, train = perm[:n_val], perm[n_val:]

        model = TwoTowerNet(self.featurizer.dim, cfg.dim, cfg.hidden, cfg.dropout, cfg.n_gnn_layers if self.use_gnn else 0)
        with torch.no_grad():
            mean = y[train].mean().clamp(_EPS, 1 - _EPS)
            model.global_bias.fill_(math.log(mean / (1 - mean)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        train_graph = self._graph(src[train], dst[train], y[train])

        best_loss, best_state, bad_epochs = float("inf"), None, 0
        self.history = []
        for epoch in range(epochs or cfg.epochs):
            model.train()
            order = train[torch.randperm(len(train), generator=gen)]
            if self.use_gnn:
                # A rating is either a message or a label in each epoch, never both, so the
                # model can't learn to read the answer off the graph.
                n_msg = int(len(order) * cfg.message_fraction)
                msg, sup = order[:n_msg], order[n_msg:]
                graph = RatingGraph.build(src[msg], dst[msg], y[msg], len(self.ids))
            else:
                sup, graph = order, None

            total = 0.0
            for batch in sup.split(cfg.batch_size):
                emb = model.embed(self.x, graph)
                loss = F.binary_cross_entropy_with_logits(model.pair_logits(emb, src[batch], dst[batch]), y[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            record = {"epoch": epoch + 1, "train_bce": total / max(len(sup), 1)}

            if n_val:
                model.eval()
                with torch.no_grad():
                    emb = model.embed(self.x, train_graph)
                    val_loss = F.binary_cross_entropy_with_logits(model.pair_logits(emb, src[val], dst[val]), y[val]).item()
                record["val_bce"] = val_loss
                if val_loss < best_loss - 1e-5:
                    best_loss, bad_epochs = val_loss, 0
                    best_state = {name: t.clone() for name, t in model.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.patience:
                        self.history.append(record)
                        break
            self.history.append(record)

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self.graph = self._graph(src, dst, y)
        return self

    def set_graph(self, ratings: Iterable[Rating | Triple]) -> None:
        """Replace the GNN's message edges without retraining. A no-op for the two-tower model."""
        if self.use_gnn:
            self.graph = self._graph(*self._encode(ratings))

    @torch.no_grad()
    def like_matrix(self) -> np.ndarray:
        """Entry [i, j] is P(ids[i] -> ids[j])."""
        if self.model is None:
            raise RuntimeError("call fit first")
        emb = self.model.embed(self.x, self.graph)
        return torch.sigmoid(self.model.logit_matrix(emb)).numpy()

    @torch.no_grad()
    def predict(self, pairs: Iterable[tuple[str, str]]) -> np.ndarray:
        """P(rater -> target) for each (rater_id, target_id)."""
        if self.model is None:
            raise RuntimeError("call fit first")
        pairs = list(pairs)
        src = torch.tensor([self.index[a] for a, _ in pairs], dtype=torch.long)
        dst = torch.tensor([self.index[b] for _, b in pairs], dtype=torch.long)
        emb = self.model.embed(self.x, self.graph)
        return torch.sigmoid(self.model.pair_logits(emb, src, dst)).numpy()

    def mutual_probability(self, u: str, v: str) -> float:
        p_uv, p_vu = self.predict([(u, v), (v, u)])
        return float(p_uv * p_vu)


def _as_triple(rating: Rating | Triple) -> Triple:
    if isinstance(rating, Rating):
        return rating.rater_id, rating.target_id, rating.score
    rater_id, target_id, score = rating
    return rater_id, target_id, float(score)


# Matchers


class TwoTowerMatcher(Matcher):
    """Proposal 1. Shows each user the compatible, unseen profile with the best mutual chance.

    The mutual chance is P(u -> v) * P(v -> u). If v has already rated u in this run, the
    observed score replaces the predicted P(v -> u), which gives pairs a quick chance to
    finish a match.

    Until min_ratings ratings are known, and on an explore share of later steps, it
    picks at random among compatible profiles, like CompatibleRandomMatcher. Exploring
    keeps the training data from covering only what the model already likes.

    pretrain_ratings is optional public data from earlier runs, such as the random
    baseline. The model then starts trained, and those ratings stay in its training set.
    It retrains from scratch every retrain_every ratings in this run. The GNN version
    also refreshes its graph every refresh_every ratings, which is much cheaper.

    Use a new matcher for each run, and run_simulation with n_workers=1 or a small number,
    so it learns from each rating soon after it's made.
    """

    name = "two-tower"
    use_gnn = False

    def __init__(
        self,
        rng: random.Random,
        config: TrainConfig = TrainConfig(),
        pretrain_ratings: Iterable[Rating | Triple] = (),
        min_ratings: int = 1000,
        retrain_every: int = 2000,
        refresh_every: int = 200,
        explore: float = 0.1,
        use_observed_reverse: bool = True,
    ) -> None:
        self._rng = rng
        self._config = config
        self._pretrain = [_as_triple(r) for r in pretrain_ratings]
        self._min_ratings = min_ratings
        self._retrain_every = retrain_every
        self._refresh_every = refresh_every
        self._explore = explore
        self._use_observed_reverse = use_observed_reverse

        self._scorer: AsymmetricScorer | None = None
        self._compatible: dict[str, list[str]] = {}
        self._run_ratings: list[Triple] = []
        self._received: dict[str, dict[str, float]] = defaultdict(dict)  # target -> {rater: score}
        self._like: np.ndarray | None = None  # [i, j] = P(i -> j)
        self._since_train = 0
        self._since_refresh = 0
        self.train_seconds = 0.0
        self.n_retrains = 0

    def _setup(self, view: PublicView) -> None:
        profiles = view.list_profiles()
        self._scorer = AsymmetricScorer(profiles, self._config, use_gnn=self.use_gnn)
        for user in profiles:
            self._compatible[user.user_id] = sorted(
                p.user_id for p in profiles if p.user_id != user.user_id and is_compatible(user, p)
            )
        self._maybe_retrain(force=True)

    def _all_ratings(self) -> list[Triple]:
        return self._pretrain + self._run_ratings

    def _maybe_retrain(self, force: bool = False) -> None:
        assert self._scorer is not None
        ratings = self._all_ratings()
        if len(ratings) < self._min_ratings:
            return
        if force or not self._scorer.is_trained or self._since_train >= self._retrain_every:
            start = time.perf_counter()
            self._scorer.fit(ratings)
            self.train_seconds += time.perf_counter() - start
            self.n_retrains += 1
            self._since_train = self._since_refresh = 0
            self._like = self._scorer.like_matrix()
        elif self.use_gnn and self._since_refresh >= self._refresh_every:
            self._scorer.set_graph(ratings)
            self._since_refresh = 0
            self._like = self._scorer.like_matrix()

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        if self._scorer is None:
            self._setup(view)
        seen = view.already_rated(user_id)
        candidates = [c for c in self._compatible[user_id] if c not in seen]
        if not candidates:
            return None
        if self._like is None or self._rng.random() < self._explore:
            return self._rng.choice(candidates)

        index = self._scorer.index
        u = index[user_id]
        cand = np.array([index[c] for c in candidates])
        forward = self._like[u, cand]
        reverse = self._like[cand, u].copy()
        if self._use_observed_reverse:
            received = self._received[user_id]
            for j, candidate in enumerate(candidates):
                if candidate in received:
                    reverse[j] = received[candidate]
        mutual = forward * reverse
        best = np.flatnonzero(mutual >= mutual.max() - 1e-9)
        return candidates[self._rng.choice(list(best))]

    def observe(self, rating: Rating) -> None:
        self._run_ratings.append(_as_triple(rating))
        self._received[rating.target_id][rating.rater_id] = rating.score
        self._since_train += 1
        self._since_refresh += 1
        if self._scorer is not None:
            self._maybe_retrain()


class TwoTowerGNNMatcher(TwoTowerMatcher):
    """Proposal 2. TwoTowerMatcher, with a GNN layer over the directed rating graph."""

    name = "two-tower-gnn"
    use_gnn = True


# Offline evaluation


def load_ratings_csv(path: str | Path) -> list[Triple]:
    """Read a ratings export (.csv or .csv.gz) with rater_id, target_id, and score columns."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        return [(row["rater_id"], row["target_id"], float(row["score"])) for row in csv.DictReader(f)]


def load_profiles(main_db: str | Path) -> list[PublicProfile]:
    """Read every profile from main.db, without the agent flag."""
    from store import Store

    return [PublicProfile.model_validate(p.model_dump(exclude={"agent"})) for p in Store(main_db).list_all_profiles()]


def split_by_pair(ratings: list[Triple], test_fraction: float, seed: int) -> tuple[list[Triple], list[Triple]]:
    """Split so both directions of a pair land on the same side.

    Then the reverse rating can't leak into the test, and pairs rated both ways in the
    test set can be used to measure mutual matches.
    """
    rng = random.Random(seed)
    pairs = sorted({tuple(sorted((a, b))) for a, b, _ in ratings})
    test_pairs = set(rng.sample(pairs, int(len(pairs) * test_fraction)))
    train, test = [], []
    for rating in ratings:
        (test if tuple(sorted(rating[:2])) in test_pairs else train).append(rating)
    return train, test


def _bias_baseline(train: list[Triple], shrink: float = 5.0):
    """Rater mean + target mean - global mean, each shrunk toward the global mean."""
    scores = np.array([s for _, _, s in train])
    g = scores.mean()
    by_rater, by_target = defaultdict(list), defaultdict(list)
    for a, b, s in train:
        by_rater[a].append(s)
        by_target[b].append(s)

    def shrunk(values: dict[str, list[float]]) -> dict[str, float]:
        return {key: (sum(v) + shrink * g) / (len(v) + shrink) - g for key, v in values.items()}

    rater, target = shrunk(by_rater), shrunk(by_target)
    return lambda pairs: np.clip([g + rater.get(a, 0.0) + target.get(b, 0.0) for a, b in pairs], _EPS, 1 - _EPS)


def _rank(x: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(x))
    ranks[np.argsort(x, kind="stable")] = np.arange(len(x))
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() > 0 and b.std() > 0 else float("nan")


def score_predictions(test: list[Triple], pred: np.ndarray, top_k: int = 3) -> dict[str, float]:
    """How well predictions match held-out scores, for single ratings and for mutual pairs.

    top{k}_lift: for each rater with at least 2k test ratings, the mean true score of the
    k profiles ranked highest, over the mean of all their test ratings. 1.0 is random.
    mutual_top10_lift: over pairs rated both ways in the test set, the mean true match
    chance (product of the two scores) of the top tenth by predicted chance, over the mean.
    """
    y = np.array([s for _, _, s in test])
    p = np.clip(pred, _EPS, 1 - _EPS)
    out = {
        "bce": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "rmse": float(np.sqrt(((y - p) ** 2).mean())),
        "pearson": _corr(p, y),
        "spearman": _corr(_rank(p), _rank(y)),
    }

    per_rater: dict[str, list[int]] = defaultdict(list)
    for i, (a, _, _) in enumerate(test):
        per_rater[a].append(i)
    lifts, tops = [], []
    for idx in per_rater.values():
        if len(idx) >= 2 * top_k:
            idx = np.array(idx)
            top = idx[np.argsort(-p[idx], kind="stable")[:top_k]]
            tops.append(y[top].mean())
            lifts.append(y[top].mean() / max(y[idx].mean(), _EPS))
    out[f"top{top_k}_mean_score"] = float(np.mean(tops)) if tops else float("nan")
    out[f"top{top_k}_lift"] = float(np.mean(lifts)) if lifts else float("nan")

    where = {(a, b): i for i, (a, b, _) in enumerate(test)}
    pairs = [(i, where[(b, a)]) for (a, b), i in where.items() if a < b and (b, a) in where]
    out["mutual_pairs"] = len(pairs)
    if len(pairs) >= 10:
        i, j = np.array(pairs).T
        true_m, pred_m = y[i] * y[j], p[i] * p[j]
        n_top = max(1, len(pairs) // 10)
        top = np.argsort(-pred_m, kind="stable")[:n_top]
        out["mutual_spearman"] = _corr(_rank(pred_m), _rank(true_m))
        out["mutual_top10_lift"] = float(true_m[top].mean() / max(true_m.mean(), _EPS))
    return out


def evaluate_offline(
    profiles: list[PublicProfile],
    ratings: list[Triple],
    test_fraction: float = 0.2,
    seed: int = 0,
    config: TrainConfig = TrainConfig(),
) -> dict[str, dict[str, float]]:
    """Train each model on the same pair split and score it on the held-out pairs."""
    train, test = split_by_pair(ratings, test_fraction, seed)
    test_pairs = [(a, b) for a, b, _ in test]
    config = replace(config, seed=seed)
    results: dict[str, dict[str, float]] = {}

    g = np.mean([s for _, _, s in train])
    results["global-mean"] = score_predictions(test, np.full(len(test), g))
    results["rater+target-mean"] = score_predictions(test, _bias_baseline(train)(test_pairs))

    for name, use_gnn in (("two-tower", False), ("two-tower-gnn", True)):
        start = time.perf_counter()
        scorer = AsymmetricScorer(profiles, config, use_gnn=use_gnn).fit(train)
        results[name] = score_predictions(test, scorer.predict(test_pairs))
        results[name]["epochs"] = len(scorer.history)
        results[name]["train_seconds"] = time.perf_counter() - start
    return results


def format_results(results: dict[str, dict[str, float]]) -> str:
    columns = list(dict.fromkeys(key for r in results.values() for key in r))
    width = max(len(c) for c in columns) + 2
    lines = [" " * width + "".join(f"{name:>20}" for name in results)]
    for column in columns:
        cells = []
        for r in results.values():
            value = r.get(column, float("nan"))
            cells.append(f"{value:>20.4f}" if isinstance(value, float) else f"{value:>20}")
        lines.append(f"{column:<{width}}" + "".join(cells))
    return "\n".join(lines)


def main() -> None:
    from paths import MAIN_DB

    parser = argparse.ArgumentParser(description="Compare the proposed matchers offline on a ratings export.")
    parser.add_argument("--ratings", required=True, nargs="+", help="One or more ratings CSVs (.csv or .csv.gz).")
    parser.add_argument("--db", default=str(MAIN_DB), help="main.db with the profiles.")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--dim", type=int, default=TrainConfig.dim)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    profiles = load_profiles(args.db)
    ratings = [r for path in args.ratings for r in load_ratings_csv(path)]
    print(f"{len(profiles)} profiles, {len(ratings)} ratings")
    config = TrainConfig(epochs=args.epochs, dim=args.dim)
    print(format_results(evaluate_offline(profiles, ratings, args.test_fraction, args.seed, config)))


if __name__ == "__main__":
    main()
