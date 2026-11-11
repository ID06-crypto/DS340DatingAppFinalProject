# Proposal 3: Bayesian per-user models with reciprocal matching

Proposals 1 and 2 (`matching_proposals.py`) train one neural network on everyone's ratings. Proposal 3 gives every user their own small Bayesian linear model of their taste, with a pooled population model as the prior. The models' uncertainty drives three things: which profiles a new user rates first, how the feed explores, and how cautious the final matching is. All of it is closed-form numpy/scipy, with no deep learning.

| File | What's in it |
| --- | --- |
| `matching_algorithms/other_proposal.py` | The algorithm: hard filters, features, Bayesian user models, population model, calibration, onboarding, policies, reciprocal matching, SQLite storage, the engine, and a `Matcher` for `run_simulation` |
| `matching_algorithms/other_proposal_sim.py` | The simulator and its oracle, metrics, baselines, ablations, and the experiment CLI |
| `matching_algorithms/other_proposal_plots.py` | Figures made from the experiment CSVs |
| `tests/test_other_proposal.py` | 26 tests |
| `exports/other_proposal/` | Results: `runs.csv`, `summary_*.csv`, `summary.md`, figures |

## How to run

```bash
python3 -m pytest tests/test_other_proposal.py                        # ~2 minutes
python3 matching_algorithms/other_proposal_sim.py --quick             # 60 users, baselines, ~1 minute
python3 matching_algorithms/other_proposal_sim.py --seeds 5 --workers 10   # Everything (see Results)
python3 matching_algorithms/other_proposal_plots.py exports/other_proposal # Redraw the figures
```

It needs numpy, scipy, scikit-learn, pandas, pydantic 2, and matplotlib. On this machine, `python3.13` (Anaconda) has them all. The other tests in `tests/` also need `typesafe_sdk` and `anthropic`.

To run it as a live matcher against Jev, exactly like the other proposals:

```python
from matching_algorithms.other_proposal import BayesianReciprocalMatcher, ProposalConfig, ProposalStorage

matcher = BayesianReciprocalMatcher(
    ProposalConfig(),
    pretrain_ratings=old_ratings,          # Optional, e.g. the random-baseline export.
    storage=ProposalStorage("data/proposal3.db"),  # Optional log with propensities.
)
run_simulation(manager, matcher, JevRater(...), run_id="bayes-1", n_steps=10_000, seed=1, n_workers=1)
```

## Design

### 1. Hard filters

A candidate is eligible when they aren't the viewer, the two pass `matching.is_compatible` (each is the gender the other is interested in), and their `looking_for` values are compatible. Candidates the viewer already rated are excluded too.

**The looking_for rule:** the only incompatible pair is `long-term` with `short-term`, because those goals directly conflict. `just a date` is compatible with everything, since a first date is how either kind of relationship starts. It's set in `INCOMPATIBLE_LOOKING_FOR` and can be turned off with `check_looking_for=False`.

### 2. Features φ(A, B): 34 of them, in a fixed order

| Block | Features |
| --- | --- |
| Pair (12) | intercept, cand_age, cand_height (standardized), age_diff and height_diff (B − A, antisymmetric), abs_age_diff, same_religion, same_school, same_major, same_looking_for, interest_jaccard, club_jaccard |
| Candidate religion (11) | one-hot over the dataset's religions, plus `other` |
| Candidate major group (6) | stem, health, business, social_science, arts_humanities, `other` |
| Candidate job group (4) | academic, service, professional, `other` |
| Candidate school (vocab + 1) | one-hot over the dataset's schools, plus `other` |

`FeatureBuilder.names` gives the order. The vocabularies and the standardization are built once from the dataset and then frozen, and unseen values land in `other`. That's 34 features on the simulator, above the 15-25 guideline: religion stays at full resolution so the religion audit can read a weight for each religion, and grouping religions would have hidden exactly what the audit is meant to show. The Jaccard similarity of two empty club lists is 0, not 1, because sharing no clubs isn't a shared interest.

### 3. Per-user Bayesian linear regression

rating ≈ θ_Aᵀ φ(A, B), with ratings on [0, 1], prior θ_A ~ N(θ_pop, τ²I), and noise σ². `BayesianUserModel` keeps the precision P and b = P·mean, so an update is `P += φφᵀ/σ²` and `b += φr/σ²`. It exposes `mean`, `covariance`, `predict`, `ucb_scores`, `thompson_scores`, and `conservative_scores`. The linear algebra uses one Cholesky factor per version:

* predictive variance is ‖L⁻¹φ‖²
* Thompson draws are mean + L⁻ᵀz
* `covariance` is only formed when something asks for it

**τ and σ are fitted by empirical Bayes** (`fit_noise_and_prior_scale`). Each population retrain picks the grid point that maximizes the marginal likelihood of every user's ratings, y_u ~ N(Φ_u θ_pop, σ²I + τ²Φ_uΦ_uᵀ), rather than hand-tuning them.

**Population prior:** `PopulationModel.fit_regression` is a scikit-learn ridge regression on everyone's pooled, calibrated ratings, with the intercept unpenalized. It's versioned and saved to the `population_model` table on every retrain. The simulator retrains it after onboarding and then every 10 rounds.

### 4. Calibration (pluggable)

Each regression calibrator is affine per user: calibrated = a·r + c. That lets the cache keep raw statistics (X^Tr and Σφ), with the calibrated ones computed exactly as a·X^Tr + c·Σφ. Swapping the calibrator, or watching a user's offset change as they rate more, never invalidates anything.

* `none`
* `zscore`: rescales each user to the population's mean and within-user spread, with no shrinkage.
* `shrunk_intercept` (default): the offset is a Bayesian average, (n·mean_u + k·mean_pop)/(n + k) − mean_pop, with k = 5.
* `shrunk_intercept_scale`: the same, plus the user's spread shrunk toward the population's in the same way.
* `pairwise`: Bradley-Terry. Each pair of a user's ratings that differ by more than 0.05 becomes "i was preferred to j", and the model is Bayesian logistic regression on φ_i − φ_j. Each user's pairs are weighted so their n ratings count as n observations, not n². Because offset and scale cancel in the difference, a user's posterior is a Laplace approximation, refitted by Newton's method on each rating. The population model is the same pooled fit. Its scores are utilities in logit units, mapped to [0, 1] with a sigmoid.

### 5. Onboarding

Greedy D-optimal design: pick the eligible candidate with the largest φᵀΣφ, then do `P += φφᵀ/σ²` and repeat. No ratings are needed to choose, so the whole list (15 by default) is planned up front. The `random` strategy is there for the ablation. `n_anchors > 0` shows fixed anchor profiles first, taken from one seeded global order, so users with the same orientation and goals share them. The shrunk-intercept calibrator then estimates each user's offset from their anchor ratings, compared with everyone else's ratings of the same profiles. That estimate doesn't suffer from selection bias.

### 6–7. Bandit feed and reciprocal matching

Every pick logs its **propensity**, the probability that the whole policy had of showing that candidate:

π = u/n + (1 − u)·(ε·π_learn + (1 − ε)·π_match)

* **u = 0.02** is a uniform floor, so every propensity is above zero, which inverse-propensity estimates need.
* **ε = 0.15** is the exploration share.
* **π_learn** comes from learning mode: Thompson sampling by default, or UCB with α = 1. Learning mode ranks by the viewer's own predicted rating plus uncertainty. Thompson propensities are estimated from 100 Monte Carlo posterior draws, and UCB is deterministic, so its propensity comes from the mixture.
* **π_match** is the deterministic argmax of matching mode. Matching mode ranks by combine(s(A→B), s(B→A)). Each side is that rater's conservative score, mean − β·std with β = 1, calibrated and clipped to [0, 1]. B's score comes from B's own model, so a B with few ratings falls back on the population prior on its own.

The combination is `harmonic` by default, with `geometric` and `min` as alternatives.

**Exposure control:**
* Candidates shown more than 3× as often as the average profile are skipped.
* Below that cap, each multiple of the average above 1 costs 0.05.

**Output:**
* `top_k(user)` is the ranked top-10 in matching mode.
* `one_to_one()` is the optional assignment. It uses the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on women seeking men × men seeking women. Same-gender pools aren't bipartite, so they're paired greedily.
* A **true match** requires both ratings ≥ 70.

### Storage

`ProposalStorage` uses a separate SQLite file (or `:memory:`), not `main.db`, so the project's existing schema is untouched. It has four tables:

* `profiles`
* `ratings`: `rater_id, ratee_id, score 0-100, timestamp, phase (onboarding|bandit|exploit), policy_id, propensity, session_id`
* `user_model_state`
* `population_model`

**`user_model_state` is a cache of raw sufficient statistics.** P = XᵀX and b = Xᵀr, plus Σφ, Σr, Σr², n_ratings, and version. Neither P nor b includes the prior or the calibration, so retraining the population model doesn't invalidate the cache. `rebuild_user_states()` recomputes the cache from `ratings`, and every experiment checks the two are identical: the maximum difference was 0 in every run.

## Assumptions

* **File layout.** The algorithm is in `other_proposal.py`, as asked. The spec's `matching/` package wasn't possible: the repo already has a root module named `matching.py`, which a package of the same name would shadow, breaking `from matching import Matcher`. The simulator is in its own file, so the algorithm file has no import path to the oracle. A test checks that by parsing the file, and another walks the engine's object graph to check it holds no reference to the simulator.
* **Scales.** Ratings are [0, 1] internally, 0–100 in the spec's SQLite tables and in the simulator, and [0, 1] in `run_simulation`, which Jev already uses.
* **Held-out evaluation.** For each user, a random 30% of eligible candidates are held out. They're never shown, and every ranking metric is computed on them. Evaluating on whatever happened to stay unrated would penalize good policies, which leave only weak candidates behind.
* **Fixed budget.** Each user rates 45 profiles in total. The onboarding-size ablation keeps the total fixed and trades onboarding for bandit rounds.
* **Regret** is measured against the true match probability P(r_AB ≥ 70)·P(r_BA ≥ 70) of the best candidate still available, using the oracle.
* **Baselines** use random onboarding and no exposure control. In matching mode they rank by their own score: the population model (global), shrunk mean received rating (popularity), or random. They explore uniformly, since they have no uncertainty.

RESULTS_PLACEHOLDER
