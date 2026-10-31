# Reciprocal matching: experiment framework

A local framework for training a reciprocal matching algorithm with AI agents as simulated users. In each step, a simulated user is shown one profile chosen by a matcher, rates it, and the rating is saved. The matcher is currently random, and simulated users rate through [Jev](https://docs.typesafe.ai/introduction/quickstart). The current users are all synthetic, as described in [Synthetic data](#synthetic-data).

| File | Purpose |
| --- | --- |
| `models.py` | Pydantic models: `Profile`, `Rating`, `Persona`, `RatingTranscript`, `Run` |
| `store.py` | Profiles, ratings, and runs, in `main.db` |
| `personas.py` | Hidden data, in `personas.db`: personas and rating transcripts |
| `db.py` | Connection and WAL helpers shared by both stores |
| `manager.py` | `ProfileManager`, the storage interface for the rest of the project |
| `matching.py` | `Matcher` interface, `PublicView`, `RandomMatcher`, `CompatibleRandomMatcher` |
| `raters.py` | `Rater` interface, `build_rater_input`, `FakeRater`, `JevRater` |
| `simulation.py` | `run_simulation` and `RunSummary` |
| `metrics.py` | `compute_metrics`: match rate, compatibility, and the learning curve for a run |
| `scripts/run_baseline.py` | An end-to-end run on fake users |
| `scripts/run_trial.py` | Tries a rater on a small sample of the real users and reports speed and scores |
| `scripts/run_experiment.py` | Runs a long Jev baseline in parallel, with resuming and backups |
| `scripts/backup_data.py` | Backs up both databases and exports a run's ratings |
| `paths.py` | Where the current dataset is stored |
| `scripts/generate_profiles.py` | Generates the original 10,000 profiles in the root `main.db` |
| `Data Generation/` | The profile and persona generators, and the lists they sample from. `written_personas.py` builds the current dataset. |
| `notebooks/persona_test.ipynb` | Tests how different kinds of persona change Jev's ratings, with charts |
| `notebooks/written_personas.ipynb` | Writes sample personas with Claude and tests them on Jev, with charts |
| `notebooks/final_dataset_check.ipynb` | Validates the final 2,000 users and their personas on Jev, with charts |
| `notebooks/random_baseline_metrics.ipynb` | Metrics and charts for the 100,000-rating random baseline |
| `SYNTHETIC_DATA_REPORT.txt` | How the data was made, everything tested, and the final results |

## Install

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the tests

```bash
pytest
```

## Run the baseline

```bash
python scripts/run_baseline.py
```

This builds five fake users in a temporary database, runs `RandomMatcher` with `FakeRater`, and prints the `RunSummary`.

## Running an experiment

```python
import random

from manager import ProfileManager
from matching import RandomMatcher
from paths import MAIN_DB, PERSONA_DB
from raters import FakeRater
from simulation import run_simulation

manager = ProfileManager(MAIN_DB, PERSONA_DB)  # The 2,000 users. Only users with a persona can rate.

summary = run_simulation(
    manager,
    RandomMatcher(random.Random(1)),
    FakeRater(random.Random(1)),
    run_id="baseline-1",
    n_steps=100,
    seed=1,
)
print(summary)

ratings = manager.ratings_in_run("baseline-1")  # In step order, for learning curves.
transcripts = manager.transcripts_in_run("baseline-1")  # What each rater saw and answered.
```

A run needs a new `run_id` each time. The same seeds and the same profile IDs give the same run.

### Adding a real matcher

Subclass `Matcher`. `run_simulation` doesn't need to change.

```python
from matching import Matcher, PublicView
from models import Rating


class MyMatcher(Matcher):
    name = "my-matcher"  # Saved with the run.

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        ...  # Return a profile ID, or None if there's nothing left for this user.

    def observe(self, rating: Rating) -> None:
        ...  # Optional. Called after every saved rating.
```

### Rating with Jev

`JevRater` sends each rating to TypeSafe's Jev model. It needs an API key from the [TypeSafe console](https://console.typesafe.ai/keys), in an environment variable rather than in the code:

```bash
export TYPESAFE_API_KEY="your-key"
```

`_call_jev` in `raters.py` is the only place the API gets called:

- **The request:** the rater's profile and persona and the target's profile go in as `state`, without user IDs or the `agent` flag. `JEV_QUESTION` is a yes/no [Noul](https://docs.typesafe.ai/primitives/noul.md) question: would the rater like this profile? It expects personas written as plain sentences.
- **The score:** a rating's `score` is Jev's probability of a yes, from 0 to 1. Train on it directly with logistic loss and soft labels. To get likes and passes shaped like real swipes, draw them from the probability with a seeded random generator.
- **Saved output:** `raw_output` keeps Jev's model version, its answer, and input tokens.
- **Errors:** the SDK retries rate limits, timeouts, and server errors. A request that still fails counts as a failed step.
- **Model version:** `JEV_MODEL` is pinned, so a new Jev release can't change scores partway through an experiment.

Check a change on a small sample before a long run:

```bash
python scripts/run_trial.py --rater fake  # Dry run on a temporary copy of the data.
python scripts/run_trial.py --rater jev --run-id jev-trial-1  # Saved to the real databases.
```

### Long runs

```bash
python scripts/run_experiment.py --run-id random-baseline-100k --steps 100000
```

- **Saving:** each rating is saved as soon as it arrives, and both databases are backed up to `backups/` at the start and end. The end of the run also writes `exports/<run_id>_ratings.csv.gz`.
- **Resuming:** if a run stops, the same command continues from the last saved rating, up to the same total. In code, pass `resume=True` to `run_simulation`.
- **Outages:** a run stops with `RunAbortedError` after 50 failures in a row, rather than using up its steps.
- **Parallel ratings:** `--workers` sets how many ratings run at once, and the default is 8. A matcher can't learn from a batch until it finishes, so one that learns from each rating should use 1 or a small number.
- **Storage:** ratings live in their own table rather than on profiles, so a user can have many across different runs without their profile changing.

### Measuring a run

```python
from metrics import compute_metrics

print(compute_metrics(manager.store, "baseline-1"))
```

A pair's chance of matching is the product of their two scores, which assumes each person's like is independent of the other's. `expected_matches` adds that up over pairs who rated each other, and `match_rate` is its average per pair. `compatible_share` is the share of ratings between people interested in each other's gender. `learning_curve` is the mean score across equal parts of the run, in order. Compare a matcher against `CompatibleRandomMatcher`, which picks randomly among compatible profiles.

### How a run behaves

- **Baseline to beat:** `random-baseline-100k` reached 4.14 expected matches per 1,000 ratings. `notebooks/random_baseline_metrics.ipynb` has the details.
- **Failed ratings:** a rater error, invalid rater output, or failed save is logged and counted in `failures`, and the run continues. The step is used up, so step numbers have a gap there. A matcher returning an unknown or already-rated profile is handled the same way, so it can't overwrite a rating.
- **Matcher exceptions:** an exception raised by `next_profile` or `observe` is a bug in the algorithm, so it stops the run. `finished_at` stays empty.
- **Scores:** a score is the probability that the rater likes the target. Out-of-range scores are clamped to 0.0 to 1.0 with a warning, and `raw_output` keeps what the model actually said. NaN, infinity, and non-numbers count as failures.
- **Exhausted users:** a user with nothing left to see is dropped from the pool without using a step. `users_exhausted` only counts users the run actually discovered this way, so it can be lower if `n_steps` runs out first.
- **Users without a persona:** they can't rate, and they're skipped with a warning, but matchers can still show their profile to others.

## Storage usage

```python
import random

from manager import ProfileManager

# rng makes generated IDs reproducible. Omit it for random IDs.
manager = ProfileManager("main.db", "personas.db", rng=random.Random(42))

ana = manager.add_profile({
    "name": "Ana",
    "age": 29,
    "height_cm": 168,
    "religion": "none",
    "school": "BU",
    "major": "nursing",
    "job": "nurse",
    "interests": ["Hiking", " chess "],
    "clubs": ["Outdoors Club"],  # Can be empty.
    "gender": "woman",  # Or "man".
    "interested_in": "men",  # Or "women".
    "looking_for": "long-term",  # Or "short-term" or "just a date".
    "agent": True,  # True for a simulated user, False for a real person.
})
ben = manager.add_profile({
    "name": "Ben",
    "age": 31,
    "height_cm": 180,
    "religion": "catholic",
    "school": "MIT",
    "major": "physics",
    "job": "teacher",
    "interests": ["chess"],
    "clubs": [],
    "gender": "man",
    "interested_in": "women",
    "looking_for": "just a date",
    "agent": True,
})
print(ana.interests)  # ['hiking', 'chess']

manager.edit_profile(ana.user_id, job="doctor")
manager.add_persona(ana.user_id, "Values kindness over ambition.", {"kindness": 0.9})

manager.add_rating(ana.user_id, ben.user_id, 0.735, run_id="run-1", model_version="model-1")
manager.add_rating(ben.user_id, ana.user_id, 0.85, run_id="run-1")

print(manager.ratings_of_target(ben.user_id, run_id="run-1")[0].score)  # 0.735
print(len(manager.mutual_pairs(0.7, run_id="run-1")))  # 1
print(len(manager.already_rated_pairs("run-1")))  # 2

manager.store.export_profiles_json("profiles.json")
manager.personas.export_personas_json("personas.json")

# Removes Ana's profile, ratings in every run, and persona.
manager.delete_profile(ana.user_id)
```

To import your datasets, include a `user_id` in the dict passed to `add_profile`. It must be a lowercase, hyphenated UUID4 string, and a duplicate raises `store.DuplicateProfileError`.

Invalid data raises `pydantic.ValidationError`, and an unknown user raises `store.ProfileNotFoundError`.

## Keeping personas away from the matching algorithm

- **Matchers only get a `PublicView`.** It's built from the main store alone, so it holds no route to `personas.db` or `ProfileManager`. It's also limited to the current run, and its profiles leave out the `agent` flag.
- **`main.db` holds no persona text.** `input_snapshot` is exactly what the rater was shown, which includes the rater's persona, and `raw_output` may repeat it. `add_rating` therefore saves both to `personas.db` as a `RatingTranscript`.
- **Keep matching code away from `personas.db`.** Don't give it a `ProfileManager` or a `PersonaStore`. Python can't stop code from deliberately opening a file, so this protects against accidents rather than deliberate access.

## Behavior to know about

- **Re-rating:** a rating replaces an earlier one with the same rater, target, and run, including its transcript. If `raw_output` and `input_snapshot` aren't passed again, the old transcript is deleted.
- **The `agent` flag:** matchers and raters aren't shown it. `PublicView` returns each profile as a `PublicProfile`, which doesn't have the field. Because matchers can't see it, they also can't avoid real people's profiles, so keep agents and real people in separate databases.
- **`gender` and `interested_in`:** `RandomMatcher` ignores these, so it can show someone a profile outside their stated interest. `CompatibleRandomMatcher` only shows profiles where the interest goes both ways, and a real matcher can read both from `PublicView`.
- **Mutual pairs across runs:** with `run_id=None`, both scores must still come from the same run. A pair that qualifies in several runs appears once per run, and each result has a `run_id`.
- **Seeded IDs:** IDs repeat only if the seed and the order of `add_profile` calls both match. A failed validation also uses up an ID, and running the same seed against an existing database raises `DuplicateProfileError`.
- **Deleting across two files:** the two databases can't share a transaction. Hidden data is deleted first, so if a delete fails partway, calling it again finishes the job.

## Synthetic data

The current dataset is 2,000 simulated Boston University students in `data/main.db` and `data/personas.db`. None of it describes a real person, and every profile has `agent` set to `True`. `paths.py` says where it lives. The full history, including every test and result, is in `SYNTHETIC_DATA_REPORT.txt`.

### Profiles

The 2,000 users are a seeded sample of 10,000 profiles made by `scripts/generate_profiles.py`, kept unchanged in the root `main.db`. Each value is drawn uniformly unless the table says otherwise.

| Field | How it's generated |
| --- | --- |
| `user_id` | A UUID4 from the seeded generator, created by `add_profile` |
| `gender` | `man` or `woman`, 50/50 |
| `name` | One of 80 names for that gender, from `profile_lists.py` |
| `age` | 18 to 24 |
| `height_cm` | 150 to 200, for all genders |
| `religion` | One of 10 options |
| `school` | Always `BU` |
| `major` | One of 42 majors, from `profile_lists.py` |
| `job` | One of 8 student jobs |
| `interests` | 1 to 3 from `hobbylist.csv`, which has 383 unique hobbies once lowercased |
| `clubs` | 0 to 3 BU organizations from `bu_clubs.py`, leaving out men-only clubs for women and women-only clubs for men |
| `interested_in` | The user's own gender 3% of the time (`GAY_RATE`), otherwise the opposite gender |
| `looking_for` | `short-term`, `long-term`, or `just a date` |

### Personas

`Data Generation/written_personas.py` draws five traits uniformly from 0 to 1 (extraversion, openness, conscientiousness, traditionalism, and achievement drive). Claude Haiku 5.5 then writes 2 to 3 sentences from those traits and the profile, covering what the person is like and what they want in a partner.

Written personas replaced 25 trait scores per user (`Data Generation/synthetic_personas.py`), because Jev barely responded to numbers. The test results are in `notebooks/`.

### How it's stored

| Data | File | Table | Format |
| --- | --- | --- | --- |
| Profiles | `data/main.db` | `profiles` | One row per user. `interests` and `clubs` are JSON lists, and `agent` is 0 or 1. |
| Personas | `data/personas.db` | `personas` | One row per user. `description` is the written persona, and `generation_seeds` records the seed, model, and traits. |

```python
from manager import ProfileManager
from paths import MAIN_DB, PERSONA_DB

manager = ProfileManager(MAIN_DB, PERSONA_DB)
print(manager.get_persona(user_id).description)
```

### Regenerating the data

This needs the `ANTHROPIC_API_KEY` environment variable and the root `main.db`:

```bash
python "Data Generation/written_personas.py"
```

Users who already have a persona are skipped, so running it again finishes an interrupted run. The sample of users and the traits repeat for the same seed, but **Claude's wording doesn't, and databases aren't in Git. Back up `data/`, because the exact personas can't be recreated.**

To rebuild the original 10,000 users, delete the root `main.db*` and `personas.db*` files, then run `python scripts/generate_profiles.py` and `python "Data Generation/synthetic_personas.py"`, in that order.

### Limitations

- **Simulated behavior:** ratings are Jev's estimates about invented people, and they haven't been compared with real people's choices.
- **Low like rates:** written personas are picky. In the final check, the mean like probability was 0.31, and 60% of ratings were 0.3 or below.
- **Unrealistic profiles:** most fields are independent and uniform, so they don't reflect BU's actual students. Height doesn't vary by gender, for example.
- **Small same-gender pool:** only 50 of the 2,000 users are interested in their own gender.
- **One gender of interest:** `interested_in` holds a single value, so the data has no users interested in both men and women.

## Changing the schema

There are no migrations. Tables are only created if they're missing, so after changing a table, delete the `.db` files, along with their `-wal` and `-shm` files, and re-import your data.

To add a field to profiles or ratings:

1. Add the field to the model in `models.py`.
2. Add its column to `SCHEMA` in `store.py`.

The insert and update statements pick up the new field automatically. A field that isn't a simple value, like `interests` or `clubs`, also needs converting in `_profile_to_row` and `_row_to_profile`.

Personas are written out column by column, so a new persona field also needs adding to `save_persona` in `personas.py`.
