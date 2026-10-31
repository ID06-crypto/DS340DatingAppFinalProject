# Reciprocal matching: experiment framework

A local framework for training a reciprocal matching algorithm with AI agents as simulated users. In each step, a simulated user is shown one profile chosen by a matcher, rates it, and the rating is saved. The matcher is currently random and the Jev rater is a stub. Persona generation isn't part of this code.

| File | Purpose |
| --- | --- |
| `models.py` | Pydantic models: `Profile`, `Rating`, `Persona`, `RatingTranscript`, `Run` |
| `store.py` | Profiles, ratings, and runs, in `main.db` |
| `personas.py` | Hidden data, in `personas.db`: personas and rating transcripts |
| `db.py` | Connection and WAL helpers shared by both stores |
| `manager.py` | `ProfileManager`, the storage interface for the rest of the project |
| `matching.py` | `Matcher` interface, `PublicView`, `RandomMatcher` |
| `raters.py` | `Rater` interface, `build_rater_input`, `FakeRater`, `JevRater` (stub) |
| `simulation.py` | `run_simulation` and `RunSummary` |
| `scripts/run_baseline.py` | An end-to-end run on fake users |

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
from raters import FakeRater
from simulation import run_simulation

manager = ProfileManager("main.db", "personas.db")
# Add profiles and personas here. Only users with a persona can rate.

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

### Connecting Jev

Implement `_call_jev` in `raters.py`. It's the only place the API gets called, and it takes the `build_rater_input` dict and returns `(score, raw_output)`.

### How a run behaves

- **Failed ratings:** a rater error, invalid rater output, or failed save is logged and counted in `failures`, and the run continues. The step is used up, so step numbers have a gap there. A matcher returning an unknown or already-rated profile is handled the same way, so it can't overwrite a rating.
- **Matcher exceptions:** an exception raised by `next_profile` or `observe` is a bug in the algorithm, so it stops the run. `finished_at` stays empty.
- **Scores:** out-of-range scores are clamped to 1.0 to 10.0 with a warning, and `raw_output` keeps what the model actually said. NaN, infinity, and non-numbers count as failures.
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
    "job": "nurse",
    "interests": ["Hiking", " chess "],
})
ben = manager.add_profile({
    "name": "Ben",
    "age": 31,
    "height_cm": 180,
    "religion": "catholic",
    "school": "MIT",
    "job": "teacher",
    "interests": ["chess"],
})
print(ana.interests)  # ['hiking', 'chess']

manager.edit_profile(ana.user_id, job="doctor")
manager.add_persona(ana.user_id, "Values kindness over ambition.", {"kindness": 0.9})

manager.add_rating(ana.user_id, ben.user_id, 7.35, run_id="run-1", model_version="model-1")
manager.add_rating(ben.user_id, ana.user_id, 8.5, run_id="run-1")

print(manager.ratings_of_target(ben.user_id, run_id="run-1")[0].score)  # 7.35
print(len(manager.mutual_pairs(7.0, run_id="run-1")))  # 1
print(len(manager.already_rated_pairs("run-1")))  # 2

manager.store.export_profiles_json("profiles.json")
manager.personas.export_personas_json("personas.json")

# Removes Ana's profile, ratings in every run, and persona.
manager.delete_profile(ana.user_id)
```

To import your datasets, include a `user_id` in the dict passed to `add_profile`. It must be a lowercase, hyphenated UUID4 string, and a duplicate raises `store.DuplicateProfileError`.

Invalid data raises `pydantic.ValidationError`, and an unknown user raises `store.ProfileNotFoundError`.

## Keeping personas away from the matching algorithm

- **Matchers only get a `PublicView`.** It's built from the main store alone, so it holds no route to `personas.db` or `ProfileManager`. It's also limited to the current run.
- **`main.db` holds no persona text.** `input_snapshot` is exactly what the rater was shown, which includes the rater's persona, and `raw_output` may repeat it. `add_rating` therefore saves both to `personas.db` as a `RatingTranscript`.
- **Keep matching code away from `personas.db`.** Don't give it a `ProfileManager` or a `PersonaStore`. Python can't stop code from deliberately opening a file, so this protects against accidents rather than deliberate access.

## Behavior to know about

- **Re-rating:** a rating replaces an earlier one with the same rater, target, and run, including its transcript. If `raw_output` and `input_snapshot` aren't passed again, the old transcript is deleted.
- **Mutual pairs across runs:** with `run_id=None`, both scores must still come from the same run. A pair that qualifies in several runs appears once per run, and each result has a `run_id`.
- **Seeded IDs:** IDs repeat only if the seed and the order of `add_profile` calls both match. A failed validation also uses up an ID, and running the same seed against an existing database raises `DuplicateProfileError`.
- **Deleting across two files:** the two databases can't share a transaction. Hidden data is deleted first, so if a delete fails partway, calling it again finishes the job.

## Changing the schema

There are no migrations. Tables are only created if they're missing, so after changing a table, delete the `.db` files, along with their `-wal` and `-shm` files, and re-import your data.

To add a field to profiles or ratings:

1. Add the field to the model in `models.py`.
2. Add its column to `SCHEMA` in `store.py`.

The insert and update statements pick up the new field automatically. A field that isn't a simple value, like `interests`, also needs converting in `_profile_to_row` and `_row_to_profile`.

Personas are written out column by column, so a new persona field also needs adding to `save_persona` in `personas.py`.
