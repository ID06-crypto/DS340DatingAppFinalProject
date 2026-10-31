# Reciprocal matching: experiment framework

A local framework for training a reciprocal matching algorithm with AI agents as simulated users. In each step, a simulated user is shown one profile chosen by a matcher, rates it, and the rating is saved. The matcher is currently random and the Jev rater is a stub. The current users are all synthetic, as described in [Synthetic data](#synthetic-data).

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
| `scripts/generate_profiles.py` | Generates the synthetic profiles and saves them to `main.db` |
| `Data Generation/` | The profile and persona generators, and the lists they sample from |

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

- **Matchers only get a `PublicView`.** It's built from the main store alone, so it holds no route to `personas.db` or `ProfileManager`. It's also limited to the current run, and its profiles leave out the `agent` flag.
- **`main.db` holds no persona text.** `input_snapshot` is exactly what the rater was shown, which includes the rater's persona, and `raw_output` may repeat it. `add_rating` therefore saves both to `personas.db` as a `RatingTranscript`.
- **Keep matching code away from `personas.db`.** Don't give it a `ProfileManager` or a `PersonaStore`. Python can't stop code from deliberately opening a file, so this protects against accidents rather than deliberate access.

## Behavior to know about

- **Re-rating:** a rating replaces an earlier one with the same rater, target, and run, including its transcript. If `raw_output` and `input_snapshot` aren't passed again, the old transcript is deleted.
- **The `agent` flag:** matchers and raters aren't shown it. `PublicView` returns each profile as a `PublicProfile`, which doesn't have the field. Because matchers can't see it, they also can't avoid real people's profiles, so keep agents and real people in separate databases.
- **`gender` and `interested_in`:** these are only stored. `RandomMatcher` ignores them, so it can show someone a profile outside their stated interest. A real matcher can read both from `PublicView` and filter on them.
- **Mutual pairs across runs:** with `run_id=None`, both scores must still come from the same run. A pair that qualifies in several runs appears once per run, and each result has a `run_id`.
- **Seeded IDs:** IDs repeat only if the seed and the order of `add_profile` calls both match. A failed validation also uses up an ID, and running the same seed against an existing database raises `DuplicateProfileError`.
- **Deleting across two files:** the two databases can't share a transaction. Hidden data is deleted first, so if a delete fails partway, calling it again finishes the job.

## Synthetic data

The dataset is 10,000 simulated Boston University students, all randomly generated, so none of it describes a real person. Every user has one profile and one persona, linked by `user_id`, and every profile has `agent` set to `True`.

### Profiles

`scripts/generate_profiles.py` calls `generate_profile` in `Data Generation/synthetic_data.py` once per user, with seed 42. Each value is drawn uniformly unless the table says otherwise.

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

`Data Generation/synthetic_personas.py` gives each agent profile a persona of 25 trait scores from 0 to 1:

| Category | Traits |
| --- | --- |
| `core_personality` | openness, conscientiousness, extraversion, agreeableness, neuroticism, honesty_humility |
| `thinking_style` | analytical_thinking, need_for_cognition, abstract_thinking, attention_to_detail, cognitive_flexibility, tolerance_for_ambiguity |
| `decision_making` | risk_tolerance, impulsivity, future_orientation |
| `emotional_style` | rumination, optimism, emotional_expressiveness |
| `social_style` | empathy, trust_in_others, conformity, assertiveness |
| `motivation_and_values` | achievement_drive, autonomy_need, traditionalism |

Each score is drawn from a normal distribution with mean 0.5 and SD 0.2, then rounded to two decimal places. Values outside 0 to 1 are redrawn rather than clipped, so the actual SD is about 0.19. Each user's scores depend only on seed 42 and their `user_id`.

### How it's stored

| Data | File | Table | Format |
| --- | --- | --- | --- |
| Profiles | `main.db` | `profiles` | One row per user. `interests` and `clubs` are JSON lists, and `agent` is 0 or 1. |
| Personas | `personas.db` | `personas` | One row per user. `description` holds the scores as JSON, and `generation_seeds` records the seed, mean, and SD. |

Scores are kept in `description` because that's the part of a persona raters are shown. To read them:

```python
scores = json.loads(manager.get_persona(user_id).description)["scores"]
```

Both files are SQLite databases in the project root. `.gitignore` excludes them, so they aren't in the repository and each copy of the project has to generate its own.

### Regenerating the data

```bash
rm -f main.db* personas.db*
python scripts/generate_profiles.py
python "Data Generation/synthetic_personas.py"
```

Run the steps in this order, because personas are generated for profiles that already exist. The same seeds and lists give the same data, but editing a list changes the profiles generated from then on.

Running `generate_profiles.py` against an existing `main.db` with the same seed raises `DuplicateProfileError`. Running `synthetic_personas.py` again only adds personas for profiles that don't have one, so it doesn't change existing ones.

### Limitations

- **Fields are unrelated:** apart from `name`, `clubs`, and `interested_in` depending on `gender`, each field is drawn independently. Height doesn't vary by gender, and majors don't affect jobs or clubs.
- **Not representative:** the uniform draws don't reflect BU's real demographics, so each religion is about 10% of users, for example.
- **Personas are unrelated to profiles:** a user's trait scores don't depend on their profile, including their interests or `looking_for`.
- **Small same-gender pool:** only 296 users (about 3%) are interested in their own gender, so these users have far fewer compatible profiles than others.
- **One gender of interest:** `interested_in` holds a single value, so the data has no users interested in both men and women.

## Changing the schema

There are no migrations. Tables are only created if they're missing, so after changing a table, delete the `.db` files, along with their `-wal` and `-shm` files, and re-import your data.

To add a field to profiles or ratings:

1. Add the field to the model in `models.py`.
2. Add its column to `SCHEMA` in `store.py`.

The insert and update statements pick up the new field automatically. A field that isn't a simple value, like `interests` or `clubs`, also needs converting in `_profile_to_row` and `_row_to_profile`.

Personas are written out column by column, so a new persona field also needs adding to `save_persona` in `personas.py`.
