# Reciprocal matching: storage layer

Local storage for training a reciprocal matching algorithm with AI agents as simulated users. The algorithm, agent simulation, and persona generation aren't part of this code.

| File | Purpose |
| --- | --- |
| `models.py` | Pydantic models: `Profile`, `Rating`, `Persona` |
| `store.py` | Profiles and ratings, in `main.db` |
| `personas.py` | Hidden personas, in `personas.db` |
| `db.py` | Connection and WAL helpers shared by both stores |
| `manager.py` | `ProfileManager`, the interface for the rest of the project |

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

## Usage

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

- **Use `Store`, not `ProfileManager`.** `ProfileManager` opens the persona database, so give the matching algorithm a `store.Store("main.db")`, which can only reach profiles and ratings.
- **Check what goes into ratings.** `raw_output` and `input_snapshot` are stored in `main.db`. If a rater's prompt or reasoning includes persona text, it ends up where the algorithm can read it, and the code can't detect this.

## Behavior to know about

- **Re-rating:** a rating replaces every field of an earlier one with the same rater, target, and run, including setting `raw_output` to empty if it's not passed again.
- **Mutual pairs across runs:** with `run_id=None`, both scores must still come from the same run. A pair that qualifies in several runs appears once per run, and each result has a `run_id`.
- **Seeded IDs:** IDs repeat only if the seed and the order of `add_profile` calls both match. A failed validation also uses up an ID, and running the same seed against an existing database raises `DuplicateProfileError`.
- **Deleting across two files:** the two databases can't share a transaction. The persona is deleted first, so if a delete fails partway, calling it again finishes the job.

## Changing the schema

There are no migrations. Tables are only created if they're missing, so after changing a table, delete the `.db` files, along with their `-wal` and `-shm` files, and re-import your data.

To add a field to profiles or ratings:

1. Add the field to the model in `models.py`.
2. Add its column to `SCHEMA` in `store.py`.

The insert and update statements pick up the new field automatically. A field that isn't a simple value, like `interests`, also needs converting in `_profile_to_row` and `_row_to_profile`.

Personas are written out column by column, so a new persona field also needs adding to `save_persona` in `personas.py`.
