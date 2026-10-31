# Reciprocal matching: data layer

Local storage for training a reciprocal matching algorithm with AI agents as simulated users. It holds profiles, ratings, and hidden agent personas in SQLite. The algorithm and the agent simulation aren't part of this code.

| File | Purpose |
| --- | --- |
| `models.py` | Pydantic models: `Profile`, `Rating`, `Persona` |
| `store.py` | SQLite persistence |
| `manager.py` | `ProfileManager`, the public interface for the rest of the project |

## Setup and tests

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Usage

```python
from manager import ProfileManager

# rng makes generated IDs reproducible. Omit it for random IDs.
manager = ProfileManager("training.db", rng=42)

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

manager.add_rating(ana.user_id, ben.user_id, 8, run_id="run-1")
manager.add_rating(ben.user_id, ana.user_id, 9, run_id="run-1")

print(len(manager.ratings_of(ben.user_id, run_id="run-1")))  # 1
print(len(manager.mutual_pairs("run-1", min_score=7)))  # 1

manager.export_profiles_json("profiles.json")

# Removes Ana's profile, persona, and ratings in every run.
manager.delete_profile(ana.user_id)
```

Invalid data raises `pydantic.ValidationError`, and an unknown user raises `store.ProfileNotFoundError`.

## Behavior to know about

- **Re-rating:** each run keeps one rating per rater and target. Rating again in the same run replaces the earlier score.
- **Deleting a profile:** this removes the user's ratings from every run, which changes what past runs replay. Pass `anonymize_ratings=True` to keep the scores. The deleted user's ID then reads back as `models.ANONYMIZED_ID`, and those ratings never appear in mutual pairs.
- **Seeded IDs:** IDs repeat only if the seed and the order of `add_profile` calls both match, so failed validations and multi-threaded calls can change them. Reusing a seed on a database that already has those IDs raises `sqlite3.IntegrityError`, so use a new database file per seeded run.
- **Personas:** these hold the ground truth agents rate by, so they must not reach the matching algorithm. Agent simulation code reads them through `store.Store`. Matching code should import only `manager`. They share a database file with profiles, so this boundary is a convention and isn't enforced.

## Changing the schema

There are no migrations. Opening a database created with a different schema version raises `SchemaVersionError`, and the fix is to delete the file or migrate it by hand.

To add a profile field:

1. Add the field to `Profile` in `models.py`.
2. Add its column to the `profiles` table in `SCHEMA` in `store.py`.
3. Bump `SCHEMA_VERSION` in `store.py`.

Insert and update statements pick up the new field automatically. A field that isn't a simple value, like `interests`, also needs converting in `_profile_to_row` and `_row_to_profile`.
