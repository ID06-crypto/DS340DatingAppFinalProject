# Reciprocal matching: experiment framework

A local framework for training a reciprocal matching algorithm with AI agents as simulated users. In each step, a simulated user is shown one profile chosen by a matcher, rates it, and the rating is saved. The matcher is currently random, and simulated users rate through [Jev](https://docs.typesafe.ai/introduction/quickstart). The current users are all synthetic, as described in [Synthetic data](#synthetic-data).

It also includes a JSON backend for the live study with about 100 invited BU students. That's covered in [The live app](#the-live-app).

| File | Purpose |
| --- | --- |
| `models.py` | Pydantic models: `Profile`, `Preferences`, `Rating`, `Persona`, `RatingTranscript`, `Run` |
| `store.py` | Profiles, private preferences, ratings, runs, and the eligible pool, in `main.db` |
| `live_schema.py`, `live_store.py` | The live app's tables in `main.db`: accounts, sessions, matches, messages, blocks, reports, photos, impressions, and the audit log |
| `app/` | The FastAPI backend for the live app |
| `consent/v1.md` | The consent text users agree to, by version |
| `scripts/seed_dev.py` | Creates a few fake approved accounts for local testing |
| `scripts/migrate_schema.py` | Upgrades databases made before preferences moved off profiles |
| `personas.py` | Hidden data, in `personas.db`: personas and rating transcripts |
| `db.py` | Connection and WAL helpers shared by both stores |
| `manager.py` | `ProfileManager`, the storage interface for the rest of the project |
| `matching.py` | `Matcher` interface, `PublicView`, `RandomMatcher`, `CompatibleRandomMatcher` |
| `raters.py` | `Rater` interface, `build_rater_input`, `FakeRater`, `JevRater` |
| `simulation.py` | `run_simulation` and `RunSummary` |
| `metrics.py` | Pure metric functions for comparing matchers: classification, AUC, precision at k, cluster bootstrap, group suppression |
| `run_metrics.py` | `compute_metrics`: match rate, compatibility, and the learning curve for a simulation run |
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
| `evaluation.py` | Pair-level evaluation: user splits, classification metrics, calibration, bias, and ranking |
| `notebooks/model_evaluation.ipynb` | Evaluates any model on "do these two people like each other?", with charts |
| `scripts/build_test_pairs.py` | Rates pairs of test users both ways, to make a larger test set |
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

    def score_candidates(self, user_id: str, view: PublicView, candidates: list[str]) -> dict[str, float] | None:
        ...  # Optional. A 0 to 1 chance of a positive rating for each candidate, or None.
```

Also give it a `description` in plain language, which users see on the transparency page. In the app, scores from `score_candidates` are logged with each impression (`predicted_score`, and the whole pool in `impression_candidates`), so the comparison page can show precision, recall, F1, and AUC. Register it in `app/matchers.py` to use it in the app.

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

### Evaluating a model

Add your model to `MODELS` in `notebooks/model_evaluation.ipynb` and run it. Test users are fixed in `exports/user_splits.json`, so train only on `train`, tune only on `validation`, and treat `test` as one-time. Results go to `exports/evaluation_results.json`, and new models are logged in `exports/evaluation_log.csv`.

### Measuring a run

```python
from run_metrics import compute_metrics

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
    "gender": "woman",  # Or "man" or "nonbinary".
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
    "looking_for": "just a date",
    "agent": True,
})
print(ana.interests)  # ['hiking', 'chess']

# Private: who each person wants to be shown. Never part of a profile or a PublicView.
manager.set_preferences(ana.user_id, ["man"])
manager.set_preferences(ben.user_id, ["woman", "nonbinary"])
print(manager.eligible_candidates(ana.user_id, "run-1"))  # [Ben's user_id]

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
- **The eligible pool:** matchers must choose from `view.eligible_candidates(user_id)`. It contains only people with mutual gender interest and approved accounts, excluding the user themselves, anyone blocked in either direction, active matches, and anyone already rated in this run. In the live app, it also excludes anyone already shown. Simulated users have no account, and count as approved in simulations. If a matcher picks outside the pool, the simulation counts that step as a failure, and the app refuses the pick.
- **Preferences are private:** they decide the pool but never appear in a profile or a `PublicView`. `CompatibleRandomMatcher` is now the same as `RandomMatcher`, and is kept so older runs can still be resumed and compared.
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
| `gender` | `man` or `woman`, 50/50. Profiles also accept `nonbinary`, which the generator doesn't use. |
| `name` | One of 80 names for that gender, from `profile_lists.py` |
| `age` | 18 to 24 |
| `height_cm` | 150 to 200, for all genders |
| `religion` | One of 10 options |
| `school` | Always `BU` |
| `major` | One of 42 majors, from `profile_lists.py` |
| `job` | One of 8 student jobs |
| `interests` | 1 to 3 from `hobbylist.csv`, which has 383 unique hobbies once lowercased |
| `clubs` | 0 to 3 BU organizations from `bu_clubs.py`, leaving out men-only clubs for women and women-only clubs for men |
| Preferences | The user's own gender 3% of the time (`GAY_RATE`), otherwise the opposite gender. These are stored privately, not on the profile. |
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
- **One gender of interest:** each synthetic user is interested in a single gender, so the data has no users interested in more than one, and no nonbinary users.

## The live app

A FastAPI app for the study, with server-rendered pages and a JSON API that share the same rules in `app/actions.py`. It handles magic-link sign-in with BU emails, onboarding and an admin approval queue, Discover (one person at a time), matches and chat, blocks and reports, account settings, and admin analytics.

### Design rules

This is a free student research project, built as a response to engagement-driven recommendation systems. These rules stay true:

- No streaks, no "someone liked you" teasers, no badge counts, and no infinite scroll.
- No online, typing, or read indicators.
- No re-engagement or reminder emails. The only emails are sign-in links and one "new match" email.
- No skip button. Reloading Discover shows the same person again.
- No ads, no tracking scripts, and nothing loaded from other sites. HTMX, Pico.css, and Chart.js are copied into `static/vendor/`, at pinned versions checked against npm's published checksums.
- One person at a time, with a calm end state when nobody's left, and no prompts to come back.

### Running it locally

```bash
python scripts/seed_dev.py   # Optional: 40 fake approved users, with some matches.
EMAIL_SENDER=console COOKIE_SECURE=false ADMIN_EMAILS=you@bu.edu EXPORT_SALT=dev-salt \
    uvicorn --factory app.main:create_app --reload --no-access-log
```

Open http://localhost:8000 in a browser, enter an email (`dev.user01@bu.edu` for a seeded user, or your admin email), and open the sign-in link the terminal prints. Run the tests with `pytest`.

**Pages and the API share URLs.** A browser asking for HTML, or an HTMX request, gets the page at `/me`, `/matches`, `/admin`, and so on. Any other request gets the JSON API at the same URL, unchanged. Internally the pages live under `/ui`. API clients must send an `Origin` header matching `SITE_URL` with every POST, PUT, and DELETE. With curl, add `-H "Origin: http://localhost:8000"`.

### Pages

| Area | What's there |
| --- | --- |
| Public | `/`, which explains the project and takes an email, and `/transparency`, which covers what's collected, who sees what, the algorithms, and these rules |
| Onboarding, in order | Consent, profile, who you'd like to meet, photos, then "under review" until an admin approves you |
| Discover | One profile at a time, with a 1 to 10 slider and number box and no default value. Shows how many people are left. Report and block are in a menu. |
| Matches | Each match's first name, photo, and last message time. Chat polls every 5 seconds only while it's open. Unmatch, block, and report are in a menu. |
| Me | Edit your profile, photos, and preferences, see who you've blocked, set match emails, opt out of the experiment, pause, download your data, or delete your account |
| Admin | Overview, approvals, reports, algorithm comparison, controls, audit log, and export. Each needs a sign-in from the last `ADMIN_STEP_UP_MINUTES`. |

**Surveys:** "Was this worth your time?" from 1 to 5 appears on Discover when nobody's left, and otherwise after every 25th rating, at most once a day. "Was this match worth your time?" appears once per match, in its chat, once the match is 3 days old. Both can be dismissed.

**Opting out:** users who opt out always get `DEFAULT_MATCHER`, and their data is left out of all analytics and exports. **Pausing** hides a user from everyone's pool without deleting anything.

### Analytics

The admin comparison page computes everything with the pure functions in `metrics.py`, over rated impressions where neither person opted out.

- **Positive rating:** `POSITIVE_THRESHOLD` (7.0) or above, on the 1 to 10 scale. This is separate from `MATCH_THRESHOLD`.
- **Positive prediction:** `PREDICTION_THRESHOLD` (0.5) or above. Matchers predict on a 0 to 1 scale. Matchers without predictions show "n/a (no predictions)" for classification metrics.
- **Confidence intervals:** 95% bootstrap intervals that resample users, because one person's ratings are correlated. Every number shows n users and n ratings.
- **Groups:** results by age band (18-20, 21-23, 24+) and gender hide any group with fewer than `MIN_GROUP_SIZE` users. When exactly one group is hidden, the next smallest is too, so it can't be worked out from the total.
- **Rapid ratings:** ratings made in under `RAPID_RATING_MS` are flagged, not removed.
- **Surveys:** a matcher is chosen for each impression, so survey answers are credited to the matcher that served the person's last impression before answering.
- **Selection bias:** a matcher that chooses who to show is evaluated on the people it chose, so its classification metrics aren't comparable with the random matcher's. The random matcher's data is the unbiased sample for offline evaluation.

The export (`/admin/export.csv` and `.json`) has one row per impression. People appear as salted hashes using `EXPORT_SALT`, with age bands and gender. It has no emails, names, free text, photos, religion, or exact ages. Each export is audited with its row count.

**Run it as one process.** Rate limits, the matcher registry, and matcher state live in memory, so several workers would each have their own copy. SQLite in WAL mode handles one process's concurrent requests.

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `SITE_URL` | `http://localhost:8000` | The site's address. Used in sign-in links and the CSRF check. |
| `ALLOWED_EMAIL_DOMAINS` | `bu.edu` | Comma-separated. The domain must match exactly, so subdomains aren't allowed. |
| `ADMIN_EMAILS` | (none) | Comma-separated. These become admins, approved on their first sign-in. |
| `APP_MAIN_DB` | `data/live/main.db` | The live database, kept apart from the simulated users. |
| `APP_PERSONA_DB` | `data/live/personas.db` | Required by `ProfileManager`. Live users have no personas. |
| `PHOTO_DIR` | `data/live/photos` | Photo storage. Never serve this folder publicly. |
| `CONSENT_DIR` | `consent` | Folder of consent texts, named `v<version>.md`. |
| `CONSENT_VERSION` | `1` | The version users must agree to. Raising it asks everyone to agree again. |
| `REQUIRE_PHOTO` | `true` | Whether approval needs at least one photo. |
| `MATCH_THRESHOLD` | `7.0` | Both people must rate each other at least this, on the 1 to 10 scale, to match. |
| `COOKIE_SECURE` | `true` | Only send the session cookie over HTTPS. Set `false` for local http. |
| `SESSION_DAYS` | `14` | How long a session lasts. |
| `LOGIN_TOKEN_MINUTES` | `15` | How long a sign-in link works. |
| `ENABLED_MATCHERS` | `random` | Comma-separated names from `app/matchers.py`, switched on when the database is new. After that, admins set them and their weights on the controls page. |
| `DEFAULT_MATCHER` | `random` | Serves users who opt out of the experiment. |
| `POSITIVE_THRESHOLD` | `7.0` | A rating at or above this counts as positive in analytics (1 to 10). |
| `PREDICTION_THRESHOLD` | `0.5` | A prediction at or above this counts as positive (0 to 1). |
| `MIN_GROUP_SIZE` | `10` | Analytics groups with fewer users than this are suppressed. |
| `RAPID_RATING_MS` | `800` | Ratings faster than this are flagged in analytics. |
| `ADMIN_STEP_UP_MINUTES` | `60` | Admins must have signed in within this many minutes. |
| `EXPORT_SALT` | (none) | Required for exports. Keep it secret and fixed, so pseudonyms stay stable. |
| `SOURCE_REPO_URL` | (none) | Optional link to the source code on the transparency page. |
| `MATCHER_SEED` | (random) | Seeds that choice, for reproducible tests. |
| `EMAIL_SENDER` | `console` | `console` prints links, which is for development only. `smtp` sends real email. |
| `SMTP_HOST`, `SMTP_PORT` | (none), `587` | The SMTP server. |
| `SMTP_USERNAME`, `SMTP_PASSWORD` | (none) | SMTP sign-in, if needed. |
| `SMTP_FROM` | (none) | The sender address. |
| `SMTP_STARTTLS` | `true` | Use STARTTLS. |

### Endpoints

| Area | Endpoints | Who can call them |
| --- | --- | --- |
| Sign-in | `POST /auth/request-link`, `GET /auth/verify`, `GET` and `POST /auth/consent`, `POST /auth/logout` | Anyone, then signed in |
| Your account | `GET /me`, `DELETE /me` | Signed in. Deleting never needs consent or approval. |
| Your data | `POST`, `GET`, and `PUT /me/profile`, `GET` and `PUT /me/preferences`, `POST /me/photos`, `DELETE /me/photos/{id}`, `PUT /me/photos` (reorder) | Signed in, with consent. Approval isn't needed. |
| Photos | `GET /photos/{id}` | The owner, admins, and approved users who may see the owner |
| Feed and ratings | `GET /feed/next`, `POST /ratings` | Approved, with a profile |
| Matches and chat | `GET /matches`, `POST /matches/{id}/unmatch`, `GET` and `POST /matches/{id}/messages` | Approved participants of an active match with no block |
| Safety | `POST`, `GET`, and `DELETE /blocks`, `POST /reports` | Approved |
| Admin | `GET /admin/accounts`, `POST /admin/accounts/{id}/approve`, `reject`, or `suspend`, `GET /admin/reports`, `POST /admin/reports/{id}/resolve`, `GET /admin/reports/{id}/messages` | Admins. Every action is written to `audit_log`. |

### Ratings in the app

People rate from 1 to 10. Ratings are stored on the same 0 to 1 scale as Jev's, as (score - 1) / 9, with `run_id` set to `live`. The impression decides the target and the matcher, so a client can never choose who it rates or which matcher gets credit. A match is made when both have rated each other at or above `MATCH_THRESHOLD`, unless either has blocked the other. The reply to a rating has the same shape whether or not the other person has rated you, unless a match was made.

### Privacy design

| Who | Can see |
| --- | --- |
| **Anyone** | Nothing except the consent text. Requesting a link gives the same reply for any address. |
| **A signed-in user** | Their own email, profile, preferences, photos, and approval status, through `/me`. |
| **Other approved users** | Your public profile and photos, but only once you're in their pool, have been shown to them, or have matched with them, and never if either of you has blocked the other. They never see your email, preferences, ratings, or whether you rated them, unless you match. |
| **Your match** | Messages in your match, while it's active and neither of you has blocked the other. |
| **Admins** | Emails, profiles, and photos in the approval queue, and reports. They can read a match's messages only through a report about it, and each read is logged. |
| **Matchers** | Public profiles, this run's ratings, and the eligible pool's user IDs. Never preferences, emails, or personas. |

- **Email:** stored only in `accounts`, never on profiles.
- **Sign-in:** login tokens and session ids are stored only as SHA-256 hashes. A link works once and expires after 15 minutes. The access log is off in the command above, and a filter removes tokens from server logs.
- **Cookies:** `HttpOnly` and `SameSite=Lax`, and `Secure` unless turned off for local http.
- **CSRF:** API requests that change anything must have an `Origin`, or else a `Referer`, matching `SITE_URL`. Every page form and HTMX request also carries a per-session CSRF token, derived from the session id, so nothing extra is stored. Pages send `Referrer-Policy: no-referrer`, which makes browsers send `Origin: null` on ordinary form posts. Those are accepted only with a valid token, and a foreign `Origin` is always refused.
- **Security headers:** every response gets `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and a strict `Content-Security-Policy` that allows only this site's own files, with no inline scripts or styles. Pages and API responses get `Cache-Control: no-store`. Photos get `Cache-Control: private`, and static files can be cached.
- **Errors:** pages show friendly 403, 404, and 500 pages without internal details. Validation errors never repeat the values that were sent.
- **Photos:** each upload is decoded by Pillow and re-encoded as a new JPEG. That rejects anything that isn't a real image and removes EXIF data such as GPS location. Photos are stored under random names, at most 1,600 pixels on the long side, with at most 5 photos of 5 MB each. Anyone not allowed to see a photo gets 404, so its existence isn't revealed.
- **Deleting an account:** this removes the profile, preferences, photos and their files, ratings, impressions, matches and their messages, blocks, reports the user filed, sessions, and sign-in links. Reports about the user are kept as safety records, with the user removed. Audit rows are kept, with the user's ids replaced by `[deleted]`.
- **Real people and agents stay apart:** the app uses its own database (`data/live/`), so simulated users never appear in real people's feeds.

## Changing the schema

Tables are only created if they're missing, so after changing a table, delete the `.db` files, along with their `-wal` and `-shm` files, and re-import your data.

One migration exists: `scripts/migrate_schema.py` upgrades databases made before preferences moved off profiles, keeping every rating. It backs up each database first. `Store` refuses to open an old database and names the command to run.

To add a field to profiles or ratings:

1. Add the field to the model in `models.py`.
2. Add its column to `SCHEMA` in `store.py`.

The insert and update statements pick up the new field automatically. A field that isn't a simple value, like `interests` or `clubs`, also needs converting in `_profile_to_row` and `_row_to_profile`.

Personas are written out column by column, so a new persona field also needs adding to `save_persona` in `personas.py`.
