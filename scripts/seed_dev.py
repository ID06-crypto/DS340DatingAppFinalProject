"""Fill the live app with about 40 fake, approved users for trying it locally. Never run in production.

    python scripts/seed_dev.py

It uses the same settings as the server (APP_MAIN_DB, PHOTO_DIR, and so on), and does
nothing if the fake users already exist. Some users have no religion, and a few pairs have
matched and messaged, so every page has something to show. Sign in as any printed email:
request a link, and the console email sender prints it.
"""

import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from app.config import Settings  # noqa: E402
from app.photos import MAX_PHOTOS, LocalDiskPhotoStorage, clean_photo, new_storage_key  # noqa: E402
from live_store import LiveStore  # noqa: E402
from manager import ProfileManager  # noqa: E402
from models import LIVE_RUN_ID  # noqa: E402

N_USERS = 40
SEED = 0
FIRST_NAMES = ["Alex", "Blake", "Casey", "Drew", "Emery", "Finley", "Gray", "Harper", "Indy", "Jordan",
               "Kai", "Logan", "Morgan", "Noor", "Oakley", "Parker", "Quinn", "Riley", "Sage", "Taylor"]
LAST_NAMES = ["Kim", "Patel", "Garcia", "Okafor", "Nguyen", "Smith", "Cohen", "Silva", "Ali", "Brown"]
MAJORS = ["biology", "economics", "computer science", "history", "psychology", "art history", "physics"]
INTERESTS = ["hiking", "chess", "cooking", "film", "running", "music", "climbing", "board games", "poetry"]
RELIGIONS = ["catholic", "jewish", "muslim", "hindu", "buddhist", "atheist", "agnostic"]
COLORS = ["steelblue", "indianred", "seagreen", "goldenrod", "slateblue", "darkorange", "teal", "sienna"]


def placeholder_photo(color: str, label: str) -> bytes:
    image = Image.new("RGB", (600, 800), color)
    ImageDraw.Draw(image).text((40, 40), label, fill="white")
    output = io.BytesIO()
    image.save(output, "JPEG")
    return clean_photo(output.getvalue())


def main() -> None:
    settings = Settings.from_env()
    settings.main_db.parent.mkdir(parents=True, exist_ok=True)
    manager = ProfileManager(settings.main_db, settings.persona_db)
    live = LiveStore(settings.main_db)
    storage = LocalDiskPhotoStorage(settings.photo_dir)
    rng = random.Random(SEED)

    if live.account_for_email("dev.user01@bu.edu") is not None:
        print("Fake users already exist. Delete the database to start over.")
        return

    users = []
    for i in range(1, N_USERS + 1):
        gender = rng.choices(["man", "woman", "nonbinary"], weights=[45, 45, 10])[0]
        interested_in = {
            "man": rng.choices([["woman"], ["man"], ["woman", "nonbinary"]], weights=[80, 10, 10])[0],
            "woman": rng.choices([["man"], ["woman"], ["man", "nonbinary"]], weights=[80, 10, 10])[0],
            "nonbinary": ["man", "woman", "nonbinary"],
        }[gender]
        email = f"dev.user{i:02d}@bu.edu"
        account, _ = live.get_or_create_account(email, admin=False)
        profile = manager.add_profile({
            "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "age": rng.randint(18, 24),
            "height_cm": rng.randint(152, 195),
            "religion": None if rng.random() < 0.3 else rng.choice(RELIGIONS),  # Some leave it blank.
            "school": "BU",
            "major": rng.choice(MAJORS),
            "job": rng.choice(["student", "tutor", "barista", "research assistant"]),
            "interests": rng.sample(INTERESTS, rng.randint(1, 3)),
            "clubs": [],
            "gender": gender,
            "looking_for": rng.choice(["long-term", "short-term", "just a date"]),
            "agent": False,
        })
        live.link_profile(account.account_id, profile.user_id)
        manager.set_preferences(profile.user_id, interested_in)
        live.record_consent(account.account_id, settings.consent_version)
        for n in range(rng.randint(1, 3)):
            key = new_storage_key()
            storage.save(key, placeholder_photo(rng.choice(COLORS), f"{profile.name.split()[0]}, photo {n + 1}"))
            live.add_photo(profile.user_id, key, MAX_PHOTOS)
        live.set_status(account.account_id, "approved", "seed_dev")
        users.append((email, profile.user_id))

    # A few rated impressions in both directions, some ending in matches with messages.
    matches = 0
    for email, user_id in users:
        for target in manager.eligible_candidates(user_id, LIVE_RUN_ID)[:3]:
            impression = live.create_impression(user_id, target, "random", len(manager.eligible_candidates(user_id)))
            live.claim_impression(impression.impression_id, user_id)
            score = rng.uniform(1, 10)
            manager.add_rating(user_id, target, (score - 1) / 9, LIVE_RUN_ID, matcher="random")
            reverse = [r for r in manager.ratings_by_rater(target, LIVE_RUN_ID) if r.target_id == user_id]
            if reverse and score >= settings.match_threshold and 1 + 9 * reverse[0].score >= settings.match_threshold:
                match = live.create_match(user_id, target)
                live.add_message(match.match_id, user_id, "Hi! Nice to match with you.")
                live.add_message(match.match_id, target, "Hey, likewise. How's your week going?")
                matches += 1

    # Then a handful of guaranteed matches: pairs who are in each other's pools rate each other highly.
    paired: set[str] = set()
    for email, user_id in users:
        if matches >= 8 or user_id in paired:
            continue
        partner = next(
            (t for t in manager.eligible_candidates(user_id, LIVE_RUN_ID)
             if t not in paired and user_id in manager.eligible_candidates(t, LIVE_RUN_ID)),
            None,
        )
        if partner is None:
            continue
        for rater, target in ((user_id, partner), (partner, user_id)):
            impression = live.create_impression(rater, target, "random", len(manager.eligible_candidates(rater)))
            live.claim_impression(impression.impression_id, rater)
            manager.add_rating(rater, target, (rng.uniform(8, 10) - 1) / 9, LIVE_RUN_ID, matcher="random")
        match = live.create_match(user_id, partner)
        live.add_message(match.match_id, user_id, "Hi! Nice to match with you.")
        live.add_message(match.match_id, partner, "Hey, likewise. How's your week going?")
        paired.update({user_id, partner})
        matches += 1

    print(f"Created {len(users)} approved users and {matches} matches in {settings.main_db}")
    print("Sign in as any of dev.user01@bu.edu to dev.user40@bu.edu. The console prints the link.")


if __name__ == "__main__":
    main()
