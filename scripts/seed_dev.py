"""Create a few fake, approved accounts for trying the app locally. Never run this in production.

    python scripts/seed_dev.py

It uses the same settings as the server (APP_MAIN_DB, PHOTO_DIR, and so on). Sign in as any
of the printed emails: request a link, and the console email sender prints it.
"""

import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app.config import Settings  # noqa: E402
from app.photos import LocalDiskPhotoStorage, clean_photo, new_storage_key  # noqa: E402
from app.photos import MAX_PHOTOS  # noqa: E402
from live_store import LiveStore  # noqa: E402
from manager import ProfileManager  # noqa: E402

PEOPLE = [
    # (email, name, gender, interested_in, looking_for, major, color)
    ("dev.alex@bu.edu", "Alex", "man", ["woman"], "long-term", "economics", "steelblue"),
    ("dev.blake@bu.edu", "Blake", "woman", ["man"], "short-term", "biology", "indianred"),
    ("dev.casey@bu.edu", "Casey", "nonbinary", ["man", "woman", "nonbinary"], "just a date", "art history", "seagreen"),
    ("dev.drew@bu.edu", "Drew", "woman", ["man", "nonbinary"], "long-term", "computer science", "goldenrod"),
    ("dev.emery@bu.edu", "Emery", "man", ["woman", "nonbinary"], "short-term", "history", "slateblue"),
    ("dev.finley@bu.edu", "Finley", "woman", ["woman"], "long-term", "psychology", "darkorange"),
]


def placeholder_photo(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (600, 800), color).save(output, "JPEG")
    return clean_photo(output.getvalue())


def main() -> None:
    settings = Settings.from_env()
    settings.main_db.parent.mkdir(parents=True, exist_ok=True)
    manager = ProfileManager(settings.main_db, settings.persona_db)
    live = LiveStore(settings.main_db)
    storage = LocalDiskPhotoStorage(settings.photo_dir)
    rng = random.Random(0)

    for email, name, gender, interested_in, looking_for, major, color in PEOPLE:
        account, created = live.get_or_create_account(email, admin=False)
        if not created and account.user_id is not None:
            print(f"{email}: already exists")
            continue
        profile = manager.add_profile({
            "name": name,
            "age": rng.randint(18, 24),
            "height_cm": rng.randint(155, 195),
            "religion": rng.choice(["none", "catholic", "jewish", "muslim", "hindu"]),
            "school": "BU",
            "major": major,
            "job": "student",
            "interests": rng.sample(["hiking", "chess", "cooking", "film", "running", "music"], 2),
            "clubs": [],
            "gender": gender,
            "looking_for": looking_for,
            "agent": False,
        })
        live.link_profile(account.account_id, profile.user_id)
        manager.set_preferences(profile.user_id, interested_in)
        live.record_consent(account.account_id, settings.consent_version)
        key = new_storage_key()
        storage.save(key, placeholder_photo(color))
        live.add_photo(profile.user_id, key, MAX_PHOTOS)
        live.set_status(account.account_id, "approved", "seed_dev")
        print(f"{email}: approved ({gender}, interested in {', '.join(interested_in)})")

    print(f"\nDatabase: {settings.main_db}")
    print("Sign in by requesting a link for one of these emails. The console prints it.")


if __name__ == "__main__":
    main()
