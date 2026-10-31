"""The public interface for profiles and ratings.

Code outside the data layer, especially the matching algorithm, should import
this module and not store.py. It deliberately offers no access to personas.
"""

import random
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import Profile, Rating, generate_user_id
from store import MutualPair, ProfileNotFoundError, Store


class ProfileManager:
    def __init__(
        self, db_path: str | Path, rng: random.Random | int | None = None
    ) -> None:
        """Open or create the database at db_path.

        Pass a random.Random or an int seed as rng to make generated user IDs
        reproducible. The same seed and the same sequence of add_profile calls
        give the same IDs.
        """
        self._store = Store(db_path)
        self._rng = random.Random(rng) if isinstance(rng, int) else rng

    # Profiles

    def add_profile(self, data: Mapping[str, Any]) -> Profile:
        """Validate data, generate a user_id, and store the new profile."""
        if "user_id" in data:
            raise ValueError("user_id is generated and cannot be supplied")
        profile = Profile.model_validate(
            {**data, "user_id": generate_user_id(self._rng)}
        )
        self._store.insert_profile(profile)
        return profile

    # user_id is positional-only so that passing user_id= as a change reaches the
    # check below instead of raising a confusing TypeError.
    def edit_profile(self, user_id: str, /, **changes: Any) -> Profile:
        """Apply changes and re-validate the full profile. user_id can't be changed."""
        if "user_id" in changes:
            raise ValueError("user_id is immutable")

        def apply(current: Profile) -> Profile:
            # model_copy(update=...) would skip validation.
            return Profile.model_validate({**current.model_dump(), **changes})

        return self._store.update_profile(user_id, apply)

    def delete_profile(self, user_id: str, anonymize_ratings: bool = False) -> None:
        """Delete a profile, its persona, and its ratings in every run.

        With anonymize_ratings=True, the user's ratings are kept with their ID
        replaced, and read back as models.ANONYMIZED_ID. Use this to preserve the
        scores of past runs. No references to the deleted user remain either way.
        """
        self._store.delete_profile(user_id, anonymize_ratings=anonymize_ratings)

    def get_profile(self, user_id: str) -> Profile:
        return self._store.get_profile(user_id)

    def list_profiles(self) -> list[Profile]:
        return self._store.list_profiles()

    def export_profiles_json(self, path: str | Path) -> None:
        self._store.export_profiles_json(path)

    # Ratings

    def add_rating(
        self,
        rater_id: str,
        target_id: str,
        score: int,
        run_id: str,
        timestamp: datetime | None = None,
    ) -> Rating:
        """Record a rating, replacing any earlier one for the same run and pair.

        The timestamp defaults to now. A simulation can pass its own clock so that
        runs can be replayed.
        """
        rating = Rating(
            rater_id=rater_id,
            target_id=target_id,
            score=score,
            timestamp=timestamp or datetime.now(timezone.utc),
            run_id=run_id,
        )
        for user_id in (rater_id, target_id):
            if not self._store.profile_exists(user_id):
                raise ProfileNotFoundError(f"no profile with user_id {user_id!r}")
        self._store.upsert_rating(rating)
        return rating

    def ratings_by(self, rater_id: str, run_id: str | None = None) -> list[Rating]:
        return self._store.ratings_by(rater_id, run_id)

    def ratings_of(self, target_id: str, run_id: str | None = None) -> list[Rating]:
        return self._store.ratings_of(target_id, run_id)

    def mutual_pairs(self, run_id: str, min_score: int) -> list[MutualPair]:
        return self._store.mutual_pairs(run_id, min_score)
