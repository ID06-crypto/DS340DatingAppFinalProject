"""The matcher interface, and what a matcher is allowed to see.

To add a real algorithm, subclass Matcher. Nothing else needs to change.
"""

import random
from abc import ABC, abstractmethod

from models import Profile, PublicProfile, Rating
from store import Store


_GENDER_OF_INTEREST = {"men": "man", "women": "woman"}


def _to_public(profile: Profile) -> PublicProfile:
    return PublicProfile.model_validate(profile.model_dump(exclude={"agent"}))


def is_compatible(user: PublicProfile, other: PublicProfile) -> bool:
    """Whether each of the two is the gender the other is interested in."""
    return (
        _GENDER_OF_INTEREST[user.interested_in] == other.gender
        and _GENDER_OF_INTEREST[other.interested_in] == user.gender
    )


class PublicView:
    """Read-only access to public data, limited to one run.

    It's built from the main store alone, so it holds no reference to personas.db or
    to ProfileManager. Profiles come back as PublicProfile, without the agent flag.

    Only this run's ratings are visible, so a run's results don't depend on what
    earlier runs left in the database.
    """

    __slots__ = ("_store", "_run_id")

    def __init__(self, store: Store, run_id: str) -> None:
        self._store = store
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def list_profiles(self) -> list[PublicProfile]:
        return [_to_public(profile) for profile in self._store.list_all_profiles()]

    def get_profile(self, user_id: str) -> PublicProfile:
        return _to_public(self._store.get_profile(user_id))

    def ratings_by_rater(self, rater_id: str) -> list[Rating]:
        return self._store.ratings_by_rater(rater_id, self._run_id)

    def ratings_of_target(self, target_id: str) -> list[Rating]:
        return self._store.ratings_of_target(target_id, self._run_id)

    def already_rated(self, user_id: str) -> set[str]:
        """The profile IDs this user has already rated in this run."""
        ratings = self._store.ratings_by_rater(user_id, self._run_id)
        return {rating.target_id for rating in ratings}


class Matcher(ABC):
    """Chooses which profile each simulated user sees next."""

    name: str

    @abstractmethod
    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        """Return the profile to show this user next, or None if there's nothing left."""

    def observe(self, rating: Rating) -> None:
        """Called after every saved rating."""


class RandomMatcher(Matcher):
    """Picks uniformly among profiles this user hasn't rated in this run."""

    name = "random"

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        seen = view.already_rated(user_id) | {user_id}
        # Sorted so that the choice doesn't depend on the order profiles were stored.
        candidates = sorted(p.user_id for p in view.list_profiles() if p.user_id not in seen)
        return self._rng.choice(candidates) if candidates else None


class CompatibleRandomMatcher(Matcher):
    """Picks uniformly among compatible profiles this user hasn't rated in this run.

    This is the baseline a real matcher has to beat. Filtering by gender and interest
    is easy, so a matcher shouldn't get credit just for doing that.

    Profiles are read once and kept, which is much faster on long runs. Use a new matcher
    for each run, and don't edit profiles during one.
    """

    name = "compatible-random"

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._profiles: list[PublicProfile] | None = None
        self._compatible: dict[str, list[str]] = {}

    def _compatible_with(self, user_id: str, view: PublicView) -> list[str]:
        if user_id not in self._compatible:
            if self._profiles is None:
                self._profiles = view.list_profiles()
            user = next(p for p in self._profiles if p.user_id == user_id)
            self._compatible[user_id] = sorted(
                p.user_id for p in self._profiles if p.user_id != user_id and is_compatible(user, p)
            )
        return self._compatible[user_id]

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        seen = view.already_rated(user_id)
        candidates = [c for c in self._compatible_with(user_id, view) if c not in seen]
        return self._rng.choice(candidates) if candidates else None
