"""The matcher interface, and what a matcher is allowed to see.

To add a real algorithm, subclass Matcher. Nothing else needs to change.
"""

import random
from abc import ABC, abstractmethod

from models import Profile, Rating
from store import Store


class PublicView:
    """Read-only access to public data, limited to one run.

    It's built from the main store alone, so it holds no reference to personas.db or
    to ProfileManager.

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

    def list_profiles(self) -> list[Profile]:
        return self._store.list_all_profiles()

    def get_profile(self, user_id: str) -> Profile:
        return self._store.get_profile(user_id)

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
