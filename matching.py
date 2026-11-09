"""The matcher interface, and what a matcher is allowed to see.

To add a real algorithm, subclass Matcher. Nothing else needs to change. A matcher must
choose from view.eligible_candidates(user_id). Anything else counts as a failed step in a
simulation and is refused by the app.
"""

import random
from abc import ABC, abstractmethod

from models import Profile, PublicProfile, Rating
from store import Store



def _to_public(profile: Profile) -> PublicProfile:
    return PublicProfile.model_validate(profile.model_dump(exclude={"agent"}))


class PublicView:
    """Read-only access to public data, limited to one run.

    It's built from the main store alone, so it holds no reference to personas.db or
    to ProfileManager. Profiles come back as PublicProfile, without the agent flag, and
    preferences are never exposed. The eligible pool applies them without revealing them.

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

    def eligible_candidates(self, user_id: str) -> list[str]:
        """The user_ids this user may be shown next, sorted. Matchers must pick from these."""
        return self._store.eligible_candidates(user_id, self._run_id)

    def ratings_by_rater(self, rater_id: str) -> list[Rating]:
        return self._store.ratings_by_rater(rater_id, self._run_id)

    def ratings_of_target(self, target_id: str) -> list[Rating]:
        return self._store.ratings_of_target(target_id, self._run_id)

    def already_rated(self, user_id: str) -> set[str]:
        """The profile IDs this user has already rated in this run."""
        ratings = self._store.ratings_by_rater(user_id, self._run_id)
        return {rating.target_id for rating in ratings}


class Matcher(ABC):
    """Chooses which profile each user sees next.

    description is plain language, shown to users on the transparency page.
    """

    name: str
    description: str = "No description yet."

    @abstractmethod
    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        """Return the profile to show this user next, or None if there's nothing left."""

    def observe(self, rating: Rating) -> None:
        """Called after every saved rating."""

    def score_candidates(
        self, user_id: str, view: PublicView, candidates: list[str]
    ) -> dict[str, float] | None:
        """Optionally score every candidate in the pool, from 0 (unlikely to like) to 1.

        Scores are logged with each impression, so offline evaluation can compare them
        with the ratings people give. Matchers that don't predict return None.
        """
        return None


class RandomMatcher(Matcher):
    """Picks uniformly from the user's eligible pool.

    This is the baseline a real matcher has to beat. The pool already applies mutual
    gender interest, so a matcher gets no credit just for that.
    """

    name = "random"
    description = (
        "Shows you someone at random from everyone you're eligible to see. It's the "
        "baseline the other algorithms are compared against."
    )

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def next_profile(self, user_id: str, view: PublicView) -> str | None:
        candidates = view.eligible_candidates(user_id)  # Already sorted, so seeds repeat.
        return self._rng.choice(candidates) if candidates else None


class CompatibleRandomMatcher(RandomMatcher):
    """The same as RandomMatcher, under the name earlier runs were recorded with.

    Before the eligible pool existed, only this matcher filtered by gender. It's kept so
    that resuming or comparing those runs still finds the matcher name they used.
    """

    name = "compatible-random"
