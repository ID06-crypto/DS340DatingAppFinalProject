"""The public interface for profiles, ratings, and personas.

ProfileManager opens the persona database, so matchers must not use it. They receive a
matching.PublicView instead.
"""

import random
from pathlib import Path
from typing import Any

from models import Persona, Profile, Rating, RatingTranscript, Run, generate_user_id
from personas import PersonaStore
from store import MutualPair, ProfileNotFoundError, Store


class ProfileManager:
    def __init__(
        self,
        main_db_path: str | Path = "main.db",
        persona_db_path: str | Path = "personas.db",
        rng: random.Random | None = None,
    ) -> None:
        """Open or create both databases. A seeded rng makes generated IDs reproducible."""
        if Path(main_db_path).resolve() == Path(persona_db_path).resolve():
            raise ValueError("personas must be stored in a separate file from main.db")
        self.store = Store(main_db_path)
        self.personas = PersonaStore(persona_db_path)
        self._rng = rng

    # Profiles

    def add_profile(self, data: dict[str, Any]) -> Profile:
        """Validate and store a profile, generating a user_id unless data supplies one."""
        if "user_id" not in data:
            data = {**data, "user_id": generate_user_id(self._rng)}
        profile = Profile.model_validate(data)
        self.store.insert_profile(profile)
        return profile

    # user_id is positional-only so that passing user_id= as a change reaches the
    # check below instead of raising a confusing TypeError.
    def edit_profile(self, user_id: str, /, **changes: Any) -> Profile:
        """Merge changes into the profile and re-validate all of it."""
        if "user_id" in changes:
            raise ValueError("user_id is immutable")

        def apply(current: Profile) -> Profile:
            # model_copy(update=...) would skip validation.
            return Profile.model_validate({**current.model_dump(), **changes})

        return self.store.update_profile(user_id, apply)

    def delete_profile(self, user_id: str) -> None:
        """Delete a profile, its ratings and their transcripts in every run, and its persona."""
        if not self.store.profile_exists(user_id):
            raise ProfileNotFoundError(f"no profile with user_id {user_id!r}")
        # The two files can't share a transaction. Deleting the hidden data first means
        # that if a later step fails, calling this again finishes the job.
        self.personas.delete_persona(user_id)
        self.personas.delete_transcripts_involving(user_id)
        self.store.delete_profile(user_id)

    def get_profile(self, user_id: str) -> Profile:
        return self.store.get_profile(user_id)

    def list_profiles(self) -> list[Profile]:
        return self.store.list_all_profiles()

    # Ratings

    def add_rating(
        self,
        rater_id: str,
        target_id: str,
        score: float,
        run_id: str,
        model_version: str | None = None,
        raw_output: str | None = None,
        input_snapshot: str | None = None,
        step: int | None = None,
    ) -> Rating:
        """Record a rating, replacing any earlier one for the same rater, target, and run.

        The rating goes in main.db. raw_output and input_snapshot can contain the rater's
        persona, so they go in personas.db, and are read back with get_transcript.
        """
        rating = Rating(
            rater_id=rater_id,
            target_id=target_id,
            score=score,
            run_id=run_id,
            model_version=model_version,
            step=step,
        )
        transcript = RatingTranscript(
            rater_id=rater_id,
            target_id=target_id,
            run_id=run_id,
            raw_output=raw_output,
            input_snapshot=input_snapshot,
        )
        for user_id in (rater_id, target_id):
            if not self.store.profile_exists(user_id):
                raise ProfileNotFoundError(f"no profile with user_id {user_id!r}")
        # The transcript goes first, so a saved rating always has its transcript. If the
        # rating then fails to save, the leftover transcript is replaced on retry.
        if raw_output is None and input_snapshot is None:
            # Otherwise a replaced rating would keep its old transcript.
            self.personas.delete_transcript(rater_id, target_id, run_id)
        else:
            self.personas.save_transcript(transcript)
        self.store.upsert_rating(rating)
        return rating

    def ratings_by_rater(self, rater_id: str, run_id: str | None = None) -> list[Rating]:
        return self.store.ratings_by_rater(rater_id, run_id)

    def ratings_of_target(
        self, target_id: str, run_id: str | None = None
    ) -> list[Rating]:
        return self.store.ratings_of_target(target_id, run_id)

    def mutual_pairs(
        self, min_score: float, run_id: str | None = None
    ) -> list[MutualPair]:
        return self.store.mutual_pairs(min_score, run_id)

    def ratings_in_run(self, run_id: str) -> list[Rating]:
        return self.store.ratings_in_run(run_id)

    def already_rated_pairs(self, run_id: str) -> set[tuple[str, str]]:
        return self.store.already_rated_pairs(run_id)

    def count_profiles(self) -> int:
        return self.store.count_profiles()

    def count_ratings(self, run_id: str | None = None) -> int:
        return self.store.count_ratings(run_id)

    # Runs

    def start_run(
        self,
        run_id: str,
        matcher_name: str,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> Run:
        """Record a new run. An existing run_id raises store.DuplicateRunError."""
        run = Run(
            run_id=run_id,
            matcher_name=matcher_name,
            seed=seed,
            config=config if config is not None else {},
        )
        self.store.insert_run(run)
        return run

    def finish_run(self, run_id: str) -> Run:
        return self.store.finish_run(run_id)

    def get_run(self, run_id: str) -> Run:
        return self.store.get_run(run_id)

    # Personas

    def add_persona(
        self,
        user_id: str,
        description: str,
        generation_seeds: dict[str, Any] | None = None,
    ) -> Persona:
        """Store a persona for an existing profile, replacing any earlier one."""
        if not self.store.profile_exists(user_id):
            raise ProfileNotFoundError(f"no profile with user_id {user_id!r}")
        persona = Persona(
            user_id=user_id,
            description=description,
            generation_seeds=generation_seeds if generation_seeds is not None else {},
        )
        self.personas.save_persona(persona)
        return persona

    def get_persona(self, user_id: str) -> Persona | None:
        return self.personas.get_persona(user_id)

    def get_transcript(
        self, rater_id: str, target_id: str, run_id: str
    ) -> RatingTranscript | None:
        return self.personas.get_transcript(rater_id, target_id, run_id)

    def transcripts_in_run(self, run_id: str) -> list[RatingTranscript]:
        return self.personas.transcripts_in_run(run_id)
