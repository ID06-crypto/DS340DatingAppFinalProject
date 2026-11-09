"""Pydantic models for profiles, preferences, ratings, and simulated-user personas."""

import random
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]

Gender = Literal["man", "woman", "nonbinary"]
GENDERS: tuple[Gender, ...] = ("man", "woman", "nonbinary")

# Ratings made by real people in the app. Simulation runs use any other run_id.
LIVE_RUN_ID = "live"


def generate_user_id(rng: random.Random | None = None) -> str:
    """Return a random UUID4 string, reproducible if given a seeded random.Random."""
    if rng is None:
        return str(uuid.uuid4())
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


class PublicProfile(BaseModel):
    """The public profile, visible to other users, agents, and the matching algorithm.

    Ratings are deliberately not stored here. "Who rated me" and "who I rated"
    are queries against the ratings table, so the two can't drift out of sync.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str
    name: NonEmptyStr
    age: int = Field(ge=18, le=100, strict=True)
    height_cm: int = Field(ge=120, le=230, strict=True)
    religion: str | None = None  # Optional. Blank means "prefer not to say".
    school: NonEmptyStr
    major: NonEmptyStr
    job: NonEmptyStr
    interests: list[str] = Field(min_length=1)
    clubs: list[str]  # Required, but can be empty for someone in no clubs.
    gender: Gender
    looking_for: Literal["short-term", "long-term", "just a date"]

    @field_validator("user_id")
    @classmethod
    def check_uuid4(cls, user_id: str) -> str:
        try:
            parsed = uuid.UUID(user_id)
        except ValueError:
            raise ValueError(f"user_id must be a UUID4 string, got {user_id!r}") from None
        # The string comparison rejects other spellings of the same UUID, such as
        # uppercase, which would otherwise count as a different user.
        if parsed.version != 4 or str(parsed) != user_id:
            raise ValueError(f"user_id must be a UUID4 string, got {user_id!r}")
        return user_id

    @field_validator("religion", mode="before")
    @classmethod
    def blank_religion_is_none(cls, religion: object) -> object:
        if isinstance(religion, str):
            return religion.strip() or None
        return religion

    @field_validator("interests", "clubs")
    @classmethod
    def normalize_items(cls, items: list[str]) -> list[str]:
        """Strip, lowercase, and deduplicate, keeping first-seen order."""
        normalized = [item.strip().lower() for item in items]
        if not all(normalized):
            raise ValueError("must not contain blank items")
        return list(dict.fromkeys(normalized))


class Profile(PublicProfile):
    """A stored profile: the public fields, plus a label the matching algorithm never sees.

    agent would be True for everyone in training and False for everyone in testing, so
    a model that could read it would learn nothing useful from it and could misbehave
    in testing. Use it for splitting and analyzing data.
    """

    agent: bool = Field(strict=True)  # True for a simulated user, False for a real person.


class Preferences(BaseModel):
    """PRIVATE: which genders a user wants to be shown. Only the user may see this.

    It's never part of a profile, a PublicView, or any other user's response. It only
    decides who appears in someone's eligible pool. "Everyone" is stored as all genders.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: NonEmptyStr
    interested_in: frozenset[Gender] = Field(min_length=1)


def mutually_interested(
    gender_a: str, interested_a: frozenset[str], gender_b: str, interested_b: frozenset[str]
) -> bool:
    """Whether each person is a gender the other wants to be shown."""
    return gender_b in interested_a and gender_a in interested_b


class Rating(BaseModel):
    """One user's score of another within a run. This is public data, kept in main.db.

    What the rater was shown and what it answered can contain its persona, so that's
    kept separately, in a RatingTranscript.
    """

    model_config = ConfigDict(extra="forbid")

    rater_id: NonEmptyStr
    target_id: NonEmptyStr
    score: float = Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False)  # P(rater likes target).
    run_id: NonEmptyStr
    timestamp: UtcDatetime = Field(default_factory=_utc_now)
    model_version: str | None = None
    step: int | None = Field(default=None, ge=1, strict=True)
    matcher: str | None = None  # The matcher that served this profile.

    @model_validator(mode="after")
    def reject_self_rating(self) -> Self:
        if self.rater_id == self.target_id:
            raise ValueError("a user cannot rate themselves")
        return self


class Run(BaseModel):
    """One matching experiment. Its ratings share its run_id."""

    model_config = ConfigDict(extra="forbid")

    run_id: NonEmptyStr
    matcher_name: NonEmptyStr
    seed: int | None = Field(default=None, strict=True)
    config: dict[str, Any] = Field(default_factory=dict)
    started_at: UtcDatetime = Field(default_factory=_utc_now)
    finished_at: UtcDatetime | None = None


class Persona(BaseModel):
    """HIDDEN data for a simulated user. Never expose this to the matching algorithm.

    A persona is the ground truth behind a simulated user's ratings. If the matching
    algorithm could see it, training would learn from leaked answers rather than from
    observed behavior. Personas are stored in their own database file for this reason.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: NonEmptyStr
    description: NonEmptyStr
    generation_seeds: dict[str, Any] = Field(default_factory=dict)
    created_at: UtcDatetime = Field(default_factory=_utc_now)


class RatingTranscript(BaseModel):
    """HIDDEN record of how a rating was produced. Never expose this to the matching algorithm.

    input_snapshot is exactly what the rater was shown, which includes its persona, and
    raw_output may repeat it. That's why this is stored in personas.db, not with the rating.
    """

    model_config = ConfigDict(extra="forbid")

    rater_id: NonEmptyStr
    target_id: NonEmptyStr
    run_id: NonEmptyStr
    raw_output: str | None = None
    input_snapshot: str | None = None
