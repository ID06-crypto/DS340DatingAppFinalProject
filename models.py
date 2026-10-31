"""Pydantic models for profiles, ratings, and simulated-agent personas."""

import random
import uuid
from datetime import datetime, timezone
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# Reported in place of a deleted user's ID on ratings kept by an anonymizing delete.
ANONYMIZED_ID = "__anonymized__"

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def generate_user_id(rng: random.Random | int | None = None) -> str:
    """Return a UUID4 string, reproducible when given a random.Random or an int seed."""
    if rng is None:
        return str(uuid.uuid4())
    if isinstance(rng, int):
        rng = random.Random(rng)
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


class Profile(BaseModel):
    """The public profile, visible to other users and agents.

    Ratings are deliberately not stored here. "Who rated me" and "who I rated"
    are queries against the ratings table, so the two can't drift out of sync.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: NonEmptyStr
    name: NonEmptyStr
    age: int = Field(ge=18, le=100, strict=True)
    height_cm: int = Field(ge=120, le=230, strict=True)
    religion: NonEmptyStr
    school: NonEmptyStr
    job: NonEmptyStr
    interests: list[str] = Field(min_length=1)

    @field_validator("interests")
    @classmethod
    def normalize_interests(cls, interests: list[str]) -> list[str]:
        """Lowercase, strip, and deduplicate, keeping first-seen order."""
        normalized = [interest.strip().lower() for interest in interests]
        if not all(normalized):
            raise ValueError("interests must not contain blank items")
        return list(dict.fromkeys(normalized))


class Rating(BaseModel):
    """One user's score of another within a training run."""

    model_config = ConfigDict(extra="forbid")

    rater_id: NonEmptyStr
    target_id: NonEmptyStr
    score: int = Field(ge=1, le=10, strict=True)
    timestamp: AwareDatetime
    run_id: NonEmptyStr

    @field_validator("timestamp")
    @classmethod
    def to_utc(cls, timestamp: datetime) -> datetime:
        return timestamp.astimezone(timezone.utc)

    @model_validator(mode="after")
    def reject_self_rating(self) -> Self:
        if self.rater_id == self.target_id:
            raise ValueError("a user cannot rate themselves")
        return self


class Persona(BaseModel):
    """A simulated agent's hidden preferences: what they secretly value.

    This is the ground truth agents use to decide how to rate others. It must
    never reach the matching algorithm, or training would learn from leaked
    answers rather than from observed ratings. That's why it lives apart from
    Profile and why manager.py exposes no way to read it.

    The fields are placeholders until the agent simulation is designed.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: NonEmptyStr
    valued_traits: dict[str, float] = Field(default_factory=dict)
    notes: str = ""
