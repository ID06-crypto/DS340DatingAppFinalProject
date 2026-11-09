"""The rater interface: how a simulated user scores a profile they're shown.

Raters are the only components allowed to read personas. Each one sees the rater's
own profile and persona, plus the target's public profile. It never sees the
target's persona, and nothing carries over between calls.
"""

import functools
import json
import math
import random
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient

from models import Persona, Profile

class RatingResult(BaseModel):
    """A rater's answer.

    The score is the probability the rater likes the target. It should be from 0.0 to 1.0, but only needs to be a finite number here.
    A real model can answer out of range, and the simulation clamps that when saving.
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(strict=True)
    raw_output: str | None = None
    input_snapshot: str | None = None
    model_version: str | None = None

    @field_validator("score")
    @classmethod
    def check_finite(cls, score: float) -> float:
        if not math.isfinite(score):
            raise ValueError(f"score must be a finite number, got {score}")
        return score


def build_rater_input(
    rater_profile: Profile, rater_persona: Persona, target_profile: Profile
) -> dict[str, Any]:
    """Build the structured input a rater model receives.

    The target appears only as a public Profile, which includes their gender, as does
    the rater's own profile. There's no parameter through which the target's persona or
    preferences could arrive. The agent flag is left out so that knowing who's simulated
    can't influence the score.
    """
    return {
        "rater_profile": rater_profile.model_dump(exclude={"agent"}),
        "rater_persona": rater_persona.description,
        "target_profile": target_profile.model_dump(exclude={"agent"}),
    }


class Rater(ABC):
    @abstractmethod
    def rate(
        self, rater_profile: Profile, rater_persona: Persona, target_profile: Profile
    ) -> RatingResult:
        """Score target_profile as the rater would. Each call stands alone."""


class FakeRater(Rater):
    """Returns random scores, for tests and dry runs."""

    MODEL_VERSION = "fake-rater-v1"

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def rate(
        self, rater_profile: Profile, rater_persona: Persona, target_profile: Profile
    ) -> RatingResult:
        rater_input = build_rater_input(rater_profile, rater_persona, target_profile)
        score = round(self._rng.uniform(0.0, 1.0), 2)
        return RatingResult(
            score=score,
            raw_output=json.dumps({"score": score}),
            input_snapshot=json.dumps(rater_input),
            model_version=self.MODEL_VERSION,
        )


# Pinned rather than "jev-latest", so a new release can't change scores partway through
# an experiment. It's saved with every rating.
JEV_MODEL = "jev-1.14.0"
JEV_PRICE_PER_MTOK = 0.042  # In US dollars, charged on input tokens only.
JEV_QUESTION_ID = "like"
JEV_QUESTION = Noul(
    instructions=(
        "`rater` is a college student using a dating app, and `rater_persona` says what "
        "they're like and what they want in a partner. Judging as this person would, and "
        "following the preferences in `rater_persona`, would `rater` like `shown_profile`?"
    ),
)


@functools.cache
def _jev_client() -> TypeSafeClient:
    """Create one client when it's first needed. It reads TYPESAFE_API_KEY."""
    return TypeSafeClient(model=JEV_MODEL, retry=RetryPolicy(max_retries=5, backoff_max=30.0))


def build_jev_state(rater_input: dict[str, Any]) -> dict[str, Any]:
    """Arrange the rater input for Jev. user_ids are left out because they mean nothing to it."""

    def without_id(profile: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in profile.items() if key != "user_id"}

    return {
        "rater": without_id(rater_input["rater_profile"]),
        "rater_persona": rater_input["rater_persona"],
        "shown_profile": without_id(rater_input["target_profile"]),
    }


def _call_jev(rater_input: dict[str, Any]) -> tuple[float, str]:
    """The only place the Jev API is called. Returns (score, raw_output).

    The score is Jev's probability that the rater likes the target. The SDK retries rate
    limits, timeouts, and server errors itself.
    """
    response = _jev_client().system_one(
        state=build_jev_state(rater_input),
        questions={JEV_QUESTION_ID: JEV_QUESTION},
    )
    score = response.answers[JEV_QUESTION_ID].noul
    raw_output = json.dumps({
        "model": response.model,
        "noul": score,
        "input_tokens": response.usage.input_tokens,
    })
    return score, raw_output


class JevRater(Rater):
    """Rates through Jev. Needs the TYPESAFE_API_KEY environment variable."""

    MODEL_VERSION = JEV_MODEL

    def rate(
        self, rater_profile: Profile, rater_persona: Persona, target_profile: Profile
    ) -> RatingResult:
        rater_input = build_rater_input(rater_profile, rater_persona, target_profile)
        score, raw_output = _call_jev(rater_input)
        return RatingResult(
            score=score,
            raw_output=raw_output,
            input_snapshot=json.dumps(rater_input),
            model_version=self.MODEL_VERSION,
        )
