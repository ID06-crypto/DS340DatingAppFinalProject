"""The rater interface: how a simulated user scores a profile they're shown.

Raters are the only components allowed to read personas. Each one sees the rater's
own profile and persona, plus the target's public profile. It never sees the
target's persona, and nothing carries over between calls.
"""

import json
import math
import random
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models import Persona, Profile

SCORE_SCALE = {
    "min": 1,
    "max": 10,
    "anchors": {
        "1": "no interest",
        "5": "neutral",
        "10": "would message immediately",
    },
}


class RatingResult(BaseModel):
    """A rater's answer.

    The score should be from 1.0 to 10.0, but only needs to be a finite number here.
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

    The target appears only as a public Profile. There's no parameter through which
    their persona could arrive. The agent flag is left out so that knowing who's
    simulated can't influence the score.
    """
    return {
        "rater_profile": rater_profile.model_dump(exclude={"agent"}),
        "rater_persona": rater_persona.description,
        "target_profile": target_profile.model_dump(exclude={"agent"}),
        "scale": SCORE_SCALE,
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
        score = round(self._rng.uniform(1.0, 10.0), 2)
        return RatingResult(
            score=score,
            raw_output=json.dumps({"score": score}),
            input_snapshot=json.dumps(rater_input),
            model_version=self.MODEL_VERSION,
        )


def _call_jev(rater_input: dict[str, Any]) -> tuple[float, str]:
    """STUB: the only place the Jev API will be called. Returns (score, raw_output)."""
    raise NotImplementedError("The Jev API integration is pending.")


class JevRater(Rater):
    """STUB: rates through Jev once _call_jev is implemented."""

    MODEL_VERSION = "jev-pending"

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
