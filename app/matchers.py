"""The registry of matchers the feed can serve from.

Each matcher is created once and kept in memory, so its state assumes one server process.
To add one, register a factory here and list its name in ENABLED_MATCHERS.
"""

import random
from collections.abc import Callable

from matching import Matcher, RandomMatcher

REGISTRY: dict[str, Callable[[random.Random], Matcher]] = {
    "random": RandomMatcher,
}


def build_matchers(names: tuple[str, ...], rng: random.Random) -> dict[str, Matcher]:
    unknown = [name for name in names if name not in REGISTRY]
    if unknown or not names:
        raise ValueError(f"ENABLED_MATCHERS must name registered matchers, got {list(names)}")
    return {name: REGISTRY[name](random.Random(rng.random())) for name in names}
