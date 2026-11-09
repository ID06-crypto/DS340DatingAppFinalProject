"""The registry of matchers the feed can serve from.

Every registered matcher is created once and kept in memory, so its state assumes one
server process. ENABLED_MATCHERS sets which are switched on when the database is new.
After that, admins switch them on and off and set their weights on the controls page.
To add one, register a factory here.
"""

import random
from collections.abc import Callable

from matching import Matcher, RandomMatcher

REGISTRY: dict[str, Callable[[random.Random], Matcher]] = {
    "random": RandomMatcher,
}


def build_matchers(enabled: tuple[str, ...], default: str, rng: random.Random) -> dict[str, Matcher]:
    """One instance of every registered matcher, after checking the configured names."""
    unknown = [name for name in enabled if name not in REGISTRY]
    if unknown or not enabled:
        raise ValueError(f"ENABLED_MATCHERS must name registered matchers, got {list(enabled)}")
    if default not in REGISTRY:
        raise ValueError(f"DEFAULT_MATCHER must name a registered matcher, got {default!r}")
    return {name: REGISTRY[name](random.Random(rng.random())) for name in sorted(REGISTRY)}
