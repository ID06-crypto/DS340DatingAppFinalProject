"""Everything the endpoints share, created once per app."""

import logging
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.config import Settings
from app.fake_names import pick_name
from app.ratelimit import RateLimiter
from live_store import Account, LiveStore
from manager import ProfileManager
from matching import Matcher
from models import PublicProfile

logger = logging.getLogger(__name__)


@dataclass
class Controls:
    """What admins can switch at runtime. Stored in app_settings."""

    matcher_weights: dict[str, float]
    signups_paused: bool
    feed_paused: bool
    rating_opens_at: datetime | None = None  # In UTC. Nobody can rate before it.
    matching_started: bool = False  # Nobody sees profiles or rates until an admin starts matching.
    results_open_at: datetime | None = None  # In UTC. Nobody sees a best match before it, or without it.


def first_name(name: str) -> str:
    return name.split()[0] if name.split() else name


@dataclass
class Services:
    settings: Settings
    manager: ProfileManager
    live: LiveStore
    matchers: dict[str, Matcher]
    rng: random.Random
    login_per_ip: RateLimiter
    # Matchers and the rng aren't thread-safe, and requests run on worker threads.
    matcher_lock: threading.Lock = field(default_factory=threading.Lock)
    # Wrong admin passwords allowed per email in 15 minutes.
    password_per_email: RateLimiter = field(default_factory=lambda: RateLimiter(5, 900))

    def consent_text(self) -> str:
        return (self.settings.consent_dir / f"v{self.settings.consent_version}.md").read_text(encoding="utf-8")

    def public_profile(self, user_id: str, own: bool = False) -> dict[str, Any]:
        """A user's public profile. Never their email, preferences, or ratings.

        The profile's name is the display name the app assigned, which is all anyone else
        sees, and religion only when it's given. The owner, and admins reviewing them
        (own=True), also see their real name, with the display name as shown_as.
        """
        profile = self.manager.get_profile(user_id)
        fields = PublicProfile.model_validate(profile.model_dump(exclude={"agent"})).model_dump()
        if own:
            fields["shown_as"] = fields["name"]
            fields["name"] = self.live.real_name(user_id) or fields["name"]
        else:
            fields["name"] = first_name(fields["name"])
            if fields["religion"] is None:
                del fields["religion"]
        return {"profile": fields}

    def pick_display_name(self, gender: str) -> str:
        with self.matcher_lock:  # The rng isn't thread-safe.
            return pick_name(gender, self.live.display_names_in_use(), self.rng)

    def give_display_names(self) -> int:
        """Give real people who joined before display names existed one, keeping their
        real name privately. Runs at startup, and does nothing once everyone has one."""
        missing = self.live.live_profiles_without_real_name()
        for user_id in missing:
            profile = self.manager.get_profile(user_id)
            self.live.set_real_name(user_id, profile.name)
            self.manager.edit_profile(user_id, name=self.pick_display_name(profile.gender))
        return len(missing)

    def controls(self) -> Controls:
        return Controls(
            matcher_weights={k: float(v) for k, v in self.live.get_setting("matcher_weights", {}).items()},
            signups_paused=bool(self.live.get_setting("signups_paused", False)),
            feed_paused=bool(self.live.get_setting("feed_paused", False)),
            rating_opens_at=datetime.fromisoformat(opens) if (opens := self.live.get_setting("rating_opens_at")) else None,
            matching_started=bool(self.live.get_setting("matching_started", False)),
            results_open_at=datetime.fromisoformat(shown) if (shown := self.live.get_setting("results_open_at")) else None,
        )

    def local_time(self, moment: datetime) -> datetime:
        return moment.astimezone(ZoneInfo(self.settings.study_timezone))

    def describe_time(self, moment: datetime) -> str:
        """Like "Monday, November 17 at 9:00 AM", in the study's time zone."""
        local = self.local_time(moment)
        return f"{local:%A, %B} {local.day} at {local:%I:%M %p}".replace(" at 0", " at ")

    def initialize_controls(self) -> None:
        """Set defaults on a new database, and log a changed MATCH_THRESHOLD once ratings exist."""
        if self.live.get_setting("matcher_weights") is None:
            self.live.set_setting("matcher_weights", {name: 1.0 for name in self.settings.enabled_matchers})
        previous = self.live.get_setting("match_threshold")
        if previous is not None and previous != self.settings.match_threshold and self.live.count_live_ratings():
            self.live.audit(
                "system",
                "match_threshold_changed",
                None,
                {"from": previous, "to": self.settings.match_threshold},
            )
        self.live.set_setting("match_threshold", self.settings.match_threshold)


    def missing_for_approval(self, account: Account) -> list[str]:
        missing = []
        if account.consent_version != self.settings.consent_version:
            missing.append("consent")
        if account.user_id is None:
            missing.append("profile")
        elif self.manager.get_preferences(account.user_id) is None:
            missing.append("preferences")
        return missing

    def delete_account(self, account: Account) -> None:
        """Delete an account and everything about it.

        Deleting the profile cascades to the account row, sessions, preferences, ratings,
        impressions, matches, blocks, surveys, and reports the user filed. Audit rows are
        kept, with the user's ids replaced.
        """
        if account.user_id is not None:
            self.manager.delete_profile(account.user_id)
        self.live.delete_account_row(account.account_id, account.email_normalized)
        self.live.anonymize_audit([i for i in (account.account_id, account.user_id) if i])
