"""Everything the endpoints share, created once per app."""

import logging
import random
import threading
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.emails import EmailSender
from app.photos import PhotoStorage
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


def first_name(name: str) -> str:
    return name.split()[0] if name.split() else name


@dataclass
class Services:
    settings: Settings
    manager: ProfileManager
    live: LiveStore
    email_sender: EmailSender
    photo_storage: PhotoStorage
    matchers: dict[str, Matcher]
    rng: random.Random
    login_per_email: RateLimiter
    login_per_ip: RateLimiter
    messages_per_user: RateLimiter
    # Matchers and the rng aren't thread-safe, and requests run on worker threads.
    matcher_lock: threading.Lock = field(default_factory=threading.Lock)

    def consent_text(self) -> str:
        return (self.settings.consent_dir / f"v{self.settings.consent_version}.md").read_text(encoding="utf-8")

    def public_profile(self, user_id: str, own: bool = False) -> dict[str, Any]:
        """A user's public profile and photo ids. Never their email, preferences, or ratings.

        Other people see only a first name, and religion only when it's given.
        """
        profile = self.manager.get_profile(user_id)
        fields = PublicProfile.model_validate(profile.model_dump(exclude={"agent"})).model_dump()
        if not own:
            fields["name"] = first_name(fields["name"])
            if fields["religion"] is None:
                del fields["religion"]
        return {
            "profile": fields,
            "photo_ids": [photo.photo_id for photo in self.live.photos_for(user_id)],
        }

    def controls(self) -> Controls:
        return Controls(
            matcher_weights={k: float(v) for k, v in self.live.get_setting("matcher_weights", {}).items()},
            signups_paused=bool(self.live.get_setting("signups_paused", False)),
            feed_paused=bool(self.live.get_setting("feed_paused", False)),
        )

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

    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email, counting a failure without raising. Never logs the address or body."""
        try:
            self.email_sender.send(to, subject, body)
            return True
        except Exception:
            logger.error("An email couldn't be sent")
            self.live.record_event("email_failure")
            return False

    def missing_for_approval(self, account: Account) -> list[str]:
        missing = []
        if account.consent_version != self.settings.consent_version:
            missing.append("consent")
        if account.user_id is None:
            missing.append("profile")
        elif self.manager.get_preferences(account.user_id) is None:
            missing.append("preferences")
        if self.settings.require_photo and (
            account.user_id is None or not self.live.photos_for(account.user_id)
        ):
            missing.append("photo")
        return missing

    def delete_account(self, account: Account) -> None:
        """Delete an account and everything about it, including photo files.

        Deleting the profile cascades to the account row, sessions, preferences, ratings,
        impressions, matches and their messages, blocks, photos rows, and reports the user
        filed. Audit rows are kept, with the user's ids replaced.
        """
        photos = self.live.photos_for(account.user_id) if account.user_id else []
        if account.user_id is not None:
            self.manager.delete_profile(account.user_id)
        self.live.delete_account_row(account.account_id, account.email_normalized)
        for photo in photos:
            self.photo_storage.delete(photo.storage_key)
        self.live.anonymize_audit([i for i in (account.account_id, account.user_id) if i])
