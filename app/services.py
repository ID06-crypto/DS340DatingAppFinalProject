"""Everything the endpoints share, created once per app."""

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

    def public_profile(self, user_id: str) -> dict[str, Any]:
        """A user's public profile and photo ids. Never their email, preferences, or ratings."""
        profile = self.manager.get_profile(user_id)
        public = PublicProfile.model_validate(profile.model_dump(exclude={"agent"}))
        return {
            "profile": public.model_dump(),
            "photo_ids": [photo.photo_id for photo in self.live.photos_for(user_id)],
        }

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
