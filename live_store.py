"""Data access for the live app's tables in main.db. The tables are defined in live_schema.py.

Store creates the tables, so create a Store (or ProfileManager) for the file first.
Times are stored as UTC ISO strings with fixed precision, so they compare correctly as text.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from db import connect, transaction

AccountStatus = Literal["pending", "approved", "rejected", "suspended", "deleted"]
ReportReason = Literal["harassment", "spam", "fake", "inappropriate", "other"]
DELETED = "[deleted]"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_text(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def sha256(value: str) -> str:
    """Hash a login token or session id. Only hashes are ever stored."""
    return hashlib.sha256(value.encode()).hexdigest()


def new_id() -> str:
    return str(uuid.uuid4())


class Account(BaseModel):
    account_id: str
    email_normalized: str
    user_id: str | None
    status: AccountStatus
    role: Literal["user", "admin"]
    consent_version: str | None
    consent_at: str | None
    created_at: str
    approved_at: str | None
    approved_by: str | None
    experiment_opt_out: bool = False
    notify_on_match: bool = True
    paused: bool = False


class Impression(BaseModel):
    impression_id: str
    viewer_id: str
    target_id: str
    matcher: str
    pool_size: int
    shown_at: str
    rated_at: str | None
    predicted_score: float | None = None
    position: int | None = None


class Match(BaseModel):
    match_id: str
    user_a: str
    user_b: str
    created_at: str
    status: Literal["active", "unmatched"]
    unmatched_by: str | None
    unmatched_at: str | None

    def other(self, user_id: str) -> str:
        return self.user_b if user_id == self.user_a else self.user_a


class Survey(BaseModel):
    survey_id: str
    user_id: str
    kind: Literal["session", "post_match"]
    match_id: str | None
    score: int
    created_at: str


class Report(BaseModel):
    report_id: str
    reporter_id: str
    reported_id: str | None
    match_id: str | None
    reason: ReportReason
    details: str
    status: Literal["open", "resolved"]
    created_at: str
    resolved_by: str | None
    resolved_at: str | None


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


class LiveStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = db_path

    # Login tokens

    def get_or_create_account(self, email: str, admin: bool) -> tuple[Account, bool]:
        """Return the account for this email, creating it if needed. Also returns whether it's new.

        A new admin account is approved straight away. A new user account is pending.
        """
        now = to_text(utc_now())
        with transaction(self.db_path) as conn:
            row = conn.execute("SELECT * FROM accounts WHERE email_normalized = ?", (email,)).fetchone()
            if row is not None:
                return Account.model_validate(dict(row)), False
            account_id = new_id()
            conn.execute(
                """
                INSERT INTO accounts (account_id, email_normalized, status, role, created_at, approved_at, approved_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    email,
                    "approved" if admin else "pending",
                    "admin" if admin else "user",
                    now,
                    now if admin else None,
                    "ADMIN_EMAILS" if admin else None,
                ),
            )
        return self.get_account(account_id), True

    def account_for_user(self, user_id: str) -> Account | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM accounts WHERE user_id = ?", (user_id,)).fetchone()
        return Account.model_validate(dict(row)) if row else None

    def account_for_email(self, email: str) -> Account | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM accounts WHERE email_normalized = ?", (email,)).fetchone()
        return Account.model_validate(dict(row)) if row else None

    def set_flags(self, account_id: str, **flags: bool) -> None:
        """Set experiment_opt_out, notify_on_match, or paused."""
        allowed = {"experiment_opt_out", "notify_on_match", "paused"}
        if not flags or not set(flags) <= allowed:
            raise ValueError(f"flags must be among {sorted(allowed)}")
        assignments = ", ".join(f"{name} = ?" for name in flags)
        with transaction(self.db_path) as conn:
            conn.execute(
                f"UPDATE accounts SET {assignments} WHERE account_id = ?",
                [int(value) for value in flags.values()] + [account_id],
            )

    def get_account(self, account_id: str) -> Account | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,)).fetchone()
        return Account.model_validate(dict(row)) if row else None

    def list_accounts(self, status: str | None) -> list[Account]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM accounts WHERE ? IS NULL OR status = ? ORDER BY created_at",
                (status, status),
            ).fetchall()
        return [Account.model_validate(dict(row)) for row in rows]

    def set_status(self, account_id: str, status: AccountStatus, by: str) -> None:
        now = to_text(utc_now())
        with transaction(self.db_path) as conn:
            if status == "approved":
                conn.execute(
                    "UPDATE accounts SET status = ?, approved_at = ?, approved_by = ? WHERE account_id = ?",
                    (status, now, by, account_id),
                )
            else:
                conn.execute("UPDATE accounts SET status = ? WHERE account_id = ?", (status, account_id))

    def record_consent(self, account_id: str, version: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "UPDATE accounts SET consent_version = ?, consent_at = ? WHERE account_id = ?",
                (version, to_text(utc_now()), account_id),
            )

    def make_admin(self, account_id: str, by: str) -> None:
        """Give an account the admin role, approved."""
        now = to_text(utc_now())
        with transaction(self.db_path) as conn:
            conn.execute(
                "UPDATE accounts SET role = 'admin', status = 'approved', approved_at = COALESCE(approved_at, ?), approved_by = COALESCE(approved_by, ?) WHERE account_id = ?",
                (now, by, account_id),
            )

    # Passwords

    def password_hash(self, account_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT password_hash FROM passwords WHERE account_id = ?", (account_id,)).fetchone()
        return row["password_hash"] if row else None

    def must_change_password(self, account_id: str) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT must_change FROM passwords WHERE account_id = ?", (account_id,)).fetchone()
        return bool(row and row["must_change"])

    def set_password_hash(self, account_id: str, password_hash: str, must_change: bool = False) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO passwords (account_id, password_hash, updated_at, must_change) VALUES (?, ?, ?, ?)
                ON CONFLICT (account_id) DO UPDATE SET password_hash = excluded.password_hash,
                    updated_at = excluded.updated_at, must_change = excluded.must_change
                """,
                (account_id, password_hash, to_text(utc_now()), int(must_change)),
            )

    def end_other_sessions(self, account_id: str, keep_hash: str | None = None) -> None:
        """Sign an account out everywhere, except the session given."""
        with transaction(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE account_id = ? AND session_hash IS NOT ?", (account_id, keep_hash))

    def adopt_admin_passwords(self) -> int:
        """Move passwords from the admin_passwords table older versions used. Idempotent."""
        with transaction(self.db_path) as conn:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'admin_passwords'").fetchone()
            if not exists:
                return 0
            moved = conn.execute(
                "INSERT OR IGNORE INTO passwords (account_id, password_hash, updated_at) SELECT account_id, password_hash, updated_at FROM admin_passwords"
            ).rowcount
            conn.execute("DROP TABLE admin_passwords")
        return moved

    # Real names. Profiles hold the assigned display name.

    def real_name(self, user_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT real_name FROM private_names WHERE user_id = ?", (user_id,)).fetchone()
        return row["real_name"] if row else None

    def set_real_name(self, user_id: str, real_name: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO private_names (user_id, real_name) VALUES (?, ?) ON CONFLICT (user_id) DO UPDATE SET real_name = excluded.real_name",
                (user_id, real_name),
            )

    def live_profiles_without_real_name(self) -> list[str]:
        """Real people's profiles made before display names existed."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT a.user_id FROM accounts a JOIN profiles p ON p.user_id = a.user_id
                WHERE p.agent = 0 AND NOT EXISTS (SELECT 1 FROM private_names n WHERE n.user_id = a.user_id)
                ORDER BY a.created_at
                """
            ).fetchall()
        return [row["user_id"] for row in rows]

    def display_names_in_use(self) -> set[str]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT p.name FROM profiles p JOIN accounts a ON a.user_id = p.user_id WHERE p.agent = 0"
            ).fetchall()
        return {row["name"] for row in rows}

    def link_profile(self, account_id: str, user_id: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute("UPDATE accounts SET user_id = ? WHERE account_id = ?", (user_id, account_id))

    def delete_account_row(self, account_id: str, email: str) -> None:
        """Delete the account, its sessions (by cascade), and its login tokens."""
        with transaction(self.db_path) as conn:
            conn.execute("DELETE FROM accounts WHERE account_id = ?", (account_id,))

    # Sessions

    def create_session(self, session_hash: str, account_id: str, expires_at: datetime) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (session_hash, account_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (session_hash, account_id, to_text(utc_now()), to_text(expires_at)),
            )

    def account_for_session(self, session_hash: str) -> Account | None:
        found = self.session(session_hash)
        return found[0] if found else None

    def session(self, session_hash: str) -> tuple[Account, datetime] | None:
        """The account for a session, and when that session started (its sign-in time)."""
        now = to_text(utc_now())
        with transaction(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            row = conn.execute(
                """
                SELECT accounts.*, sessions.created_at AS session_started
                FROM sessions JOIN accounts USING (account_id)
                WHERE session_hash = ?
                """,
                (session_hash,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        started = datetime.fromisoformat(data.pop("session_started"))
        return Account.model_validate(data), started

    def delete_session(self, session_hash: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_hash = ?", (session_hash,))

    # Impressions

    def create_impression(
        self,
        viewer_id: str,
        target_id: str,
        matcher: str,
        pool_size: int,
        predicted_score: float | None = None,
        candidate_scores: dict[str, float] | None = None,
    ) -> Impression:
        """Record that a profile was shown, as the viewer's next position.

        candidate_scores, if the matcher scored the whole pool, is saved too, so offline
        evaluation can see what it would have ranked differently.
        """
        impression_id = new_id()
        with transaction(self.db_path) as conn:
            position = conn.execute(
                "SELECT COUNT(*) + 1 FROM impressions WHERE viewer_id = ?", (viewer_id,)
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO impressions
                    (impression_id, viewer_id, target_id, matcher, pool_size, shown_at, predicted_score, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (impression_id, viewer_id, target_id, matcher, pool_size, to_text(utc_now()), predicted_score, position),
            )
            if candidate_scores:
                conn.executemany(
                    "INSERT INTO impression_candidates (impression_id, candidate_id, predicted_score) VALUES (?, ?, ?)",
                    [(impression_id, candidate, float(score)) for candidate, score in candidate_scores.items()],
                )
        return self.get_impression(impression_id)

    def candidate_scores(self, impression_id: str) -> dict[str, float]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT candidate_id, predicted_score FROM impression_candidates WHERE impression_id = ?",
                (impression_id,),
            ).fetchall()
        return {row["candidate_id"]: row["predicted_score"] for row in rows}

    def pending_impression(self, viewer_id: str) -> Impression | None:
        """The viewer's latest impression that hasn't been rated yet, if any."""
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM impressions WHERE viewer_id = ? AND rated_at IS NULL ORDER BY shown_at DESC LIMIT 1",
                (viewer_id,),
            ).fetchone()
        return Impression.model_validate(dict(row)) if row else None

    def unrated_impressions(self, viewer_id: str) -> list[Impression]:
        """Every impression the viewer hasn't rated yet, newest first."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM impressions WHERE viewer_id = ? AND rated_at IS NULL ORDER BY shown_at DESC",
                (viewer_id,),
            ).fetchall()
        return [Impression.model_validate(dict(row)) for row in rows]

    def count_rated(self, viewer_id: str) -> int:
        with connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM impressions WHERE viewer_id = ? AND rated_at IS NOT NULL", (viewer_id,)
            ).fetchone()[0]

    def get_impression(self, impression_id: str) -> Impression | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM impressions WHERE impression_id = ?", (impression_id,)).fetchone()
        return Impression.model_validate(dict(row)) if row else None

    def claim_impression(self, impression_id: str, viewer_id: str) -> Literal["claimed", "missing", "rated"]:
        """Mark an impression rated, atomically, so it can only be rated once."""
        with transaction(self.db_path) as conn:
            row = conn.execute(
                "SELECT rated_at FROM impressions WHERE impression_id = ? AND viewer_id = ?",
                (impression_id, viewer_id),
            ).fetchone()
            if row is None:
                return "missing"
            if row["rated_at"] is not None:
                return "rated"
            conn.execute(
                "UPDATE impressions SET rated_at = ? WHERE impression_id = ?",
                (to_text(utc_now()), impression_id),
            )
        return "claimed"

    def unclaim_impression(self, impression_id: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute("UPDATE impressions SET rated_at = NULL WHERE impression_id = ?", (impression_id,))

    def was_shown(self, viewer_id: str, target_id: str) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM impressions WHERE viewer_id = ? AND target_id = ?", (viewer_id, target_id)
            ).fetchone()
        return row is not None

    # Matches

    def create_match(self, a: str, b: str) -> Match:
        """Create an active match, or return the existing one if two requests race."""
        user_a, user_b = _pair(a, b)
        with transaction(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO matches (match_id, user_a, user_b, created_at, status)
                VALUES (?, ?, ?, ?, 'active')
                """,
                (new_id(), user_a, user_b, to_text(utc_now())),
            )
        return self.active_match_between(a, b)

    def get_match(self, match_id: str) -> Match | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,)).fetchone()
        return Match.model_validate(dict(row)) if row else None

    def active_match_between(self, a: str, b: str) -> Match | None:
        user_a, user_b = _pair(a, b)
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM matches WHERE user_a = ? AND user_b = ? AND status = 'active'", (user_a, user_b)
            ).fetchone()
        return Match.model_validate(dict(row)) if row else None

    def active_matches_for(self, user_id: str) -> list[Match]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM matches WHERE status = 'active' AND (user_a = ? OR user_b = ?) ORDER BY created_at",
                (user_id, user_id),
            ).fetchall()
        return [Match.model_validate(dict(row)) for row in rows]

    def unmatch(self, match_id: str, by: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "UPDATE matches SET status = 'unmatched', unmatched_by = ?, unmatched_at = ? WHERE match_id = ? AND status = 'active'",
                (by, to_text(utc_now()), match_id),
            )

    def matches_for(self, user_id: str) -> list[Match]:
        """Every match this user has been in, active or not."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM matches WHERE user_a = ? OR user_b = ? ORDER BY created_at", (user_id, user_id)
            ).fetchall()
        return [Match.model_validate(dict(row)) for row in rows]

    # Blocks

    def blocked_either_way(self, a: str, b: str) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM blocks WHERE (blocker_id = ? AND blocked_id = ?) OR (blocker_id = ? AND blocked_id = ?)",
                (a, b, b, a),
            ).fetchone()
        return row is not None

    def add_block(self, blocker_id: str, blocked_id: str) -> None:
        """Block someone and end any active match between the two, in one transaction."""
        now = to_text(utc_now())
        user_a, user_b = _pair(blocker_id, blocked_id)
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO blocks (blocker_id, blocked_id, created_at) VALUES (?, ?, ?)",
                (blocker_id, blocked_id, now),
            )
            conn.execute(
                """
                UPDATE matches SET status = 'unmatched', unmatched_by = ?, unmatched_at = ?
                WHERE user_a = ? AND user_b = ? AND status = 'active'
                """,
                (blocker_id, now, user_a, user_b),
            )

    def remove_block(self, blocker_id: str, blocked_id: str) -> bool:
        with transaction(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM blocks WHERE blocker_id = ? AND blocked_id = ?", (blocker_id, blocked_id)
            )
        return cursor.rowcount > 0

    def blocks_by(self, blocker_id: str) -> list[dict[str, str]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT blocked_id, created_at FROM blocks WHERE blocker_id = ? ORDER BY created_at",
                (blocker_id,),
            ).fetchall()
        return [{"user_id": row["blocked_id"], "created_at": row["created_at"]} for row in rows]

    # Reports

    def add_report(
        self, reporter_id: str, reported_id: str, match_id: str | None, reason: ReportReason, details: str
    ) -> Report:
        report_id = new_id()
        with transaction(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO reports (report_id, reporter_id, reported_id, match_id, reason, details, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
                """,
                (report_id, reporter_id, reported_id, match_id, reason, details, to_text(utc_now())),
            )
        return self.get_report(report_id)

    def get_report(self, report_id: str) -> Report | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,)).fetchone()
        return Report.model_validate(dict(row)) if row else None

    def list_reports(self, status: str | None) -> list[Report]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM reports WHERE ? IS NULL OR status = ? ORDER BY created_at", (status, status)
            ).fetchall()
        return [Report.model_validate(dict(row)) for row in rows]

    def resolve_report(self, report_id: str, by: str) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "UPDATE reports SET status = 'resolved', resolved_by = ?, resolved_at = ? WHERE report_id = ?",
                (by, to_text(utc_now()), report_id),
            )

    # Surveys

    def add_survey(self, user_id: str, kind: str, score: int, match_id: str | None = None) -> Survey:
        survey_id = new_id()
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO surveys (survey_id, user_id, kind, match_id, score, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (survey_id, user_id, kind, match_id, score, to_text(utc_now())),
            )
        return Survey(survey_id=survey_id, user_id=user_id, kind=kind, match_id=match_id, score=score, created_at=to_text(utc_now()))

    def record_survey_prompt(self, user_id: str, kind: str, match_id: str | None = None) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO survey_prompts (user_id, kind, match_id, shown_at) VALUES (?, ?, ?, ?)",
                (user_id, kind, match_id, to_text(utc_now())),
            )

    def last_survey_prompt(self, user_id: str, kind: str, match_id: str | None = None) -> datetime | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT MAX(shown_at) FROM survey_prompts
                WHERE user_id = ? AND kind = ? AND (? IS NULL OR match_id = ?)
                """,
                (user_id, kind, match_id, match_id),
            ).fetchone()
        return datetime.fromisoformat(row[0]) if row[0] else None

    # Settings and events

    def get_setting(self, key: str, default: Any = None) -> Any:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value, sort_keys=True)),
            )

    def record_event(self, kind: str) -> None:
        """Count something that went wrong, such as a failed email. Stores no personal data."""
        with transaction(self.db_path) as conn:
            conn.execute("INSERT INTO app_events (kind, created_at) VALUES (?, ?)", (kind, to_text(utc_now())))

    def count_events(self) -> dict[str, int]:
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT kind, COUNT(*) FROM app_events GROUP BY kind").fetchall()
        return {row[0]: row[1] for row in rows}

    def count_live_ratings(self) -> int:
        with connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM ratings WHERE run_id = 'live'").fetchone()[0]

    # Audit log

    def audit(self, actor_account_id: str, action: str, target: str | None, details: dict[str, Any]) -> None:
        with transaction(self.db_path) as conn:
            conn.execute(
                "INSERT INTO audit_log (actor_account_id, action, target, details, created_at) VALUES (?, ?, ?, ?, ?)",
                (actor_account_id, action, target, json.dumps(details, sort_keys=True), to_text(utc_now())),
            )

    def audit_entries(self, action: str | None = None, actor: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Audit rows, oldest first, optionally filtered. With limit, the newest ones."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM audit_log
                    WHERE (? IS NULL OR action = ?) AND (? IS NULL OR actor_account_id = ?)
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id
                """,
                (action, action, actor, actor, limit if limit is not None else -1),
            ).fetchall()
        return [dict(row) for row in rows]

    def anonymize_audit(self, identifiers: list[str]) -> None:
        """Keep audit rows about a deleted user, with their ids replaced by a placeholder."""
        with transaction(self.db_path) as conn:
            for identifier in identifiers:
                conn.execute("UPDATE audit_log SET actor_account_id = ? WHERE actor_account_id = ?", (DELETED, identifier))
                conn.execute("UPDATE audit_log SET target = ? WHERE target = ?", (DELETED, identifier))
                conn.execute(
                    "UPDATE audit_log SET details = REPLACE(details, ?, ?) WHERE instr(details, ?) > 0",
                    (identifier, DELETED, identifier),
                )
