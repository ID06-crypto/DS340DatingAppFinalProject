"""Tables for the live app, stored in main.db alongside profiles and ratings.

Emails exist only in accounts. Login tokens and session ids are stored as SHA-256
hashes, never as the raw values.

Deleting a profile cascades to everything about that user here. Reports about a deleted
user are kept, with their ids cleared, because they're safety records.
"""

LIVE_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS accounts (
        account_id       TEXT PRIMARY KEY,
        email_normalized TEXT NOT NULL UNIQUE,
        user_id          TEXT UNIQUE REFERENCES profiles (user_id) ON DELETE CASCADE,
        status           TEXT NOT NULL
            CHECK (status IN ('pending', 'approved', 'rejected', 'suspended', 'deleted')),
        role             TEXT NOT NULL CHECK (role IN ('user', 'admin')),
        consent_version  TEXT,
        consent_at       TEXT,
        created_at       TEXT NOT NULL,
        approved_at      TEXT,
        approved_by      TEXT,
        experiment_opt_out INTEGER NOT NULL DEFAULT 0 CHECK (experiment_opt_out IN (0, 1)),
        notify_on_match    INTEGER NOT NULL DEFAULT 1 CHECK (notify_on_match IN (0, 1)),
        paused             INTEGER NOT NULL DEFAULT 0 CHECK (paused IN (0, 1))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_accounts_status ON accounts (status)",
    """
    CREATE TABLE IF NOT EXISTS login_tokens (
        token_hash       TEXT PRIMARY KEY,
        email_normalized TEXT NOT NULL,
        created_at       TEXT NOT NULL,
        expires_at       TEXT NOT NULL,
        used_at          TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_login_tokens_email ON login_tokens (email_normalized)",
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_hash TEXT PRIMARY KEY,
        account_id   TEXT NOT NULL REFERENCES accounts (account_id) ON DELETE CASCADE,
        created_at   TEXT NOT NULL,
        expires_at   TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_account ON sessions (account_id)",
    """
    CREATE TABLE IF NOT EXISTS matches (
        match_id     TEXT PRIMARY KEY,
        user_a       TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        user_b       TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        created_at   TEXT NOT NULL,
        status       TEXT NOT NULL CHECK (status IN ('active', 'unmatched')),
        unmatched_by TEXT,
        unmatched_at TEXT,
        CHECK (user_a < user_b)
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_matches_one_active
        ON matches (user_a, user_b) WHERE status = 'active'
    """,
    "CREATE INDEX IF NOT EXISTS idx_matches_user_b ON matches (user_b)",
    """
    CREATE TABLE IF NOT EXISTS messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id   TEXT NOT NULL REFERENCES matches (match_id) ON DELETE CASCADE,
        sender_id  TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        body       TEXT NOT NULL CHECK (length(body) BETWEEN 1 AND 2000),
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_match ON messages (match_id, message_id)",
    """
    CREATE TABLE IF NOT EXISTS blocks (
        blocker_id TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        blocked_id TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        PRIMARY KEY (blocker_id, blocked_id),
        CHECK (blocker_id <> blocked_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_blocks_blocked ON blocks (blocked_id)",
    """
    CREATE TABLE IF NOT EXISTS reports (
        report_id   TEXT PRIMARY KEY,
        reporter_id TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        reported_id TEXT REFERENCES profiles (user_id) ON DELETE SET NULL,
        match_id    TEXT REFERENCES matches (match_id) ON DELETE SET NULL,
        reason      TEXT NOT NULL
            CHECK (reason IN ('harassment', 'spam', 'fake', 'inappropriate', 'other')),
        details     TEXT NOT NULL CHECK (length(details) <= 1000),
        status      TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
        created_at  TEXT NOT NULL,
        resolved_by TEXT,
        resolved_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS photos (
        photo_id    TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        position    INTEGER NOT NULL,
        storage_key TEXT NOT NULL UNIQUE,
        created_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_photos_user ON photos (user_id, position)",
    """
    CREATE TABLE IF NOT EXISTS impressions (
        impression_id TEXT PRIMARY KEY,
        viewer_id     TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        target_id     TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        matcher       TEXT NOT NULL,
        pool_size     INTEGER NOT NULL,
        shown_at      TEXT NOT NULL,
        rated_at      TEXT,
        predicted_score REAL,   -- The matcher's score for this profile, if it gives one.
        position        INTEGER -- This is the viewer's nth impression, starting at 1.
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_impressions_viewer ON impressions (viewer_id, target_id)",
    "CREATE INDEX IF NOT EXISTS idx_impressions_target ON impressions (target_id)",
    """
    CREATE TABLE IF NOT EXISTS impression_candidates (
        impression_id   TEXT NOT NULL REFERENCES impressions (impression_id) ON DELETE CASCADE,
        candidate_id    TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        predicted_score REAL NOT NULL,
        PRIMARY KEY (impression_id, candidate_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS surveys (
        survey_id  TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        kind       TEXT NOT NULL CHECK (kind IN ('session', 'post_match')),
        match_id   TEXT REFERENCES matches (match_id) ON DELETE SET NULL,
        score      INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS survey_prompts (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    TEXT NOT NULL REFERENCES profiles (user_id) ON DELETE CASCADE,
        kind       TEXT NOT NULL CHECK (kind IN ('session', 'post_match')),
        match_id   TEXT REFERENCES matches (match_id) ON DELETE CASCADE,
        shown_at   TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_survey_prompts_user ON survey_prompts (user_id, kind)",
    """
    CREATE TABLE IF NOT EXISTS app_settings (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    # Counts of errors and failed emails, with no personal data, for the admin overview.
    """
    CREATE TABLE IF NOT EXISTS app_events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        kind       TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        actor_account_id TEXT NOT NULL,
        action           TEXT NOT NULL,
        target           TEXT,
        details          TEXT NOT NULL,
        created_at       TEXT NOT NULL
    )
    """,
)
