"""Persona storage in its own SQLite file (personas.db).

Personas are hidden data, so they're kept out of main.db. Code that only has the
main database, through store.py, can't read them. There's no foreign key to main.db
because SQLite can't enforce one across files. ProfileManager keeps the two in step.
"""

import json
import sqlite3
from pathlib import Path

from db import connect, initialize, transaction
from models import Persona

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS personas (
        user_id          TEXT PRIMARY KEY,
        description      TEXT NOT NULL,
        generation_seeds TEXT NOT NULL,
        created_at       TEXT NOT NULL
    )
    """,
)


def _row_to_persona(row: sqlite3.Row) -> Persona:
    data = dict(row)
    data["generation_seeds"] = json.loads(data["generation_seeds"])
    return Persona.model_validate(data)


class PersonaStore:
    def __init__(self, db_path: str | Path = "personas.db") -> None:
        self.db_path = db_path
        initialize(db_path, SCHEMA)

    def save_persona(self, persona: Persona) -> None:
        """Store a persona, replacing any existing one for this user_id."""
        with transaction(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO personas (user_id, description, generation_seeds, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (user_id) DO UPDATE SET
                    description = excluded.description,
                    generation_seeds = excluded.generation_seeds,
                    created_at = excluded.created_at
                """,
                (
                    persona.user_id,
                    persona.description,
                    json.dumps(persona.generation_seeds),
                    persona.created_at.isoformat(timespec="microseconds"),
                ),
            )

    def get_persona(self, user_id: str) -> Persona | None:
        """Return this user's persona, or None, since not every user has one."""
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM personas WHERE user_id = ?", (user_id,)
            ).fetchone()
        return None if row is None else _row_to_persona(row)

    def delete_persona(self, user_id: str) -> None:
        """Delete this user's persona, if they have one."""
        with transaction(self.db_path) as conn:
            conn.execute("DELETE FROM personas WHERE user_id = ?", (user_id,))

    def list_personas(self) -> list[Persona]:
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM personas ORDER BY rowid").fetchall()
        return [_row_to_persona(row) for row in rows]

    def export_personas_json(self, path: str | Path) -> None:
        personas = [persona.model_dump(mode="json") for persona in self.list_personas()]
        Path(path).write_text(json.dumps(personas, indent=2), encoding="utf-8")
