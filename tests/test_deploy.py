"""The pieces that matter for running on a real server."""

import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import backups
from app.config import Settings
from app.main import create_app

ROOT = Path(__file__).resolve().parent.parent


def safe_settings(tmp_path: Path) -> Settings:
    return Settings(
        site_url="https://study.example.org",
        app_env="production",
        main_db=tmp_path / "disk" / "main.db",
        persona_db=tmp_path / "disk" / "personas.db",
        cookie_secure=True,
        export_salt="a-long-random-secret-value",
        admin_emails=("lead@bu.edu",),
        backup_every_hours=24,
    )


def test_production_starts_with_safe_settings_and_backs_up(tmp_path: Path) -> None:
    app = create_app(safe_settings(tmp_path))
    assert TestClient(app).get("/healthz").json() == {"ok": True}
    app.state.backups.stop.set()
    app.state.backups.join(timeout=10)
    made = list((tmp_path / "disk" / "backups").glob("main-*.db"))
    assert len(made) == 1  # One straight away at startup.


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"site_url": "http://study.example.org"}, "https://"),
        ({"cookie_secure": False}, "COOKIE_SECURE"),
        ({"export_salt": ""}, "EXPORT_SALT"),
        ({"admin_emails": ()}, "ADMIN_EMAILS"),
        ({"main_db": ROOT / "data" / "live" / "main.db"}, "persistent disk"),
        ({"backup_every_hours": 0}, "BACKUP_EVERY_HOURS"),
    ],
)
def test_production_refuses_unsafe_settings(tmp_path: Path, change: dict, problem: str) -> None:
    with pytest.raises(RuntimeError, match=problem):
        create_app(replace(safe_settings(tmp_path), **change))


def test_development_skips_the_checks(tmp_path: Path) -> None:
    settings = Settings(main_db=tmp_path / "m.db", persona_db=tmp_path / "p.db")
    assert settings.production_problems()  # It would fail them,
    create_app(settings)  # but development doesn't check.


def test_backups_are_consistent_and_only_the_newest_are_kept(tmp_path: Path) -> None:
    source = tmp_path / "main.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE t (x)")
        conn.execute("INSERT INTO t VALUES (42)")
    folder = tmp_path / "backups"
    for i in range(4):
        (folder / f"main-2000010{i}-000000.db").parent.mkdir(exist_ok=True)
        backups.backup_to(source, folder / f"main-2000010{i}-000000.db")
    newest = backups.backup_now(source, folder, keep=3)
    kept = sorted(p.name for p in folder.glob("main-*.db"))
    assert len(kept) == 3 and newest.name == kept[-1]
    assert sorted(p.name for p in folder.iterdir()) == kept  # No -wal or -shm files left behind.
    with sqlite3.connect(newest) as conn:
        assert conn.execute("SELECT x FROM t").fetchall() == [(42,)]


def test_the_seed_script_refuses_to_run_in_production(tmp_path: Path) -> None:
    environment = {**os.environ, "APP_ENV": "production", "APP_MAIN_DB": str(tmp_path / "m.db"), "APP_PERSONA_DB": str(tmp_path / "p.db")}
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "seed_dev.py")], env=environment, capture_output=True, text=True)
    assert done.returncode != 0 and "production" in done.stderr
    assert not (tmp_path / "m.db").exists()


def test_the_server_needs_only_the_app_requirements() -> None:
    wanted = {line.split("==")[0].lower() for line in (ROOT / "requirements-app.txt").read_text().splitlines() if "==" in line}
    assert wanted == {"fastapi", "uvicorn", "python-multipart", "jinja2", "pydantic"}
    render = (ROOT / "render.yaml").read_text()
    assert "requirements-app.txt" in render and "plan: starter" in render and "mountPath: /var/data" in render
