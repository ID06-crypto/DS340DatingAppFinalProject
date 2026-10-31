import json
import random
import sys
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Data Generation"))

import written_personas
from manager import ProfileManager
from models import Persona
from raters import JEV_QUESTION, build_jev_state, build_rater_input
from written_personas import (
    CLAUDE_MODEL,
    PERSONA_TRAITS,
    PersonaWritingError,
    build_prompt,
    copy_sampled_users,
    generate_personas,
    sample_traits,
    write_persona,
)


def profile_data(name: str = "Sam") -> dict[str, Any]:
    return {
        "name": name,
        "age": 22,
        "height_cm": 175,
        "religion": "catholic",
        "school": "BU",
        "major": "biology",
        "job": "tutor",
        "interests": ["hiking"],
        "clubs": ["chess club"],
        "gender": "man",
        "interested_in": "women",
        "looking_for": "long-term",
        "agent": True,
    }


def claude_reply(
    text: str = "They are thoughtful. They want a reader.", stop_reason: str = "end_turn"
) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": CLAUDE_MODEL,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 300, "output_tokens": 100},
        },
    )


def fake_claude(respond: Any) -> tuple[anthropic.Anthropic, list[Any]]:
    """A real client with a fake network layer, so no API call can be made."""
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return respond(requests)

    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return client, requests


@pytest.fixture
def manager(tmp_path: Path) -> ProfileManager:
    return ProfileManager(tmp_path / "main.db", tmp_path / "personas.db")


def test_traits_are_in_range_and_repeat_for_a_user() -> None:
    traits = sample_traits("user-1")
    assert tuple(traits) == PERSONA_TRAITS
    assert all(0 <= value <= 1 for value in traits.values())
    assert sample_traits("user-1") == traits
    assert sample_traits("user-2") != traits


def test_traits_are_drawn_evenly() -> None:
    values = [sample_traits(f"user-{i}")["openness"] for i in range(2000)]
    assert sum(v < 0.2 or v > 0.8 for v in values) / len(values) == pytest.approx(0.4, abs=0.05)


def test_prompt_has_profile_and_traits_but_no_name_id_or_agent_flag(
    manager: ProfileManager,
) -> None:
    profile = manager.add_profile(profile_data("UNIQUE_NAME"))
    prompt = build_prompt(profile, {"openness": 0.91})
    assert '"major": "biology"' in prompt
    assert "0.91" in prompt
    assert "UNIQUE_NAME" not in prompt
    assert profile.user_id not in prompt
    assert '"agent"' not in prompt


def test_write_persona_returns_text_and_sends_expected_settings(
    manager: ProfileManager,
) -> None:
    client, requests = fake_claude(lambda _: claude_reply())
    profile = manager.add_profile(profile_data())
    text, input_tokens, output_tokens = write_persona(client, profile, sample_traits(profile.user_id))
    assert text == "They are thoughtful. They want a reader."
    assert (input_tokens, output_tokens) == (300, 100)
    assert requests[0]["model"] == CLAUDE_MODEL
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unfinished_persona_raises(manager: ProfileManager, stop_reason: str) -> None:
    client, _ = fake_claude(lambda _: claude_reply(stop_reason=stop_reason))
    profile = manager.add_profile(profile_data())
    with pytest.raises(PersonaWritingError, match=stop_reason):
        write_persona(client, profile, {"openness": 0.5})


def test_copy_keeps_ids_is_seeded_and_runs_once(tmp_path: Path) -> None:
    source = ProfileManager(tmp_path / "s.db", tmp_path / "s_p.db", rng=random.Random(1))
    ids = {source.add_profile(profile_data()).user_id for _ in range(10)}
    first = ProfileManager(tmp_path / "a.db", tmp_path / "a_p.db")
    second = ProfileManager(tmp_path / "b.db", tmp_path / "b_p.db")

    assert copy_sampled_users(source, first, n_users=4) == 4
    assert copy_sampled_users(source, second, n_users=4) == 4
    assert copy_sampled_users(source, first, n_users=4) == 0

    first_ids = [p.user_id for p in first.list_profiles()]
    assert len(first_ids) == 4
    assert set(first_ids) <= ids
    assert first_ids == [p.user_id for p in second.list_profiles()]


def test_copy_refuses_too_few_source_profiles(tmp_path: Path) -> None:
    source = ProfileManager(tmp_path / "s.db", tmp_path / "s_p.db")
    source.add_profile(profile_data())
    target = ProfileManager(tmp_path / "a.db", tmp_path / "a_p.db")
    with pytest.raises(ValueError, match="need 2"):
        copy_sampled_users(source, target, n_users=2)


def test_generate_saves_persona_with_traits(manager: ProfileManager) -> None:
    profile = manager.add_profile(profile_data())
    client, _ = fake_claude(lambda _: claude_reply("They like maps."))
    stats = generate_personas(manager, client)
    persona = manager.get_persona(profile.user_id)
    assert persona.description == "They like maps."
    assert persona.generation_seeds["traits"] == sample_traits(profile.user_id)
    assert persona.generation_seeds["model"] == CLAUDE_MODEL
    assert stats == {"written": 1, "failed": 0, "input_tokens": 300, "output_tokens": 100}


def test_generate_skips_existing_personas_and_respects_limit(manager: ProfileManager) -> None:
    users = [manager.add_profile(profile_data()).user_id for _ in range(5)]
    manager.add_persona(users[0], "KEEP ME")
    client, requests = fake_claude(lambda _: claude_reply())

    assert generate_personas(manager, client, limit=2)["written"] == 2
    assert len(requests) == 2
    assert manager.get_persona(users[0]).description == "KEEP ME"
    assert generate_personas(manager, client)["written"] == 2
    assert len(requests) == 4
    assert generate_personas(manager, client)["written"] == 0
    assert len(requests) == 4


def test_temporary_failures_are_skipped_and_finished_by_a_second_run(
    manager: ProfileManager,
) -> None:
    user = manager.add_profile(profile_data()).user_id
    client, _ = fake_claude(lambda _: httpx2.Response(429, json={}))
    assert generate_personas(manager, client)["failed"] == 1
    assert manager.get_persona(user) is None

    client, _ = fake_claude(lambda _: claude_reply())
    assert generate_personas(manager, client)["written"] == 1
    assert manager.get_persona(user) is not None


def test_a_bad_key_stops_the_run(manager: ProfileManager) -> None:
    manager.add_profile(profile_data())
    client, _ = fake_claude(lambda _: httpx2.Response(401, json={}))
    with pytest.raises(anthropic.AuthenticationError):
        generate_personas(manager, client)


def test_jev_receives_the_written_persona_as_plain_text(manager: ProfileManager) -> None:
    rater = manager.add_profile(profile_data())
    target = manager.add_profile(profile_data("Tess"))
    persona = Persona(user_id=rater.user_id, description="They want someone who reads.")
    state = build_jev_state(build_rater_input(rater, persona, target))
    assert state["rater_persona"] == "They want someone who reads."
    assert "`rater_persona`" in JEV_QUESTION.instructions
    assert "trait" not in JEV_QUESTION.instructions.lower()


def test_generator_writes_to_the_current_dataset() -> None:
    assert written_personas.MAIN_DB.parent == written_personas.DATA_DIR
    assert written_personas.PERSONA_DB.parent == written_personas.DATA_DIR


def test_ratings_export_and_backup_hold_the_right_data(
    manager: ProfileManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import csv
    import gzip

    import backup_data

    alice = manager.add_profile(profile_data()).user_id
    bob = manager.add_profile(profile_data()).user_id
    manager.add_persona(alice, "HIDDEN_PERSONA_TEXT")
    manager.start_run("run-1", "test", 1, {})
    manager.add_rating(alice, bob, 0.5, "run-1", input_snapshot="HIDDEN_PERSONA_TEXT", step=1)

    path = backup_data.export_ratings(manager.store, "run-1", tmp_path / "exports")
    with gzip.open(path, "rt", encoding="utf-8") as file:
        text = file.read()
    rows = list(csv.DictReader(text.splitlines()))
    assert [(r["rater_id"], r["target_id"], float(r["score"])) for r in rows] == [(alice, bob, 0.5)]
    assert "HIDDEN_PERSONA_TEXT" not in text

    monkeypatch.setattr(backup_data, "MAIN_DB", manager.store.db_path)
    monkeypatch.setattr(backup_data, "PERSONA_DB", manager.personas.db_path)
    destination = backup_data.backup_databases(tmp_path / "backups")
    copy = ProfileManager(destination / "main.db", destination / "personas.db")
    assert copy.count_ratings("run-1") == 1
    assert copy.get_persona(alice).description == "HIDDEN_PERSONA_TEXT"
