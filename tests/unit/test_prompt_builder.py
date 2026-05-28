"""Unit tests for PromptBuilder — no LLM calls, pure string logic."""

import pytest

from app.agent.models.chat_models import Persona
from app.agent.prompt_builder import PromptBuilder


@pytest.fixture
def builder():
    return PromptBuilder()


# ── build_system_prompt ───────────────────────────────────────────────────────


def test_system_prompt_contains_persona_section(builder):
    prompt = builder.build_system_prompt(Persona.historian)
    assert "## Your persona" in prompt


@pytest.mark.parametrize("persona", list(Persona))
def test_system_prompt_renders_for_every_persona(builder, persona):
    """Every persona template must exist and produce non-empty output."""
    prompt = builder.build_system_prompt(persona)
    assert len(prompt) > 100, f"Prompt for {persona} is suspiciously short"


def test_system_prompt_different_per_persona(builder):
    historian = builder.build_system_prompt(Persona.historian)
    storyteller = builder.build_system_prompt(Persona.storyteller)
    assert historian != storyteller


# ── build_user_message ────────────────────────────────────────────────────────


def test_user_message_with_coordinates(builder):
    msg = builder.build_user_message(latitude=48.853, longitude=2.349, location_name=None)
    assert "48.853" in msg
    assert "2.349" in msg


def test_user_message_with_location_name(builder):
    msg = builder.build_user_message(latitude=48.853, longitude=2.349, location_name="Notre-Dame de Paris")
    assert "Notre-Dame de Paris" in msg


def test_user_message_with_explicit_language(builder):
    msg = builder.build_user_message(latitude=None, longitude=None, location_name=None, language="en")
    assert "Language: English" in msg


def test_user_message_language_german(builder):
    msg = builder.build_user_message(latitude=None, longitude=None, location_name=None, language="de")
    assert "Language: German" in msg


def test_user_message_auto_language_not_injected(builder):
    """language='auto' should produce no Language: line."""
    msg = builder.build_user_message(latitude=None, longitude=None, location_name=None, language="auto")
    assert "Language:" not in msg


def test_user_message_none_language_not_injected(builder):
    msg = builder.build_user_message(latitude=None, longitude=None, location_name=None, language=None)
    assert "Language:" not in msg


def test_user_message_with_question(builder):
    msg = builder.build_user_message(
        latitude=50.110, longitude=8.682, location_name="Römerberg", message="Who built this square?"
    )
    assert "Who built this square?" in msg


def test_user_message_no_location(builder):
    """No coords → no Coordinates line, but message still included."""
    msg = builder.build_user_message(
        latitude=None, longitude=None, location_name=None, message="Расскажи про Колизей"
    )
    assert "Coordinates:" not in msg
    assert "Расскажи про Колизей" in msg


def test_user_message_all_fields(builder):
    msg = builder.build_user_message(
        latitude=41.89,
        longitude=12.492,
        location_name="Colosseum, Rome",
        message="When was it built?",
        language="en",
    )
    assert "41.89" in msg
    assert "Colosseum, Rome" in msg
    assert "Language: English" in msg
    assert "When was it built?" in msg
