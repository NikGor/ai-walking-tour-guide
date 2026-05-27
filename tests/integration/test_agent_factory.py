"""Integration tests for AgentFactory — real LLM calls, mocked Google tools.

Strategy:
  - OpenRouter/GPT-4.1 is called for real to test the actual pipeline end-to-end.
  - Google Search and Google Places are mocked to avoid cost and flakiness,
    but we verify whether the LLM *decided* to call them.
  - Content assertions are intentionally minimal: if we got a non-empty string
    back without an exception, the pipeline is working.

Run:  poetry run pytest tests/integration/ -v
Skip: unset OPENROUTER_API_KEY to skip all tests in this file.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.agent.agent_factory import AgentFactory
from app.agent.models.models import ChatRequest, Persona

# ── Locations ─────────────────────────────────────────────────────────────────

COLOSSEUM = {"latitude": 41.8902, "longitude": 12.4922}
VIEUX_PORT = {"latitude": 43.2965, "longitude": 5.3698}
ROEMER = {"latitude": 50.1104, "longitude": 8.6821}


# ── Helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture
def factory():
    return AgentFactory()


def _stub_tools():
    """Patch execute_tool in the loop with a lightweight stub."""

    async def _fake(name: str, args: dict, lat: float, lon: float) -> str:
        if name == "google_places_search":
            return "- Chez Fonfon: Famous bouillabaisse restaurant (rating: 4.6)"
        return "Historical records confirm the site dates to antiquity."

    return patch("app.agent.loop.execute_tool", new=AsyncMock(side_effect=_fake))


def _spy_tools():
    """Like _stub_tools but also records which tools were called."""
    calls: list[str] = []

    async def _fake(name: str, args: dict, lat: float, lon: float) -> str:
        calls.append(name)
        if name == "google_places_search":
            return "- Chez Fonfon: Famous bouillabaisse restaurant (rating: 4.6)"
        return "Historical records confirm the site dates to antiquity."

    return calls, patch("app.agent.loop.execute_tool", new=AsyncMock(side_effect=_fake))


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_full_pipeline_returns_text(factory):
    """Smoke test: pipeline completes and returns a non-empty string."""
    request = ChatRequest(**COLOSSEUM, persona=Persona.historian)
    with _stub_tools():
        result = await factory.run(request)
    assert result.parsed_content.text


@pytest.mark.integration
async def test_places_tool_called_for_restaurant_query(factory):
    """LLM must invoke google_places_search when user asks to find a nearby venue."""
    request = ChatRequest(
        **VIEUX_PORT,
        persona=Persona.historian,
        message="Где поблизости можно поесть буйабес?",
    )
    calls, stub = _spy_tools()
    with stub:
        result = await factory.run(request)

    assert "google_places_search" in calls
    assert result.parsed_content.text


@pytest.mark.integration
async def test_search_tool_not_called_for_basic_history(factory):
    """LLM should answer well-known history from its own knowledge without calling google_search."""
    request = ChatRequest(
        **COLOSSEUM,
        persona=Persona.historian,
        message="Расскажи про историю Колизея",
    )
    calls, stub = _spy_tools()
    with stub:
        result = await factory.run(request)

    assert "google_search" not in calls
    assert result.parsed_content.text


@pytest.mark.integration
async def test_language_override(factory):
    """language='en' must not raise and must return a non-empty response."""
    request = ChatRequest(
        **ROEMER,
        persona=Persona.historian,
        message="Tell me about this place",
        language="en",
    )
    with _stub_tools():
        result = await factory.run(request)
    assert result.parsed_content.text


@pytest.mark.integration
async def test_chat_history_injected(factory):
    """Pipeline must not fail when history is passed."""
    history = [
        {"role": "user", "content": "Tell me about the Colosseum"},
        {"role": "assistant", "content": '{"text": "The Colosseum was built in 70-80 AD."}'},
    ]
    request = ChatRequest(
        **COLOSSEUM,
        persona=Persona.historian,
        message="How many spectators could it hold?",
    )
    with _stub_tools():
        result = await factory.run(request, history=history)
    assert result.parsed_content.text


@pytest.mark.integration
async def test_llm_trace_populated(factory):
    """Response must always include token usage data."""
    request = ChatRequest(**COLOSSEUM, persona=Persona.historian)
    with _stub_tools():
        result = await factory.run(request)

    assert result.llm_trace.input_tokens > 0
    assert result.llm_trace.output_tokens > 0
    assert result.llm_trace.total_tokens == result.llm_trace.input_tokens + result.llm_trace.output_tokens


@pytest.mark.integration
async def test_no_location_pipeline(factory):
    """Pipeline must work when no GPS coordinates are provided."""
    request = ChatRequest(persona=Persona.historian, message="Расскажи про Колизей")
    with _stub_tools():
        result = await factory.run(request)
    assert result.parsed_content.text


@pytest.mark.integration
async def test_dark_tourism_persona(factory):
    """dark_tourism persona must complete the pipeline without errors."""
    request = ChatRequest(**COLOSSEUM, persona=Persona.dark_tourism)
    with _stub_tools():
        result = await factory.run(request)
    assert result.parsed_content.text
