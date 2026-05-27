"""Integration tests for AgentFactory — real LLM calls, mocked Google tools.

Strategy:
  - OpenRouter/GPT-4.1 is called for real to test actual LLM behaviour.
  - Google Search and Google Places are mocked to avoid cost and flakiness,
    but we verify whether the LLM *decided* to call them.

Run:  poetry run pytest tests/integration/ -v
Skip: set OPENROUTER_API_KEY="" to skip all tests in this file.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.agent.agent_factory import AgentFactory
from app.agent.models.models import ChatRequest, Persona

# ── Fixtures ──────────────────────────────────────────────────────────────────

COLOSSEUM = {"latitude": 41.8902, "longitude": 12.4922}
VIEUX_PORT = {"latitude": 43.2965, "longitude": 5.3698}
ROEMER = {"latitude": 50.1104, "longitude": 8.6821}


@pytest.fixture
def factory():
    return AgentFactory()


def _stub_search(answer: str = "Historical records show the site dates to antiquity."):
    """Patch execute_tool in loop to spy on calls and return stub data."""
    return patch(
        "app.agent.loop.execute_tool",
        new=AsyncMock(side_effect=_fake_execute(answer)),
    )


def _fake_execute(search_answer: str):
    async def _inner(name: str, args: dict, lat: float, lon: float) -> str:
        if name == "google_places_search":
            return (
                "- Ristorante Il Gladiatore: Roman cuisine near the arena (rating: 4.5)\n"
                "- La Taverna dei Fori Imperiali: Traditional trattoria (rating: 4.3)"
            )
        if name == "google_search":
            return search_answer
        return f"Unknown tool: {name}"

    return _inner


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_full_pipeline_returns_nonempty_text(factory):
    """Basic smoke test: location → coherent narrative text."""
    request = ChatRequest(**COLOSSEUM, persona=Persona.historian)

    with _stub_search():
        result = await factory.run(request)

    text = result.parsed_content.text
    assert isinstance(text, str)
    assert len(text) > 100, "Response text is suspiciously short"


@pytest.mark.integration
async def test_response_mentions_location(factory):
    """LLM should reference the Colosseum when given its coordinates."""
    request = ChatRequest(**COLOSSEUM, persona=Persona.historian)

    with _stub_search():
        result = await factory.run(request)

    text = result.parsed_content.text.lower()
    assert any(word in text for word in ["colosseum", "coliseum", "колизей", "amphitheatre", "амфитеатр"])


@pytest.mark.integration
async def test_places_tool_called_for_restaurant_query(factory):
    """LLM must call google_places_search when user asks to find a nearby venue."""
    request = ChatRequest(
        **VIEUX_PORT,
        persona=Persona.historian,
        message="Где поблизости можно поесть буйабес?",
    )
    tool_calls: list[str] = []

    async def spy_execute(name: str, args: dict, lat: float, lon: float) -> str:
        tool_calls.append(name)
        if name == "google_places_search":
            return "- Chez Fonfon: Famous bouillabaisse restaurant (rating: 4.6)"
        return "No result"

    with patch("app.agent.loop.execute_tool", new=AsyncMock(side_effect=spy_execute)):
        result = await factory.run(request)

    assert "google_places_search" in tool_calls, (
        "LLM should have called google_places_search for a 'find restaurant' query"
    )
    assert result.parsed_content.text


@pytest.mark.integration
async def test_search_tool_not_called_for_basic_history(factory):
    """LLM should answer well-known history from its own knowledge, not call google_search."""
    request = ChatRequest(
        **COLOSSEUM,
        persona=Persona.historian,
        message="Расскажи про историю Колизея",
    )
    tool_calls: list[str] = []

    async def spy_execute(name: str, args: dict, lat: float, lon: float) -> str:
        tool_calls.append(name)
        return "Some search result"

    with patch("app.agent.loop.execute_tool", new=AsyncMock(side_effect=spy_execute)):
        result = await factory.run(request)

    assert "google_search" not in tool_calls, (
        "LLM should NOT call google_search for well-known Colosseum history"
    )
    assert result.parsed_content.text


@pytest.mark.integration
async def test_language_override_returns_english(factory):
    """language='en' must produce an English response regardless of message language."""
    request = ChatRequest(
        **ROEMER,
        persona=Persona.historian,
        message="Расскажи об этом месте",
        language="en",
    )

    with _stub_search():
        result = await factory.run(request)

    text = result.parsed_content.text
    # Heuristic: English response should contain common English function words
    english_words = ["the", "was", "is", "in", "of", "and", "this", "that", "which", "were"]
    found = sum(1 for w in english_words if f" {w} " in text.lower())
    assert found >= 3, f"Response doesn't look like English. Got: {text[:200]}"


@pytest.mark.integration
async def test_chat_history_is_used_in_followup(factory):
    """LLM should use injected history to answer follow-up questions."""
    history = [
        {"role": "user", "content": "Coordinates: 41.8902, 12.4922\nUser message: Tell me about this place"},
        {
            "role": "assistant",
            "content": (
                '{"text": "The Colosseum, also known as the Flavian Amphitheatre, '
                "was built between 70-80 AD under emperors Vespasian and Titus. "
                'It could hold 50,000 to 80,000 spectators."}'
            ),
        },
    ]

    request = ChatRequest(
        **COLOSSEUM,
        persona=Persona.historian,
        message="How many emperors were involved in its construction?",
    )

    with _stub_search():
        result = await factory.run(request, history=history)

    text = result.parsed_content.text.lower()
    # LLM should reference at least one emperor mentioned in the injected history
    emperor_names = ["vespasian", "titus", "веспасиан", "тит", "флавий", "flavian"]
    assert any(name in text for name in emperor_names), (
        f"LLM should have used history context about the emperors. Got: {text[:300]}"
    )


@pytest.mark.integration
async def test_llm_trace_populated(factory):
    """ParsedLLMResponse must always include token usage data."""
    request = ChatRequest(**COLOSSEUM, persona=Persona.historian)

    with _stub_search():
        result = await factory.run(request)

    trace = result.llm_trace
    assert trace.input_tokens > 0
    assert trace.output_tokens > 0
    assert trace.total_tokens == trace.input_tokens + trace.output_tokens


@pytest.mark.integration
async def test_dark_tourism_persona_different_tone(factory):
    """dark_tourism persona should produce notably different content than historian."""
    historian_req = ChatRequest(**COLOSSEUM, persona=Persona.historian)
    dark_req = ChatRequest(**COLOSSEUM, persona=Persona.dark_tourism)

    with _stub_search():
        historian_result = await factory.run(historian_req)
    with _stub_search():
        dark_result = await factory.run(dark_req)

    historian_text = historian_result.parsed_content.text
    dark_text = dark_result.parsed_content.text

    # Texts should be different (different persona → different output)
    assert historian_text != dark_text, "Different personas should produce different narratives"
    assert len(dark_text) > 50
