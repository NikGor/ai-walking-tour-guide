"""Unit tests for LLM response parsing and cost calculation."""

import json

import pytest

from app.agent.models.models import ChatResponse
from app.utils.llm_parser import (
    ParsedLLMResponse,
    calculate_token_cost,
    parse_openrouter_response,
)

# ── Mock response helpers ─────────────────────────────────────────────────────


def _make_response(
    content: str,
    model: str = "openai/gpt-4.1",
    input_tokens: int = 100,
    output_tokens: int = 50,
):
    """Build a minimal mock of an OpenRouter chat.completions response."""

    class Usage:
        prompt_tokens = input_tokens
        completion_tokens = output_tokens
        total_tokens = input_tokens + output_tokens
        prompt_tokens_details = None
        completion_tokens_details = None

    class Message:
        pass

    class Choice:
        pass

    class Response:
        pass

    msg = Message()
    msg.content = content

    choice = Choice()
    choice.message = msg

    resp = Response()
    resp.usage = Usage()
    resp.model = model
    resp.choices = [choice]
    resp.id = "test-response-id"

    return resp


# ── calculate_token_cost ──────────────────────────────────────────────────────


def test_cost_calculation_known_model():
    cost = calculate_token_cost("openai/gpt-4.1", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(10.0)  # $2 input + $8 output


def test_cost_calculation_zero_tokens():
    assert calculate_token_cost("openai/gpt-4.1", 0, 0) == 0.0


def test_cost_calculation_unknown_model_returns_zero():
    assert calculate_token_cost("some/unknown-model", 1000, 500) == 0.0


def test_cost_calculation_prefix_match():
    """Versioned model names should match by prefix."""
    cost_versioned = calculate_token_cost("openai/gpt-4.1-2025-04-14", 1_000_000, 0)
    cost_base = calculate_token_cost("openai/gpt-4.1", 1_000_000, 0)
    assert cost_versioned == cost_base


# ── parse_openrouter_response ─────────────────────────────────────────────────


def test_parse_returns_parsed_llm_response():
    payload = json.dumps(
        {"text": "The Colosseum was built in 70-80 AD.", "suggestions": [], "recommended_personas": []}
    )
    resp = _make_response(payload)
    result = parse_openrouter_response(resp, ChatResponse)
    assert isinstance(result, ParsedLLMResponse)


def test_parse_extracts_chat_response_text():
    payload = json.dumps(
        {"text": "Built by Emperor Vespasian.", "suggestions": [], "recommended_personas": []}
    )
    resp = _make_response(payload)
    result = parse_openrouter_response(resp, ChatResponse)
    assert isinstance(result.parsed_content, ChatResponse)
    assert result.parsed_content.text == "Built by Emperor Vespasian."


def test_parse_fills_llm_trace_tokens():
    payload = json.dumps({"text": "test", "suggestions": [], "recommended_personas": []})
    resp = _make_response(payload, input_tokens=200, output_tokens=80)
    result = parse_openrouter_response(resp, ChatResponse)
    assert result.llm_trace.input_tokens == 200
    assert result.llm_trace.output_tokens == 80
    assert result.llm_trace.total_tokens == 280


def test_parse_calculates_cost():
    payload = json.dumps({"text": "test", "suggestions": [], "recommended_personas": []})
    resp = _make_response(payload, model="openai/gpt-4.1", input_tokens=1_000_000, output_tokens=0)
    result = parse_openrouter_response(resp, ChatResponse)
    assert result.llm_trace.total_cost == pytest.approx(2.0)


def test_parse_raises_on_empty_choices():
    resp = _make_response("{}")
    resp.choices = []
    with pytest.raises(ValueError, match="no choices"):
        parse_openrouter_response(resp, ChatResponse)


def test_parse_raises_on_empty_content():
    resp = _make_response("")
    resp.choices[0].message.content = ""
    with pytest.raises(ValueError, match="content is empty"):
        parse_openrouter_response(resp, ChatResponse)


def test_parse_stores_response_id():
    payload = json.dumps({"text": "hello", "suggestions": [], "recommended_personas": []})
    resp = _make_response(payload)
    result = parse_openrouter_response(resp, ChatResponse)
    assert result.response_id == "test-response-id"
