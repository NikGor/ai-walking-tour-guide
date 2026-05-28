"""Unit tests for OpenRouterClient — schema building, no real API calls."""

from unittest.mock import MagicMock, patch

import pytest

from app.agent.models.chat_models import ChatResponse
from app.backend.openrouter_client import OpenRouterClient


@pytest.fixture
def client():
    """OpenRouterClient with the OpenAI HTTP client mocked out — no API key needed."""
    with patch("app.backend.openrouter_client.OpenAI", return_value=MagicMock()):
        return OpenRouterClient()


# ── _enforce_no_additional_properties (static — no instance needed) ───────────


def test_enforce_adds_additional_properties_to_object():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = OpenRouterClient._enforce_no_additional_properties(schema)
    assert result["additionalProperties"] is False


def test_enforce_does_not_overwrite_existing():
    schema = {"type": "object", "additionalProperties": True}
    result = OpenRouterClient._enforce_no_additional_properties(schema)
    assert result["additionalProperties"] is True


def test_enforce_recurses_into_properties():
    schema = {
        "type": "object",
        "properties": {
            "nested": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }
    result = OpenRouterClient._enforce_no_additional_properties(schema)
    assert result["properties"]["nested"]["additionalProperties"] is False


# ── _build_create_kwargs ──────────────────────────────────────────────────────


def test_build_kwargs_basic(client):
    msgs = [{"role": "user", "content": "hello"}]
    kwargs = client._build_create_kwargs(msgs, "openai/gpt-4.1")
    assert kwargs["model"] == "openai/gpt-4.1"
    assert kwargs["messages"] == msgs
    assert "response_format" not in kwargs
    assert "tools" not in kwargs


def test_build_kwargs_with_response_format(client):
    msgs = [{"role": "user", "content": "hello"}]
    kwargs = client._build_create_kwargs(msgs, "openai/gpt-4.1", response_format=ChatResponse)
    assert "response_format" in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "ChatResponse"
    assert kwargs["response_format"]["json_schema"]["strict"] is True


def test_build_kwargs_wraps_tools_in_function_type(client):
    tools = [{"name": "google_search", "description": "...", "parameters": {}}]
    kwargs = client._build_create_kwargs([], "model", tools=tools)
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tools"][0]["function"]["name"] == "google_search"


def test_build_kwargs_response_format_strips_internal_fields(client):
    """llm_trace and response_id should not appear in the JSON schema sent to the LLM."""
    kwargs = client._build_create_kwargs([], "model", response_format=ChatResponse)
    schema_props = kwargs["response_format"]["json_schema"]["schema"].get("properties", {})
    assert "llm_trace" not in schema_props
    assert "response_id" not in schema_props
