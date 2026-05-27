"""Shared fixtures and configuration for all tests."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# OpenAI SDK raises at import time if api_key is absent (even when we use OpenRouter).
# Set a dummy value so unit tests that don't make real API calls can import freely.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-for-tests")
os.environ.setdefault("OPENROUTER_API_KEY", "sk-dummy-for-tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: real LLM API calls (requires OPENROUTER_API_KEY)")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests when OPENROUTER_API_KEY is not set."""
    if os.getenv("OPENROUTER_API_KEY"):
        return
    skip = pytest.mark.skip(reason="OPENROUTER_API_KEY not set")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
