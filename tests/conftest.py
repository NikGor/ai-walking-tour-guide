"""Shared fixtures and configuration for all tests."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


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
