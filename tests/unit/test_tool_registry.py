"""Unit tests for tool registry — schema correctness and location filtering."""

from app.utils.registry_utils import _TOOLS, get_tools


def test_get_tools_with_location_returns_all():
    tools = get_tools(has_location=True)
    names = {t["name"] for t in tools}
    assert "google_search" in names
    assert "google_places_search" in names


def test_get_tools_without_location_excludes_places():
    tools = get_tools(has_location=False)
    names = {t["name"] for t in tools}
    assert "google_search" in names
    assert "google_places_search" not in names


def test_all_tools_have_required_fields():
    for tool in _TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["parameters"]["type"] == "object"
        assert "properties" in tool["parameters"]
        assert "required" in tool["parameters"]


def test_google_search_requires_query():
    search_tool = next(t for t in _TOOLS if t["name"] == "google_search")
    assert "query" in search_tool["parameters"]["required"]


def test_google_places_requires_query():
    places_tool = next(t for t in _TOOLS if t["name"] == "google_places_search")
    assert "query" in places_tool["parameters"]["required"]


def test_tools_have_additional_properties_false():
    """Strict JSON schema mode requires additionalProperties: false."""
    for tool in _TOOLS:
        assert tool["parameters"].get("additionalProperties") is False
