"""Unit tests for tool dispatcher — routing logic, mocked implementations."""

from unittest.mock import AsyncMock, patch

import pytest

from app.utils.dispatcher_utils import execute_tool


@pytest.fixture
def search_result_ok():
    return {"success": True, "answer": "The Römerberg dates back to 1405."}


@pytest.fixture
def places_result_ok():
    return {
        "success": True,
        "places": [
            {"name": "Zum Storch", "description": "Traditional Frankfurt restaurant", "rating": 4.7},
            {"name": "Fichte Kränzi", "description": "Local tavern", "rating": 4.2},
        ],
    }


async def test_execute_tool_routes_to_google_search(search_result_ok):
    with patch(
        "app.utils.dispatcher_utils.google_search_tool",
        new=AsyncMock(return_value=search_result_ok),
    ) as mock_search:
        result, extra = await execute_tool("google_search", {"query": "Römerberg history"}, lat=0, lon=0)

    mock_search.assert_awaited_once_with("Römerberg history")
    assert "1405" in result
    assert extra is None


async def test_execute_tool_routes_to_google_places(places_result_ok):
    with patch(
        "app.utils.dispatcher_utils.google_places_search_tool",
        new=AsyncMock(return_value=places_result_ok),
    ) as mock_places:
        result, extra = await execute_tool(
            "google_places_search",
            {"query": "restaurant", "radius_meters": 300},
            lat=50.110,
            lon=8.682,
        )

    mock_places.assert_awaited_once()
    call_kwargs = mock_places.call_args.kwargs
    assert call_kwargs["location_lat"] == 50.110
    assert call_kwargs["location_lng"] == 8.682
    assert call_kwargs["radius_meters"] == 300
    assert "Zum Storch" in result
    assert "4.7" in result
    assert extra is None


async def test_execute_tool_uses_default_radius_when_not_provided(places_result_ok):
    with patch(
        "app.utils.dispatcher_utils.google_places_search_tool",
        new=AsyncMock(return_value=places_result_ok),
    ) as mock_places:
        await execute_tool("google_places_search", {"query": "cafe"}, lat=0.0, lon=0.0)

    assert mock_places.call_args.kwargs["radius_meters"] == 500


async def test_execute_tool_returns_error_string_on_search_failure():
    with patch(
        "app.utils.dispatcher_utils.google_search_tool",
        new=AsyncMock(return_value={"success": False, "message": "API quota exceeded"}),
    ):
        result, _ = await execute_tool("google_search", {"query": "test"}, lat=0, lon=0)

    assert "Search failed" in result
    assert "API quota exceeded" in result


async def test_execute_tool_returns_no_places_found_on_empty_result():
    with patch(
        "app.utils.dispatcher_utils.google_places_search_tool",
        new=AsyncMock(return_value={"success": True, "places": []}),
    ):
        result, _ = await execute_tool("google_places_search", {"query": "museum"}, lat=0, lon=0)

    assert result == "No places found."


async def test_execute_tool_unknown_name_returns_error():
    result, _ = await execute_tool("nonexistent_tool", {}, lat=0, lon=0)
    assert "Unknown tool" in result
    assert "nonexistent_tool" in result
