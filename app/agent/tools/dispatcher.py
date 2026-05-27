"""Tool dispatcher — routes LLM tool-call requests to their implementations."""

import logging
from typing import Any

from app.agent.tools.google_places_search_tool import google_places_search_tool
from app.agent.tools.google_search_tool import google_search_tool

logger = logging.getLogger(__name__)


async def execute_tool(name: str, args: dict[str, Any], lat: float, lon: float) -> str:
    """Execute a named tool and return its result as a plain string."""
    if name == "google_search":
        result = await google_search_tool(args["query"])
        if result.get("success"):
            return result.get("answer", "No answer returned.")
        return f"Search failed: {result.get('message', 'unknown error')}"

    if name == "google_places_search":
        result = await google_places_search_tool(
            query=args["query"],
            max_results=3,
            location_lat=lat,
            location_lng=lon,
            radius_meters=args.get("radius_meters", 500),
        )
        if result.get("success") and result.get("places"):
            lines = [
                f"- {p['name']}: {p.get('description', '')} (rating: {p.get('rating', 'n/a')})"
                for p in result["places"]
            ]
            return "\n".join(lines)
        return "No places found."

    return f"Unknown tool: {name}"
