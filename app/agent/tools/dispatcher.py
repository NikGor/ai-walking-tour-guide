"""Tool dispatcher — routes LLM tool-call requests to their implementations."""

import json
import logging
from typing import Any

from app.agent.tools.city_tour_tool import city_tour_tool
from app.agent.tools.google_places_search_tool import google_places_search_tool
from app.agent.tools.google_search_tool import google_search_tool

logger = logging.getLogger(__name__)


async def execute_tool(
    name: str,
    args: dict[str, Any],
    lat: float,
    lon: float,
) -> tuple[str, bytes | None]:
    """Execute a named tool.

    Returns:
        (context_str, map_png) — context_str goes into the LLM message;
        map_png is side-channel bytes (e.g. a map image) passed out of band.
    """
    if name == "google_search":
        result = await google_search_tool(args["query"])
        if result.get("success"):
            return result.get("answer", "No answer returned."), None
        return f"Search failed: {result.get('message', 'unknown error')}", None

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
            return "\n".join(lines), None
        return "No places found.", None

    if name == "plan_city_tour":
        result = await city_tour_tool(
            city=args["city"],
            poi_names=args["poi_names"],
            start_time=args.get("start_time", "10:00"),
        )
        map_png: bytes | None = result.pop("_map_png", None)
        if not result.get("success"):
            return f"Tour planning failed: {result.get('message', 'unknown error')}", None
        # Provide structured summary to LLM — exclude raw bytes
        llm_summary = json.dumps(
            {k: v for k, v in result.items() if k != "_map_png"},
            ensure_ascii=False,
            indent=2,
        )
        return llm_summary, map_png

    return f"Unknown tool: {name}", None
