"""Tool schema registry — OpenAI function-calling format.

Defines all tools available to the LLM and returns the appropriate
subset depending on whether a GPS location is known.
"""

_TOOLS: list[dict] = [
    {
        "name": "google_search",
        "description": (
            "Search the web for specific facts about a place. "
            "Use ONLY when you need a concrete detail you don't confidently know — "
            "an exact date, an obscure local landmark, a specific person or event tied to this location. "
            "Do NOT use if you can give a solid general answer from your own knowledge. "
            "Default to answering directly; search only to fill a specific gap."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Precise query, e.g. 'Goldene Waage Frankfurt Römerberg history'",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "google_places_search",
        "description": (
            "Find nearby venues — restaurants, cafés, museums, bars, shops, etc. "
            "ALWAYS call this tool when the user asks where to eat, drink, visit, or find any type of venue. "
            "NEVER answer venue questions from your own knowledge — "
            "venue names and details must come from this tool. "
            "Invented or guessed venue names are a critical failure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Place type or name to search for",
                },
                "radius_meters": {
                    "type": "number",
                    "description": "Search radius in meters (default 500)",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


def get_tools(has_location: bool) -> list[dict]:
    """Return tool list, omitting google_places_search when no GPS location is available."""
    if has_location:
        return _TOOLS
    return [t for t in _TOOLS if t["name"] != "google_places_search"]
