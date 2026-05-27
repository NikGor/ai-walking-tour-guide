import json
import logging
from typing import Any, cast

from app.agent.models.models import ChatRequest, ChatResponse
from app.agent.prompt_builder import PromptBuilder
from app.agent.tools.google_places_search_tool import google_places_search_tool
from app.agent.tools.google_search_tool import google_search_tool
from app.backend.openrouter_client import OpenRouterClient
from app.utils.geocoder import reverse_geocode
from app.utils.llm_parser import ParsedLLMResponse, parse_openrouter_response

logger = logging.getLogger(__name__)

_MODEL = "openai/gpt-4.1"

# ── Tool definitions (OpenAI function calling format) ─────────────────────────

_TOOLS = [
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
            "Find a specific nearby venue — restaurant, museum, café, etc. "
            "Use ONLY when the user explicitly asks to find or recommend a type of place nearby. "
            "Never use for historical or architectural questions."
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


async def _execute_tool(name: str, args: dict[str, Any], lat: float, lon: float) -> str:
    """Execute a tool call and return its result as a string."""
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
            lines = []
            for p in result["places"]:
                lines.append(f"- {p['name']}: {p.get('description', '')} (rating: {p.get('rating', 'n/a')})")
            return "\n".join(lines)
        return "No places found."

    return f"Unknown tool: {name}"


class AgentFactory:
    def __init__(self):
        self._client = OpenRouterClient()
        self._prompt_builder = PromptBuilder()

    async def run(
        self,
        request: ChatRequest,
        history: list[dict[str, Any]] | None = None,
    ) -> ParsedLLMResponse:
        has_location = request.latitude is not None and request.longitude is not None

        location_name: str | None = None
        if has_location:
            assert request.latitude is not None and request.longitude is not None
            logger.info("=== STEP 2.5: Geocode ===")
            location_name = await reverse_geocode(request.latitude, request.longitude)

        system_prompt = self._prompt_builder.build_system_prompt(request.persona.value)
        user_message = self._prompt_builder.build_user_message(
            latitude=request.latitude,
            longitude=request.longitude,
            location_name=location_name,
            message=request.message,
            language=request.language,
        )

        current_user_msg: str | list[dict[str, Any]] = (
            user_message
            if not request.photo_url
            else [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": request.photo_url}},
            ]
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
            logger.info(
                "\033[36mHIST ›\033[0m injecting \033[33m%d\033[0m previous messages",
                len(history),
            )
        messages.append({"role": "user", "content": current_user_msg})

        logger.info("=== STEP 3: AI Processing ===")
        logger.info(
            "\033[35mLLM  ›\033[0m persona=\033[35m%s\033[0m  model=\033[36m%s\033[0m",
            request.persona.value,
            _MODEL,
        )

        # ── Round 1: with tools (places only if location known) ──────────────
        tools = _TOOLS if has_location else [t for t in _TOOLS if t["name"] != "google_places_search"]
        raw = await self._client.create_completion(
            messages=messages,
            model=_MODEL,
            tools=tools,
        )

        # ── Handle tool calls ─────────────────────────────────────────────────
        choice = raw.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)

        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                logger.info(
                    "\033[32mTOOL ›\033[0m \033[1m%s\033[0m  args=%s",
                    tc.function.name,
                    args,
                )
                lat = request.latitude or 0.0
                lon = request.longitude or 0.0
                result = await _execute_tool(tc.function.name, args, lat, lon)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            # ── Round 2: final answer with structured output ───────────────────
            logger.info("\033[35mLLM  ›\033[0m round 2 — structured output")
            raw = await self._client.create_completion(
                messages=messages,
                model=_MODEL,
                response_format=ChatResponse,
            )
        else:
            # No tools called — re-ask for structured output
            logger.info("\033[35mLLM  ›\033[0m no tools called, requesting structured output")
            messages.append({"role": "assistant", "content": choice.message.content})
            messages.append({"role": "user", "content": "Now format your answer as JSON."})
            raw = await self._client.create_completion(
                messages=messages,
                model=_MODEL,
                response_format=ChatResponse,
            )

        parsed = parse_openrouter_response(raw, ChatResponse)
        result = cast(ChatResponse, parsed.parsed_content)

        logger.info(
            "\033[35mLLM  ›\033[0m done  words=\033[33m%d\033[0m  tokens=\033[33m%d\033[0m  \033[2m$%.4f\033[0m",  # noqa: E501
            len(result.text.split()),
            parsed.llm_trace.total_tokens,
            parsed.llm_trace.total_cost,
        )
        return parsed
