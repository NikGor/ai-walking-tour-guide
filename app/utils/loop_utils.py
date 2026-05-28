"""Agentic loop — multi-round LLM orchestration with tool use.

Round 1: LLM receives the conversation with tools available.
         If it calls tools → execute them, append results, go to round 2.
         If it answers directly → re-ask for structured JSON output.
Round 2: LLM produces the final structured response.
"""

import json
import logging
from typing import Any

from app.agent.models.models import ChatRequest, ChatResponse
from app.backend.openrouter_client import OpenRouterClient
from app.utils.dispatcher_utils import execute_tool
from app.utils.llm_parser_utils import ParsedLLMResponse, parse_openrouter_response

logger = logging.getLogger(__name__)


async def run_agentic_loop(
    client: OpenRouterClient,
    messages: list[dict[str, Any]],
    tools: list[dict],
    request: ChatRequest,
    model: str,
) -> ParsedLLMResponse:
    """Run the agentic loop and return a fully parsed response with map if produced."""
    map_png: bytes | None = None

    # ── Round 1: with tools ───────────────────────────────────────────────────
    raw = await client.create_completion(messages=messages, model=model, tools=tools)
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

        lat = request.latitude or 0.0
        lon = request.longitude or 0.0
        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            logger.info("\033[32mTOOL ›\033[0m \033[1m%s\033[0m  args=%s", tc.function.name, args)
            result_str, tool_map_png = await execute_tool(tc.function.name, args, lat, lon)
            if tool_map_png:
                map_png = tool_map_png  # last map wins (normally only one tour tool per request)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

        # ── Round 2: structured output after tool results ─────────────────────
        logger.info("\033[35mLLM  ›\033[0m round 2 — structured output")
        raw = await client.create_completion(messages=messages, model=model, response_format=ChatResponse)

    else:
        # No tools called — re-ask with structured output constraint
        logger.info("\033[35mLLM  ›\033[0m no tools called, requesting structured output")
        messages.append({"role": "assistant", "content": choice.message.content})
        messages.append({"role": "user", "content": "Now format your answer as JSON."})
        raw = await client.create_completion(messages=messages, model=model, response_format=ChatResponse)

    parsed = parse_openrouter_response(raw, ChatResponse)
    parsed.map_image = map_png
    return parsed
