import logging
import time
from typing import Any, cast

from app.agent.function_runner import run_agentic_loop
from app.agent.models.chat_models import ChatRequest, ChatResponse
from app.agent.prompt_builder import PromptBuilder
from app.backend.openrouter_client import OpenRouterClient
from app.utils.geocoder_utils import LocationContext, get_location_context
from app.utils.llm_parser_utils import ParsedLLMResponse
from app.utils.registry_utils import get_tools

logger = logging.getLogger(__name__)

_MODEL = "openai/gpt-4.1"


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

        t0 = time.perf_counter()

        # ── Geocode + enrich ──────────────────────────────────────────────────
        location_ctx: LocationContext | None = None
        if has_location:
            assert request.latitude is not None and request.longitude is not None
            location_ctx = await get_location_context(request.latitude, request.longitude)
        t_geo = time.perf_counter()

        # ── Build messages ────────────────────────────────────────────────────
        messages = self._build_messages(request, history, location_ctx)
        t_prompt = time.perf_counter()

        logger.info(
            "\033[35mLLM  ›\033[0m persona=\033[35m%s\033[0m  model=\033[36m%s\033[0m",
            request.persona.value,
            _MODEL,
        )

        # ── Agentic loop (tools → structured output) ──────────────────────────
        parsed = await run_agentic_loop(
            client=self._client,
            messages=messages,
            tools=get_tools(has_location),
            request=request,
            model=_MODEL,
        )
        t_llm = time.perf_counter()

        result = cast(ChatResponse, parsed.parsed_content)

        logger.info(
            "\033[35mTIME ›\033[0m total=\033[33m%.1fs\033[0m  "
            "geo=\033[33m%.1fs\033[0m  prompt=\033[33m%.0fms\033[0m  llm=\033[33m%.1fs\033[0m  "
            "words=\033[33m%d\033[0m  tokens=\033[33m%d\033[0m  \033[2m$%.4f\033[0m",
            t_llm - t0,
            t_geo - t0,
            (t_prompt - t_geo) * 1000,
            t_llm - t_prompt,
            len(result.text.split()),
            parsed.llm_trace.total_tokens,
            parsed.llm_trace.total_cost,
        )
        return parsed

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(
        self,
        request: ChatRequest,
        history: list[dict[str, Any]] | None,
        location_ctx: LocationContext | None,
    ) -> list[dict[str, Any]]:
        system_prompt = self._prompt_builder.build_system_prompt(request.persona.value)
        user_message = self._prompt_builder.build_user_message(
            latitude=request.latitude,
            longitude=request.longitude,
            location_ctx=location_ctx,
            message=request.message,
            language=request.language,
            response_format=request.response_format,
        )

        user_content: str | list[dict[str, Any]] = (
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
        messages.append({"role": "user", "content": user_content})
        return messages
