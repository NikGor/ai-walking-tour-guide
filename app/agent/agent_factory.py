import asyncio
import base64
import logging
import time
from typing import Any, cast

import httpx

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
        wiki_image: bytes | None = None
        commons_image: bytes | None = None
        if has_location:
            assert request.latitude is not None and request.longitude is not None
            location_ctx = await get_location_context(request.latitude, request.longitude)
            wiki_image, commons_image = await asyncio.gather(
                self._maybe_fetch_image(location_ctx.wikipedia_image_url),
                self._maybe_fetch_image(location_ctx.commons_image_url),
            )
        t_geo = time.perf_counter()

        # ── Build messages ────────────────────────────────────────────────────
        messages = self._build_messages(request, history, location_ctx, wiki_image, commons_image)
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
        parsed.wiki_image = wiki_image
        parsed.commons_image = commons_image

        img_flags = ("  📷wiki" if wiki_image else "") + ("  🏛commons" if commons_image else "")
        logger.info(
            "\033[35mTIME ›\033[0m total=\033[33m%.1fs\033[0m  "
            "geo=\033[33m%.1fs\033[0m  prompt=\033[33m%.0fms\033[0m  llm=\033[33m%.1fs\033[0m  "
            "words=\033[33m%d\033[0m  tokens=\033[33m%d\033[0m  \033[2m$%.4f\033[0m%s",
            t_llm - t0,
            t_geo - t0,
            (t_prompt - t_geo) * 1000,
            t_llm - t_prompt,
            len(result.text.split()),
            parsed.llm_trace.total_tokens,
            parsed.llm_trace.total_cost,
            img_flags,
        )
        return parsed

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(
        self,
        request: ChatRequest,
        history: list[dict[str, Any]] | None,
        location_ctx: LocationContext | None,
        wiki_image: bytes | None = None,
        commons_image: bytes | None = None,
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

        # Build user content — text + optional images
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_message}]
        if request.photo_url:
            content_parts.append({"type": "image_url", "image_url": {"url": request.photo_url}})
        if wiki_image:
            b64 = base64.b64encode(wiki_image).decode()
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            logger.info("\033[36mWIKI ›\033[0m image attached to LLM context (%d bytes)", len(wiki_image))
        if commons_image:
            content_parts.append(
                {"type": "text", "text": "[Archival photo from Wikimedia Commons — historical reference]"}
            )
            b64 = base64.b64encode(commons_image).decode()
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            logger.info(
                "\033[36mCOMM ›\033[0m archival photo attached to LLM context (%d bytes)", len(commons_image)
            )

        user_content: str | list[dict[str, Any]] = user_message if len(content_parts) == 1 else content_parts

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
            logger.info(
                "\033[36mHIST ›\033[0m injecting \033[33m%d\033[0m previous messages",
                len(history),
            )
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    async def _maybe_fetch_image(url: str | None) -> bytes | None:
        return await AgentFactory._fetch_image(url) if url else None

    @staticmethod
    async def _fetch_image(url: str) -> bytes | None:
        headers = {"User-Agent": "SolarisPliny/1.0 (github.com/NikGor/ai-walking-tour-guide)"}
        try:
            async with httpx.AsyncClient(headers=headers) as client:
                resp = await client.get(url, timeout=8.0, follow_redirects=True)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            logger.warning("wiki_img: failed to fetch %s — %s", url, e)
            return None
