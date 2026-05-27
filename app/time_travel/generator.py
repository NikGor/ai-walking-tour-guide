"""Orchestrates the Time Travel Lens generation pipeline.

Flow:
  1. Geocode coordinates → location name + context
  2. Call LLM to generate historical description + image prompt (JSON)
  3. Call Gemini to generate the image
  4. Return combined TimeTravelResponse
"""

import json
import logging
from typing import Any

from app.backend.openrouter_client import OpenRouterClient
from app.time_travel.image_gen import generate_image
from app.time_travel.models import TimeTravelRequest, TimeTravelResponse
from app.utils.geocoder import LocationContext, get_location_context

logger = logging.getLogger(__name__)

_LLM_MODEL = "openai/gpt-4.1"
_client: OpenRouterClient | None = None


def _get_client() -> OpenRouterClient:
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client


# ── Era formatting ─────────────────────────────────────────────────────────────


def _format_era(year: int, era: str, language: str) -> str:
    """Return a human-readable era label in the requested language."""
    abs_year = abs(year)
    if language == "ru":
        suffix = "до н.э." if era == "BCE" else "н.э."
        return f"{abs_year:,} {suffix}".replace(",", " ")
    else:
        suffix = " BCE" if era == "BCE" else " CE"
        return f"{abs_year:,}{suffix}"


def _year_for_prompt(year: int, era: str) -> str:
    """Return a prompt-friendly year string like '1462 CE' or '2560 BCE'."""
    return f"{abs(year)} {era}"


# ── LLM prompt ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a visual historian. Given a location and a historical year, you produce:
1. A vivid 2-paragraph historical narrative (100–150 words) describing what a person standing
   at that exact location would have seen, heard, and smelled in that era.
   Write in the requested language. Be specific — name materials, architecture styles,
   who would have been there and why.
2. A detailed English image generation prompt (1 paragraph, ~60–80 words) describing
   the visual scene for an AI image generator. Focus on: setting/buildings, time of day,
   atmosphere, people, what is visible at ground level. No meta-commentary.

Return ONLY a JSON object with exactly these keys:
{
  "historical_text": "<narrative in requested language>",
  "image_prompt": "<English image gen prompt>"
}"""


def _build_user_message(
    location_ctx: LocationContext,
    year: int,
    era: str,
    style: str,
    language: str,
) -> str:
    year_str = _year_for_prompt(year, era)
    lang_label = "Russian" if language == "ru" else "English"

    lines = [
        f"Location: {location_ctx.name}",
        f"Coordinates: ({location_ctx.name})",
        f"Target year: {year_str}",
        f"Art style requested: {style}",
        f"Language for historical_text: {lang_label}",
        "",
    ]

    if location_ctx.wikipedia_summary:
        lines.append(f"Wikipedia context: {location_ctx.wikipedia_summary[:800]}")
    if location_ctx.start_date:
        lines.append(f"Construction date: {location_ctx.start_date}")
    if location_ctx.architect:
        lines.append(f"Architect: {location_ctx.architect}")
    if location_ctx.historic:
        lines.append(f"Historic type: {location_ctx.historic}")

    lines.append("\nGenerate the JSON with historical_text and image_prompt as specified.")
    return "\n".join(lines)


async def _call_llm(
    location_ctx: LocationContext,
    year: int,
    era: str,
    style: str,
    language: str,
) -> dict[str, Any]:
    """Call GPT-4.1 to get historical narrative + image prompt."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_message(location_ctx, year, era, style, language),
        },
    ]
    response = await _get_client().create_completion(
        messages=messages,
        model=_LLM_MODEL,
    )
    raw = response.choices[0].message.content or "{}"

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("time_travel_gen_001: LLM returned non-JSON: %s", raw[:200])
        return {
            "historical_text": raw,
            "image_prompt": f"Historical view of {location_ctx.name} in {_year_for_prompt(year, era)}",
        }


# ── Main entry point ───────────────────────────────────────────────────────────


async def generate_time_travel(request: TimeTravelRequest) -> TimeTravelResponse:
    """Full pipeline: geocode → LLM → image → response."""
    logger.info(
        "time_travel_gen_002: lat=%.4f lon=%.4f year=%d %s style=%s",
        request.latitude,
        request.longitude,
        request.year,
        request.era,
        request.style,
    )

    # Step 1: geocode
    try:
        location_ctx = await get_location_context(request.latitude, request.longitude)
    except Exception as e:
        logger.error("time_travel_gen_error_001: geocode failed: %s", e)
        location_ctx = LocationContext(name=f"{request.latitude:.4f}, {request.longitude:.4f}")

    era_label = _format_era(request.year, request.era, request.language)
    logger.info("time_travel_gen_003: location=%r  era=%s", location_ctx.name, era_label)

    # Step 2: LLM → historical text + image prompt
    try:
        llm_result = await _call_llm(
            location_ctx=location_ctx,
            year=request.year,
            era=request.era,
            style=request.style,
            language=request.language,
        )
        historical_text = llm_result.get("historical_text", "")
        image_prompt = llm_result.get("image_prompt", "")
    except Exception as e:
        logger.error("time_travel_gen_error_002: LLM failed: %s", e, exc_info=True)
        historical_text = ""
        image_prompt = (
            f"Historical view of {location_ctx.name}, {_year_for_prompt(request.year, request.era)}"
        )

    logger.info(
        "time_travel_gen_004: LLM done  text_len=%d  prompt_len=%d",
        len(historical_text),
        len(image_prompt),
    )

    # Step 3: image generation
    image_data, image_mime = await generate_image(
        image_prompt=image_prompt,
        style=request.style,
        reference_image_b64=request.reference_image_b64,
    )

    if image_data:
        logger.info("time_travel_gen_005: image generated  mime=%s  b64_len=%d", image_mime, len(image_data))
    else:
        logger.warning("time_travel_gen_006: image generation returned None")

    return TimeTravelResponse(
        image_data=image_data,
        image_mime=image_mime or "image/jpeg",
        historical_text=historical_text,
        image_prompt=image_prompt,
        era_label=era_label,
        location_name=location_ctx.name,
    )
