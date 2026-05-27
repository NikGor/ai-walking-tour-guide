"""Orchestrates the Time Travel Lens generation pipeline.

Flow:
  1. Geocode coordinates → location name + context
  2. Gemini 3 Pro Preview (+ Google Search Grounding) → historical analysis + visual_prompt JSON
  3. Gemini image model → rendered historical image
  4. Return combined TimeTravelResponse

Prompt system adapted from a production Time Travel app.
"""

import json
import logging
import os
from typing import Any

from app.time_travel.image_gen import generate_image
from app.time_travel.models import LuckyResponse, TimeTravelRequest, TimeTravelResponse
from app.utils.geocoder import LocationContext, get_location_context

logger = logging.getLogger(__name__)

# ── Model selection ────────────────────────────────────────────────────────────

_ANALYSIS_MODEL = "gemini-3-pro-preview"  # text + Google Search Grounding
_ANALYSIS_FALLBACK = "gemini-2.5-pro-preview-05-06"  # fallback if 3-pro unavailable

# ── Season helper ──────────────────────────────────────────────────────────────


def _get_season(month: int | None) -> str:
    if month is None:
        return "Unknown"
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


# ── Era helpers ────────────────────────────────────────────────────────────────


def _format_era(year: int, era: str, language: str) -> str:
    abs_year = abs(year)
    if language == "ru":
        suffix = "до н.э." if era == "BCE" else "н.э."
        return f"{abs_year:,} {suffix}".replace(",", " ")
    suffix = " BCE" if era == "BCE" else " CE"
    return f"{abs_year:,}{suffix}"


def _year_str(year: int, era: str) -> str:
    return f"{abs(year)} {era}"


# ── Role modules ───────────────────────────────────────────────────────────────

_ROLE_STANDARD = """\
ROLE: Expert Historical Photographer & Geographer.
INPUT DATA:
Location: {lat}, {lon}
Details: "{location_name}"
Date: {year} {era}
Season: {season} (Month: {month})

TASK: Analyze the location and time period. Create a highly detailed visual description for an AI image generator.

VISUAL GUIDELINES:
1. Historical Accuracy: What buildings existed? What was the landscape? (e.g., dirt roads vs paved, specific architecture styles).
2. Photography Style:
   - 2024+: Digital, sharp, 4k.
   - 1980s: Film grain, Kodachrome palette.
   - 1860s: Daguerreotype, vignette, long exposure blur.
   - Pre-1839: Hyper-realistic cinematic render or matte painting.
3. Environment: Strictly apply the season ({season}). If Winter, show snow/mud/bare trees. If Summer, lush vegetation/dust.
4. Lighting & Atmosphere: Define the mood (e.g., "Golden Hour", "Overcast", "Gaslamp lit").\
"""

_ROLE_SELFIE = """\
ROLE: Time Travel Scenographer.
INPUT DATA:
Location: {lat}, {lon}
Details: "{location_name}"
Date: {year} {era}
Season: {season} (Month: {month})

TASK: Create a visual description for a "Time Travel Selfie".
VISUAL GUIDELINES:
1. Subject: A typical person from this era (e.g., Roman Centurion, Medieval Peasant, 1920s Flapper).
2. Pose: The subject is in the foreground, looking into the "lens", arm extended or holding a device (if applicable) or just posing close-up.
3. Background: The specific historical location must be visible and recognizable behind the subject.
4. Style: Consistent with the era's visual technology (or hyper-realistic render if pre-camera).\
"""

_ROLE_ART = """\
ROLE: Art Historian.
INPUT DATA:
Location: {lat}, {lon}
Details: "{location_name}"
Date: {year} {era}
Season: {season} (Month: {month})

TASK: Describe an artwork that could have been created in this place at this time.
VISUAL GUIDELINES:
1. Medium & Style: Choose the dominant style of the era (e.g., Cave Painting, Egyptian Fresco, Roman Mosaic,
   Medieval Tapestry, Oil Painting, Ukiyo-e).
2. Technique: Describe brushstrokes, material texture (canvas, stone, papyrus), and color palette.
3. Subject: Depict the location or a typical event through the eyes of an artist of that time.\
"""

_FINAL_OUTPUT = """\

FINAL OUTPUT INSTRUCTION:
Provide a JSON object containing the historical details and a **visual_prompt** field.
The 'visual_prompt' field must contain the FULL, FINAL, ENGLISH text description to be sent
directly to the Image Generator.

Required JSON Structure:
{{
  "title": "Russian title for the historical card (max 10 words)",
  "description": "Russian historical description (3 sentences, vivid and specific)",
  "visual_prompt": "Detailed English text prompt describing the scene, objects, lighting, style, and camera. Include ALL historical details here.",
  "suggestions": [
    {{"label": "Russian label", "year": 1200, "era": "CE", "type": "time", "lat": {lat}, "lng": {lon}}},
    {{"label": "Russian label", "year": 44, "era": "BCE", "type": "time", "lat": {lat}, "lng": {lon}}}
  ]
}}
OUTPUT JSON ONLY. No markdown fences, no commentary.\
"""

_STRUCTURAL_BLOCK = """\
CRITICAL INSTRUCTION: STRUCTURAL & COMPOSITIONAL PRESERVATION.
The provided image is the GEOMETRIC BLUEPRINT. You MUST strictly adhere to the composition,
perspective, camera angle, and building layout of this image.

YOUR TASK: "Re-skin" this exact scene to the year {year} {era}.
1. KEEP: The main shapes, silhouettes, and spatial arrangement of objects. Do not rotate the camera.
2. REPLACE: Modern materials with era-appropriate ones (e.g., asphalt -> dirt/cobblestone,
   glass -> wood/stone/air).
3. ADAPT: If there are modern objects (cars, poles), replace them with era-equivalents
   (carriages, trees) or remove them, but maintain the depth and scale of the scene.

TARGET STYLE DESCRIPTION:
{visual_prompt}\
"""

_LUCKY_PROMPT = """\
Task: Pick a random, visually spectacular, and non-banal historical event or location.
Avoid cliché examples like Pyramids or Eiffel Tower. Think: "Tunguska Event",
"Library of Alexandria at peak", "Woodstock 1969", "Tenochtitlan 1519", "Constantinople 537 CE",
"Pompeii hours before eruption", "Battle of Agincourt", "Opening of Suez Canal".

Return strictly JSON:
{
  "lat": number,
  "lng": number,
  "year": number,
  "era": "BCE" or "CE",
  "month": number (1-12, optional),
  "hasSpecificDate": boolean,
  "locationDetail": "Specific name of the place/event in Russian"
}
OUTPUT JSON ONLY.\
"""


# ── Build prompt ───────────────────────────────────────────────────────────────


def _build_analysis_prompt(
    lat: float,
    lon: float,
    location_name: str,
    year: int,
    era: str,
    month: int | None,
    style: str,
) -> str:
    season = _get_season(month)
    month_str = str(month) if month else "unknown"
    year_s = _year_str(year, era)

    role_template = {"photorealistic": _ROLE_STANDARD, "selfie": _ROLE_SELFIE, "art": _ROLE_ART}.get(
        style, _ROLE_STANDARD
    )
    role = role_template.format(
        lat=lat,
        lon=lon,
        location_name=location_name,
        year=year_s,
        era=era,
        season=season,
        month=month_str,
    )
    final = _FINAL_OUTPUT.format(lat=lat, lon=lon)
    return role + "\n\n" + final


# ── Gemini analysis call ───────────────────────────────────────────────────────


async def _call_gemini_analysis(prompt: str) -> dict[str, Any]:
    """Call Gemini 3 Pro (with Google Search Grounding) for historical analysis."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Try the primary model, fall back on 404/unavailable
    for model in (_ANALYSIS_MODEL, _ANALYSIS_FALLBACK):
        try:
            logger.info("time_travel_gen_001: analysis with \033[36m%s\033[0m", model)
            resp = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.7,
                ),
            )
            raw = resp.candidates[0].content.parts[0].text or "{}"
            break
        except Exception as e:
            logger.warning("time_travel_gen_002: %s failed (%s), trying fallback", model, e)
    else:
        # Both Gemini models failed — fall back to OpenRouter GPT-4.1
        logger.warning("time_travel_gen_003: Gemini analysis unavailable, using OpenRouter fallback")
        raw = await _openrouter_fallback(prompt)

    return _parse_json(raw)


async def _openrouter_fallback(prompt: str) -> str:
    """Fallback: use OpenRouter GPT-4.1 for analysis (no grounding)."""
    from app.backend.openrouter_client import OpenRouterClient

    client = OpenRouterClient()
    resp = await client.create_completion(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-4.1",
    )
    return resp.choices[0].message.content or "{}"


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("time_travel_gen_004: non-JSON response (len=%d)", len(raw))
        return {"title": "", "description": raw[:400], "visual_prompt": raw, "suggestions": []}


# ── Lucky (random event) ───────────────────────────────────────────────────────


async def generate_lucky() -> LuckyResponse:
    """Pick a random spectacular historical event."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")

    from google import genai

    client = genai.Client(api_key=api_key)

    for model in (_ANALYSIS_MODEL, _ANALYSIS_FALLBACK):
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=_LUCKY_PROMPT,
            )
            raw = resp.candidates[0].content.parts[0].text or "{}"
            break
        except Exception as e:
            logger.warning("time_travel_gen_lucky_001: %s failed: %s", model, e)
    else:
        raise RuntimeError("All Gemini models unavailable for lucky request")

    data = _parse_json(raw)
    return LuckyResponse(
        lat=float(data["lat"]),
        lng=float(data["lng"]),
        year=int(data["year"]),
        era=data.get("era", "CE"),
        location_detail=data.get("locationDetail", ""),
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────


async def generate_time_travel(request: TimeTravelRequest) -> TimeTravelResponse:
    """Full pipeline: geocode → Gemini analysis → image → response."""
    logger.info(
        "time_travel_gen_005: lat=%.4f lon=%.4f year=%d %s style=%s model=%s",
        request.latitude,
        request.longitude,
        request.year,
        request.era,
        request.style,
        request.image_model,
    )

    # Step 1: geocode
    try:
        location_ctx = await get_location_context(request.latitude, request.longitude)
    except Exception as e:
        logger.error("time_travel_gen_error_001: geocode failed: %s", e)
        location_ctx = LocationContext(name=f"{request.latitude:.4f}, {request.longitude:.4f}")

    era_label = _format_era(request.year, request.era, request.language)
    logger.info("time_travel_gen_006: location=%r  era=%s", location_ctx.name, era_label)

    # Step 2: Gemini analysis → {title, description, visual_prompt, suggestions}
    try:
        prompt = _build_analysis_prompt(
            lat=request.latitude,
            lon=request.longitude,
            location_name=location_ctx.name,
            year=request.year,
            era=request.era,
            month=request.month,
            style=request.style,
        )
        llm = await _call_gemini_analysis(prompt)
    except Exception as e:
        logger.error("time_travel_gen_error_002: analysis failed: %s", e, exc_info=True)
        llm = {
            "title": "",
            "description": "",
            "visual_prompt": (
                f"Historical view of {location_ctx.name}, "
                f"{_year_str(request.year, request.era)}, "
                f"{request.style} style"
            ),
            "suggestions": [],
        }

    visual_prompt: str = llm.get("visual_prompt", "")
    title: str = llm.get("title", "")
    description: str = llm.get("description", "")
    suggestions: list[dict[str, Any]] = llm.get("suggestions", [])

    logger.info(
        "time_travel_gen_007: analysis done  title=%r  prompt_len=%d  suggestions=%d",
        title,
        len(visual_prompt),
        len(suggestions),
    )

    # Step 3: for reference photos, wrap prompt with structural preservation block
    image_prompt = visual_prompt
    if request.reference_image_b64:
        image_prompt = _STRUCTURAL_BLOCK.format(
            year=_year_str(request.year, request.era),
            era=request.era,
            visual_prompt=visual_prompt,
        )

    # Step 4: image generation
    image_data, image_mime = await generate_image(
        image_prompt=image_prompt,
        style=request.style,
        reference_image_b64=request.reference_image_b64,
        model_tier=request.image_model,
    )

    if image_data:
        logger.info("time_travel_gen_008: image ok  mime=%s  b64_len=%d", image_mime, len(image_data))
    else:
        logger.warning("time_travel_gen_009: image generation returned None")

    return TimeTravelResponse(
        image_data=image_data,
        image_mime=image_mime or "image/jpeg",
        title=title,
        historical_text=description,
        image_prompt=image_prompt,
        era_label=era_label,
        location_name=location_ctx.name,
        suggestions=suggestions,
    )
