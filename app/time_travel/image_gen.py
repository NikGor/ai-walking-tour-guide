"""OpenRouter-powered image generation for the Time Travel Lens.

Replaces the former google-genai SDK implementation.
Uses OPENROUTER_API_KEY only — no GEMINI_API_KEY / GOOGLE_API_KEY needed.
"""

import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model tiers mapped to OpenRouter model IDs (Gemini image family)
_IMAGE_MODELS: dict[str, str] = {
    "fast": "google/gemini-2.5-flash-image",
    "balanced": "google/gemini-3.1-flash-image-preview",
    "quality": "google/gemini-3-pro-image-preview",
}

# Fallback order when the primary model fails
_FALLBACK_CHAIN: list[str] = [
    "google/gemini-2.5-flash-image",
    "google/gemini-3.1-flash-image-preview",
    "google/gemini-3-pro-image-preview",
]


def _style_suffix(style: str) -> str:
    """Return an art-style description appended to the image prompt."""
    if style == "selfie":
        return (
            " Shot as a first-person selfie photo taken by a person standing there. "
            "Smartphone camera quality, authentic candid feel."
        )
    if style == "art":
        return (
            " Rendered in the dominant art style of the era — "
            "illuminated manuscript style for medieval, oil painting for Renaissance, "
            "daguerreotype or sepia photo for 19th century, modern digital art for future. "
            "High artistic quality."
        )
    # photorealistic (default)
    return (
        " Photorealistic, as if captured by a high-resolution camera. "
        "Natural lighting, accurate historical details, no anachronisms."
    )


def _build_messages(prompt: str, reference_image_b64: str | None) -> list[dict]:
    """Build the OpenRouter messages payload."""
    if reference_image_b64:
        # img2img: send text prompt + reference image together
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{reference_image_b64}"},
                    },
                ],
            }
        ]
    return [{"role": "user", "content": prompt}]


def _extract_image(response_json: dict) -> tuple[str | None, str]:
    """Pull the first image out of an OpenRouter chat-completions response.

    Response content is an array of typed parts:
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    or occasionally a plain string when only text was returned.
    """
    try:
        choices = response_json.get("choices", [])
        if not choices:
            logger.warning("time_travel_img_008: no choices in response")
            return None, ""

        content = choices[0].get("message", {}).get("content")
        if not content:
            logger.warning("time_travel_img_008b: empty message content")
            return None, ""

        # content can be a list of parts or a plain string
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]

        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                # url is "data:<mime>;base64,<data>"
                m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
                if m:
                    mime, img_b64 = m.group(1), m.group(2).strip()
                    logger.info(
                        "time_travel_img_009: extracted image  mime=%s  b64_len=%d",
                        mime,
                        len(img_b64),
                    )
                    return img_b64, mime

        logger.warning("time_travel_img_010: no image_url part found in response")
        return None, ""

    except Exception as e:
        logger.error("time_travel_img_error_004: extraction failed: %s", e, exc_info=True)
        return None, ""


async def _call_openrouter(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    api_key: str,
) -> dict:
    """Single OpenRouter chat-completions call. Returns parsed JSON or raises."""
    payload = {
        "model": model,
        "modalities": ["image", "text"],
        "messages": messages,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = await client.post(_OPENROUTER_URL, json=payload, headers=headers, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


async def generate_image(
    image_prompt: str,
    style: str = "photorealistic",
    reference_image_b64: str | None = None,
    model_tier: str = "balanced",
) -> tuple[str | None, str]:
    """Generate a historical scene image via OpenRouter.

    Args:
        image_prompt: Full text prompt for the scene.
        style: "photorealistic" | "selfie" | "art"
        reference_image_b64: Optional base64 reference photo for img2img.
        model_tier: "fast" | "balanced" | "quality"

    Returns:
        (base64_image_data, mime_type) — or (None, "") on failure.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("time_travel_img_001: OPENROUTER_API_KEY is not set")
        return None, ""

    # For reference photos the structural preservation block is already in the prompt;
    # don't append a style suffix as it may conflict with those structural instructions.
    full_prompt = image_prompt if reference_image_b64 else image_prompt + _style_suffix(style)

    messages = _build_messages(full_prompt, reference_image_b64)

    # Build fallback chain: requested model first, then the rest
    primary = _IMAGE_MODELS.get(model_tier, _IMAGE_MODELS["balanced"])
    candidates = [primary] + [m for m in _FALLBACK_CHAIN if m != primary]

    async with httpx.AsyncClient() as client:
        for model in candidates:
            logger.info(
                "time_travel_img_002: generating with \033[36m%s\033[0m  img2img=%s  prompt_len=%d",
                model,
                bool(reference_image_b64),
                len(full_prompt),
            )
            try:
                response_json = await _call_openrouter(client, model, messages, api_key)
                img_b64, mime = _extract_image(response_json)
                if img_b64:
                    return img_b64, mime
                logger.warning("time_travel_img_003: %s returned no image — trying next", model)
            except httpx.HTTPStatusError as e:
                logger.warning(
                    "time_travel_img_003: %s HTTP %d — trying next",
                    model,
                    e.response.status_code,
                )
            except Exception as e:
                logger.warning("time_travel_img_003: %s failed (%s) — trying next", model, e)

    logger.error("time_travel_img_error_001: all image models exhausted, returning None")
    return None, ""
