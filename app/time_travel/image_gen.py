"""Gemini-powered image generation for the Time Travel Lens."""

import base64
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Gemini model that natively generates images alongside text.
# Falls back to Imagen 3 if unavailable.
_GEMINI_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"
_IMAGEN_MODEL = "imagen-3.0-generate-002"


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


async def generate_image(
    image_prompt: str,
    style: str = "photorealistic",
    reference_image_b64: str | None = None,
) -> tuple[str | None, str]:
    """Generate a historical scene image.

    Returns (base64_image_data, mime_type).
    Returns (None, "") on failure.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("time_travel_img_001: GEMINI_API_KEY not set")
        return None, ""

    full_prompt = image_prompt + _style_suffix(style)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        if reference_image_b64:
            # img2img: use reference photo + transform to historical era
            return await _generate_with_reference(client, types, full_prompt, reference_image_b64)
        else:
            return await _generate_text_to_image(client, types, full_prompt)

    except Exception as e:
        logger.error("time_travel_img_error_001: %s", e, exc_info=True)
        return None, ""


async def _generate_text_to_image(
    client: Any,
    types: Any,
    prompt: str,
) -> tuple[str | None, str]:
    """Generate image from text prompt using Gemini image generation."""
    logger.info(
        "time_travel_img_002: Generating image with \033[36m%s\033[0m  prompt_len=%d",
        _GEMINI_IMAGE_MODEL,
        len(prompt),
    )

    try:
        response = await client.aio.models.generate_content(
            model=_GEMINI_IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        return _extract_image_from_response(response)
    except Exception as e:
        logger.warning(
            "time_travel_img_003: %s failed (%s), trying Imagen 3",
            _GEMINI_IMAGE_MODEL,
            e,
        )
        return await _generate_with_imagen3(client, types, prompt)


async def _generate_with_reference(
    client: Any,
    types: Any,
    prompt: str,
    reference_image_b64: str,
) -> tuple[str | None, str]:
    """img2img: transform a reference photo to a historical era."""
    logger.info("time_travel_img_004: img2img generation with reference photo")

    try:
        image_bytes = base64.b64decode(reference_image_b64)
        contents = [
            types.Part.from_text(
                "Using the attached reference photo as the composition and viewpoint, " + prompt
            ),
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ]
        response = await client.aio.models.generate_content(
            model=_GEMINI_IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        return _extract_image_from_response(response)
    except Exception as e:
        logger.error("time_travel_img_error_002: img2img failed: %s", e, exc_info=True)
        return None, ""


async def _generate_with_imagen3(
    client: Any,
    types: Any,
    prompt: str,
) -> tuple[str | None, str]:
    """Fallback: generate with Imagen 3."""
    logger.info("time_travel_img_005: Trying Imagen 3 fallback")
    try:
        response = await client.aio.models.generate_images(
            model=_IMAGEN_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
            ),
        )
        imgs = response.generated_images
        if imgs and imgs[0].image and imgs[0].image.image_bytes:
            img_b64 = base64.b64encode(imgs[0].image.image_bytes).decode()
            logger.info("time_travel_img_006: Imagen 3 succeeded, %d bytes", len(imgs[0].image.image_bytes))
            return img_b64, "image/jpeg"
        logger.warning("time_travel_img_007: Imagen 3 returned no images")
        return None, ""
    except Exception as e:
        logger.error("time_travel_img_error_003: Imagen 3 failed: %s", e, exc_info=True)
        return None, ""


def _extract_image_from_response(response: object) -> tuple[str | None, str]:
    """Pull the first image part out of a Gemini generate_content response."""
    try:
        candidates = getattr(response, "candidates", [])
        if not candidates:
            logger.warning("time_travel_img_008: No candidates in response")
            return None, ""

        for part in candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                mime = getattr(inline, "mime_type", "image/jpeg")
                # data may already be base64 string or raw bytes
                if isinstance(data, (bytes, bytearray)):
                    img_b64 = base64.b64encode(data).decode()
                else:
                    img_b64 = data  # already base64
                logger.info(
                    "time_travel_img_009: Extracted image  mime=%s  b64_len=%d",
                    mime,
                    len(img_b64),
                )
                return img_b64, mime

        logger.warning("time_travel_img_010: No inline_data found in response parts")
        return None, ""
    except Exception as e:
        logger.error("time_travel_img_error_004: extraction failed: %s", e, exc_info=True)
        return None, ""
