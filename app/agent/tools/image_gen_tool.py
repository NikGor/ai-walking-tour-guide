"""On-demand image generation tool for the agentic loop."""

import base64
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = "x-ai/grok-imagine-image-quality"


async def generate_image_tool(prompt: str) -> bytes | None:
    """Generate an image via Grok Imagine and return raw JPEG bytes."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("image_gen_tool: OPENROUTER_API_KEY not set")
        return None

    payload = {
        "model": _MODEL,
        "modalities": ["image"],
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info("image_gen_tool_001: generating  model=%s  prompt=%r", _MODEL, prompt[:80])
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(_OPENROUTER_URL, json=payload, headers=headers, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("choices", [{}])[0].get("message", {})
        for img in msg.get("images") or []:
            url = (img.get("image_url") or {}).get("url", "")
            m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
            if m:
                image_bytes = base64.b64decode(m.group(2).strip())
                logger.info("image_gen_tool_002: generated %d bytes", len(image_bytes))
                return image_bytes

        logger.warning("image_gen_tool_003: no image in response")
        return None
    except Exception as e:
        logger.error("image_gen_tool_error: %s", e, exc_info=True)
        return None
