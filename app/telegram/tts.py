"""Text-to-Speech via OpenRouter audio API.

Returns raw MP3 bytes ready to be sent as a Telegram voice message.
Requires OPENROUTER_API_KEY — no OPENAI_API_KEY needed.
"""

import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

_OPENROUTER_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"

_TTS_MODEL = "openai/gpt-4o-mini-tts"
_TTS_VOICE = "alloy"  # neutral/academic; good for Russian long-form narrative

# Style instruction — a single English sentence works best even for Russian content.
_TTS_INSTRUCTIONS = (
    "You are a documentary narrator — calm, authoritative, and engaging. "
    "Speak clearly at a measured pace. Preserve the original language of the text."
)

# ~1500 chars ≈ 70–80 s of audio — comfortable Telegram voice message length.
_MAX_CHARS = 1500


def _strip_markup(text: str) -> str:
    """Remove HTML/Markdown/entities before sending to TTS."""
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    text = re.sub(r"\*{1,3}|_{1,3}|~~|`+", "", text)  # markdown bold/italic/code
    text = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", text)  # HTML entities
    return text.strip()


async def synthesise(text: str) -> bytes | None:
    """Convert text to MP3 bytes via OpenRouter TTS.

    Returns None if OPENROUTER_API_KEY is not set or synthesis fails.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("tts_001: OPENROUTER_API_KEY not set — voice disabled")
        return None

    clean = _strip_markup(text)
    if not clean:
        return None
    if len(clean) > _MAX_CHARS:
        clean = clean[:_MAX_CHARS].rsplit(" ", 1)[0] + "…"

    payload = {
        "model": _TTS_MODEL,
        "voice": _TTS_VOICE,
        "input": clean,
        "instructions": _TTS_INSTRUCTIONS,
        "response_format": "mp3",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        logger.info("tts_002: synthesising %d chars  model=%s  voice=%s", len(clean), _TTS_MODEL, _TTS_VOICE)
        async with httpx.AsyncClient() as client:
            resp = await client.post(_OPENROUTER_TTS_URL, json=payload, headers=headers, timeout=30.0)
            resp.raise_for_status()
            audio = resp.content
        logger.info("tts_003: audio ready  %d bytes", len(audio))
        return audio
    except Exception as e:
        logger.error("tts_error_001: TTS failed: %s", e, exc_info=True)
        return None
