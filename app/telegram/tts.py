"""Text-to-Speech via OpenAI TTS API.

Returns raw MP3 bytes ready to be sent as a Telegram voice message.
Requires OPENAI_API_KEY (not OpenRouter — TTS is only on the direct API).
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# Voice that works well for Russian narrative text.
_TTS_VOICE = "nova"
_TTS_MODEL = "tts-1"
# Telegram voice messages cap at ~1 min for good UX; ~1000 chars ≈ 60 s.
_MAX_CHARS = 1000


def _strip_markup(text: str) -> str:
    """Remove HTML tags and markdown symbols before sending to TTS."""
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    text = re.sub(r"\*{1,3}|_{1,3}|~~|`+", "", text)  # markdown bold/italic/code
    return text.strip()


async def synthesise(text: str) -> bytes | None:
    """Convert text to MP3 bytes via OpenAI TTS.

    Returns None if OPENAI_API_KEY is not set or synthesis fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("tts_001: OPENAI_API_KEY not set — voice disabled")
        return None

    clean = _strip_markup(text)
    if len(clean) > _MAX_CHARS:
        clean = clean[:_MAX_CHARS].rsplit(" ", 1)[0] + "…"

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        logger.info("tts_002: synthesising %d chars  voice=%s", len(clean), _TTS_VOICE)
        response = await client.audio.speech.create(
            model=_TTS_MODEL,
            voice=_TTS_VOICE,
            input=clean,
            response_format="mp3",
        )
        audio = response.read()
        logger.info("tts_003: audio ready  %d bytes", len(audio))
        return audio
    except Exception as e:
        logger.error("tts_error_001: TTS failed: %s", e, exc_info=True)
        return None
