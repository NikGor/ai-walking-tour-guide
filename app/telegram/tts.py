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

_TTS_MODEL = "openai/gpt-4o-mini-tts-2025-12-15"
_TTS_VOICE = "alloy"  # neutral/academic; good for Russian long-form narrative

# ~1500 chars ≈ 70–80 s of audio — comfortable Telegram voice message length.
_MAX_CHARS = 1500

_PERSONA_INSTRUCTIONS: dict[str, str] = {
    "historian": (
        "You are a composed academic historian narrating in Russian. "
        "Speak clearly and steadily, with measured pacing. Thoughtful, authoritative tone."
    ),
    "architecture_expert": (
        "You are an architecture expert narrating in Russian. "
        "Speak with focused precision and quiet enthusiasm, as if explaining a blueprint."
    ),
    "roman_empire": (
        "You are narrating in Russian about the Roman Empire. "
        "Speak with gravitas and grandeur, like a senator recounting ancient glory."
    ),
    "storyteller": (
        "You are a theatrical storyteller narrating in Russian. "
        "Be expressive and vivid — vary your pace, build tension, let moments land."
    ),
    "medieval_resident": (
        "You are narrating in Russian as a medieval townsperson. "
        "Speak in a warm, earthy tone, as if recounting life from memory."
    ),
    "military_expert": (
        "You are a military historian narrating in Russian. "
        "Speak crisply and directly, with the disciplined cadence of a field briefing."
    ),
    "deep_time": (
        "You are narrating vast geological timescales in Russian. "
        "Speak slowly and with awe, as if contemplating epochs beyond human memory."
    ),
}


def _strip_markup(text: str) -> str:
    """Remove HTML/Markdown/entities before sending to TTS."""
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    text = re.sub(r"\*{1,3}|_{1,3}|~~|`+", "", text)  # markdown bold/italic/code
    text = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", text)  # HTML entities
    return text.strip()


async def synthesise(text: str, persona: str = "historian") -> bytes | None:
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
        "response_format": "mp3",
        "instructions": _PERSONA_INSTRUCTIONS.get(persona, _PERSONA_INSTRUCTIONS["historian"]),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        logger.info("tts_002: synthesising %d chars  model=%s  voice=%s", len(clean), _TTS_MODEL, _TTS_VOICE)
        async with httpx.AsyncClient() as client:
            resp = await client.post(_OPENROUTER_TTS_URL, json=payload, headers=headers, timeout=30.0)
            if not resp.is_success:
                logger.error("tts_error_001: HTTP %d — %s", resp.status_code, resp.text)
                resp.raise_for_status()
            audio = resp.content
        logger.info("tts_003: audio ready  %d bytes", len(audio))
        return audio
    except httpx.HTTPStatusError:
        return None
    except Exception as e:
        logger.error("tts_error_002: TTS failed: %s", e, exc_info=True)
        return None
