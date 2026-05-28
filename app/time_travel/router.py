"""FastAPI routes for the Time Travel Lens Telegram Mini App."""

import base64
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from urllib.parse import parse_qsl, unquote

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.time_travel.generator import generate_lucky, generate_time_travel
from app.time_travel.models import LuckyResponse, SendToChatRequest, TimeTravelRequest, TimeTravelResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/time-travel", tags=["time-travel"])

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


@router.get("", response_class=HTMLResponse, include_in_schema=False)
async def time_travel_app() -> HTMLResponse:
    """Serve the Telegram Mini App HTML page."""
    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    # Inject the API base URL so the frontend knows where to call
    api_base = os.getenv("APP_BASE_URL", "")
    html = html.replace("__API_BASE__", api_base)
    return HTMLResponse(content=html)


@router.post("/generate", response_model=TimeTravelResponse)
async def generate(request: TimeTravelRequest) -> TimeTravelResponse:
    """Generate a historical image + text for a location and era."""
    logger.info(
        "time_travel_route_001: generate  lat=%.4f lon=%.4f year=%d %s",
        request.latitude,
        request.longitude,
        request.year,
        request.era,
    )
    try:
        return await generate_time_travel(request)
    except Exception as e:
        logger.error("time_travel_route_error_001: %s", e, exc_info=True)
        return TimeTravelResponse(
            historical_text="",
            era_label="",
            location_name="",
            error=str(e),
        )


@router.post("/lucky", response_model=LuckyResponse)
async def lucky() -> LuckyResponse:
    """Return a random spectacular historical event (location + date)."""
    try:
        return await generate_lucky()
    except Exception as e:
        logger.error("time_travel_route_error_lucky: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ── Send to Telegram chat ──────────────────────────────────────────────────────


def _validate_init_data(init_data: str) -> int | None:
    """Validate Telegram WebApp initData and return chat_id (= user_id for private chats).

    Returns None if invalid or bot token not set.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not bot_token or not init_data:
        return None

    parsed = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = parsed.pop("hash", "")

    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(parsed.items()))

    secret_key = hmac.new(
        key=b"WebAppData",
        msg=bot_token.encode(),
        digestmod=hashlib.sha256,
    ).digest()
    expected_hash = hmac.new(
        key=secret_key,
        msg=data_check_string.encode(),
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(received_hash, expected_hash):
        logger.warning(
            "time_travel_route_002: initData hash mismatch  recv=%.8s  want=%.8s",
            received_hash,
            expected_hash,
        )
        return None

    try:
        user = json.loads(unquote(parsed.get("user", "{}")))
        user_id = user.get("id")
        logger.info("time_travel_route_002b: initData valid  user_id=%s", user_id)
        return user_id
    except Exception as e:
        logger.warning("time_travel_route_002c: user parse failed: %s", e)
        return None


@router.post("/send-to-chat")
async def send_to_chat(request: SendToChatRequest) -> dict:
    """Send the generated image to the user's Telegram chat."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        raise HTTPException(status_code=503, detail="Bot token not configured")

    # Authenticate via initData (skip validation in local dev if token matches env)
    chat_id = _validate_init_data(request.init_data)
    if not chat_id:
        logger.warning(
            "time_travel_route_send_403: init_data len=%d  bot_token_set=%s",
            len(request.init_data),
            bool(bot_token),
        )
        raise HTTPException(status_code=403, detail="Invalid or missing initData")

    logger.info(
        "time_travel_route_003: sending photo to chat_id=%d  era=%s  loc=%r",
        chat_id,
        request.era_label,
        request.location_name,
    )

    # Build caption (Telegram photo captions: max 1024 chars)
    caption_parts = [
        f"🕰 <b>{request.era_label}</b>",
        f"📍 {request.location_name}",
        "",
        request.historical_text[:900],
    ]
    caption = "\n".join(caption_parts)

    # Decode image
    try:
        image_bytes = base64.b64decode(request.image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_data: {e}") from e

    ext = "png" if "png" in request.image_mime.lower() else "jpg"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"photo": (f"time_travel.{ext}", image_bytes, request.image_mime)},
            )
        result = resp.json()
        if not result.get("ok"):
            logger.error("time_travel_route_error_002: Telegram API error: %s", result)
            raise HTTPException(status_code=502, detail=result.get("description", "Telegram error"))
        msg_id = result.get("result", {}).get("message_id")
        logger.info("time_travel_route_004: photo sent ok  message_id=%s", msg_id)
        return {"ok": True}
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="Telegram API timeout") from e
