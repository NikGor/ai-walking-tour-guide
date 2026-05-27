"""FastAPI routes for the Time Travel Lens Telegram Mini App."""

import logging
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.time_travel.generator import generate_time_travel
from app.time_travel.models import TimeTravelRequest, TimeTravelResponse

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
