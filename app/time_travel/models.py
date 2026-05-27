"""Pydantic models for the Time Travel Lens feature."""

from typing import Literal

from pydantic import BaseModel, Field


class TimeTravelRequest(BaseModel):
    latitude: float
    longitude: float
    year: int = Field(..., description="Year number (positive = CE/AD, negative = BCE)")
    era: Literal["BCE", "CE"] = "CE"
    style: Literal["photorealistic", "selfie", "art"] = "photorealistic"
    reference_image_b64: str | None = None  # base64 JPEG for img2img "my street" mode
    language: str = "ru"


class TimeTravelResponse(BaseModel):
    image_data: str | None = None  # base64-encoded image bytes
    image_mime: str = "image/jpeg"
    historical_text: str
    image_prompt: str = ""  # prompt used for image gen (for debug/transparency)
    era_label: str  # e.g. "1462 н.э." or "2560 до н.э."
    location_name: str
    error: str | None = None
