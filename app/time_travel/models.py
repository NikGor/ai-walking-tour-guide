"""Pydantic models for the Time Travel Lens feature."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TimeTravelRequest(BaseModel):
    latitude: float
    longitude: float
    year: int = Field(..., description="Year number (positive = CE/AD, negative = BCE)")
    era: Literal["BCE", "CE"] = "CE"
    month: int | None = None  # 1-12; used for season detection
    style: Literal["photorealistic", "selfie", "art"] = "photorealistic"
    # Image generation model tier
    image_model: Literal["fast", "balanced", "quality"] = "balanced"
    reference_image_b64: str | None = None  # base64 JPEG for img2img "my street" mode
    language: str = "ru"


class SendToChatRequest(BaseModel):
    image_data: str  # base64
    image_mime: str = "image/jpeg"
    era_label: str
    location_name: str
    historical_text: str
    init_data: str = Field(default="", description="Telegram WebApp initData for auth")


class TimeTravelResponse(BaseModel):
    image_data: str | None = None  # base64-encoded image bytes
    image_mime: str = "image/jpeg"
    title: str = ""  # short Russian title for the result card
    historical_text: str  # 3-sentence Russian description
    image_prompt: str = ""  # final English prompt sent to image generator
    era_label: str  # e.g. "1462 н.э." or "2560 до н.э."
    location_name: str
    suggestions: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class LuckyResponse(BaseModel):
    lat: float
    lng: float
    year: int
    era: Literal["BCE", "CE"]
    location_detail: str  # Russian name of the event/place
