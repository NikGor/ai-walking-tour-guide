from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Persona(str, Enum):
    historian = "historian"
    dark_tourism = "dark_tourism"
    architecture_expert = "architecture_expert"
    roman_empire = "roman_empire"
    ww2_context = "ww2_context"
    cyberpunk = "cyberpunk"
    fantasy_bard = "fantasy_bard"
    local_grandpa = "local_grandpa"


class ChatRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    photo_url: Optional[str] = None
    persona: Persona = Persona.historian
    message: Optional[str] = None


class TimelineEvent(BaseModel):
    year: str
    event: str


class ChatResponse(BaseModel):
    title: str
    summary: str
    history: str
    facts: list[str]
    timeline: list[TimelineEvent]
    related_people: list[str]
    sources: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
