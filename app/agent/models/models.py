from datetime import datetime
from datetime import timezone as dt_timezone
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Persona ───────────────────────────────────────────────────────────────────

class Persona(str, Enum):
    historian          = "historian"
    dark_tourism       = "dark_tourism"
    architecture_expert = "architecture_expert"
    roman_empire       = "roman_empire"
    ww2_context        = "ww2_context"
    cyberpunk          = "cyberpunk"
    storyteller        = "storyteller"
    local_grandpa      = "local_grandpa"


# ── Tour response ─────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """LLM output — narrative text about the location."""

    text: str



# ── Content ───────────────────────────────────────────────────────────────────

class Content(BaseModel):
    """Plain-text message content."""

    text: str


# ── Token-usage models ────────────────────────────────────────────────────────

class InputTokensDetails(BaseModel):
    """Details about input token usage."""

    cached_tokens: int = Field(default=0, description="Number of cached tokens used")


class OutputTokensDetails(BaseModel):
    """Details about output token usage."""

    reasoning_tokens: int = Field(
        default=0, description="Number of reasoning tokens generated"
    )


class LllmTrace(BaseModel):
    """Complete LLM usage trace for cost tracking and analytics."""

    model: str = Field(description="Name of the LLM model used")
    input_tokens: int = Field(description="Number of input tokens processed")
    input_tokens_details: InputTokensDetails = Field(
        default_factory=InputTokensDetails,
        description="Details about input token usage",
    )
    output_tokens: int = Field(description="Number of output tokens generated")
    output_tokens_details: OutputTokensDetails = Field(
        default_factory=OutputTokensDetails,
        description="Details about output token usage",
    )
    total_tokens: int = Field(description="Total number of tokens used")
    total_cost: float = Field(default=0.0, description="Total cost of the API call in USD")


# ── Pipeline tracing ──────────────────────────────────────────────────────────

class PipelineStep(BaseModel):
    """Single pipeline step timing record."""

    step: str = Field(description="Name of the pipeline step")
    status: str = Field(description="Status of the step")
    duration_ms: int = Field(description="Duration of the step in milliseconds")


class StepTrace(BaseModel):
    """Timing and LLM usage data for a single pipeline stage."""

    duration_ms: int = Field(description="Duration of the stage in milliseconds")
    llm_trace: Optional[LllmTrace] = Field(
        default=None,
        description="LLM usage trace for this stage, if it involved an LLM call",
    )


class PipelineTrace(BaseModel):
    """Aggregated trace for all stages of a single agent request."""

    total_ms: int = Field(description="Total duration of the request in milliseconds")
    steps: list[StepTrace] = Field(default_factory=list, description="Per-stage traces")


# ── Chat models ───────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """Individual chat message in a conversation."""

    message_id: Optional[str] = Field(
        default=None, description="Unique identifier for the message"
    )
    role: Literal["user", "assistant", "system"] = Field(
        description="Role of the message sender"
    )
    content: Content = Field(description="Structured content of the message")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(dt_timezone.utc),
        description="Timestamp when the message was created",
    )
    conversation_id: Optional[str] = Field(
        default=None, description="ID of the conversation this message belongs to"
    )
    previous_message_id: Optional[str] = Field(
        default=None, description="ID of the previous message for threading"
    )
    model: Optional[str] = Field(
        default=None, description="LLM model used to generate this message"
    )
    llm_trace: Optional[LllmTrace] = Field(
        default=None, description="LLM usage trace for this message"
    )
    pipeline_steps: list[PipelineStep] = Field(
        default_factory=list,
        description="Pipeline step timings for this message",
    )
    pipeline_trace: Optional[PipelineTrace] = Field(
        default=None,
        description="Detailed per-stage pipeline trace from the AI agent",
    )


class Conversation(BaseModel):
    """Complete conversation with messages and metadata."""

    conversation_id: str = Field(description="Unique identifier for the conversation")
    title: str = Field(default="Walking Tour", description="Title of the conversation")
    messages: Optional[List[ChatMessage]] = Field(
        default=None, description="List of messages in the conversation"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(dt_timezone.utc),
        description="Timestamp when the conversation was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(dt_timezone.utc),
        description="Timestamp when the conversation was last updated",
    )
    llm_trace: Optional[LllmTrace] = Field(
        default=None, description="Aggregated LLM usage trace for the conversation"
    )
    total_input_tokens: int = Field(
        default=0, description="Total input tokens used in this conversation"
    )
    total_output_tokens: int = Field(
        default=0, description="Total output tokens generated in this conversation"
    )
    total_tokens: int = Field(
        default=0, description="Total tokens used in this conversation"
    )
    total_cost: float = Field(
        default=0.0, description="Total cost of this conversation in USD"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.messages:
            from .utils import calculate_conversation_llm_trace
            self.llm_trace = calculate_conversation_llm_trace(self.messages)


# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Walking tour chat request."""

    latitude: Optional[float] = Field(default=None, ge=-90, le=90, description="GPS latitude")
    longitude: Optional[float] = Field(default=None, ge=-180, le=180, description="GPS longitude")
    photo_url: Optional[str] = Field(
        default=None, description="Optional photo URL for scene detection"
    )
    persona: Persona = Field(
        default=Persona.historian, description="Narrator persona"
    )
    response_format: Literal["plain", "markdown", "html", "ssml"] = Field(
        default="markdown", description="Format of the text content in the response"
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional user question; also determines response language",
    )
    user_name: Optional[str] = Field(
        default=None, description="Name of the user"
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Continue an existing conversation; auto-created if absent"
    )
    previous_message_id: Optional[str] = Field(
        default=None, description="ID of the previous message for threading"
    )
    language: Optional[str] = Field(
        default=None,
        description="Force response language: 'ru', 'en', 'de'. None = auto-detect from message.",
    )


class ConversationRequest(BaseModel):
    """Request model for creating a new conversation."""

    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional custom conversation ID. Auto-generated if not provided.",
    )
    title: Optional[str] = Field(
        default="Walking Tour", description="Title of the conversation"
    )


class ConversationResponse(BaseModel):
    """Response model for conversation creation."""

    conversation_id: str = Field(description="ID of the created conversation")
    title: str = Field(description="Title of the conversation")
    created_at: datetime = Field(description="Timestamp when the conversation was created")
    message: str = Field(
        default="Conversation created successfully", description="Success message"
    )


class ChatHistoryMessage(BaseModel):
    """Simplified message model for chat history."""

    role: Literal["user", "assistant", "system"] = Field(
        description="Role of the message sender"
    )
    content: Content = Field(description="Structured content of the message")


class ChatHistoryResponse(BaseModel):
    """Response model for chat history endpoint."""

    messages: List[ChatHistoryMessage] = Field(
        description="List of messages in the conversation"
    )
