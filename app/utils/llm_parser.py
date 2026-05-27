"""LLM response parsing and token-cost tracking."""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agent.models.models import (
    InputTokensDetails,
    LllmTrace,
    OutputTokensDetails,
)

logger = logging.getLogger(__name__)

# ── Token pricing ─────────────────────────────────────────────────────────────
# Prices in USD per 1 000 000 tokens (input / output).
# Prefix matching is used for versioned names (e.g. gpt-4.1-2025-04-14).
MODEL_TOKEN_PRICES: dict[str, dict[str, float]] = {
    "openai/gpt-4.1":     {"input": 2.00, "output": 8.00},
    "gpt-4.1":            {"input": 2.00, "output": 8.00},
    "openai/gpt-4o":      {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


# ── Result wrapper ────────────────────────────────────────────────────────────

class ParsedLLMResponse(BaseModel):
    """Parsed response from an LLM provider."""

    model_config = {"arbitrary_types_allowed": True}

    parsed_content: Any
    llm_trace: LllmTrace
    response_id: str | None = None


# ── Cost helper ───────────────────────────────────────────────────────────────

def calculate_token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return cost in USD for the given token counts.

    Uses prefix matching so versioned names like ``gpt-4.1-2025-04-14``
    match the ``gpt-4.1`` key.
    """
    prices = MODEL_TOKEN_PRICES.get(model) or next(
        (v for k, v in MODEL_TOKEN_PRICES.items() if model.startswith(k)), None
    )
    if prices is None:
        return 0.0
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]


# ── OpenRouter parser ─────────────────────────────────────────────────────────

def parse_openrouter_response(
    raw_response: Any,
    expected_type: type[BaseModel],
) -> ParsedLLMResponse:
    """Parse an OpenRouter ``chat.completions`` response.

    Extracts token usage, calculates cost, and validates the JSON content
    into ``expected_type``.
    """
    logger.info("llm_parser_001: Parsing OpenRouter response")

    usage = raw_response.usage
    input_tokens  = getattr(usage, "prompt_tokens",     0) or 0 if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
    total_tokens  = getattr(usage, "total_tokens",      0) or 0 if usage else 0

    prompt_details     = getattr(usage, "prompt_tokens_details",     None) if usage else None
    completion_details = getattr(usage, "completion_tokens_details", None) if usage else None

    model = raw_response.model or "unknown"
    llm_trace = LllmTrace(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        total_cost=calculate_token_cost(model, input_tokens, output_tokens),
        input_tokens_details=InputTokensDetails(
            cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0
        ),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0
        ),
    )

    logger.info(
        "llm_parser_002: cost=\033[33m$%.6f\033[0m "
        "in=\033[33m%d\033[0m out=\033[33m%d\033[0m model=\033[36m%s\033[0m",
        llm_trace.total_cost, input_tokens, output_tokens, model,
    )

    if not raw_response.choices:
        raise ValueError("OpenRouter response has no choices")
    content = raw_response.choices[0].message.content
    if not content:
        raise ValueError("OpenRouter response content is empty")

    parsed_content = expected_type.model_validate(json.loads(content))
    logger.info(
        "llm_parser_003: parsed \033[36m%s\033[0m",
        type(parsed_content).__name__,
    )

    return ParsedLLMResponse(
        parsed_content=parsed_content,
        llm_trace=llm_trace,
        response_id=getattr(raw_response, "id", None),
    )
