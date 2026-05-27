"""Utility functions for agent models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ChatMessage, LllmTrace


def calculate_conversation_llm_trace(
    messages: list[ChatMessage],
) -> LllmTrace | None:
    """Aggregate LLM traces from all messages in a conversation.

    Returns a combined ``LllmTrace`` summing tokens and cost across all
    assistant messages, or ``None`` if no traces are present.
    """
    from .models import InputTokensDetails, LllmTrace, OutputTokensDetails

    traces = [m.llm_trace for m in messages if m.llm_trace is not None]
    if not traces:
        return None

    total_input = sum(t.input_tokens for t in traces)
    total_output = sum(t.output_tokens for t in traces)

    return LllmTrace(
        model=traces[-1].model,
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
        total_cost=sum(t.total_cost for t in traces),
        input_tokens_details=InputTokensDetails(
            cached_tokens=sum(t.input_tokens_details.cached_tokens for t in traces)
        ),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=sum(t.output_tokens_details.reasoning_tokens for t in traces)
        ),
    )
