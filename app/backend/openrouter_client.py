"""OpenRouter client for API interactions using OpenAI SDK."""

import logging
import os
from typing import Any
from openai import (
    APIConnectionError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel
from app.utils.retry_utils import call_with_retry


logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Client for OpenRouter API interactions using OpenAI SDK."""

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        logger.info("openrouter_client_001: Initialized OpenRouter client")

    def _log_usage(self, response: Any) -> None:
        """Log token usage from response."""
        try:
            usage = response.usage
            if usage:
                input_tok = getattr(usage, "prompt_tokens", 0) or 0
                output_tok = getattr(usage, "completion_tokens", 0) or 0
                total_tok = getattr(usage, "total_tokens", 0) or 0
                logger.info(
                    f"openrouter_client_004: Usage - Input: \033[33m{input_tok}\033[0m | "
                    f"Output: \033[33m{output_tok}\033[0m | "
                    f"Total: \033[33m{total_tok}\033[0m"
                )
            logger.info(
                f"openrouter_client_005: Model: \033[36m{response.model}\033[0m"
            )
        except Exception as e:
            logger.warning(f"openrouter_client_warning_001: Could not log usage: {e}")

    @staticmethod
    def _enforce_no_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively add ``additionalProperties: false`` to every object in schema.

        Required for strict JSON schema mode (Azure/OpenAI enforce this).
        """
        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        for section in ("properties", "$defs", "definitions"):
            for sub in schema.get(section, {}).values():
                if isinstance(sub, dict):
                    OpenRouterClient._enforce_no_additional_properties(sub)
        if isinstance(schema.get("items"), dict):
            OpenRouterClient._enforce_no_additional_properties(schema["items"])
        return schema

    def _build_create_kwargs(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs dict for chat.completions.create calls."""
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if response_format:
            schema = response_format.model_json_schema()
            if "properties" in schema:
                schema["properties"].pop("llm_trace", None)
                schema["properties"].pop("response_id", None)
                if "required" in schema:
                    schema["required"] = [
                        field
                        for field in schema["required"]
                        if field not in ["llm_trace", "response_id"]
                    ]
            self._enforce_no_additional_properties(schema)
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }
        if tools:
            create_kwargs["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]
        return create_kwargs

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: type[BaseModel] | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """
        Create a completion using OpenRouter API with structured outputs.

        Note: previous_response_id is accepted for interface compatibility
        but not used (OpenRouter doesn't support it yet).
        """
        msg_breakdown = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            msg_breakdown[role] = msg_breakdown.get(role, 0) + 1
        logger.info(
            f"openrouter_client_002: Calling \033[36m{model}\033[0m with \033[33m{len(messages)}\033[0m msgs "
            f"(system: {msg_breakdown.get('system', 0)}, user: {msg_breakdown.get('user', 0)}, assistant: {msg_breakdown.get('assistant', 0)})"
        )
        if previous_response_id:
            logger.info(
                "openrouter_client_003: previous_response_id ignored (not supported by OpenRouter)"
            )
        try:
            create_kwargs = self._build_create_kwargs(
                messages, model, response_format, tools
            )
            response = await call_with_retry(
                lambda: self.client.chat.completions.create(**create_kwargs),
                retryable_exceptions=(
                    RateLimitError,
                    APIConnectionError,
                    InternalServerError,
                ),
                context="openrouter_client",
            )
            self._log_usage(response)
            return response
        except Exception as e:
            logger.error(f"openrouter_client_error_001: \033[31m{e!s}\033[0m")
            raise

