import logging
import os

from openai import AsyncOpenAI

from app.agent.prompt_builder import PromptBuilder
from app.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MODEL = "openai/gpt-4.1"

_PLACEHOLDER_CONTEXT = (
    "[Retrieval pipeline not yet connected. "
    "Use your best available knowledge of this location, "
    "but set confidence to reflect that no external sources were retrieved.]"
)


class AgentFactory:
    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=_OPENROUTER_BASE_URL,
        )
        self._prompt_builder = PromptBuilder()

    async def run(self, request: ChatRequest) -> ChatResponse:
        system_prompt = self._prompt_builder.build_system_prompt(request.persona.value)
        user_message = self._prompt_builder.build_user_message(
            latitude=request.latitude,
            longitude=request.longitude,
            retrieved_context=_PLACEHOLDER_CONTEXT,
            message=request.message,
        )

        logger.info("=== STEP 3: AI Processing ===")
        logger.info("agent_001: Persona: \033[35m%s\033[0m", request.persona.value)
        logger.info("agent_002: Calling \033[36m%s\033[0m", _MODEL)

        completion = await self._client.beta.chat.completions.parse(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=ChatResponse,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("OpenRouter returned no parsed response")

        logger.info(
            "agent_003: Response len: \033[33m%d\033[0m words, confidence: \033[33m%.2f\033[0m",
            len(result.history.split()), result.confidence,
        )
        return result
