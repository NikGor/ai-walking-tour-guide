import logging

from openai import AsyncOpenAI

from app.agent.prompt_builder import PromptBuilder
from app.models import ChatRequest, ChatResponse

logger = logging.getLogger("solaris.agent")

_PLACEHOLDER_CONTEXT = (
    "[Retrieval pipeline not yet connected. "
    "Use your best available knowledge of this location, "
    "but set confidence to reflect that no external sources were retrieved.]"
)


class AgentFactory:
    def __init__(self):
        self._client = AsyncOpenAI()
        self._prompt_builder = PromptBuilder()

    async def run(self, request: ChatRequest) -> ChatResponse:
        system_prompt = self._prompt_builder.build_system_prompt(request.persona.value)
        user_message = self._prompt_builder.build_user_message(
            latitude=request.latitude,
            longitude=request.longitude,
            retrieved_context=_PLACEHOLDER_CONTEXT,
            message=request.message,
        )

        logger.info("calling openai model=gpt-4.1")
        completion = await self._client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=ChatResponse,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("OpenAI returned no parsed response")

        logger.info("response ready: title=%r confidence=%.2f", result.title, result.confidence)
        return result
