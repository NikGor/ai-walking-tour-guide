import logging

from app.backend.openrouter_client import OpenRouterClient
from app.agent.prompt_builder import PromptBuilder
from app.agent.models.models import ChatRequest, ChatResponse
from app.utils.llm_parser import ParsedLLMResponse, parse_openrouter_response

logger = logging.getLogger(__name__)

_MODEL = "openai/gpt-4.1"

_PLACEHOLDER_CONTEXT = (
    "[Retrieval pipeline not yet connected. "
    "Use your best available knowledge of this location, "
    "but set confidence to reflect that no external sources were retrieved.]"
)


class AgentFactory:
    def __init__(self):
        self._client = OpenRouterClient()
        self._prompt_builder = PromptBuilder()

    async def run(self, request: ChatRequest) -> ParsedLLMResponse:
        system_prompt = self._prompt_builder.build_system_prompt(request.persona.value)
        user_message = self._prompt_builder.build_user_message(
            latitude=request.latitude,
            longitude=request.longitude,
            retrieved_context=_PLACEHOLDER_CONTEXT,
            message=request.message,
        )

        logger.info("=== STEP 3: AI Processing ===")
        logger.info("\033[35mLLM  ›\033[0m persona=\033[35m%s\033[0m  model=\033[36m%s\033[0m", request.persona.value, _MODEL)

        raw = await self._client.create_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=_MODEL,
            response_format=ChatResponse,
        )

        parsed = parse_openrouter_response(raw, ChatResponse)
        result: ChatResponse = parsed.parsed_content

        logger.info(
            "\033[35mLLM  ›\033[0m done  words=\033[33m%d\033[0m  tokens=\033[33m%d\033[0m  \033[2m$%.4f\033[0m",
            len(result.text.split()), parsed.llm_trace.total_tokens, parsed.llm_trace.total_cost,
        )
        return parsed
