import logging

from dotenv import load_dotenv

from app.agent.agent_factory import AgentFactory
from app.models import ChatRequest, ChatResponse

load_dotenv()

logger = logging.getLogger(__name__)

_agent = AgentFactory()


async def handle_chat(request: ChatRequest) -> ChatResponse:
    logger.info("=== STEP 2: Chat Request ===")
    logger.info(
        "controller_001: lat=\033[33m%.4f\033[0m lon=\033[33m%.4f\033[0m persona=\033[35m%s\033[0m",
        request.latitude, request.longitude, request.persona.value,
    )

    result = await _agent.run(request)

    logger.info("=== STEP 4: Response Ready ===")
    logger.info(
        "controller_002: title=\033[32m%r\033[0m confidence=\033[33m%.2f\033[0m",
        result.title, result.confidence,
    )
    return result
