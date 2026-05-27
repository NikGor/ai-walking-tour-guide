import logging

from dotenv import load_dotenv

from app.agent.agent_factory import AgentFactory
from app.models import ChatRequest, ChatResponse

load_dotenv()

logger = logging.getLogger("solaris.controller")

_agent = AgentFactory()


async def handle_chat(request: ChatRequest) -> ChatResponse:
    logger.info("chat request: lat=%.4f lon=%.4f persona=%s", request.latitude, request.longitude, request.persona.value)
    return await _agent.run(request)
