import logging
import uuid

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.agent_factory import AgentFactory
from app.agent.models.models import ChatMessage, ChatRequest, ChatResponse, Content
from app.db.repository import get_or_create_conversation, save_message

load_dotenv()

logger = logging.getLogger(__name__)

_agent = AgentFactory()


def _render(result: ChatResponse, fmt: str) -> str:
    if fmt == "html":
        return f"<h1>{result.title}</h1>\n<p>{result.text}</p>"
    if fmt == "ssml":
        return f"<speak><p>{result.title}. {result.text}</p></speak>"
    return result.text


def _user_content_text(request: ChatRequest) -> str:
    lines = [f"📍 {request.latitude}, {request.longitude} | {request.persona.value}"]
    if request.message:
        lines.append(request.message)
    return "\n".join(lines)


async def handle_chat(request: ChatRequest, db: AsyncSession) -> ChatMessage:
    logger.info("=== STEP 2: Chat Request ===")
    logger.info(
        "\033[33mREQ  ›\033[0m lat=%.4f lon=%.4f  persona=\033[35m%s\033[0m  fmt=\033[36m%s\033[0m",
        request.latitude, request.longitude, request.persona.value, request.response_format,
    )

    # ── DB: get or create conversation ────────────────────────────────────────
    parsed_result = await _agent.run(request)
    result: ChatResponse = parsed_result.parsed_content

    conv = await get_or_create_conversation(db, request.conversation_id, title=result.title)

    # ── Save user message ──────────────────────────────────────────────────────
    await save_message(
        db,
        conversation_id=conv.id,
        role="user",
        content_text=_user_content_text(request),
    )

    # ── Save assistant message ─────────────────────────────────────────────────
    content_text = _render(result, request.response_format)
    await save_message(
        db,
        conversation_id=conv.id,
        role="assistant",
        content_text=content_text,
        llm_trace=parsed_result.llm_trace,
        model=parsed_result.llm_trace.model,
    )

    await db.commit()

    message = ChatMessage(
        message_id=str(uuid.uuid4()),
        role="assistant",
        content=Content(text=content_text),
        conversation_id=conv.id,
        model=parsed_result.llm_trace.model,
        llm_trace=parsed_result.llm_trace,
    )

    logger.info("=== STEP 4: Response Ready ===")
    logger.info(
        "\033[32mRESP ›\033[0m \033[1m%s\033[0m  \033[2mconv:%s\033[0m",
        result.title, conv.id[:8],
    )
    return message
