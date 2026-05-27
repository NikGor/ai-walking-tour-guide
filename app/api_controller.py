import logging
import uuid

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.agent_factory import AgentFactory
from app.agent.models.models import ChatMessage, ChatRequest, ChatResponse, Content
from app.db.repository import get_or_create_conversation, get_recent_messages, save_message

load_dotenv()

logger = logging.getLogger(__name__)

_agent = AgentFactory()


def _conv_title(request: ChatRequest) -> str:
    if request.latitude is not None and request.longitude is not None:
        return f"{request.latitude:.4f}, {request.longitude:.4f} | {request.persona.value}"
    return f"chat | {request.persona.value}"


def _render(result: ChatResponse, fmt: str) -> str:
    if fmt == "ssml":
        return f"<speak>{result.text}</speak>"
    # html and markdown: LLM formats the text directly, return as-is
    return result.text


def _user_content_text(request: ChatRequest) -> str:
    if request.latitude is not None and request.longitude is not None:
        lines = [f"📍 {request.latitude}, {request.longitude} | {request.persona.value}"]
    else:
        lines = [f"💬 {request.persona.value}"]
    if request.message:
        lines.append(request.message)
    return "\n".join(lines)


async def handle_chat(request: ChatRequest, db: AsyncSession) -> ChatMessage:
    logger.info("=== STEP 2: Chat Request ===")
    if request.latitude is not None and request.longitude is not None:
        logger.info(
            "\033[33mREQ  ›\033[0m lat=%.4f lon=%.4f  persona=\033[35m%s\033[0m  fmt=\033[36m%s\033[0m",
            request.latitude,
            request.longitude,
            request.persona.value,
            request.response_format,
        )
    else:
        logger.info(
            "\033[33mREQ  ›\033[0m no-location  persona=\033[35m%s\033[0m  fmt=\033[36m%s\033[0m",
            request.persona.value,
            request.response_format,
        )

    # ── DB: get or create conversation, load history ──────────────────────────
    conv = await get_or_create_conversation(db, request.conversation_id, title=_conv_title(request))
    history = await get_recent_messages(db, conv.id, limit=12)

    parsed_result = await _agent.run(request, history=history)
    result: ChatResponse = parsed_result.parsed_content

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
        suggestions=result.suggestions or None,
    )

    await db.commit()

    message = ChatMessage(
        message_id=str(uuid.uuid4()),
        role="assistant",
        content=Content(text=content_text),
        suggestions=result.suggestions,
        recommended_personas=result.recommended_personas,
        conversation_id=conv.id,
        model=parsed_result.llm_trace.model,
        llm_trace=parsed_result.llm_trace,
    )

    logger.info("=== STEP 4: Response Ready ===")
    logger.info(
        "\033[32mRESP ›\033[0m words=\033[33m%d\033[0m  \033[2mconv:%s\033[0m",
        len(content_text.split()),
        conv.id[:8],
    )
    return message
