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
    """Render a structured ChatResponse into the requested text format."""
    if fmt == "plain":
        return _render_plain(result)
    if fmt == "html":
        return _render_html(result)
    if fmt == "ssml":
        return _render_ssml(result)
    return _render_markdown(result)


def _render_markdown(result: ChatResponse) -> str:
    lines = [f"# {result.title}", "", result.summary, "", result.history]
    if result.facts:
        lines += ["", "## Факты"]
        lines += [f"- {f}" for f in result.facts]
    if result.timeline:
        lines += ["", "## Хронология"]
        lines += [f"- **{e.year}** — {e.event}" for e in result.timeline]
    if result.related_people:
        lines += ["", f"**Люди:** {', '.join(result.related_people)}"]
    if result.sources:
        lines += ["", "## Источники"]
        lines += [f"- {s}" for s in result.sources]
    lines += ["", f"*Достоверность: {result.confidence:.0%}*"]
    return "\n".join(lines)


def _render_plain(result: ChatResponse) -> str:
    lines = [result.title, "", result.summary, "", result.history]
    if result.facts:
        lines += ["", "Факты:"]
        lines += [f"  • {f}" for f in result.facts]
    if result.timeline:
        lines += ["", "Хронология:"]
        lines += [f"  {e.year}: {e.event}" for e in result.timeline]
    if result.related_people:
        lines += ["", f"Связанные люди: {', '.join(result.related_people)}"]
    lines += ["", f"Достоверность: {result.confidence:.0%}"]
    return "\n".join(lines)


def _render_html(result: ChatResponse) -> str:
    parts = [f"<h1>{result.title}</h1>", f"<p>{result.summary}</p>", f"<p>{result.history}</p>"]
    if result.facts:
        items = "".join(f"<li>{f}</li>" for f in result.facts)
        parts.append(f"<h2>Факты</h2><ul>{items}</ul>")
    if result.timeline:
        items = "".join(f"<li><strong>{e.year}</strong> — {e.event}</li>" for e in result.timeline)
        parts.append(f"<h2>Хронология</h2><ul>{items}</ul>")
    if result.related_people:
        parts.append(f"<p><strong>Люди:</strong> {', '.join(result.related_people)}</p>")
    parts.append(f"<p><em>Достоверность: {result.confidence:.0%}</em></p>")
    return "\n".join(parts)


def _render_ssml(result: ChatResponse) -> str:
    text = f"{result.title}. {result.summary} {result.history}"
    if result.related_people:
        text += f" Связанные люди: {', '.join(result.related_people)}."
    return f"<speak><p>{text}</p></speak>"


def _user_content_text(request: ChatRequest) -> str:
    lines = [f"📍 {request.latitude}, {request.longitude} | {request.persona.value}"]
    if request.message:
        lines.append(request.message)
    return "\n".join(lines)


async def handle_chat(request: ChatRequest, db: AsyncSession) -> ChatMessage:
    logger.info("=== STEP 2: Chat Request ===")
    logger.info(
        "controller_001: lat=\033[33m%.4f\033[0m lon=\033[33m%.4f\033[0m "
        "persona=\033[35m%s\033[0m fmt=\033[36m%s\033[0m",
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
        "controller_002: title=\033[32m%r\033[0m confidence=\033[33m%.2f\033[0m conv=\033[36m%s\033[0m",
        result.title, result.confidence, conv.id,
    )
    return message
