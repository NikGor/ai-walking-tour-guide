import logging
import uuid

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.models.models import (
    ChatMessage,
    Content,
    Conversation,
    InputTokensDetails,
    LllmTrace,
    OutputTokensDetails,
)
from app.db.orm_models import ConversationORM, MessageORM, UserSettingsORM
from app.db.session import DATABASE_URL

logger = logging.getLogger(__name__)


def _dialect_insert(table):
    """Return a dialect-aware INSERT construct that supports ON CONFLICT clauses."""
    if "postgresql" in DATABASE_URL:
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert(table)


async def get_or_create_conversation(
    db: AsyncSession,
    conversation_id: str | None,
    title: str = "Walking Tour",
) -> ConversationORM:
    new_id = conversation_id or str(uuid.uuid4())
    is_resume = bool(conversation_id)

    # INSERT ... ON CONFLICT DO NOTHING — race-safe for concurrent Telegram retries
    stmt = (
        _dialect_insert(ConversationORM)
        .values(id=new_id, title=title)
        .on_conflict_do_nothing(index_elements=["id"])
    )
    await db.execute(stmt)
    await db.flush()

    result = await db.execute(select(ConversationORM).where(ConversationORM.id == new_id))
    conv = result.scalar_one()

    action = "resume " if is_resume else "new    "
    logger.info("\033[36mCONV ›\033[0m %s  %s  \033[2m%s\033[0m", action, conv.id[:8], conv.title[:50])
    return conv


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content_text: str,
    llm_trace: LllmTrace | None = None,
    model: str | None = None,
) -> MessageORM:
    msg = MessageORM(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=role,
        content_text=content_text,
        model=model,
        input_tokens=llm_trace.input_tokens if llm_trace else 0,
        output_tokens=llm_trace.output_tokens if llm_trace else 0,
        total_tokens=llm_trace.total_tokens if llm_trace else 0,
        total_cost=llm_trace.total_cost if llm_trace else 0.0,
    )
    db.add(msg)

    if llm_trace:
        await db.execute(
            update(ConversationORM)
            .where(ConversationORM.id == conversation_id)
            .values(
                total_input_tokens=ConversationORM.total_input_tokens + msg.input_tokens,
                total_output_tokens=ConversationORM.total_output_tokens + msg.output_tokens,
                total_tokens=ConversationORM.total_tokens + msg.total_tokens,
                total_cost=ConversationORM.total_cost + msg.total_cost,
            )
        )

    await db.flush()
    logger.info(
        "\033[35mMSG  ›\033[0m %-9s %s  \033[2mconv:%s\033[0m",
        role,
        msg.id[:8],
        conversation_id[:8],
    )
    return msg


async def get_recent_messages(
    db: AsyncSession,
    conversation_id: str,
    limit: int = 12,
) -> list[dict]:
    """Return the last `limit` messages as OpenAI-format dicts for LLM context injection."""
    result = await db.execute(
        select(MessageORM)
        .where(MessageORM.conversation_id == conversation_id)
        .order_by(desc(MessageORM.created_at))
        .limit(limit)
    )
    rows = result.scalars().all()
    # reverse to chronological order before returning
    return [{"role": m.role, "content": m.content_text} for m in reversed(rows)]


async def get_user_settings(db: AsyncSession, chat_id: int) -> dict:
    """Load persisted Telegram user settings. Returns defaults if not found."""
    result = await db.execute(select(UserSettingsORM).where(UserSettingsORM.chat_id == chat_id))
    row = result.scalar_one_or_none()
    if not row:
        return {"persona": "historian", "lang": "auto", "fmt": "html", "lat": None, "lon": None}
    return {"persona": row.persona, "lang": row.lang, "fmt": row.fmt, "lat": row.lat, "lon": row.lon}


async def upsert_user_settings(
    db: AsyncSession,
    chat_id: int,
    persona: str,
    lang: str,
    fmt: str,
    lat: float | None,
    lon: float | None,
) -> None:
    """Insert or update user settings for a Telegram chat — race-safe upsert."""
    values = dict(chat_id=chat_id, persona=persona, lang=lang, fmt=fmt, lat=lat, lon=lon)
    stmt = (
        _dialect_insert(UserSettingsORM)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["chat_id"],
            set_=dict(persona=persona, lang=lang, fmt=fmt, lat=lat, lon=lon),
        )
    )
    await db.execute(stmt)
    await db.flush()


async def get_conversation(
    db: AsyncSession,
    conversation_id: str,
) -> Conversation | None:
    result = await db.execute(
        select(ConversationORM)
        .where(ConversationORM.id == conversation_id)
        .options(selectinload(ConversationORM.messages))
    )
    conv_orm = result.scalar_one_or_none()
    if not conv_orm:
        return None

    messages = [
        ChatMessage(
            message_id=m.id,
            role=m.role,
            content=Content(text=m.content_text),
            conversation_id=conversation_id,
            model=m.model,
            created_at=m.created_at,
            llm_trace=LllmTrace(
                model=m.model or "unknown",
                input_tokens=m.input_tokens,
                output_tokens=m.output_tokens,
                total_tokens=m.total_tokens,
                total_cost=m.total_cost,
                input_tokens_details=InputTokensDetails(),
                output_tokens_details=OutputTokensDetails(),
            )
            if m.model
            else None,
        )
        for m in conv_orm.messages
    ]

    return Conversation(
        conversation_id=conv_orm.id,
        title=conv_orm.title,
        messages=messages,
        created_at=conv_orm.created_at,
        updated_at=conv_orm.updated_at,
        total_input_tokens=conv_orm.total_input_tokens,
        total_output_tokens=conv_orm.total_output_tokens,
        total_tokens=conv_orm.total_tokens,
        total_cost=conv_orm.total_cost,
    )
