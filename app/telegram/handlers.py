import logging
import os

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    CallbackQuery,
    ReplyKeyboardMarkup,
)
from sqlalchemy import select, func

from app.agent.models.models import ChatRequest, Persona
from app.api_controller import handle_chat
from app.db.orm_models import ConversationORM, MessageORM
from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)

router = Router()

# In-memory session: chat_id → {lat, lon, persona, lang}
_sessions: dict[int, dict] = {}

# ── Keyboards ─────────────────────────────────────────────────────────────────

_location_button = KeyboardButton(text="📍 Отправить локацию", request_location=True)
_location_kb = ReplyKeyboardMarkup(
    keyboard=[[_location_button]], resize_keyboard=True, one_time_keyboard=True
)

_PERSONA_LABELS = {
    Persona.historian:           "📜 Историк",
    Persona.dark_tourism:        "💀 Тёмный туризм",
    Persona.architecture_expert: "🏛 Архитектор",
    Persona.roman_empire:        "⚔️ Римская империя",
    Persona.ww2_context:         "🪖 Вторая мировая",
    Persona.cyberpunk:           "🤖 Киберпанк",
    Persona.storyteller:         "🎭 Сказитель",
    Persona.local_grandpa:       "👴 Местный дед",
}

_LANG_LABELS = {
    "auto": "🌐 Авто (по сообщению)",
    "ru":   "🇷🇺 Русский",
    "en":   "🇬🇧 English",
    "de":   "🇩🇪 Deutsch",
}


def _modes_kb(current: Persona) -> InlineKeyboardMarkup:
    buttons = []
    for persona, label in _PERSONA_LABELS.items():
        check = "✅ " if persona == current else ""
        buttons.append([InlineKeyboardButton(
            text=f"{check}{label}",
            callback_data=f"mode:{persona.value}",
        )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _lang_kb(current: str) -> InlineKeyboardMarkup:
    buttons = []
    for lang, label in _LANG_LABELS.items():
        check = "✅ " if lang == current else ""
        buttons.append([InlineKeyboardButton(
            text=f"{check}{label}",
            callback_data=f"lang:{lang}",
        )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ── Help text ─────────────────────────────────────────────────────────────────

_HELP = (
    "🗺 <b>Solaris Pliny</b> — твой личный историк\n\n"
    "<b>Как пользоваться:</b>\n"
    "1. Отправь 📍 геолокацию — получи историю места\n"
    "2. Напиши вопрос — уточни детали или спроси о чём-то рядом\n"
    "3. Отправь фото — бот определит объект и расскажет его историю\n\n"
    "<b>Команды:</b>\n"
    "/whereami — история текущего места\n"
    "/modes — стиль рассказа\n"
    "/lang — язык ответов\n"
    "/new — начать новый разговор\n"
    "/history — статистика сессии\n"
    "/settings — все настройки\n"
    "/help — эта справка"
)


# ── Command handlers ──────────────────────────────────────────────────────────

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(_HELP, reply_markup=_location_kb)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(_HELP)


@router.message(Command("whereami"))
async def cmd_whereami(message: Message) -> None:
    chat_id = message.chat.id
    session = _sessions.get(chat_id, {})
    if "lat" not in session:
        await message.answer("Сначала отправь геолокацию 📍", reply_markup=_location_kb)
        return
    await _dispatch(message, lat=session["lat"], lon=session["lon"], user_message=None)


@router.message(Command("modes"))
async def cmd_modes(message: Message) -> None:
    chat_id = message.chat.id
    persona = _sessions.get(chat_id, {}).get("persona", Persona.historian)
    await message.answer("Выбери стиль рассказа:", reply_markup=_modes_kb(persona))


@router.message(Command("lang"))
async def cmd_lang(message: Message) -> None:
    chat_id = message.chat.id
    lang = _sessions.get(chat_id, {}).get("lang", "auto")
    await message.answer("Выбери язык ответов:", reply_markup=_lang_kb(lang))


@router.message(Command("new"))
async def cmd_new(message: Message) -> None:
    chat_id = message.chat.id
    session = _sessions.get(chat_id, {})
    # Keep persona and lang, clear location
    _sessions[chat_id] = {
        "persona": session.get("persona", Persona.historian),
        "lang": session.get("lang", "auto"),
    }
    logger.info("\033[34mTG   ›\033[0m new conversation  chat=\033[36m%d\033[0m", chat_id)
    await message.answer(
        "🔄 <b>Новый разговор</b>\n\nЛокация и история сброшены. Отправь 📍 чтобы начать.",
        reply_markup=_location_kb,
    )


@router.message(Command("history"))
async def cmd_history(message: Message) -> None:
    chat_id = message.chat.id
    async with AsyncSessionLocal() as db:
        conv_result = await db.execute(
            select(ConversationORM).where(ConversationORM.id == str(chat_id))
        )
        conv = conv_result.scalar_one_or_none()

        user_msg_count = 0
        if conv:
            count_result = await db.execute(
                select(func.count(MessageORM.id))
                .where(MessageORM.conversation_id == str(chat_id))
                .where(MessageORM.role == "user")
            )
            user_msg_count = count_result.scalar() or 0

    if not conv:
        await message.answer("История пуста. Отправь геолокацию 📍 чтобы начать.")
        return

    lines = [
        "🗂 <b>Статистика сессии</b>",
        f"💬 Вопросов задано: <b>{user_msg_count}</b>",
        f"🪙 Токенов использовано: {conv.total_tokens:,}",
        f"💵 Потрачено: <b>${conv.total_cost:.4f}</b>",
        f"📅 Начат: {conv.created_at.strftime('%d.%m.%Y %H:%M')}",
        f"🔄 Последнее: {conv.updated_at.strftime('%d.%m.%Y %H:%M')}",
        "",
        "/new — начать новый разговор",
    ]
    await message.answer("\n".join(lines))


@router.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
    chat_id = message.chat.id
    session = _sessions.get(chat_id, {})
    persona = session.get("persona", Persona.historian)
    lang = session.get("lang", "auto")
    lat = session.get("lat")
    lon = session.get("lon")

    location_str = f"{lat:.4f}, {lon:.4f}" if lat else "не задана"
    lines = [
        "⚙️ <b>Настройки</b>",
        f"🎭 Стиль: <b>{_PERSONA_LABELS[persona]}</b>",
        f"🌐 Язык: <b>{_LANG_LABELS[lang]}</b>",
        f"📍 Последняя локация: <code>{location_str}</code>",
        f"🆔 Chat ID: <code>{chat_id}</code>",
        "",
        "/modes — сменить стиль",
        "/lang — сменить язык",
        "/new — начать новый разговор",
    ]
    await message.answer("\n".join(lines))


# ── Callbacks ─────────────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("mode:"))
async def cb_mode(callback: CallbackQuery) -> None:
    chat_id = callback.message.chat.id
    persona = Persona(callback.data.split(":", 1)[1])
    _sessions.setdefault(chat_id, {})["persona"] = persona
    label = _PERSONA_LABELS[persona]
    await callback.message.edit_text(
        f"✅ Стиль: <b>{label}</b>\n\nОтправь локацию или /whereami чтобы получить рассказ.",
        reply_markup=None,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("lang:"))
async def cb_lang(callback: CallbackQuery) -> None:
    chat_id = callback.message.chat.id
    lang = callback.data.split(":", 1)[1]
    _sessions.setdefault(chat_id, {})["lang"] = lang
    label = _LANG_LABELS[lang]
    await callback.message.edit_text(
        f"✅ Язык: <b>{label}</b>",
        reply_markup=None,
    )
    await callback.answer()


# ── Location & text ───────────────────────────────────────────────────────────

@router.message(F.location)
async def handle_location(message: Message) -> None:
    chat_id = message.chat.id
    lat = message.location.latitude
    lon = message.location.longitude

    session = _sessions.setdefault(chat_id, {"persona": Persona.historian, "lang": "auto"})
    session["lat"] = lat
    session["lon"] = lon

    logger.info(
        "\033[34mTG   ›\033[0m location  chat=\033[36m%d\033[0m  lat=%.4f lon=%.4f",
        chat_id, lat, lon,
    )
    await _dispatch(message, lat=lat, lon=lon, user_message=None)


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    chat_id = message.chat.id
    session = _sessions.get(chat_id, {})
    if "lat" not in session:
        await message.answer("Сначала отправь геолокацию 📍, потом фото.", reply_markup=_location_kb)
        return

    photo = message.photo[-1]
    from app.telegram.bot import get_bot
    bot = get_bot()
    file = await bot.get_file(photo.file_id)
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    photo_url = f"https://api.telegram.org/file/bot{token}/{file.file_path}"
    caption = message.caption.strip() if message.caption else None

    logger.info(
        "\033[34mTG   ›\033[0m photo  chat=\033[36m%d\033[0m  file=%s",
        chat_id, file.file_path,
    )
    await _dispatch(message, lat=session["lat"], lon=session["lon"],
                    user_message=caption, photo_url=photo_url)


@router.message(F.text)
async def handle_text(message: Message) -> None:
    chat_id = message.chat.id
    text = message.text.strip()
    session = _sessions.get(chat_id, {})
    await _dispatch(
        message,
        lat=session.get("lat"),
        lon=session.get("lon"),
        user_message=text,
    )


# ── Core dispatch ─────────────────────────────────────────────────────────────

async def _dispatch(
    message: Message,
    lat: float | None,
    lon: float | None,
    user_message: str | None,
    photo_url: str | None = None,
) -> None:
    chat_id = message.chat.id
    session = _sessions.get(chat_id, {})
    persona = session.get("persona", Persona.historian)
    lang = session.get("lang", "auto")

    request = ChatRequest(
        latitude=lat,
        longitude=lon,
        persona=persona,
        message=user_message or "",
        photo_url=photo_url,
        conversation_id=str(chat_id),
        user_name=message.from_user.first_name if message.from_user else None,
        language=None if lang == "auto" else lang,
    )

    thinking = await message.answer("⏳")

    try:
        async with AsyncSessionLocal() as db:
            response = await handle_chat(request, db)
        reply = response.content.text
    except Exception as e:
        logger.exception("TG dispatch error for chat %d", chat_id)
        reply = f"⚠️ Ошибка: {e}"

    await thinking.delete()
    await message.answer(reply)
