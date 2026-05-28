import json
import logging
import os
from pathlib import Path

from aiogram import F, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    WebAppInfo,
)
from sqlalchemy import func, select

from app.agent.models.models import ChatRequest, Persona
from app.api_controller import handle_chat
from app.config import RESPONSE_FORMAT
from app.db.orm_models import ConversationORM, MessageORM
from app.db.repository import get_user_settings, upsert_user_settings
from app.db.session import AsyncSessionLocal

# ── Debug mode ────────────────────────────────────────────────────────────────

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("1", "true", "yes")

_PARSE_MODES: dict[str, ParseMode] = {
    "html": ParseMode.HTML,
    "markdown": ParseMode.MARKDOWN,
}


def _parse_mode(fmt: str) -> ParseMode | None:
    return _PARSE_MODES.get(fmt)


async def _safe_edit(message, text: str, **kwargs) -> None:
    """Edit a text message's content — or, if the message has no text (e.g. voice),
    remove its inline keyboard and send a new text message instead."""
    try:
        await message.edit_text(text, **kwargs)
    except TelegramBadRequest:
        try:
            await message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        await message.answer(text, **kwargs)


logger = logging.getLogger(__name__)

router = Router()

# Write-through cache: chat_id → {lat, lon, persona, lang}
# Populated lazily from DB on first access; written to DB on every change.
_sessions: dict[int, dict] = {}


async def _get_session(chat_id: int) -> dict:
    """Return session dict, loading from DB if not yet cached."""
    if chat_id not in _sessions:
        async with AsyncSessionLocal() as db:
            _sessions[chat_id] = await get_user_settings(db, chat_id)
    return _sessions[chat_id]


async def _persist_session(chat_id: int) -> None:
    """Write current in-memory session to DB."""
    s = _sessions.get(chat_id, {})
    async with AsyncSessionLocal() as db:
        await upsert_user_settings(
            db,
            chat_id=chat_id,
            persona=s.get("persona", "historian"),
            lang=s.get("lang", "auto"),
            fmt=s.get("fmt", RESPONSE_FORMAT),
            lat=s.get("lat"),
            lon=s.get("lon"),
            voice=s.get("voice", False),
        )
        await db.commit()


# ── Keyboards ─────────────────────────────────────────────────────────────────

_location_button = KeyboardButton(text="📍 Локация", request_location=True)

# Time Travel Mini App button — only shown when APP_BASE_URL is configured (HTTPS required by Telegram).
_APP_BASE_URL = os.getenv("APP_BASE_URL", "").rstrip("/")
if _APP_BASE_URL:
    _time_travel_button = KeyboardButton(
        text="🕰 Машина времени",
        web_app=WebAppInfo(url=f"{_APP_BASE_URL}/time-travel"),
    )
    _kb_row = [_location_button, _time_travel_button]
else:
    _kb_row = [_location_button]

_location_kb = ReplyKeyboardMarkup(keyboard=[_kb_row], resize_keyboard=True, is_persistent=True)

# ── Debug samples keyboard ────────────────────────────────────────────────────

_SAMPLES_PATH = Path(__file__).parent.parent.parent / "samples" / "requests.json"

_SAMPLE_LABELS = [
    "🏛 Ватикан",
    "🌉 Tower Bridge",
    "🎨 Монмартр",
    "🗼 Эйфелева башня",
    "🖼 Лувр",
    "⚔️ Колизей",
    "👑 Букингемский",
    "🏛 Парламент",
]


def _load_samples() -> list[dict]:
    if not _SAMPLES_PATH.exists():
        logger.warning("debug: samples/requests.json not found at %s", _SAMPLES_PATH)
        return []
    with _SAMPLES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


_SAMPLES: list[dict] = _load_samples() if DEBUG_MODE else []


def _debug_kb() -> InlineKeyboardMarkup:
    """2-column inline keyboard with sample locations (debug mode only)."""
    rows = []
    for i in range(0, len(_SAMPLES), 2):
        row = []
        for j in range(i, min(i + 2, len(_SAMPLES))):
            label = _SAMPLE_LABELS[j] if j < len(_SAMPLE_LABELS) else f"Sample {j}"
            row.append(InlineKeyboardButton(text=label, callback_data=f"sample:{j}"))
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _location_markup() -> InlineKeyboardMarkup | ReplyKeyboardMarkup:
    """Return debug sample buttons or the real location keyboard depending on DEBUG_MODE."""
    return _debug_kb() if DEBUG_MODE else _location_kb


_PERSONA_LABELS = {
    Persona.historian: "📜 Историк",
    Persona.architecture_expert: "🏛 Архитектор",
    Persona.roman_empire: "⚔️ Римская империя",
    Persona.storyteller: "🎭 Сказитель",
    Persona.medieval_resident: "🏚 Житель средних веков",
    Persona.military_expert: "🗡 Военный историк",
    Persona.deep_time: "🧊 Геолог / ледниковый период",
}

_LANG_LABELS = {
    "auto": "🌐 Авто (по сообщению)",
    "ru": "🇷🇺 Русский",
    "en": "🇬🇧 English",
    "de": "🇩🇪 Deutsch",
}


def _modes_kb(current: Persona) -> InlineKeyboardMarkup:
    buttons = []
    for persona, label in _PERSONA_LABELS.items():
        check = "✅ " if persona == current else ""
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{check}{label}",
                    callback_data=f"mode:{persona.value}",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _lang_kb(current: str) -> InlineKeyboardMarkup:
    buttons = []
    for lang, label in _LANG_LABELS.items():
        check = "✅ " if lang == current else ""
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{check}{label}",
                    callback_data=f"lang:{lang}",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=buttons)


_FMT_LABELS = {
    "html": "📄 HTML (рекомендуется)",
    "markdown": "✏️ Markdown",
    "plain": "📝 Простой текст",
}


def _fmt_kb(current: str) -> InlineKeyboardMarkup:
    buttons = []
    for fmt, label in _FMT_LABELS.items():
        check = "✅ " if fmt == current else ""
        buttons.append([InlineKeyboardButton(text=f"{check}{label}", callback_data=f"fmt:{fmt}")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


_PERSONA_SWITCH_LABELS = {
    "historian": "📜 Спросить историка",
    "architecture_expert": "🏛 Спросить архитектора",
    "roman_empire": "⚔️ Спросить о Риме",
    "storyteller": "🎭 Услышать историю",
    "medieval_resident": "🏚 Жизнь в Средние века",
    "military_expert": "🗡 Военный взгляд",
    "deep_time": "🧊 Взгляд геолога",
}


def _cb_data(prefix: str, text: str) -> str:
    """Build a callback_data string that fits in Telegram's 64-byte hard limit.

    Truncates `text` in UTF-8 byte space, then decodes safely to avoid splitting
    a multi-byte codepoint.  Persona slugs are ASCII so no truncation needed there.
    """
    budget = 64 - len(prefix.encode())
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return prefix + text
    # Drop bytes from the end until we have a valid UTF-8 sequence
    return prefix + encoded[:budget].decode("utf-8", errors="ignore")


def _suggestions_kb(
    suggestions: list[str], recommended_personas: list[str] | None = None
) -> InlineKeyboardMarkup:
    """Build a row-per-suggestion keyboard from LLM place suggestions.

    Up to 3 place buttons come first; up to 2 persona-switch buttons are appended last.
    """
    buttons = [[InlineKeyboardButton(text=s, callback_data=_cb_data("place:", s))] for s in suggestions[:3]]
    for slug in (recommended_personas or [])[:2]:
        if slug in _PERSONA_SWITCH_LABELS:
            label = _PERSONA_SWITCH_LABELS[slug]
            buttons.append([InlineKeyboardButton(text=label, callback_data=_cb_data("mode:", slug))])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ── Help text ─────────────────────────────────────────────────────────────────

_HELP = (
    "🗺 <b>Solaris Pliny</b> — твой личный историк\n\n"
    "<b>Как пользоваться:</b>\n"
    "1. Отправь 📍 геолокацию — получи историю места\n"
    "2. Напиши вопрос — уточни детали или спроси о чём-то рядом\n"
    "3. Отправь фото — бот определит объект и расскажет его историю\n"
    "4. Нажми 🕰 <b>Машина времени</b> — посмотри как выглядело место в любую эпоху\n\n"
    "<b>Команды:</b>\n"
    "/whereami — история текущего места\n"
    "/continue — продолжить рассказ\n"
    "/tour — пешеходный тур по городу 🗺\n"
    "/modes — стиль рассказа\n"
    "/lang — язык ответов\n"
    "/fmt — формат текста (HTML / Markdown)\n"
    "/voice — включить / выключить голосовые ответы 🔊\n"
    "/new — начать новый разговор\n"
    "/history — статистика сессии\n"
    "/settings — все настройки\n"
    "/help — эта справка"
)


# ── Command handlers ──────────────────────────────────────────────────────────


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(_HELP, reply_markup=_location_markup())


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(_HELP)


@router.message(Command("whereami"))
async def cmd_whereami(message: Message) -> None:
    chat_id = message.chat.id
    session = await _get_session(chat_id)
    if session.get("lat") is None:
        await message.answer("Сначала отправь геолокацию 📍", reply_markup=_location_markup())
        return
    await _dispatch(message, lat=session["lat"], lon=session["lon"], user_message=None)


@router.message(Command("continue"))
async def cmd_continue(message: Message) -> None:
    chat_id = message.chat.id
    session = await _get_session(chat_id)
    if session.get("lat") is None:
        await message.answer("Сначала отправь геолокацию 📍", reply_markup=_location_markup())
        return
    await _dispatch(
        message,
        lat=session["lat"],
        lon=session["lon"],
        user_message="Продолжи рассказ",
    )


@router.message(Command("tour"))
async def cmd_tour(message: Message) -> None:
    """Plan a walking city tour.

    - With GPS location: ask the LLM to determine the city from coordinates and plan a tour.
    - Without GPS (or with city name in command): use the provided city name or session context.

    Usage: /tour          — uses current GPS location to determine city
           /tour Rome     — explicit city name
           /tour Рим      — also works
    """
    chat_id = message.chat.id
    session = await _get_session(chat_id)

    # Extract optional city argument from the command, e.g. "/tour Rome" → "Rome"
    text = (message.text or "").strip()
    # Strip "/tour" prefix and any bot username (@username)
    parts = text.split(None, 1)
    city_arg = parts[1].strip() if len(parts) > 1 else None

    lat = session.get("lat")
    lon = session.get("lon")

    if city_arg:
        # City name provided explicitly in the command
        user_message = (
            f"Составь пешеходный тур на целый день по городу {city_arg}. Используй инструмент plan_city_tour."
        )
        await _dispatch(message, lat=lat, lon=lon, user_message=user_message)
    elif lat is not None and lon is not None:
        # Use GPS — LLM will resolve city from coordinates
        user_message = (
            "Составь пешеходный тур на целый день по этому городу. "
            "Определи город по координатам и используй инструмент plan_city_tour."
        )
        await _dispatch(message, lat=lat, lon=lon, user_message=user_message)
    else:
        await message.answer(
            "🗺 <b>Тур по городу</b>\n\n"
            "Отправь локацию 📍 чтобы я определил город автоматически, "
            "или укажи город явно:\n\n"
            "<code>/tour Рим</code>\n"
            "<code>/tour Saint Petersburg</code>",
            reply_markup=_location_markup(),
        )


@router.message(Command("modes"))
async def cmd_modes(message: Message) -> None:
    chat_id = message.chat.id
    session = await _get_session(chat_id)
    persona = Persona(session.get("persona", Persona.historian))
    await message.answer("Выбери стиль рассказа:", reply_markup=_modes_kb(persona))


@router.message(Command("lang"))
async def cmd_lang(message: Message) -> None:
    chat_id = message.chat.id
    session = await _get_session(chat_id)
    lang = session.get("lang", "auto")
    await message.answer("Выбери язык ответов:", reply_markup=_lang_kb(lang))


@router.message(Command("fmt"))
async def cmd_fmt(message: Message) -> None:
    chat_id = message.chat.id
    session = await _get_session(chat_id)
    fmt = session.get("fmt", RESPONSE_FORMAT)
    await message.answer("Выбери формат текста:", reply_markup=_fmt_kb(fmt))


@router.message(Command("voice"))
async def cmd_voice(message: Message) -> None:
    chat_id = message.chat.id
    session = await _get_session(chat_id)
    new_state = not session.get("voice", False)
    session["voice"] = new_state
    await _persist_session(chat_id)
    if new_state:
        await message.answer(
            "🔊 <b>Голосовой режим включён</b>\n\n"
            "Ответы будут отправляться как голосовые сообщения.\n\n"
            "/voice — выключить"
        )
    else:
        await message.answer("🔇 <b>Голосовой режим выключен</b>\n\nОтветы снова текстовые.")


@router.message(Command("new"))
async def cmd_new(message: Message) -> None:
    chat_id = message.chat.id
    session = await _get_session(chat_id)
    # Keep persona and lang, clear location
    session["lat"] = None
    session["lon"] = None
    await _persist_session(chat_id)
    logger.info("\033[34mTG   ›\033[0m new conversation  chat=\033[36m%d\033[0m", chat_id)
    await message.answer(
        "🔄 <b>Новый разговор</b>\n\nЛокация и история сброшены. Отправь 📍 чтобы начать.",
        reply_markup=_location_markup(),
    )


@router.message(Command("history"))
async def cmd_history(message: Message) -> None:
    chat_id = message.chat.id
    async with AsyncSessionLocal() as db:
        conv_result = await db.execute(select(ConversationORM).where(ConversationORM.id == str(chat_id)))
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
    session = await _get_session(chat_id)
    persona = Persona(session.get("persona", Persona.historian))
    lang = session.get("lang", "auto")
    lat = session.get("lat")
    lon = session.get("lon")

    fmt = session.get("fmt", RESPONSE_FORMAT)
    voice = session.get("voice", False)
    location_str = f"{lat:.4f}, {lon:.4f}" if lat else "не задана"
    lines = [
        "⚙️ <b>Настройки</b>",
        f"🎭 Стиль: <b>{_PERSONA_LABELS[persona]}</b>",
        f"🌐 Язык: <b>{_LANG_LABELS[lang]}</b>",
        f"📄 Формат: <b>{_FMT_LABELS.get(fmt, fmt)}</b>",
        f"{'🔊' if voice else '🔇'} Голос: <b>{'включён' if voice else 'выключен'}</b>",
        f"📍 Последняя локация: <code>{location_str}</code>",
        f"🆔 Chat ID: <code>{chat_id}</code>",
        "",
        "/modes — стиль  •  /lang — язык  •  /fmt — формат  •  /voice — голос",
        "/new — начать новый разговор",
    ]
    await message.answer("\n".join(lines))


# ── Callbacks ─────────────────────────────────────────────────────────────────


@router.callback_query(F.data.startswith("mode:"))
async def cb_mode(callback: CallbackQuery) -> None:
    chat_id = callback.message.chat.id
    persona = Persona(callback.data.split(":", 1)[1])
    session = await _get_session(chat_id)
    session["persona"] = persona.value
    await _persist_session(chat_id)
    label = _PERSONA_LABELS[persona]
    await _safe_edit(
        callback.message,
        f"✅ Стиль: <b>{label}</b>\n\n"
        "Отправь локацию или /whereami чтобы получить рассказ.\n"
        "/continue — продолжить с новой персоной.",
        reply_markup=None,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("lang:"))
async def cb_lang(callback: CallbackQuery) -> None:
    chat_id = callback.message.chat.id
    lang = callback.data.split(":", 1)[1]
    session = await _get_session(chat_id)
    session["lang"] = lang
    await _persist_session(chat_id)
    label = _LANG_LABELS[lang]
    await _safe_edit(
        callback.message,
        f"✅ Язык: <b>{label}</b>",
        reply_markup=None,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("place:"))
async def cb_place(callback: CallbackQuery) -> None:
    if not callback.message or not hasattr(callback.message, "answer"):
        await callback.answer("⚠️ Сообщение недоступно")
        return

    place_name = callback.data.split(":", 1)[1]
    logger.info(
        "\033[34mTG   ›\033[0m place tap  chat=\033[36m%d\033[0m  place=%r",
        callback.message.chat.id,
        place_name,
    )
    await callback.answer()

    # Remove buttons — ignore errors (message may already be edited or expired)
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    session = await _get_session(callback.message.chat.id)
    await _dispatch(
        callback.message,
        lat=session.get("lat"),
        lon=session.get("lon"),
        user_message=f"Расскажи подробнее про {place_name}",
    )


@router.callback_query(F.data.startswith("fmt:"))
async def cb_fmt(callback: CallbackQuery) -> None:
    chat_id = callback.message.chat.id
    fmt = callback.data.split(":", 1)[1]
    session = await _get_session(chat_id)
    session["fmt"] = fmt
    await _persist_session(chat_id)
    label = _FMT_LABELS[fmt]
    await _safe_edit(
        callback.message,
        f"✅ Формат: <b>{label}</b>",
        reply_markup=None,
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("sample:"))
async def cb_sample(callback: CallbackQuery) -> None:
    """Debug-mode: inject a sample location and trigger the standard dispatch pipeline."""
    idx = int(callback.data.split(":", 1)[1])
    if idx >= len(_SAMPLES):
        await callback.answer("⚠️ Сэмпл не найден")
        return

    sample = _SAMPLES[idx]
    lat: float = sample["latitude"]
    lon: float = sample["longitude"]
    chat_id = callback.message.chat.id
    label = _SAMPLE_LABELS[idx] if idx < len(_SAMPLE_LABELS) else sample.get("_location", f"Sample {idx}")

    logger.info(
        "\033[33mDBG  ›\033[0m sample tap  chat=\033[36m%d\033[0m  #%d  %s  lat=%.4f lon=%.4f",
        chat_id,
        idx,
        label,
        lat,
        lon,
    )

    session = await _get_session(chat_id)
    session["lat"] = lat
    session["lon"] = lon
    await _persist_session(chat_id)

    await callback.answer(f"📍 {label}")
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    await _dispatch(callback.message, lat=lat, lon=lon, user_message=None)


# ── Location & text ───────────────────────────────────────────────────────────


@router.message(F.location)
async def handle_location(message: Message) -> None:
    chat_id = message.chat.id
    lat = message.location.latitude
    lon = message.location.longitude

    session = await _get_session(chat_id)
    session["lat"] = lat
    session["lon"] = lon
    await _persist_session(chat_id)

    logger.info(
        "\033[34mTG   ›\033[0m location  chat=\033[36m%d\033[0m  lat=%.4f lon=%.4f",
        chat_id,
        lat,
        lon,
    )
    await _dispatch(message, lat=lat, lon=lon, user_message=None)


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    chat_id = message.chat.id
    session = await _get_session(chat_id)
    if session.get("lat") is None:
        await message.answer("Сначала отправь геолокацию 📍, потом фото.", reply_markup=_location_markup())
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
        chat_id,
        file.file_path,
    )
    await _dispatch(
        message, lat=session["lat"], lon=session["lon"], user_message=caption, photo_url=photo_url
    )


@router.message(F.text)
async def handle_text(message: Message) -> None:
    chat_id = message.chat.id
    text = message.text.strip()
    session = await _get_session(chat_id)
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
    session = await _get_session(chat_id)
    persona = Persona(session.get("persona", Persona.historian))
    lang = session.get("lang", "auto")
    fmt = session.get("fmt", RESPONSE_FORMAT)

    request = ChatRequest(
        latitude=lat,
        longitude=lon,
        persona=persona,
        message=user_message or "",
        photo_url=photo_url,
        conversation_id=str(chat_id),
        user_name=message.from_user.first_name if message.from_user else None,
        language=None if lang == "auto" else lang,
        response_format=fmt,
    )

    thinking = await message.answer("⏳")

    map_image: bytes | None = None
    conv_id = str(chat_id)
    try:
        async with AsyncSessionLocal() as db:
            response = await handle_chat(request, db)
        reply = response.content.text
        suggestions = response.suggestions
        map_image = response.map_image
        # Don't recommend the persona that's already active
        # Filter out the currently active persona from recommendations
        recommended_personas = [p for p in response.recommended_personas if p != persona.value]
    except Exception as e:
        logger.exception("TG dispatch error for chat %d", chat_id)
        reply = f"⚠️ Ошибка: {e}"
        suggestions = []
        recommended_personas = []
        # Save error to DB so the conversation history is complete
        try:
            async with AsyncSessionLocal() as db:
                from app.db.repository import get_or_create_conversation, save_message

                conv = await get_or_create_conversation(db, conv_id)
                await save_message(db, conversation_id=conv.id, role="assistant", content_text=reply)
                await db.commit()
        except Exception:
            logger.warning("TG dispatch: failed to save error message to DB", exc_info=True)

    has_buttons = bool(suggestions) or bool(recommended_personas)
    # No location yet → show the persistent location keyboard so user can share.
    # Location is known → show inline suggestions (reply keyboard persists independently).
    if not session.get("lat"):
        markup = _location_markup()
    elif has_buttons:
        markup = _suggestions_kb(suggestions, recommended_personas)
    else:
        markup = None

    await thinking.delete()

    # ── Map image (city tour) ─────────────────────────────────────────────────
    if map_image:
        map_file = BufferedInputFile(map_image, filename="tour_map.png")
        await message.answer_photo(photo=map_file)

    # ── Voice mode ────────────────────────────────────────────────────────────
    if session.get("voice") and reply and not reply.startswith("⚠️"):
        from app.telegram.tts import synthesise

        audio = await synthesise(reply)
        if audio:
            voice_file = BufferedInputFile(audio, filename="voice.mp3")
            await message.answer_voice(voice=voice_file, reply_markup=markup)
            return
        # TTS failed → fall back to text silently
        logger.warning("tts_fallback: synthesis failed for chat %d, sending text", chat_id)

    await message.answer(reply, parse_mode=_parse_mode(fmt), reply_markup=markup)
