import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from app.telegram.handlers import router

logger = logging.getLogger(__name__)

_bot: Bot | None = None
_dp: Dispatcher | None = None


def get_bot() -> Bot:
    global _bot
    if _bot is None:
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        _bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    return _bot


def get_dispatcher() -> Dispatcher:
    global _dp
    if _dp is None:
        _dp = Dispatcher()
        _dp.include_router(router)
    return _dp


async def start_polling() -> None:
    from aiogram.types import BotCommand

    bot = get_bot()
    dp = get_dispatcher()

    await bot.set_my_commands(
        [
            BotCommand(command="whereami", description="История текущего места"),
            BotCommand(command="continue", description="Продолжить рассказ"),
            BotCommand(command="modes", description="Стиль рассказа (персона)"),
            BotCommand(command="lang", description="Язык ответов"),
            BotCommand(command="fmt", description="Формат текста (HTML / Markdown)"),
            BotCommand(command="new", description="Начать новый разговор"),
            BotCommand(command="history", description="Статистика сессии"),
            BotCommand(command="settings", description="Все настройки"),
            BotCommand(command="help", description="Помощь"),
        ]
    )

    bot_info = await bot.get_me()
    logger.info(
        "\033[34mTG   ›\033[0m polling started  @\033[1m%s\033[0m",
        bot_info.username,
    )
    await dp.start_polling(bot, handle_signals=False)
