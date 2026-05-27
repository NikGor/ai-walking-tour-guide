import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.endpoints import router
from app.time_travel.router import router as time_travel_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from alembic.config import Config

    from alembic import command

    logger.info("=== STEP 1: App Init ===")
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.attributes["skip_logging"] = True

    # Pass the runtime DB URL to Alembic, stripping async drivers so the
    # sync Alembic engine works (asyncpg/aiosqlite are async-only).
    from app.db.session import DATABASE_URL

    sync_url = DATABASE_URL.replace("+asyncpg", "").replace("+aiosqlite", "")
    alembic_cfg.set_main_option("sqlalchemy.url", sync_url)

    command.upgrade(alembic_cfg, "head")
    logger.info("main_001: DB migrations applied")
    logger.info("main_002: Solaris Pliny ready on \033[36m0.0.0.0:8000\033[0m")

    tg_task = None
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        from app.telegram.bot import start_polling

        tg_task = asyncio.create_task(start_polling())
    else:
        logger.info("\033[34mTG   ›\033[0m TELEGRAM_BOT_TOKEN not set, skipping")

    yield

    if tg_task:
        tg_task.cancel()
        try:
            await tg_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Solaris Pliny",
    description="Location-aware AI historian bot",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(time_travel_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
