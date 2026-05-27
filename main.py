import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.endpoints import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== STEP 1: App Init ===")
    logger.info("main_001: Solaris Pliny ready on \033[36m0.0.0.0:8000\033[0m")
    yield


app = FastAPI(
    title="Solaris Pliny",
    description="Location-aware AI historian bot",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
