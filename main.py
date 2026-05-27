import logging

import uvicorn
from fastapi import FastAPI

from app.endpoints import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("solaris")

app = FastAPI(
    title="Solaris Pliny",
    description="Location-aware AI historian bot",
    version="0.1.0",
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
