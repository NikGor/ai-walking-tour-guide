"""
Quick dev sandbox — run directly to test the full request pipeline.
Usage: poetry run python sandbox.py
"""
import asyncio
import json
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.api_controller import handle_chat
from app.models import ChatRequest, Persona

# Frankfurt Römerberg — a good historically rich test location
TEST_REQUEST = ChatRequest(
    latitude=50.1104,
    longitude=8.6821,
    persona=Persona.historian,
    message="Кто построил этот собор и сколько лет шло строительство?",
)


async def main():
    print(f"Sending request: lat={TEST_REQUEST.latitude}, lon={TEST_REQUEST.longitude}, persona={TEST_REQUEST.persona.value}\n")
    response = await handle_chat(TEST_REQUEST)
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
