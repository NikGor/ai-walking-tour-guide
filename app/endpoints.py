from fastapi import APIRouter

from app.api_controller import handle_chat
from app.models import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await handle_chat(request)
