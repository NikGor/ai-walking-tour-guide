from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.models.chat_models import ChatMessage, ChatRequest, Conversation
from app.api_controller import handle_chat
from app.db.repository import get_conversation
from app.db.session import get_db

router = APIRouter()


@router.post("/chat", response_model=ChatMessage)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    return await handle_chat(request, db)


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation_history(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    conv = await get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv
