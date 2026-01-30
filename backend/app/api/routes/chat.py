"""Chat routes."""
from fastapi import APIRouter, HTTPException

from app.services import ChatService
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()
chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a chat message."""
    try:
        result = chat_service.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            model_name=request.model,
            embedding_model=request.embedding_model
        )
        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"]
        )
    except ValueError as e:
        # No documents uploaded
        return ChatResponse(
            response="Please upload documents first, then I can answer your questions.",
            conversation_id=request.conversation_id or chat_service.create_conversation()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
