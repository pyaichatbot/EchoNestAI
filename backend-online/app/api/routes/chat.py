from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse
import asyncio
import json

from app.db.database import get_db
from app.db.schemas.chat import (
    ChatRequest, ChatResponse, ChatSession, 
    ChatMessage, ChatFeedback, SupportedLanguage,
    LanguageDetectionRequest, LanguageDetectionResponse
)
from app.api.deps.auth import get_current_active_user
from app.db.crud.chat import (
    create_chat_session, get_chat_session,
    create_chat_message, create_chat_feedback
)
from app.llm.chat_service import process_chat_query
from app.sse.event_manager import chat_stream_manager
from app.services.language_service import detect_language, get_supported_languages

router = APIRouter(tags=["chat"])

@router.post("/chat/query", response_model=ChatResponse)
async def chat_query(
    chat_request: ChatRequest,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Process a chat query and return a response.
    """
    # Create or get session
    session_id = chat_request.session_id
    if not session_id:
        session = await create_chat_session(
            db, 
            child_id=chat_request.child_id, 
            group_id=chat_request.group_id
        )
        session_id = session.id
    
    # Process query
    response = await process_chat_query(
        db,
        query=chat_request.input,
        session_id=session_id,
        child_id=chat_request.child_id,
        group_id=chat_request.group_id,
        document_scope=chat_request.document_scope,
        language=chat_request.language
    )
    
    # Save user message
    await create_chat_message(
        db,
        session_id=session_id,
        is_user=True,
        content=chat_request.input
    )
    
    # Save assistant message
    await create_chat_message(
        db,
        session_id=session_id,
        is_user=False,
        content=response.response,
        source_documents=json.dumps(response.source_documents),
        confidence=response.confidence
    )
    
    return response

@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    session_id: str,
    current_user: Any = Depends(get_current_active_user)
) -> Any:
    """
    Stream chat responses using Server-Sent Events.
    """
    async def event_generator():
        client_id = f"user_{current_user.id}_session_{session_id}"
        queue = chat_stream_manager.register_client(client_id)
        
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                message = await queue.get()
                if message == "CLOSE":
                    break
                
                yield {
                    "event": "message",
                    "data": message
                }
                
                await asyncio.sleep(0.1)
        finally:
            chat_stream_manager.remove_client(client_id)
    
    return EventSourceResponse(event_generator())

@router.post("/chat/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback_data: ChatFeedback,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Submit feedback for a chat message.
    """
    return await create_chat_feedback(db, obj_in=feedback_data)

@router.get("/chat/sessions/{child_id}", response_model=List[ChatSession])
async def get_chat_sessions(
    child_id: str,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get chat sessions for a child.
    """
    # Implementation would check if current user has access to this child
    return await get_chat_session(db, child_id=child_id)

@router.get("/chat/languages", response_model=List[SupportedLanguage])
async def list_supported_languages() -> Any:
    """
    Get list of supported languages for chat.
    """
    return get_supported_languages()

@router.post("/chat/detect-language", response_model=LanguageDetectionResponse)
async def detect_text_language(
    request: LanguageDetectionRequest
) -> Any:
    """
    Detect language of input text.
    """
    return await detect_language(request.text)
