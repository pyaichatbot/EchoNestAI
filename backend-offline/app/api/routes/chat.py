from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.schemas.chat import (
    ChatRequest, ChatResponse, ChatSession, 
    ChatMessage, ChatFeedback
)
from app.api.deps.auth import get_current_active_user, get_device_auth
from app.db.crud.chat import (
    create_chat_session, get_chat_session, get_chat_messages,
    create_chat_message, create_chat_feedback, delete_chat_session
)
from app.llm.chat_service import process_chat_query

router = APIRouter(tags=["chat"])

@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Process a chat query and return a response.
    """
    # Create or get session
    if not request.session_id:
        session = await create_chat_session(
            db, 
            child_id=request.child_id, 
            group_id=request.group_id
        )
        session_id = session.id
    else:
        session_id = request.session_id
    
    # Process query
    response = await process_chat_query(
        db,
        query=request.input,
        session_id=session_id,
        child_id=request.child_id,
        group_id=request.group_id,
        document_scope=request.document_scope,
        language=request.language
    )
    
    # Save user message
    await create_chat_message(
        db,
        session_id=session_id,
        is_user=True,
        content=request.input
    )
    
    # Save assistant message
    await create_chat_message(
        db,
        session_id=session_id,
        is_user=False,
        content=response.response,
        source_documents=str(response.source_documents),
        confidence=response.confidence
    )
    
    return response

@router.get("/sessions", response_model=List[ChatSession])
async def get_sessions(
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    limit: int = 10,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get chat sessions for a child or group.
    """
    sessions = await get_chat_session(
        db,
        child_id=child_id,
        group_id=group_id,
        limit=limit
    )
    return sessions

@router.get("/sessions/{session_id}", response_model=List[ChatMessage])
async def get_session_messages(
    session_id: str = Path(...),
    limit: int = 50,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get messages for a chat session.
    """
    messages = await get_chat_messages(
        db,
        session_id=session_id,
        limit=limit
    )
    return messages

@router.post("/feedback", response_model=ChatFeedback)
async def provide_feedback(
    feedback: ChatFeedback,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Provide feedback for a chat message.
    """
    feedback = await create_chat_feedback(
        db,
        obj_in=feedback
    )
    return feedback

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str = Path(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Delete a chat session and all its messages.
    """
    await delete_chat_session(db, session_id=session_id)
    return {"message": "Session deleted successfully"}
