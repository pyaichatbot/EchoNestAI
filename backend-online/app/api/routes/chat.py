from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
from fastapi.responses import StreamingResponse
from datetime import datetime

from app.db.database import get_db
from app.db.schemas.chat import (
    ChatRequest, ChatResponse, ChatSession, 
    ChatMessage, ChatFeedback, SupportedLanguage,
    LanguageDetectionRequest, LanguageDetectionResponse
)
from app.api.deps.auth import get_current_active_user
from app.db.crud.chat import (
    create_chat_session, get_chat_session,
    create_chat_message, create_chat_feedback,
    get_chat_messages
)
from app.llm.chat_service import process_chat_query, stream_chat_response
from app.sse.event_manager import chat_stream_manager
from app.services.language_service import detect_language, get_supported_languages

router = APIRouter(tags=["chat"])

@router.post("/chat/rag", response_model=ChatResponse)
async def chat_rag(
    chat_request: ChatRequest,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Alias for /chat/query to support frontend expectations. Accepts an optional 'context' field (used if provided).
    """
    session_id = chat_request.session_id
    if not session_id:
        session = await create_chat_session(
            db, 
            child_id=chat_request.child_id, 
            group_id=chat_request.group_id
        )
        session_id = session.id
    
    response = await process_chat_query(
        db,
        query=chat_request.input,
        session_id=session_id,
        child_id=chat_request.child_id,
        group_id=chat_request.group_id,
        document_scope=chat_request.document_scope,
        language=chat_request.language,
        context=chat_request.context  # Pass context to service
    )
    await create_chat_message(
        db,
        session_id=session_id,
        is_user=True,
        content=chat_request.input
    )
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

@router.get("/chat/rag/stream")
async def chat_rag_stream_get(
    request: Request,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    session_id: Optional[str] = None,
    language: str = "en",
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    SSE stream for chat responses (GET). Accepts query params for child_id, group_id, session_id, language.
    """
    async def event_generator():
        # For production, you may want to use a more robust client/session key
        client_id = f"user_{current_user.id}_session_{session_id or 'none'}"
        queue = chat_stream_manager.register_client(client_id)
        try:
            async for token in stream_chat_response(
                db=db,
                query="",  # No initial query for GET
                session_id=session_id or "",
                language=language,
                child_id=child_id,
                group_id=group_id
            ):
                yield {"event": "message", "data": token}
                await asyncio.sleep(0.1)
        finally:
            chat_stream_manager.remove_client(client_id)
    return EventSourceResponse(event_generator())

@router.post("/chat/rag/stream")
async def chat_rag_stream_post(
    request: Request,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    session_id: Optional[str] = None,
    language: str = "en",
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_current_active_user)
) -> StreamingResponse:
    """
    SSE stream for chat responses (POST). Accepts JSON body with input, context, document_scope.
    """
    body = await request.json()
    input_text = body.get("input", "")
    context = body.get("context")
    document_scope = body.get("document_scope")

    async def event_generator():
        async for token in stream_chat_response(
            db=db,
            query=input_text,
            session_id=session_id or "",
            language=language,
            child_id=child_id,
            group_id=group_id,
            document_scope=document_scope
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/chat/history")
async def get_chat_history_endpoint(
    session_id: Optional[str] = None,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    limit: int = 50,
    before: Optional[str] = None,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get chat history for a session, child, or group. Supports pagination via 'before' (ISO timestamp or message id).
    Returns messages in frontend format, with hasMore and nextCursor.
    """
    # Validate input
    if not session_id and not (child_id or group_id):
        raise HTTPException(status_code=400, detail="Must provide sessionId, childId, or groupId.")

    # Find session(s)
    sessions = []
    if session_id:
        sessions = await get_chat_session(db, id=session_id)
    elif child_id or group_id:
        sessions = await get_chat_session(db, child_id=child_id, group_id=group_id, limit=10)
    if not sessions:
        return {"messages": [], "hasMore": False}

    # For now, only support one session (latest if multiple)
    session = sessions[0]
    all_msgs = await get_chat_messages(db, session_id=session.id, limit=1000)
    # Sort by created_at ascending
    all_msgs.sort(key=lambda m: m.created_at)

    # Pagination: if 'before' is provided, only include messages before that
    if before:
        try:
            before_dt = datetime.fromisoformat(before)
            msgs = [m for m in all_msgs if m.created_at < before_dt]
        except Exception:
            # fallback: treat as message id
            idx = next((i for i, m in enumerate(all_msgs) if m.id == before), None)
            msgs = all_msgs[:idx] if idx is not None else all_msgs
    else:
        msgs = all_msgs

    # Paginate
    paged_msgs = msgs[-limit:] if limit else msgs
    has_more = len(msgs) > len(paged_msgs)
    next_cursor = paged_msgs[0].created_at.isoformat() if has_more and paged_msgs else None

    # Format for frontend
    def msg_to_frontend(m):
        return {
            "id": m.id,
            "senderId": session.child_id if m.is_user else "assistant",
            "text": m.content,
            "timestamp": int(m.created_at.timestamp() * 1000),
            "status": "sent",
            "sourceDocuments": json.loads(m.source_documents) if m.source_documents else [],
            "confidence": m.confidence
        }
    messages = [msg_to_frontend(m) for m in paged_msgs]
    return {"messages": messages, "hasMore": has_more, "nextCursor": next_cursor}
