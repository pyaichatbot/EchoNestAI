from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete
from typing import Any, Dict, Optional, List, Union
import json

from app.db.models.models import ChatSession, ChatMessage, ChatFeedback
from app.db.schemas.chat import ChatMessageCreate, ChatFeedbackCreate

async def create_chat_session(
    db: AsyncSession,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None
) -> ChatSession:
    """
    Create a new chat session.
    """
    db_obj = ChatSession(
        child_id=child_id,
        group_id=group_id
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_chat_session(
    db: AsyncSession,
    id: Optional[str] = None,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    limit: int = 10
) -> List[ChatSession]:
    """
    Get chat sessions with optional filters.
    """
    if id:
        result = await db.execute(select(ChatSession).filter(ChatSession.id == id))
        session = result.scalars().first()
        if session:
            return [session]
        return []
    
    query = select(ChatSession).order_by(ChatSession.created_at.desc()).limit(limit)
    
    if child_id:
        query = query.filter(ChatSession.child_id == child_id)
    
    if group_id:
        query = query.filter(ChatSession.group_id == group_id)
    
    result = await db.execute(query)
    return result.scalars().all()

async def get_chat_messages(
    db: AsyncSession,
    session_id: str,
    limit: int = 50
) -> List[ChatMessage]:
    """
    Get chat messages for a session.
    """
    query = select(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def create_chat_message(
    db: AsyncSession,
    session_id: str,
    is_user: bool,
    content: str,
    source_documents: Optional[str] = None,
    confidence: Optional[float] = None
) -> ChatMessage:
    """
    Create a new chat message.
    """
    db_obj = ChatMessage(
        session_id=session_id,
        is_user=is_user,
        content=content,
        source_documents=source_documents,
        confidence=confidence
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_chat_feedback(
    db: AsyncSession,
    message_id: str
) -> Optional[ChatFeedback]:
    """
    Get feedback for a chat message.
    """
    result = await db.execute(
        select(ChatFeedback).filter(ChatFeedback.message_id == message_id)
    )
    return result.scalars().first()

async def create_chat_feedback(
    db: AsyncSession,
    obj_in: ChatFeedbackCreate
) -> ChatFeedback:
    """
    Create feedback for a chat message.
    """
    # Check if feedback already exists
    existing = await get_chat_feedback(db, message_id=obj_in.message_id)
    if existing:
        # Update existing feedback
        stmt = (
            update(ChatFeedback)
            .where(ChatFeedback.message_id == obj_in.message_id)
            .values(
                rating=obj_in.rating,
                comment=obj_in.comment,
                flagged=obj_in.flagged,
                flag_reason=obj_in.flag_reason
            )
        )
        await db.execute(stmt)
        await db.commit()
        await db.refresh(existing)
        return existing
    
    # Create new feedback
    db_obj = ChatFeedback(
        message_id=obj_in.message_id,
        rating=obj_in.rating,
        comment=obj_in.comment,
        flagged=obj_in.flagged,
        flag_reason=obj_in.flag_reason
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def delete_chat_session(
    db: AsyncSession,
    session_id: str
) -> bool:
    """
    Delete a chat session and all its messages.
    """
    # Get all messages
    messages = await get_chat_messages(db, session_id=session_id)
    
    # Delete feedback for all messages
    for message in messages:
        await db.execute(
            delete(ChatFeedback).where(ChatFeedback.message_id == message.id)
        )
    
    # Delete messages
    await db.execute(
        delete(ChatMessage).where(ChatMessage.session_id == session_id)
    )
    
    # Delete session
    await db.execute(
        delete(ChatSession).where(ChatSession.id == session_id)
    )
    
    await db.commit()
    return True
