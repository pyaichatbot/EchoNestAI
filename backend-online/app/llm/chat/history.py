from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import setup_logging

logger = setup_logging("chat_history")

async def get_chat_history(
    db: AsyncSession,
    session_id: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get chat history for a session.
    
    Args:
        db: Database session
        session_id: Chat session ID
        limit: Maximum number of messages to retrieve
        
    Returns:
        List of chat messages with role and content
    """
    from sqlalchemy import select
    from sqlalchemy.sql import desc
    from app.db.models.models import ChatMessage
    
    query = select(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(desc(ChatMessage.created_at)).limit(limit)
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    # Convert to list of dicts and reverse to get chronological order
    history = []
    for msg in reversed(messages):
        role = "user" if msg.is_user else "assistant"
        history.append({
            "role": role,
            "content": msg.content
        })
    
    return history
