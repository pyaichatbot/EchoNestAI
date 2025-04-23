from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import setup_logging
from app.db.crud.chat import create_chat_message
from app.llm.chat_service import generate_embedding, retrieve_relevant_documents

logger = setup_logging("voice_utils")

async def save_streamed_response(
    db: AsyncSession,
    session_id: str,
    query: str,
    response: str,
    language: str,
    document_scope: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Save streamed response and retrieve source documents and confidence.
    
    Args:
        db: Database session
        session_id: Chat session ID
        query: User query
        response: Generated response
        language: Language code
        document_scope: Optional document scope
        
    Returns:
        Tuple of (source_documents, confidence)
    """
    # Retrieve relevant documents for the query
    # Generate embedding for query
    query_embedding = await generate_embedding(query, language)
    
    # Retrieve relevant documents
    context_passages, source_documents = await retrieve_relevant_documents(
        db, 
        query_embedding, 
        document_scope=document_scope,
        top_k=3
    )
    
    # Calculate confidence based on document relevance
    if context_passages:
        confidence = 0.85  # High confidence when documents are found
    else:
        confidence = 0.6   # Lower confidence when no documents are found
    
    # Save assistant message
    await create_chat_message(
        db,
        session_id=session_id,
        is_user=False,
        content=response,
        source_documents=source_documents,
        confidence=confidence
    )
    
    return source_documents, confidence
