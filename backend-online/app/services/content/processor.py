from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
import aiofiles
from fastapi import UploadFile, BackgroundTasks
import asyncio
import uuid
import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import setup_logging
from app.sse.event_manager import content_upload_manager
from app.services.document.processor import process_document
from app.services.audio.processor import process_audio
from app.services.video.processor import process_video

logger = setup_logging("content_processor")

async def process_content_async(
    content_id: str,
    file_path: str,
    content_type: str,
    language: str,
    event_manager: Any
) -> None:
    """
    Process uploaded content asynchronously.
    This includes:
    - For documents: text extraction, chunking, embedding generation
    - For audio: transcription
    - For video: audio extraction and transcription
    
    Updates are sent via SSE to the client.
    """
    try:
        # Send initial status
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 0,
                "message": "Starting content processing"
            },
            client_filter=content_id
        )
        
        # Process based on content type
        if content_type == "document":
            await process_document(content_id, file_path, language, event_manager)
        elif content_type == "audio":
            await process_audio(content_id, file_path, language, event_manager)
        elif content_type == "video":
            await process_video(content_id, file_path, language, event_manager)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Send completion status
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "completed",
                "progress": 100,
                "message": "Content processing completed"
            },
            client_filter=content_id
        )
        
    except Exception as e:
        logger.error(f"Error processing content {content_id}: {str(e)}")
        # Send error status
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "error",
                "progress": 0,
                "message": f"Error processing content: {str(e)}"
            },
            client_filter=content_id
        )

async def search_content(
    db: AsyncSession, 
    query: str, 
    language: str,
    document_scope: Optional[List[str]] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search for relevant content based on query.
    
    Args:
        db: Database session
        query: Search query
        language: Language code
        document_scope: Optional list of document IDs to search within
        top_k: Number of results to return
        
    Returns:
        List of search results with content chunks and metadata
    """
    try:
        from app.services.content.embeddings import generate_embeddings, search_embeddings
        
        # Generate embedding for query
        query_embedding = await generate_embeddings([query], language)
        
        # Search for similar content
        results = await search_embeddings(db, query_embedding[0], document_scope, top_k)
        
        return results
    except Exception as e:
        logger.error(f"Error searching content: {str(e)}")
        raise
