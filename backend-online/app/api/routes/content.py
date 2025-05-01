from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Query, Request
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import os
import shutil
from datetime import datetime
from fastapi.responses import JSONResponse

from app.db.database import get_db
from app.db.schemas.content import (
    Content, ContentCreate, ContentUpdate, 
    Child, Group, ContentAssignment,
    RelevantDocumentsRequest, RelevantDocumentsResponse,
    RelevantDocumentItem
)
from app.api.deps.auth import get_current_active_user
from app.db.crud.content import (
    create_content, get_content, get_contents,
    update_content, delete_content, assign_content,
    validate_document_ids
)
from app.core.config import settings
from app.services.content_processor import process_content_async
from app.sse.event_manager import content_upload_manager
from app.llm.chat_service import get_relevant_documents
from app.services.language_service import is_language_supported
from app.core.logging import setup_logging
from app.core.cache import cache
from app.core.rate_limiter import rate_limiter

logger = setup_logging("content_routes")

router = APIRouter(tags=["content"])

@router.post("/content/upload", response_model=Content, status_code=status.HTTP_201_CREATED)
async def upload_content(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    type: str = Form(...),
    assign_to: List[str] = Form([]),
    language: str = Form("en"),
    sync_offline: bool = Form(False),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Upload content (document, audio, video).
    """
    # Validate content type
    if type not in ["document", "audio", "video"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type. Must be document, audio, or video."
        )
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_FOLDER, unique_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get file size in MB
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # Create content object
    content_in = ContentCreate(
        title=title,
        type=type,
        language=language,
        sync_offline=sync_offline,
        assign_to=assign_to
    )
    
    # Create content in database
    content = await create_content(
        db, 
        obj_in=content_in, 
        file_path=file_path, 
        size_mb=size_mb, 
        created_by=current_user.id
    )
    
    # Process content in background (transcription, embedding, etc.)
    background_tasks.add_task(
        process_content_async, 
        content.id, 
        file_path, 
        content.type, 
        content.language,
        content_upload_manager
    )
    
    return content

@router.get("/content/list", response_model=List[Content])
async def list_content(
    assigned_to: Optional[str] = None,
    type: Optional[str] = None,
    language: Optional[str] = None,
    status: Optional[str] = None,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get list of content with optional filters.
    """
    return await get_contents(
        db, 
        user_id=current_user.id, 
        assigned_to=assigned_to, 
        content_type=type, 
        language=language,
        status=status
    )

@router.get("/content/{id}", response_model=Content)
async def get_content_by_id(
    id: str,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get content by ID.
    """
    content = await get_content(db, id=id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    return content

@router.patch("/content/{id}/sync", response_model=Content)
async def update_content_sync(
    id: str,
    sync_data: dict,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update content sync status.
    """
    content = await get_content(db, id=id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    update_data = ContentUpdate(sync_offline=sync_data.get("sync_offline"))
    return await update_content(db, db_obj=content, obj_in=update_data)

@router.patch("/content/{id}/archive", response_model=Content)
async def archive_content(
    id: str,
    archive_data: dict,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Archive content.
    """
    content = await get_content(db, id=id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    update_data = ContentUpdate(archived=archive_data.get("archived"))
    return await update_content(db, db_obj=content, obj_in=update_data)

@router.delete("/content/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_content(
    id: str,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> None:
    """
    Delete content.
    """
    content = await get_content(db, id=id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    await delete_content(db, id=id)
    return

@router.post("/content/relevant-documents", response_model=RelevantDocumentsResponse)
async def get_relevant_document_contexts(
    request: RelevantDocumentsRequest,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> RelevantDocumentsResponse:
    """
    Get relevant document contexts based on a search query.
    
    Args:
        request: Search request containing query, optional document IDs, language, and limit
        
    Returns:
        Relevant documents with their contexts and total count
    
    Raises:
        HTTPException: If validation fails or other errors occur
    """
    try:
        # Validate language code
        if not await is_language_supported(request.language):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Language '{request.language}' is not supported"
            )
        
        # Validate document IDs if provided
        document_ids = request.document_ids
        if document_ids:
            valid_docs = await validate_document_ids(db, document_ids, current_user)
            if not valid_docs:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid document IDs provided"
                )
            document_ids = valid_docs
        
        # Get relevant documents with caching
        cache_key = f"relevant_docs:{request.query}:{request.language}:{document_ids}:{request.limit}"
        cached_result = await cache.get_cache(cache_key)
        
        if cached_result:
            return RelevantDocumentsResponse(**cached_result)
        
        # Get relevant documents
        relevant_docs = await get_relevant_documents(
            db=db,
            query=request.query,
            language=request.language,
            document_ids=document_ids,
            top_k=request.limit,
            similarity_threshold=0.6  # Default threshold
        )
        
        # Convert to response format
        response = RelevantDocumentsResponse(
            documents=[
                RelevantDocumentItem(
                    id=doc.document_id,
                    title=doc.title,
                    content=doc.context,
                    relevance_score=doc.relevance_score,
                    language=doc.language,
                    type=doc.content_type
                ) for doc in relevant_docs
            ],
            total_found=len(relevant_docs)
        )
        
        # Cache the results
        await cache.set_cache(cache_key, response.dict(), expire=300)  # Cache for 5 minutes
        
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving relevant documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )
