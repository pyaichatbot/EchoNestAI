from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Path, BackgroundTasks
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import os
import aiofiles
import json
import shutil
from datetime import datetime

from app.db.database import get_db
from app.db.schemas.content import (
    Content, ContentCreate, ContentUpdate, 
    ContentAssignment, ContentAssignmentCreate
)
from app.db.crud.content import (
    get_content, get_contents, create_content, 
    update_content, delete_content, assign_content,
    get_content_for_device
)
from app.api.deps.auth import get_current_active_user, get_current_admin_user, get_device_auth
from app.core.config import settings
from app.services.content_processor import content_processor
from app.core.logging import setup_logging

logger = setup_logging("content_routes")

router = APIRouter(tags=["content"])

@router.get("/", response_model=List[Content])
async def list_contents(
    user_id: Optional[str] = None,
    assigned_to: Optional[str] = None,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get list of content with optional filters.
    """
    contents = await get_contents(
        db, 
        user_id=user_id,
        assigned_to=assigned_to,
        content_type=content_type,
        language=language,
        skip=skip, 
        limit=limit
    )
    return contents

@router.post("/", response_model=Content)
async def create_new_content(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    content_type: str = Form(...),
    language: str = Form("en"),
    sync_offline: bool = Form(False),
    assign_to: List[str] = Form([]),
    file: UploadFile = File(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Create new content with file upload.
    """
    # Validate content type
    if content_type not in ["document", "audio", "video"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type. Must be document, audio, or video."
        )
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(settings.CONTENT_FOLDER, exist_ok=True)
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = os.path.join(settings.CONTENT_FOLDER, unique_filename)
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Calculate file size in MB
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # Create content object
    content_in = ContentCreate(
        title=title,
        type=content_type,
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
    
    # Process content in background
    logger.info(f"Starting background processing for content: {content.id}")
    background_tasks.add_task(
        process_content_async,
        content.id,
        db
    )
    
    return content

@router.get("/{content_id}", response_model=Content)
async def get_content_by_id(
    content_id: str = Path(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get content by ID.
    """
    content = await get_content(db, id=content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    return content

@router.put("/{content_id}", response_model=Content)
async def update_content_by_id(
    content_in: ContentUpdate,
    content_id: str = Path(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update content by ID.
    """
    content = await get_content(db, id=content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Check if user is admin or content creator
    if current_user.role != "admin" and content.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    content = await update_content(db, db_obj=content, obj_in=content_in)
    return content

@router.delete("/{content_id}")
async def delete_content_by_id(
    content_id: str = Path(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Delete content by ID.
    """
    content = await get_content(db, id=content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Check if user is admin or content creator
    if current_user.role != "admin" and content.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Delete file from disk
    try:
        os.remove(content.file_path)
    except OSError as e:
        # Log error but continue
        logger.error(f"Error deleting file {content.file_path}: {str(e)}")
    
    await delete_content(db, id=content_id)
    return {"message": "Content deleted successfully"}

@router.post("/assign", response_model=ContentAssignment)
async def assign_content_to_target(
    assignment_in: ContentAssignmentCreate,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Assign content to a child or group.
    """
    # Check if content exists
    content = await get_content(db, id=assignment_in.content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Create assignment
    assignment = await assign_content(db, obj_in=assignment_in)
    return assignment

@router.get("/device/sync", response_model=List[Content])
async def get_device_content(
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    limit: int = 10,
    device: Any = Depends(get_device_auth),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get content for device sync.
    """
    # If no child_id or group_id provided, use the ones from the device
    if not child_id and not group_id:
        child_id = device.child_id
        group_id = device.group_id
    
    contents = await get_content_for_device(
        db,
        child_id=child_id,
        group_id=group_id,
        limit=limit
    )
    
    return contents

@router.post("/{content_id}/reprocess")
async def reprocess_content(
    content_id: str,
    background_tasks: BackgroundTasks,
    current_user: Any = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Reprocess content (admin only).
    """
    content = await get_content(db, id=content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Process content in background
    logger.info(f"Starting background reprocessing for content: {content.id}")
    background_tasks.add_task(
        process_content_async,
        content.id,
        db
    )
    
    return {"message": f"Content {content_id} reprocessing started"}

@router.get("/search", response_model=List[dict])
async def search_content(
    query: str,
    language: str = "en",
    document_scope: Optional[List[str]] = None,
    top_k: int = 5,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Search content using vector similarity.
    """
    results = await content_processor.search_content(
        db,
        query=query,
        language=language,
        document_scope=document_scope,
        top_k=top_k
    )
    
    return results

async def process_content_async(content_id: str, db: AsyncSession) -> None:
    """
    Process content asynchronously.
    
    Args:
        content_id: Content ID
        db: Database session
    """
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.db.database import get_async_session
    from sqlalchemy import update
    from app.db.models.models import Content
    
    # Create new database session
    async for session in get_async_session():
        db = session
        break
    
    try:
        # Update content status to processing
        stmt = update(Content).where(Content.id == content_id).values(
            status="processing",
            processed_at=None
        )
        await db.execute(stmt)
        await db.commit()
        
        logger.info(f"Processing content: {content_id}")
        
        # Process content
        success = await content_processor.process_document(db, content_id)
        
        # Update content status based on processing result
        status = "processed" if success else "error"
        
        stmt = update(Content).where(Content.id == content_id).values(
            status=status,
            processed_at=datetime.utcnow() if success else None
        )
        await db.execute(stmt)
        await db.commit()
        
        logger.info(f"Content processing completed: {content_id}, status: {status}")
        
    except Exception as e:
        logger.error(f"Error processing content {content_id}: {str(e)}")
        
        # Update content status to error
        stmt = update(Content).where(Content.id == content_id).values(
            status="error"
        )
        await db.execute(stmt)
        await db.commit()
    finally:
        # Close database session
        await db.close()
