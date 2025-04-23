from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import os
import shutil
from datetime import datetime

from app.db.database import get_db
from app.db.schemas.content import (
    Content, ContentCreate, ContentUpdate, 
    Child, Group, ContentAssignment
)
from app.api.deps.auth import get_current_active_user
from app.db.crud.content import (
    create_content, get_content, get_contents,
    update_content, delete_content, assign_content
)
from app.core.config import settings
from app.services.content_processor import process_content_async
from app.sse.content_events import content_upload_manager

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
        language=language
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
) -> Any:
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
    return {"status": "success"}
