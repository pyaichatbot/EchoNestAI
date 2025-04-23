import os
import json
import aiofiles
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.models import Device, Content, ContentAssignment
from app.db.schemas.device import SyncManifestResponse, SyncProfile, DocumentInfo
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("sync_service")

async def generate_sync_manifest(
    db: AsyncSession,
    device: Device,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    mode: str = "home",
    current_cache_size: float = 0.0,
    current_doc_ids: List[str] = []
) -> SyncManifestResponse:
    """
    Generate a sync manifest for a device.
    
    Args:
        db: Database session
        device: Device requesting sync
        child_id: Child ID for content filtering
        group_id: Group ID for content filtering
        mode: Device mode (home, school, hybrid)
        current_cache_size: Current cache size in MB
        current_doc_ids: List of document IDs currently on device
        
    Returns:
        Sync manifest response
    """
    # Create sync profile based on device mode and assignment
    sync_profile = SyncProfile(
        mode=mode,
        child_id=child_id,
        group_id=group_id,
        sync_limit_mb=settings.MAX_CONTENT_SIZE_MB,
        max_documents=settings.MAX_DOCUMENTS_PER_DEVICE
    )
    
    # Get content assigned to child or group
    from sqlalchemy import select, and_, or_
    from sqlalchemy.sql import desc
    
    query = select(Content).filter(
        and_(
            Content.sync_offline == True,
            Content.archived == False
        )
    ).order_by(desc(Content.created_at))
    
    if child_id:
        query = query.join(ContentAssignment).filter(ContentAssignment.child_id == child_id)
    elif group_id:
        query = query.join(ContentAssignment).filter(ContentAssignment.group_id == group_id)
    
    result = await db.execute(query)
    contents = result.scalars().all()
    
    # Determine which documents need to be synced
    documents = []
    total_size = 0
    sync_required = False
    
    for content in contents:
        # Skip if we've reached the limit
        if len(documents) >= sync_profile.max_documents:
            break
        
        # Skip if adding this document would exceed the size limit
        if total_size + content.size_mb > sync_profile.sync_limit_mb:
            continue
        
        # Check if document is already on device
        is_new = content.id not in current_doc_ids
        
        # Get assignments for this content
        assignment_query = select(ContentAssignment).filter(ContentAssignment.content_id == content.id)
        assignment_result = await db.execute(assignment_query)
        assignments = assignment_result.scalars().all()
        
        # Build assigned_to dict
        assigned_to = {"child_ids": [], "group_ids": []}
        for assignment in assignments:
            if assignment.child_id:
                assigned_to["child_ids"].append(assignment.child_id)
            if assignment.group_id:
                assigned_to["group_ids"].append(assignment.group_id)
        
        # Create document info
        doc_info = DocumentInfo(
            id=content.id,
            title=content.title,
            type=content.type,
            language=content.language,
            size_mb=content.size_mb,
            version=content.version,
            priority=is_new,  # New documents have higher priority
            url=f"/api/v1/content/{content.id}/download",
            embedding_meta={"chunk_count": 0, "embedding_dim": 384},  # Default values
            assigned_to=assigned_to,
            created_at=content.created_at,
            expires_at=content.expires_at
        )
        
        documents.append(doc_info)
        total_size += content.size_mb
        
        # If this is a new document, sync is required
        if is_new:
            sync_required = True
    
    # Create sync manifest
    manifest = SyncManifestResponse(
        device_id=device.id,
        sync_profile=sync_profile,
        documents=documents,
        sync_required=sync_required,
        timestamp=datetime.utcnow()
    )
    
    # Log sync manifest generation
    logger.info(f"Generated sync manifest for device {device.id}: {len(documents)} documents, sync required: {sync_required}")
    
    return manifest

async def log_sync_operation(
    db: AsyncSession,
    device_id: str,
    sync_type: str,
    status: str,
    message: str,
    details: Dict[str, Any] = {}
) -> None:
    """
    Log a sync operation.
    
    Args:
        db: Database session
        device_id: Device ID
        sync_type: Type of sync operation (pull or push)
        status: Status of operation (success, error, etc.)
        message: Status message
        details: Additional details
    """
    from app.db.models.models import SyncLog
    
    # Create sync log
    sync_log = SyncLog(
        device_id=device_id,
        sync_type=sync_type,
        status=status,
        message=message,
        details=json.dumps(details)
    )
    
    db.add(sync_log)
    await db.commit()
    
    logger.info(f"Logged sync operation for device {device_id}: {sync_type} - {status}")

async def update_device_sync_metrics(
    db: AsyncSession,
    device_id: str,
    used_mb: float,
    doc_count: int,
    status: str = "success",
    message: str = "Sync completed successfully"
) -> None:
    """
    Update device sync metrics.
    
    Args:
        db: Database session
        device_id: Device ID
        used_mb: Used storage in MB
        doc_count: Number of documents
        status: Sync status
        message: Status message
    """
    from app.db.crud.device import update_device_sync_status, update_device_last_sync
    
    # Update sync status
    await update_device_sync_status(
        db,
        device_id=device_id,
        obj_in={
            "used_mb": used_mb,
            "doc_count": doc_count,
            "last_sync_status": status,
            "last_sync_message": message
        }
    )
    
    # Update last sync timestamp
    await update_device_last_sync(db, device_id=device_id)
    
    logger.info(f"Updated sync metrics for device {device_id}: {used_mb} MB, {doc_count} documents")
