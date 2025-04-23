from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.schemas.device import (
    SyncManifestRequest, SyncManifestResponse, SyncProfile,
    DocumentInfo, EmbeddingMeta, AssignedTo
)
from app.db.models.models import Content, ContentAssignment, ContentEmbedding
from app.db.crud.content import get_content_for_device
from app.db.crud.device import update_device_last_sync
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("sync_service")

async def generate_sync_manifest(
    db: AsyncSession,
    device_id: str,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    mode: str = "home",
    current_cache_size: float = 0,
    current_doc_ids: List[str] = []
) -> SyncManifestResponse:
    """
    Generate a content sync manifest for a device.
    
    This determines what content should be synced to the device based on:
    - Child or group assignments
    - Device mode (home/school)
    - Current cache size and content
    - Resource constraints
    """
    logger.info(f"Generating sync manifest for device {device_id}, child: {child_id}, group: {group_id}, mode: {mode}")
    
    # Update device last sync time
    await update_device_last_sync(db, device_id=device_id)
    
    # Get content assigned to child or group
    content_list = await get_content_for_device(
        db,
        child_id=child_id,
        group_id=group_id,
        limit=settings.MAX_DOCUMENTS_PER_DEVICE
    )
    
    # Create sync profile
    sync_profile = SyncProfile(
        mode=mode,
        child_id=child_id,
        group_id=group_id,
        sync_limit_mb=settings.MAX_CONTENT_SIZE_MB,
        max_documents=settings.MAX_DOCUMENTS_PER_DEVICE
    )
    
    # Determine if sync is required and calculate total size
    sync_required = False
    total_size_mb = current_cache_size
    
    # Get embedding metadata for all content
    embedding_meta_map = await get_embedding_metadata(db, [c.id for c in content_list])
    
    # Convert content to document info
    documents = []
    for content in content_list:
        # Check if content is already on device and hasn't been updated
        if content.id in current_doc_ids:
            # Get the content from the database to check version
            stmt = select(Content).where(Content.id == content.id)
            result = await db.execute(stmt)
            db_content = result.scalars().first()
            
            # Skip if already in device and not updated
            if db_content and db_content.version == content.version:
                logger.debug(f"Content {content.id} already on device and up to date")
                continue
        
        sync_required = True
        total_size_mb += content.size_mb
        
        # Get embedding metadata
        embedding_meta = embedding_meta_map.get(content.id, EmbeddingMeta(
            chunk_count=0,
            embedding_dim=384  # Default dimension
        ))
        
        # Get assignments
        assigned_child_ids, assigned_group_ids = await get_content_assignments(db, content.id)
        
        # Create document info
        doc_info = DocumentInfo(
            id=content.id,
            title=content.title,
            type=content.type,
            language=content.language,
            size_mb=content.size_mb,
            version=content.version,
            priority=determine_priority(content, child_id, group_id),
            url=f"{settings.API_V1_STR}/content/download/{content.id}",
            embedding_meta=embedding_meta,
            assigned_to=AssignedTo(
                child_ids=assigned_child_ids,
                group_ids=assigned_group_ids
            ),
            created_at=content.created_at,
            expires_at=content.expires_at
        )
        documents.append(doc_info)
    
    # Sort documents by priority
    documents.sort(key=lambda x: (0 if x.priority else 1, x.created_at), reverse=True)
    
    # Check if we exceed size limits and trim if necessary
    if total_size_mb > settings.MAX_CONTENT_SIZE_MB:
        logger.warning(f"Content size exceeds limit: {total_size_mb}MB > {settings.MAX_CONTENT_SIZE_MB}MB")
        
        # Keep high priority documents first
        documents = trim_documents_to_size_limit(
            documents, 
            settings.MAX_CONTENT_SIZE_MB - current_cache_size
        )
    
    # Create response
    response = SyncManifestResponse(
        device_id=device_id,
        sync_profile=sync_profile,
        documents=documents,
        sync_required=sync_required,
        timestamp=datetime.utcnow()
    )
    
    logger.info(f"Sync manifest generated for device {device_id}: {len(documents)} documents, sync required: {sync_required}")
    return response

async def get_embedding_metadata(db: AsyncSession, content_ids: List[str]) -> Dict[str, EmbeddingMeta]:
    """
    Get embedding metadata for content IDs.
    
    Args:
        db: Database session
        content_ids: List of content IDs
        
    Returns:
        Dictionary mapping content IDs to embedding metadata
    """
    if not content_ids:
        return {}
    
    # Query to count chunks and get embedding dimension for each content
    stmt = select(
        ContentEmbedding.content_id,
        func.count(ContentEmbedding.id).label("chunk_count"),
        ContentEmbedding.embedding_vector
    ).where(
        ContentEmbedding.content_id.in_(content_ids)
    ).group_by(
        ContentEmbedding.content_id,
        ContentEmbedding.embedding_vector
    )
    
    result = await db.execute(stmt)
    rows = result.all()
    
    # Create mapping
    embedding_meta_map = {}
    
    for row in rows:
        content_id = row.content_id
        chunk_count = row.chunk_count
        
        # Get embedding dimension from the first embedding vector
        embedding_dim = 384  # Default
        if row.embedding_vector:
            try:
                # Parse embedding vector from JSON
                vector = json.loads(row.embedding_vector)
                if isinstance(vector, list):
                    embedding_dim = len(vector)
            except Exception as e:
                logger.error(f"Error parsing embedding vector: {str(e)}")
        
        embedding_meta_map[content_id] = EmbeddingMeta(
            chunk_count=chunk_count,
            embedding_dim=embedding_dim
        )
    
    return embedding_meta_map

async def get_content_assignments(db: AsyncSession, content_id: str) -> tuple:
    """
    Get child and group assignments for content.
    
    Args:
        db: Database session
        content_id: Content ID
        
    Returns:
        Tuple of (child_ids, group_ids)
    """
    # Query assignments
    stmt = select(ContentAssignment).where(ContentAssignment.content_id == content_id)
    result = await db.execute(stmt)
    assignments = result.scalars().all()
    
    # Extract child and group IDs
    child_ids = [a.child_id for a in assignments if a.child_id]
    group_ids = [a.group_id for a in assignments if a.group_id]
    
    return child_ids, group_ids

def determine_priority(content: Any, child_id: Optional[str], group_id: Optional[str]) -> bool:
    """
    Determine if content is high priority.
    
    Priority is based on:
    - Content type (educational content is higher priority)
    - Recent assignments
    - Usage patterns
    
    Args:
        content: Content object
        child_id: Child ID
        group_id: Group ID
        
    Returns:
        True if content is high priority
    """
    # Educational content is high priority
    if hasattr(content, 'metadata') and content.metadata:
        try:
            metadata = json.loads(content.metadata) if isinstance(content.metadata, str) else content.metadata
            if metadata.get('category') in ['educational', 'homework', 'assignment']:
                return True
        except:
            pass
    
    # Recent content is high priority (less than 7 days old)
    if content.created_at:
        days_old = (datetime.utcnow() - content.created_at).days
        if days_old < 7:
            return True
    
    # Default to medium priority
    return False

def trim_documents_to_size_limit(documents: List[DocumentInfo], max_size_mb: float) -> List[DocumentInfo]:
    """
    Trim document list to fit within size limit.
    
    Args:
        documents: List of documents
        max_size_mb: Maximum size in MB
        
    Returns:
        Trimmed list of documents
    """
    # Sort by priority (high priority first)
    sorted_docs = sorted(documents, key=lambda x: (0 if x.priority else 1, x.created_at), reverse=True)
    
    result = []
    current_size = 0
    
    for doc in sorted_docs:
        if current_size + doc.size_mb <= max_size_mb:
            result.append(doc)
            current_size += doc.size_mb
        else:
            # Skip this document as it would exceed the limit
            logger.debug(f"Skipping document {doc.id} due to size limit")
    
    return result

async def process_sync_status_update(
    db: AsyncSession,
    device_id: str,
    synced_content_ids: List[str],
    failed_content_ids: List[str],
    sync_timestamp: datetime
) -> Dict[str, Any]:
    """
    Process sync status update from device.
    
    Args:
        db: Database session
        device_id: Device ID
        synced_content_ids: List of successfully synced content IDs
        failed_content_ids: List of failed content IDs
        sync_timestamp: Sync timestamp
        
    Returns:
        Status response
    """
    from app.db.models.models import Device, DeviceContentSync
    from sqlalchemy import insert, update, delete
    
    logger.info(f"Processing sync status update for device {device_id}: {len(synced_content_ids)} synced, {len(failed_content_ids)} failed")
    
    try:
        # Update device last sync time
        stmt = update(Device).where(Device.id == device_id).values(
            last_sync=sync_timestamp,
            updated_at=datetime.utcnow()
        )
        await db.execute(stmt)
        
        # Remove existing sync records for this device
        stmt = delete(DeviceContentSync).where(DeviceContentSync.device_id == device_id)
        await db.execute(stmt)
        
        # Insert new sync records for successful syncs
        for content_id in synced_content_ids:
            stmt = insert(DeviceContentSync).values(
                device_id=device_id,
                content_id=content_id,
                sync_status="synced",
                synced_at=sync_timestamp
            )
            await db.execute(stmt)
        
        # Insert records for failed syncs
        for content_id in failed_content_ids:
            stmt = insert(DeviceContentSync).values(
                device_id=device_id,
                content_id=content_id,
                sync_status="failed",
                synced_at=sync_timestamp
            )
            await db.execute(stmt)
        
        await db.commit()
        
        return {
            "status": "success",
            "device_id": device_id,
            "synced_count": len(synced_content_ids),
            "failed_count": len(failed_content_ids),
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        await db.rollback()
        logger.error(f"Error processing sync status update: {str(e)}")
        raise
