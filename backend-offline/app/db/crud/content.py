from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete, and_, or_
from typing import Any, Dict, Optional, List, Union
import uuid
from datetime import datetime
import logging

from app.db.models.models import Content, ContentAssignment, ContentEmbedding, ContentType
from app.db.schemas.content import ContentCreate, ContentUpdate, ContentAssignmentCreate
from app.core.logging import setup_logging

logger = setup_logging("content_crud")

async def get_content(db: AsyncSession, id: str) -> Optional[Content]:
    """
    Get content by ID.
    
    Args:
        db: Database session
        id: Content ID
        
    Returns:
        Content object if found, None otherwise
    """
    try:
        result = await db.execute(select(Content).filter(Content.id == id))
        content = result.scalars().first()
        if content:
            logger.debug(f"Retrieved content: {id}")
        else:
            logger.debug(f"Content not found: {id}")
        return content
    except Exception as e:
        logger.error(f"Error retrieving content {id}: {str(e)}")
        raise

async def get_contents(
    db: AsyncSession, 
    user_id: Optional[str] = None,
    assigned_to: Optional[str] = None,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    skip: int = 0, 
    limit: int = 100
) -> List[Content]:
    """
    Get a list of content with optional filters.
    
    Args:
        db: Database session
        user_id: Filter by creator user ID
        assigned_to: Filter by assigned child or group ID
        content_type: Filter by content type
        language: Filter by language
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of Content objects
    """
    try:
        # Validate pagination parameters
        skip = max(0, skip)  # Ensure skip is not negative
        limit = min(max(1, limit), 1000)  # Ensure limit is between 1 and 1000
        
        query = select(Content).offset(skip).limit(limit)
        
        # Apply filters
        if user_id:
            query = query.filter(Content.created_by == user_id)
        
        if content_type:
            query = query.filter(Content.type == content_type)
        
        if language:
            query = query.filter(Content.language == language)
        
        if assigned_to:
            # Join with assignments to filter by assigned child or group
            query = query.join(ContentAssignment).filter(
                or_(
                    ContentAssignment.child_id == assigned_to,
                    ContentAssignment.group_id == assigned_to
                )
            )
        
        # Order by creation date (newest first)
        query = query.order_by(Content.created_at.desc())
        
        result = await db.execute(query)
        contents = result.scalars().all()
        
        logger.debug(f"Retrieved {len(contents)} content items with filters: user_id={user_id}, assigned_to={assigned_to}, content_type={content_type}, language={language}")
        return contents
    except Exception as e:
        logger.error(f"Error retrieving content list: {str(e)}")
        raise

async def create_content(
    db: AsyncSession, 
    obj_in: ContentCreate,
    file_path: str,
    size_mb: float,
    created_by: str
) -> Content:
    """
    Create new content and its assignments.
    
    Args:
        db: Database session
        obj_in: Content creation data
        file_path: Path to the content file
        size_mb: File size in MB
        created_by: User ID of the creator
        
    Returns:
        Created Content object
    """
    try:
        # Generate a unique ID for the content
        content_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        
        # Validate content type
        valid_types = ["document", "audio", "video"]
        if obj_in.type not in valid_types:
            raise ValueError(f"Invalid content type: {obj_in.type}. Must be one of {valid_types}")
        
        # Create content
        db_obj = Content(
            id=content_id,
            title=obj_in.title,
            type=obj_in.type,
            language=obj_in.language,
            file_path=file_path,
            size_mb=size_mb,
            sync_offline=obj_in.sync_offline,
            archived=False,
            status="pending",  # Initial status
            version=1,
            created_by=created_by,
            created_at=current_time,
            updated_at=current_time
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        
        logger.info(f"Created content: {content_id}, title: {obj_in.title}, type: {obj_in.type}")
        
        # Create assignments if any
        if obj_in.assign_to and len(obj_in.assign_to) > 0:
            assignment_count = 0
            for target_id in obj_in.assign_to:
                # Validate target ID format and determine if it's a child or group ID
                if target_id.startswith("child_"):
                    child_id = target_id
                    group_id = None
                    target_type = "child"
                elif target_id.startswith("group_"):
                    child_id = None
                    group_id = target_id
                    target_type = "group"
                else:
                    logger.warning(f"Invalid target ID format: {target_id}, skipping assignment")
                    continue
                
                # Create assignment
                assignment_id = str(uuid.uuid4())
                assignment = ContentAssignment(
                    id=assignment_id,
                    content_id=content_id,
                    child_id=child_id,
                    group_id=group_id,
                    created_at=current_time
                )
                db.add(assignment)
                assignment_count += 1
                
                logger.debug(f"Created assignment: {assignment_id}, content: {content_id}, {target_type}: {target_id}")
            
            if assignment_count > 0:
                await db.commit()
                logger.info(f"Created {assignment_count} assignments for content: {content_id}")
        
        return db_obj
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating content: {str(e)}")
        raise

async def update_content(
    db: AsyncSession, 
    db_obj: Content, 
    obj_in: Union[ContentUpdate, Dict[str, Any]]
) -> Content:
    """
    Update content.
    
    Args:
        db: Database session
        db_obj: Existing Content object
        obj_in: Content update data
        
    Returns:
        Updated Content object
    """
    try:
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        
        # Add updated timestamp
        update_data["updated_at"] = datetime.utcnow()
        
        # Execute update
        stmt = (
            update(Content)
            .where(Content.id == db_obj.id)
            .values(**update_data)
        )
        await db.execute(stmt)
        await db.commit()
        await db.refresh(db_obj)
        
        logger.info(f"Updated content: {db_obj.id}, fields: {', '.join(update_data.keys())}")
        return db_obj
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating content {db_obj.id}: {str(e)}")
        raise

async def delete_content(db: AsyncSession, id: str) -> bool:
    """
    Delete content and its assignments.
    
    Args:
        db: Database session
        id: Content ID
        
    Returns:
        True if deletion was successful
    """
    try:
        # Delete assignments first
        assignment_stmt = delete(ContentAssignment).where(ContentAssignment.content_id == id)
        assignment_result = await db.execute(assignment_stmt)
        assignment_count = assignment_result.rowcount
        
        # Delete embeddings
        embedding_stmt = delete(ContentEmbedding).where(ContentEmbedding.content_id == id)
        embedding_result = await db.execute(embedding_stmt)
        embedding_count = embedding_result.rowcount
        
        # Delete content
        content_stmt = delete(Content).where(Content.id == id)
        content_result = await db.execute(content_stmt)
        content_count = content_result.rowcount
        
        await db.commit()
        
        if content_count > 0:
            logger.info(f"Deleted content: {id}, with {assignment_count} assignments and {embedding_count} embeddings")
            return True
        else:
            logger.warning(f"Content not found for deletion: {id}")
            return False
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting content {id}: {str(e)}")
        raise

async def assign_content(
    db: AsyncSession, 
    obj_in: ContentAssignmentCreate
) -> ContentAssignment:
    """
    Create a content assignment.
    
    Args:
        db: Database session
        obj_in: Content assignment creation data
        
    Returns:
        Created ContentAssignment object
    """
    try:
        # Validate that content exists
        content = await get_content(db, id=obj_in.content_id)
        if not content:
            raise ValueError(f"Content not found: {obj_in.content_id}")
        
        # Validate that either child_id or group_id is provided, but not both
        if (obj_in.child_id is None and obj_in.group_id is None) or (obj_in.child_id is not None and obj_in.group_id is not None):
            raise ValueError("Either child_id or group_id must be provided, but not both")
        
        # Validate ID format
        if obj_in.child_id and not obj_in.child_id.startswith("child_"):
            raise ValueError(f"Invalid child ID format: {obj_in.child_id}")
        if obj_in.group_id and not obj_in.group_id.startswith("group_"):
            raise ValueError(f"Invalid group ID format: {obj_in.group_id}")
        
        # Check if assignment already exists
        existing_query = select(ContentAssignment).filter(
            and_(
                ContentAssignment.content_id == obj_in.content_id,
                or_(
                    and_(ContentAssignment.child_id == obj_in.child_id, obj_in.child_id != None),
                    and_(ContentAssignment.group_id == obj_in.group_id, obj_in.group_id != None)
                )
            )
        )
        existing_result = await db.execute(existing_query)
        existing_assignment = existing_result.scalars().first()
        
        if existing_assignment:
            logger.info(f"Assignment already exists for content {obj_in.content_id}")
            return existing_assignment
        
        # Create new assignment
        assignment_id = str(uuid.uuid4())
        db_obj = ContentAssignment(
            id=assignment_id,
            content_id=obj_in.content_id,
            child_id=obj_in.child_id,
            group_id=obj_in.group_id,
            created_at=datetime.utcnow()
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        
        target_type = "child" if obj_in.child_id else "group"
        target_id = obj_in.child_id if obj_in.child_id else obj_in.group_id
        logger.info(f"Created content assignment: {assignment_id}, content {obj_in.content_id} to {target_type} {target_id}")
        
        return db_obj
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating content assignment: {str(e)}")
        raise

async def get_content_for_device(
    db: AsyncSession,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    limit: int = 10
) -> List[Content]:
    """
    Get content assigned to a child or group for device sync.
    
    Args:
        db: Database session
        child_id: Child ID
        group_id: Group ID
        limit: Maximum number of records to return
        
    Returns:
        List of Content objects
    """
    try:
        # Validate that either child_id or group_id is provided
        if not (child_id or group_id):
            logger.warning("No child_id or group_id provided for device content sync")
            return []
        
        # Validate ID format
        if child_id and not child_id.startswith("child_"):
            logger.warning(f"Invalid child ID format: {child_id}")
            return []
        if group_id and not group_id.startswith("group_"):
            logger.warning(f"Invalid group ID format: {group_id}")
            return []
        
        # Validate limit
        limit = min(max(1, limit), 100)  # Ensure limit is between 1 and 100
        
        # Build base query for content that is marked for offline sync and not archived
        query = select(Content).filter(
            and_(
                Content.archived == False,
                Content.sync_offline == True,
                Content.status == "processed"  # Only return fully processed content
            )
        ).order_by(Content.updated_at.desc()).limit(limit)
        
        # Add filters for child or group assignments
        if child_id and group_id:
            # If both are provided, get content assigned to either
            query = query.join(ContentAssignment).filter(
                or_(
                    ContentAssignment.child_id == child_id,
                    ContentAssignment.group_id == group_id
                )
            )
        elif child_id:
            query = query.join(ContentAssignment).filter(ContentAssignment.child_id == child_id)
        elif group_id:
            query = query.join(ContentAssignment).filter(ContentAssignment.group_id == group_id)
        
        result = await db.execute(query)
        contents = result.scalars().all()
        
        logger.info(f"Retrieved {len(contents)} content items for device sync: child_id={child_id}, group_id={group_id}")
        return contents
    except Exception as e:
        logger.error(f"Error retrieving content for device: {str(e)}")
        raise

async def get_content_embeddings(
    db: AsyncSession,
    content_id: str
) -> List[ContentEmbedding]:
    """
    Get embeddings for a content item.
    
    Args:
        db: Database session
        content_id: Content ID
        
    Returns:
        List of ContentEmbedding objects
    """
    try:
        # Validate that content exists
        content = await get_content(db, id=content_id)
        if not content:
            logger.warning(f"Content not found for embeddings: {content_id}")
            return []
        
        query = select(ContentEmbedding).filter(ContentEmbedding.content_id == content_id)
        result = await db.execute(query)
        embeddings = result.scalars().all()
        
        logger.debug(f"Retrieved {len(embeddings)} embeddings for content: {content_id}")
        return embeddings
    except Exception as e:
        logger.error(f"Error retrieving content embeddings for {content_id}: {str(e)}")
        raise

async def create_content_embedding(
    db: AsyncSession,
    content_id: str,
    chunk_index: int,
    chunk_text: str,
    embedding_vector: str
) -> ContentEmbedding:
    """
    Create a content embedding.
    
    Args:
        db: Database session
        content_id: Content ID
        chunk_index: Index of the text chunk
        chunk_text: Text chunk
        embedding_vector: JSON string of embedding vector
        
    Returns:
        Created ContentEmbedding object
    """
    try:
        # Validate that content exists
        content = await get_content(db, id=content_id)
        if not content:
            raise ValueError(f"Content not found: {content_id}")
        
        # Check if embedding already exists for this chunk
        existing_query = select(ContentEmbedding).filter(
            and_(
                ContentEmbedding.content_id == content_id,
                ContentEmbedding.chunk_index == chunk_index
            )
        )
        existing_result = await db.execute(existing_query)
        existing_embedding = existing_result.scalars().first()
        
        if existing_embedding:
            # Update existing embedding
            existing_embedding.chunk_text = chunk_text
            existing_embedding.embedding_vector = embedding_vector
            existing_embedding.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(existing_embedding)
            
            logger.debug(f"Updated embedding for content {content_id}, chunk {chunk_index}")
            return existing_embedding
        
        # Create new embedding
        embedding_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        
        db_obj = ContentEmbedding(
            id=embedding_id,
            content_id=content_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            embedding_vector=embedding_vector,
            created_at=current_time,
            updated_at=current_time
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        
        logger.debug(f"Created embedding: {embedding_id}, content: {content_id}, chunk: {chunk_index}")
        return db_obj
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating content embedding for {content_id}: {str(e)}")
        raise

async def delete_content_embeddings(
    db: AsyncSession,
    content_id: str
) -> int:
    """
    Delete all embeddings for a content item.
    
    Args:
        db: Database session
        content_id: Content ID
        
    Returns:
        Number of embeddings deleted
    """
    try:
        stmt = delete(ContentEmbedding).where(ContentEmbedding.content_id == content_id)
        result = await db.execute(stmt)
        await db.commit()
        
        deleted_count = result.rowcount
        logger.info(f"Deleted {deleted_count} embeddings for content: {content_id}")
        return deleted_count
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting embeddings for content {content_id}: {str(e)}")
        raise

async def get_content_by_type(
    db: AsyncSession,
    content_type: ContentType,
    limit: int = 20,
    skip: int = 0
) -> List[Content]:
    """
    Get content by type.
    
    Args:
        db: Database session
        content_type: Content type
        limit: Maximum number of records to return
        skip: Number of records to skip
        
    Returns:
        List of Content objects
    """
    try:
        # Validate pagination parameters
        skip = max(0, skip)  # Ensure skip is not negative
        limit = min(max(1, limit), 100)  # Ensure limit is between 1 and 100
        
        query = select(Content).filter(
            and_(
                Content.type == content_type,
                Content.archived == False
            )
        ).order_by(Content.created_at.desc()).offset(skip).limit(limit)
        
        result = await db.execute(query)
        contents = result.scalars().all()
        
        logger.debug(f"Retrieved {len(contents)} content items of type {content_type}")
        return contents
    except Exception as e:
        logger.error(f"Error retrieving content by type {content_type}: {str(e)}")
        raise

async def get_content_assignments(
    db: AsyncSession,
    content_id: str
) -> List[ContentAssignment]:
    """
    Get all assignments for a content item.
    
    Args:
        db: Database session
        content_id: Content ID
        
    Returns:
        List of ContentAssignment objects
    """
    try:
        query = select(ContentAssignment).filter(ContentAssignment.content_id == content_id)
        result = await db.execute(query)
        assignments = result.scalars().all()
        
        logger.debug(f"Retrieved {len(assignments)} assignments for content: {content_id}")
        return assignments
    except Exception as e:
        logger.error(f"Error retrieving assignments for content {content_id}: {str(e)}")
        raise

async def remove_content_assignment(
    db: AsyncSession,
    content_id: str,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None
) -> bool:
    """
    Remove a content assignment.
    
    Args:
        db: Database session
        content_id: Content ID
        child_id: Child ID
        group_id: Group ID
        
    Returns:
        True if assignment was removed, False otherwise
    """
    try:
        # Validate that either child_id or group_id is provided
        if not (child_id or group_id):
            logger.warning("No child_id or group_id provided for assignment removal")
            return False
        
        conditions = [ContentAssignment.content_id == content_id]
        
        if child_id:
            conditions.append(ContentAssignment.child_id == child_id)
        if group_id:
            conditions.append(ContentAssignment.group_id == group_id)
        
        stmt = delete(ContentAssignment).where(and_(*conditions))
        result = await db.execute(stmt)
        await db.commit()
        
        removed_count = result.rowcount
        target_type = "child" if child_id else "group"
        target_id = child_id if child_id else group_id
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} content assignments: content {content_id} from {target_type} {target_id}")
            return True
        else:
            logger.warning(f"No assignments found to remove: content {content_id}, {target_type} {target_id}")
            return False
    except Exception as e:
        await db.rollback()
        logger.error(f"Error removing content assignment: {str(e)}")
        raise
