from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete, and_, or_
from typing import Any, Dict, Optional, List, Union
import uuid
from datetime import datetime

from app.db.models.models import Content, ContentAssignment, ContentEmbedding, ContentType
from app.db.schemas.content import ContentCreate, ContentUpdate, ContentAssignmentCreate
from app.core.logging import setup_logging

logger = setup_logging("content_crud")

async def get_content(db: AsyncSession, id: str) -> Optional[Content]:
    """
    Get content by ID.
    """
    result = await db.execute(select(Content).filter(Content.id == id))
    return result.scalars().first()

async def get_contents(
    db: AsyncSession, 
    user_id: str,
    assigned_to: Optional[str] = None,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0, 
    limit: int = 100
) -> List[Content]:
    """
    Get a list of content with optional filters.
    """
    query = select(Content).filter(Content.created_by == user_id).offset(skip).limit(limit)
    
    if content_type:
        query = query.filter(Content.type == content_type)
    
    if language:
        query = query.filter(Content.language == language)
    
    if status:
        query = query.filter(Content.status == status)
    
    if assigned_to:
        # Join with assignments to filter by assigned child or group
        query = query.join(ContentAssignment).filter(
            or_(
                ContentAssignment.child_id == assigned_to,
                ContentAssignment.group_id == assigned_to
            )
        )
    
    result = await db.execute(query)
    return result.scalars().all()

async def create_content(
    db: AsyncSession, 
    obj_in: ContentCreate,
    file_path: str,
    size_mb: float,
    created_by: str
) -> Content:
    """
    Create new content and its assignments.
    """
    # Create content
    content_id = str(uuid.uuid4())
    
    db_obj = Content(
        id=content_id,
        title=obj_in.title,
        type=obj_in.type,
        language=obj_in.language,
        file_path=file_path,
        size_mb=size_mb,
        sync_offline=obj_in.sync_offline,
        archived=False,
        status=getattr(obj_in, 'status', 'pending'),
        version=1,
        created_by=created_by,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    
    logger.info(f"Created new content: {content_id}, type: {obj_in.type}, title: {obj_in.title}")
    
    # Create assignments if any
    if obj_in.assign_to:
        for target_id in obj_in.assign_to:
            # Determine if it's a child or group ID based on prefix
            is_child = target_id.startswith("child_")
            is_group = target_id.startswith("group_")
            
            if not (is_child or is_group):
                logger.warning(f"Invalid target ID format: {target_id}, skipping assignment")
                continue
                
            assignment = ContentAssignment(
                id=str(uuid.uuid4()),
                content_id=db_obj.id,
                child_id=target_id if is_child else None,
                group_id=target_id if is_group else None,
                created_at=datetime.utcnow()
            )
            db.add(assignment)
            
            logger.info(f"Created content assignment: content {content_id} to {'child' if is_child else 'group'} {target_id}")
        
        await db.commit()
    
    return db_obj

async def update_content(
    db: AsyncSession, 
    db_obj: Content, 
    obj_in: Union[ContentUpdate, Dict[str, Any]]
) -> Content:
    """
    Update content.
    """
    if isinstance(obj_in, dict):
        update_data = obj_in
    else:
        update_data = obj_in.dict(exclude_unset=True)
    
    # Add updated timestamp
    update_data["updated_at"] = datetime.utcnow()
    
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

async def delete_content(db: AsyncSession, id: str) -> bool:
    """
    Delete content and its assignments.
    """
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
    
    await db.commit()
    
    logger.info(f"Deleted content: {id}, with {assignment_count} assignments and {embedding_count} embeddings")
    return True

async def assign_content(
    db: AsyncSession, 
    obj_in: ContentAssignmentCreate
) -> ContentAssignment:
    """
    Create a content assignment.
    """
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

async def get_content_for_device(
    db: AsyncSession,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    limit: int = 10
) -> List[Content]:
    """
    Get content assigned to a child or group for device sync.
    """
    if not (child_id or group_id):
        logger.warning("No child_id or group_id provided for device content sync")
        return []
    
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

async def get_content_by_type(
    db: AsyncSession,
    content_type: ContentType,
    limit: int = 20,
    skip: int = 0
) -> List[Content]:
    """
    Get content by type.
    """
    query = select(Content).filter(
        and_(
            Content.type == content_type,
            Content.archived == False
        )
    ).order_by(Content.created_at.desc()).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def get_content_assignments(
    db: AsyncSession,
    content_id: str
) -> List[ContentAssignment]:
    """
    Get all assignments for a content item.
    """
    query = select(ContentAssignment).filter(ContentAssignment.content_id == content_id)
    result = await db.execute(query)
    return result.scalars().all()

async def remove_content_assignment(
    db: AsyncSession,
    content_id: str,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None
) -> bool:
    """
    Remove a content assignment.
    """
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
    logger.info(f"Removed {removed_count} content assignments: content {content_id} from {target_type} {target_id}")
    
    return removed_count > 0

async def validate_document_ids(
    db: AsyncSession,
    document_ids: List[str],
    user: Any
) -> List[str]:
    """
    Validate document IDs and check user access permissions
    
    Args:
        db: Database session
        document_ids: List of document IDs to validate
        user: Current user object
        
    Returns:
        List of valid document IDs the user has access to
    """
    # Query for documents and their assignments
    query = select(Content.id).where(
        Content.id.in_(document_ids)
    )
    
    # If user is not admin, check assignments
    if not user.is_superuser:
        query = query.join(
            ContentAssignment,
            ContentAssignment.content_id == Content.id
        ).where(
            # User has direct access or access through group
            (ContentAssignment.child_id.in_(user.children_ids)) |
            (ContentAssignment.group_id.in_(user.group_ids))
        )
    
    result = await db.execute(query)
    valid_ids = result.scalars().all()
    
    return list(valid_ids)
