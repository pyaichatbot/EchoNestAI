from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete, func, and_
from uuid import uuid4
from datetime import datetime

from app.db.database import get_db
from app.db.models.models import ChatFeedback, ChatMessage
from app.db.schemas.chat import ChatFeedback as ChatFeedbackSchema
from app.db.schemas.chat import ChatFeedbackCreate, ChatFeedbackInDB, FeedbackRequest, FlagRequest
from app.api.deps.auth import get_current_user
from app.db.schemas.user import User
from app.core.logging import setup_logging

logger = setup_logging(__name__)
router = APIRouter(tags=["feedback"])

@router.post("/feedback/chat", response_model=ChatFeedbackSchema, status_code=status.HTTP_201_CREATED)
async def create_chat_feedback(
    feedback: ChatFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create feedback for a chat message.
    
    This endpoint allows users to provide feedback on AI responses, including:
    - Rating (1-5 stars)
    - Comments
    - Flagging inappropriate content
    """
    logger.info(f"Creating chat feedback for message {feedback.message_id}")
    
    # Check if message exists
    message_query = select(ChatMessage).where(ChatMessage.id == feedback.message_id)
    message_result = await db.execute(message_query)
    message = message_result.scalars().first()
    
    if not message:
        logger.error(f"Message {feedback.message_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    # Check if feedback already exists for this message
    existing_query = select(ChatFeedback).where(ChatFeedback.message_id == feedback.message_id)
    existing_result = await db.execute(existing_query)
    existing_feedback = existing_result.scalars().first()
    
    if existing_feedback:
        logger.info(f"Updating existing feedback for message {feedback.message_id}")
        # Update existing feedback
        existing_feedback.rating = feedback.rating
        existing_feedback.comment = feedback.comment
        existing_feedback.flagged = feedback.flagged
        existing_feedback.flag_reason = feedback.flag_reason
        await db.commit()
        await db.refresh(existing_feedback)
        
        return ChatFeedbackSchema(
            id=existing_feedback.id,
            message_id=existing_feedback.message_id,
            rating=existing_feedback.rating,
            comment=existing_feedback.comment,
            flagged=existing_feedback.flagged,
            flag_reason=existing_feedback.flag_reason,
            created_at=existing_feedback.created_at
        )
    
    # Create new feedback
    new_feedback = ChatFeedback(
        id=str(uuid4()),
        message_id=feedback.message_id,
        rating=feedback.rating,
        comment=feedback.comment,
        flagged=feedback.flagged,
        flag_reason=feedback.flag_reason,
        created_at=datetime.utcnow()
    )
    
    db.add(new_feedback)
    await db.commit()
    await db.refresh(new_feedback)
    
    logger.info(f"Created new feedback with ID {new_feedback.id}")
    
    return ChatFeedbackSchema(
        id=new_feedback.id,
        message_id=new_feedback.message_id,
        rating=new_feedback.rating,
        comment=new_feedback.comment,
        flagged=new_feedback.flagged,
        flag_reason=new_feedback.flag_reason,
        created_at=new_feedback.created_at
    )

@router.get("/feedback/chat/{message_id}", response_model=ChatFeedbackSchema)
async def get_chat_feedback(
    message_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get feedback for a specific chat message.
    """
    logger.info(f"Getting feedback for message {message_id}")
    
    query = select(ChatFeedback).where(ChatFeedback.message_id == message_id)
    result = await db.execute(query)
    feedback = result.scalars().first()
    
    if not feedback:
        logger.error(f"Feedback for message {message_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found for this message"
        )
    
    return ChatFeedbackSchema(
        id=feedback.id,
        message_id=feedback.message_id,
        rating=feedback.rating,
        comment=feedback.comment,
        flagged=feedback.flagged,
        flag_reason=feedback.flag_reason,
        created_at=feedback.created_at
    )

@router.get("/feedback/chat", response_model=List[ChatFeedbackSchema])
async def list_chat_feedback(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    flagged_only: bool = Query(False),
    min_rating: Optional[int] = Query(None, ge=1, le=5),
    max_rating: Optional[int] = Query(None, ge=1, le=5),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List feedback for chat messages with filtering options.
    
    Parameters:
    - limit: Maximum number of results to return
    - offset: Number of results to skip
    - flagged_only: If true, only return flagged feedback
    - min_rating: Minimum rating to filter by
    - max_rating: Maximum rating to filter by
    """
    logger.info(f"Listing chat feedback with filters: flagged_only={flagged_only}, min_rating={min_rating}, max_rating={max_rating}")
    
    query = select(ChatFeedback)
    
    # Apply filters
    if flagged_only:
        query = query.where(ChatFeedback.flagged == True)
    
    if min_rating is not None:
        query = query.where(ChatFeedback.rating >= min_rating)
    
    if max_rating is not None:
        query = query.where(ChatFeedback.rating <= max_rating)
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    feedback_list = result.scalars().all()
    
    return [
        ChatFeedbackSchema(
            id=feedback.id,
            message_id=feedback.message_id,
            rating=feedback.rating,
            comment=feedback.comment,
            flagged=feedback.flagged,
            flag_reason=feedback.flag_reason,
            created_at=feedback.created_at
        )
        for feedback in feedback_list
    ]

@router.post("/feedback/flag", status_code=status.HTTP_201_CREATED)
async def flag_content(
    flag_request: FlagRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Flag content as inappropriate or problematic.
    
    This endpoint allows users to flag any content (chat messages, documents, etc.)
    that may be inappropriate, incorrect, or problematic.
    """
    logger.info(f"Flagging content {flag_request.content_id} for reason: {flag_request.reason}")
    
    # Check if it's a chat message
    message_query = select(ChatMessage).where(ChatMessage.id == flag_request.content_id)
    message_result = await db.execute(message_query)
    message = message_result.scalars().first()
    
    if message:
        # Check if feedback already exists
        feedback_query = select(ChatFeedback).where(ChatFeedback.message_id == flag_request.content_id)
        feedback_result = await db.execute(feedback_query)
        feedback = feedback_result.scalars().first()
        
        if feedback:
            # Update existing feedback
            feedback.flagged = True
            feedback.flag_reason = flag_request.reason
            await db.commit()
            logger.info(f"Updated existing feedback for message {flag_request.content_id} with flag")
        else:
            # Create new feedback with flag
            new_feedback = ChatFeedback(
                id=str(uuid4()),
                message_id=flag_request.content_id,
                flagged=True,
                flag_reason=flag_request.reason,
                comment=flag_request.notes,
                created_at=datetime.utcnow()
            )
            db.add(new_feedback)
            await db.commit()
            logger.info(f"Created new feedback with flag for message {flag_request.content_id}")
        
        return {"status": "success", "message": "Content flagged successfully"}
    
    # If not a chat message, could implement other content type flagging here
    
    logger.error(f"Content {flag_request.content_id} not found for flagging")
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Content not found"
    )

@router.delete("/feedback/chat/{feedback_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_feedback(
    feedback_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete feedback for a chat message.
    
    This endpoint allows administrators to remove feedback entries.
    """
    logger.info(f"Deleting feedback {feedback_id}")
    
    # Check if feedback exists
    query = select(ChatFeedback).where(ChatFeedback.id == feedback_id)
    result = await db.execute(query)
    feedback = result.scalars().first()
    
    if not feedback:
        logger.error(f"Feedback {feedback_id} not found for deletion")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found"
        )
    
    # Delete the feedback
    delete_stmt = delete(ChatFeedback).where(ChatFeedback.id == feedback_id)
    await db.execute(delete_stmt)
    await db.commit()
    
    logger.info(f"Deleted feedback {feedback_id}")
    
    return None

@router.post("/feedback/general", status_code=status.HTTP_201_CREATED)
async def submit_general_feedback(
    feedback_request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submit general feedback about the system.
    
    This endpoint allows users to provide general feedback about their experience,
    including the input they provided, the response they received, and their rating.
    """
    logger.info(f"Submitting general feedback with rating {feedback_request.rating}")
    
    # For general feedback, we create a temporary chat session and message
    # to store the feedback, since we don't have a dedicated general feedback table
    
    # Create a new chat session
    session_id = str(uuid4())
    
    # Create user message
    user_message = ChatMessage(
        id=str(uuid4()),
        session_id=session_id,
        is_user=True,
        content=feedback_request.input,
        created_at=datetime.utcnow()
    )
    
    # Create AI response message
    ai_message = ChatMessage(
        id=str(uuid4()),
        session_id=session_id,
        is_user=False,
        content=feedback_request.response,
        created_at=datetime.utcnow()
    )
    
    # Create feedback for AI message
    feedback = ChatFeedback(
        id=str(uuid4()),
        message_id=ai_message.id,
        rating=feedback_request.rating,
        comment=feedback_request.comment,
        created_at=datetime.utcnow()
    )
    
    # Add all to database
    db.add(user_message)
    db.add(ai_message)
    db.add(feedback)
    await db.commit()
    
    logger.info(f"Created general feedback with ID {feedback.id}")
    
    return {
        "status": "success",
        "message": "Feedback submitted successfully",
        "feedback_id": feedback.id
    }

@router.get("/feedback/stats")
async def get_feedback_stats(
    startDate: Optional[str] = Query(None),
    endDate: Optional[str] = Query(None),
    childId: Optional[str] = Query(None),
    groupId: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get aggregated feedback statistics for chat messages.
    Supports filtering by date range, child, and group.
    """
    # Build base query
    query = select(ChatFeedback)
    # Join with ChatMessage for child/group filtering
    if childId or groupId:
        query = query.join(ChatMessage, ChatFeedback.message_id == ChatMessage.id)
        if childId:
            query = query.where(ChatMessage.session.has(child_id=childId))
        if groupId:
            query = query.where(ChatMessage.session.has(group_id=groupId))
    # Date filtering
    if startDate:
        try:
            start_dt = datetime.fromisoformat(startDate)
            query = query.where(ChatFeedback.created_at >= start_dt)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid startDate format. Use ISO8601.")
    if endDate:
        try:
            end_dt = datetime.fromisoformat(endDate)
            query = query.where(ChatFeedback.created_at <= end_dt)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid endDate format. Use ISO8601.")
    # Execute query
    result = await db.execute(query)
    feedbacks = result.scalars().all()
    # Aggregate stats
    total_feedback = len(feedbacks)
    positive_ratings = sum(1 for f in feedbacks if f.rating and f.rating > 0)
    negative_ratings = sum(1 for f in feedbacks if f.rating and f.rating < 0)
    # Placeholder for tags and response quality (not in DB)
    common_tags = []
    response_quality_breakdown = {}
    return {
        "totalFeedback": total_feedback,
        "positiveRatings": positive_ratings,
        "negativeRatings": negative_ratings,
        "commonTags": common_tags,
        "responseQualityBreakdown": response_quality_breakdown
    }

@router.get("/feedback/flags")
async def list_flagged_feedback(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all flagged feedback entries (flagged=True), with pagination.
    Returns id, message_id, flagged, flag_reason, comment, created_at, rating.
    """
    query = select(ChatFeedback).where(ChatFeedback.flagged == True).offset(offset).limit(limit)
    result = await db.execute(query)
    feedbacks = result.scalars().all()
    return [
        {
            "id": f.id,
            "message_id": f.message_id,
            "flagged": f.flagged,
            "flag_reason": f.flag_reason,
            "comment": f.comment,
            "created_at": f.created_at.isoformat() if f.created_at else None,
            "rating": f.rating
        }
        for f in feedbacks
    ]

@router.patch("/feedback/flags/{flag_id}/resolve")
async def resolve_flagged_feedback(
    flag_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Resolve a flagged feedback entry by marking it as not flagged and setting resolved_at.
    """
    # Fetch the feedback entry
    query = select(ChatFeedback).where(ChatFeedback.id == flag_id)
    result = await db.execute(query)
    feedback = result.scalars().first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Flagged feedback not found")
    if not feedback.flagged:
        raise HTTPException(status_code=400, detail="Feedback is not flagged")
    # Mark as resolved
    feedback.flagged = False
    # Optionally add a resolved_at field if present in the model
    if hasattr(feedback, "resolved_at"):
        feedback.resolved_at = datetime.utcnow()
    await db.commit()
    await db.refresh(feedback)
    return {
        "id": feedback.id,
        "message_id": feedback.message_id,
        "flagged": feedback.flagged,
        "flag_reason": feedback.flag_reason,
        "comment": feedback.comment,
        "created_at": feedback.created_at,
        "rating": feedback.rating
    }
