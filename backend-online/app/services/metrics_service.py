from typing import List, Dict, Any, Optional, Set
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status

from app.db.models.models import (
    Content, ChatMessage, Event, NotificationSetting,
    ContentAssignment, Child, Group
)
from app.db.schemas.dashboard import (
    ChildMetricsResponse, GroupMetricsResponse, EventsResponse,
    NotificationSettings, NotificationSettingsUpdate
)
from app.core.logging import setup_logging
from app.core.cache import cache
from app.core.metrics import metrics_client

logger = setup_logging("metrics_service")

# Constants
VALID_TIME_RANGES: Set[str] = {"day", "week", "month", "year"}
CACHE_TTL = 300  # 5 minutes
MAX_RETRIES = 3

class MetricsError(Exception):
    """Base exception for metrics service errors."""
    pass

async def _get_time_range(time_range: str) -> tuple[datetime, datetime]:
    """Helper to get start and end dates based on time range."""
    if time_range not in VALID_TIME_RANGES:
        raise ValueError(f"Invalid time_range. Must be one of: {VALID_TIME_RANGES}")
    
    end_date = datetime.utcnow()
    if time_range == "day":
        start_date = end_date - timedelta(days=1)
    elif time_range == "week":
        start_date = end_date - timedelta(weeks=1)
    elif time_range == "month":
        start_date = end_date - timedelta(days=30)
    else:  # year
        start_date = end_date - timedelta(days=365)
    
    return start_date, end_date

async def _fetch_content_metrics(
    db: AsyncSession,
    child_id: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, Any]]:
    """Fetch most used content metrics."""
    try:
        content_query = select(
            Content.id,
            Content.title,
            func.count(ChatMessage.id).label("usage_count")
        ).join(
            ChatMessage, ChatMessage.content_id == Content.id
        ).filter(
            and_(
                ChatMessage.created_at >= start_date,
                ChatMessage.created_at <= end_date,
                ChatMessage.child_id == child_id
            )
        ).group_by(
            Content.id
        ).order_by(
            desc("usage_count")
        ).limit(5)

        result = await db.execute(content_query)
        return [
            {"id": row.id, "title": row.title, "usage_count": row.usage_count}
            for row in result.all()
        ]
    except SQLAlchemyError as e:
        logger.error(f"Database error in _fetch_content_metrics: {str(e)}")
        raise MetricsError("Failed to fetch content metrics") from e

async def _fetch_hours_metrics(
    db: AsyncSession,
    child_id: str,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, float]:
    """Fetch active hours metrics."""
    try:
        hours_query = select(
            func.extract("hour", ChatMessage.created_at).label("hour"),
            func.count().label("count")
        ).filter(
            and_(
                ChatMessage.created_at >= start_date,
                ChatMessage.created_at <= end_date,
                ChatMessage.child_id == child_id
            )
        ).group_by("hour")

        result = await db.execute(hours_query)
        return {str(row.hour): float(row.count) for row in result.all()}
    except SQLAlchemyError as e:
        logger.error(f"Database error in _fetch_hours_metrics: {str(e)}")
        raise MetricsError("Failed to fetch hours metrics") from e

@cache(ttl=CACHE_TTL)
async def get_child_metrics(
    db: AsyncSession,
    child_id: str,
    time_range: str = "week"
) -> ChildMetricsResponse:
    """
    Get dashboard metrics for a specific child.
    
    Args:
        db: Database session
        child_id: Child ID to get metrics for
        time_range: Time range for metrics (day, week, month, year)
        
    Returns:
        ChildMetricsResponse with various metrics
        
    Raises:
        ValueError: If time_range is invalid
        HTTPException: If database operations fail
    """
    with metrics_client.timer('child_metrics_duration'):
        try:
            # Validate child exists
            child_query = select(Child).filter(Child.id == child_id)
            child = await db.scalar(child_query)
            if not child:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Child {child_id} not found"
                )

            # Get time range
            start_date, end_date = await _get_time_range(time_range)
            
            # Fetch metrics concurrently
            async with asyncio.TaskGroup() as tg:
                content_task = tg.create_task(
                    _fetch_content_metrics(db, child_id, start_date, end_date)
                )
                hours_task = tg.create_task(
                    _fetch_hours_metrics(db, child_id, start_date, end_date)
                )
                queries_task = tg.create_task(
                    _fetch_queries_metrics(db, child_id, start_date, end_date)
                )
                engagement_task = tg.create_task(
                    _fetch_engagement_metrics(db, child_id, start_date, end_date)
                )

            # Get results
            most_used_content = await content_task
            active_hours = await hours_task
            common_queries = await queries_task
            engagement_trend = await engagement_task
            
            # Get emotion heatmap if available
            emotion_heatmap = None  # Implement if emotion tracking is added

            metrics_client.increment('child_metrics_success')
            return ChildMetricsResponse(
                most_used_content=most_used_content,
                active_hours=active_hours,
                common_queries=common_queries,
                engagement_trend=engagement_trend,
                emotion_heatmap=emotion_heatmap
            )

        except ValueError as e:
            metrics_client.increment('child_metrics_validation_error')
            logger.warning(f"Validation error in get_child_metrics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except MetricsError as e:
            metrics_client.increment('child_metrics_db_error')
            logger.error(f"Metrics error in get_child_metrics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch metrics"
            )
        except Exception as e:
            metrics_client.increment('child_metrics_unknown_error')
            logger.exception(f"Unexpected error in get_child_metrics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )

async def get_group_metrics(
    db: AsyncSession,
    group_id: str,
    time_range: str = "week"
) -> GroupMetricsResponse:
    """Get dashboard metrics for a specific group."""
    # Similar to get_child_metrics but aggregates data for all children in group
    end_date = datetime.utcnow()
    if time_range == "day":
        start_date = end_date - timedelta(days=1)
    elif time_range == "week":
        start_date = end_date - timedelta(weeks=1)
    elif time_range == "month":
        start_date = end_date - timedelta(days=30)
    else:  # year
        start_date = end_date - timedelta(days=365)

    # Get children in group
    children_query = select(Child.id).filter(Child.group_id == group_id)
    result = await db.execute(children_query)
    child_ids = [row.id for row in result.all()]

    # Get most used content for group
    content_query = select(
        Content.id,
        Content.title,
        func.count(ChatMessage.id).label("usage_count")
    ).join(
        ChatMessage, ChatMessage.content_id == Content.id
    ).filter(
        and_(
            ChatMessage.created_at >= start_date,
            ChatMessage.created_at <= end_date,
            ChatMessage.child_id.in_(child_ids)
        )
    ).group_by(
        Content.id
    ).order_by(
        desc("usage_count")
    ).limit(5)

    result = await db.execute(content_query)
    most_used_content = [
        {"id": row.id, "title": row.title, "usage_count": row.usage_count}
        for row in result.all()
    ]

    # Get active hours for group
    hours_query = select(
        func.extract("hour", ChatMessage.created_at).label("hour"),
        func.count().label("count")
    ).filter(
        and_(
            ChatMessage.created_at >= start_date,
            ChatMessage.created_at <= end_date,
            ChatMessage.child_id.in_(child_ids)
        )
    ).group_by(
        "hour"
    )

    result = await db.execute(hours_query)
    active_hours = {str(row.hour): float(row.count) for row in result.all()}

    # Get common queries for group
    queries_query = select(
        ChatMessage.content.label("query"),
        func.count().label("count")
    ).filter(
        and_(
            ChatMessage.created_at >= start_date,
            ChatMessage.created_at <= end_date,
            ChatMessage.child_id.in_(child_ids),
            ChatMessage.is_user == True
        )
    ).group_by(
        ChatMessage.content
    ).order_by(
        desc("count")
    ).limit(10)

    result = await db.execute(queries_query)
    common_queries = [
        {"query": row.query, "count": str(row.count)}
        for row in result.all()
    ]

    # Get engagement trend for group
    engagement_query = select(
        func.date_trunc("day", ChatMessage.created_at).label("date"),
        func.count().label("count")
    ).filter(
        and_(
            ChatMessage.created_at >= start_date,
            ChatMessage.created_at <= end_date,
            ChatMessage.child_id.in_(child_ids)
        )
    ).group_by(
        "date"
    ).order_by(
        "date"
    )

    result = await db.execute(engagement_query)
    engagement_trend = [
        {"date": row.date.isoformat(), "count": row.count}
        for row in result.all()
    ]

    # Get emotion heatmap if available
    emotion_heatmap = None  # Implement if emotion tracking is added

    return GroupMetricsResponse(
        most_used_content=most_used_content,
        active_hours=active_hours,
        common_queries=common_queries,
        engagement_trend=engagement_trend,
        emotion_heatmap=emotion_heatmap
    )

async def get_events(
    db: AsyncSession,
    user_id: str,
    event_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> EventsResponse:
    """Get dashboard events with optional filters."""
    with metrics_client.timer('get_events_duration'):
        try:
            # Validate limit
            if limit > 100:
                raise ValueError("Limit cannot exceed 100")
            
            # Build base query
            query = select(Event)
            
            # Apply filters
            if event_type:
                query = query.filter(Event.event_type == event_type)
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total = await db.scalar(count_query)
            
            # Apply pagination
            query = query.order_by(desc(Event.timestamp)).offset(offset).limit(limit)
            
            # Execute query with retry
            for attempt in range(MAX_RETRIES):
                try:
                    result = await db.execute(query)
                    events = result.scalars().all()
                    break
                except SQLAlchemyError as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(0.1 * (attempt + 1))
            
            metrics_client.increment('get_events_success')
            return EventsResponse(events=events, total=total or 0)

        except ValueError as e:
            metrics_client.increment('get_events_validation_error')
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except SQLAlchemyError as e:
            metrics_client.increment('get_events_db_error')
            logger.error(f"Database error in get_events: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch events"
            )
        except Exception as e:
            metrics_client.increment('get_events_unknown_error')
            logger.exception(f"Unexpected error in get_events: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )

async def get_notification_settings(
    db: AsyncSession,
    user_id: str
) -> NotificationSettings:
    """Get user notification settings."""
    query = select(NotificationSetting).filter(NotificationSetting.user_id == user_id)
    result = await db.execute(query)
    settings = result.scalar_one_or_none()
    
    if not settings:
        # Create default settings if none exist
        settings = NotificationSetting(
            user_id=user_id,
            email_on_upload_complete=True,
            email_on_flags=True,
            sms_alerts=False
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    
    return NotificationSettings.model_validate(settings)

async def update_notification_settings(
    db: AsyncSession,
    user_id: str,
    obj_in: NotificationSettingsUpdate
) -> NotificationSettings:
    """Update user notification settings."""
    query = select(NotificationSetting).filter(NotificationSetting.user_id == user_id)
    result = await db.execute(query)
    settings = result.scalar_one_or_none()
    
    if not settings:
        settings = NotificationSetting(user_id=user_id)
        db.add(settings)
    
    # Update fields
    for field, value in obj_in.model_dump().items():
        setattr(settings, field, value)
    
    settings.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(settings)
    
    return NotificationSettings.model_validate(settings) 