from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.schemas.dashboard import (
    DashboardMetricsBase, ChildMetricsResponse, GroupMetricsResponse,
    EventsResponse, NotificationSettings, NotificationSettingsUpdate
)
from app.api.deps.auth import get_current_active_user
from app.services.metrics_service import (
    get_child_metrics, get_group_metrics, get_events,
    update_notification_settings, get_notification_settings
)

router = APIRouter(tags=["dashboard"])

@router.get("/dashboard/metrics/child/{child_id}", response_model=ChildMetricsResponse)
async def get_child_dashboard_metrics(
    child_id: str,
    time_range: str = "week",
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get dashboard metrics for a child.
    """
    # Implementation would check if current user has access to this child
    return await get_child_metrics(db, child_id=child_id, time_range=time_range)

@router.get("/dashboard/metrics/group/{group_id}", response_model=GroupMetricsResponse)
async def get_group_dashboard_metrics(
    group_id: str,
    time_range: str = "week",
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get dashboard metrics for a group.
    """
    # Implementation would check if current user has access to this group
    return await get_group_metrics(db, group_id=group_id, time_range=time_range)

@router.get("/dashboard/events", response_model=EventsResponse)
async def get_dashboard_events(
    event_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get dashboard events with optional filters.
    """
    return await get_events(
        db, 
        user_id=current_user.id, 
        event_type=event_type, 
        limit=limit, 
        offset=offset
    )

@router.get("/dashboard/notifications/settings", response_model=NotificationSettings)
async def get_user_notification_settings(
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get user notification settings.
    """
    return await get_notification_settings(db, user_id=current_user.id)

@router.patch("/dashboard/notifications/settings", response_model=NotificationSettings)
async def update_user_notification_settings(
    settings_update: NotificationSettingsUpdate,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update user notification settings.
    """
    return await update_notification_settings(db, user_id=current_user.id, obj_in=settings_update)
