from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.api import deps
from app.db.models.models import User, NotificationSetting, Event
from app.schemas.notification import NotificationList, NotificationSettings, NotificationSettingsUpdate

router = APIRouter()

@router.get("/events/poll", response_model=NotificationList)
def poll_notifications(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
    skip: int = 0,
    limit: int = 50,
) -> Any:
    """
    Poll for notifications.
    Returns a list of notifications and unread count for the current user.
    """
    notifications = crud.notification.get_user_notifications(
        db=db, user_id=current_user.id, skip=skip, limit=limit
    )
    unread_count = crud.notification.get_unread_count(db=db, user_id=current_user.id)
    
    return NotificationList(
        notifications=notifications,
        unread_count=unread_count
    )

@router.post("/notifications/{notification_id}/read")
def mark_notification_read(
    notification_id: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Mark a notification as read.
    """
    notification = crud.notification.mark_as_read(
        db=db, notification_id=notification_id, user_id=current_user.id
    )
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "success"}

@router.get("/settings/notifications", response_model=NotificationSettings)
def get_notification_settings(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Get notification settings for the current user.
    """
    settings = crud.notification_settings.get_by_user_id(db=db, user_id=current_user.id)
    if not settings:
        settings = crud.notification_settings.create(db=db, user_id=current_user.id)
    return settings

@router.patch("/settings/notifications", response_model=NotificationSettings)
def update_notification_settings(
    *,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
    settings_in: NotificationSettingsUpdate,
) -> Any:
    """
    Update notification settings for the current user.
    """
    settings = crud.notification_settings.get_by_user_id(db=db, user_id=current_user.id)
    if not settings:
        settings = crud.notification_settings.create(db=db, user_id=current_user.id)
    
    settings = crud.notification_settings.update(db=db, db_obj=settings, obj_in=settings_in)
    return settings 