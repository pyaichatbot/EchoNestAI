from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from app.db.models.models import EventType

class NotificationBase(BaseModel):
    event_id: str
    read: bool = False

class NotificationCreate(NotificationBase):
    user_id: str

class Notification(NotificationBase):
    id: str
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True

class NotificationSettingsBase(BaseModel):
    email_on_upload_complete: bool = True
    email_on_flags: bool = True
    sms_alerts: bool = False

class NotificationSettingsUpdate(NotificationSettingsBase):
    pass

class NotificationSettings(NotificationSettingsBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class NotificationList(BaseModel):
    notifications: List[Notification]
    unread_count: int 