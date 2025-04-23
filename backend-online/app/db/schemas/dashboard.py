from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class DashboardMetricsBase(BaseModel):
    child_id: Optional[str] = None
    group_id: Optional[str] = None
    time_range: str = "week"  # day, week, month, year

class ChildMetricsResponse(BaseModel):
    most_used_content: List[Dict[str, Any]]
    active_hours: Dict[str, float]
    common_queries: List[Dict[str, str]]
    engagement_trend: List[Dict[str, Any]]
    emotion_heatmap: Optional[Dict[str, Any]] = None

class GroupMetricsResponse(BaseModel):
    most_used_content: List[Dict[str, Any]]
    active_hours: Dict[str, float]
    common_queries: List[Dict[str, str]]
    engagement_trend: List[Dict[str, Any]]
    emotion_heatmap: Optional[Dict[str, Any]] = None

class EventBase(BaseModel):
    event_type: str  # upload_complete, sync_failed, flagged_response
    target_id: str
    user_role: str
    timestamp: datetime
    meta: Dict[str, Any] = {}

class EventCreate(EventBase):
    pass

class EventInDB(EventBase):
    id: str

    class Config:
        from_attributes = True

class Event(EventInDB):
    pass

class EventsResponse(BaseModel):
    events: List[Event]
    total: int

class NotificationSettingsBase(BaseModel):
    email_on_upload_complete: bool = True
    email_on_flags: bool = True
    sms_alerts: bool = False

class NotificationSettingsUpdate(NotificationSettingsBase):
    pass

class NotificationSettingsInDB(NotificationSettingsBase):
    id: str
    user_id: str
    updated_at: datetime

    class Config:
        from_attributes = True

class NotificationSettings(NotificationSettingsInDB):
    pass

class WebhookNotification(BaseModel):
    event: str  # upload_complete, sync_failed, flagged_response
    target_id: str
    user_role: str
    timestamp: datetime
    meta: Dict[str, Any] = {}
