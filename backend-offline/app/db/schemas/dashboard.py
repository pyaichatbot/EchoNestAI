from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class DashboardMetricsBase(BaseModel):
    """Base dashboard metrics schema."""
    time_range: str
    start_date: datetime
    end_date: datetime

class ChildMetricsResponse(DashboardMetricsBase):
    """Child metrics response schema."""
    child_id: str
    total_sessions: int
    total_questions: int
    total_content_consumed: int
    content_by_type: Dict[str, int]
    session_duration_avg: float
    topics: List[str]
    daily_activity: List[Dict[str, Any]]

class GroupMetricsResponse(DashboardMetricsBase):
    """Group metrics response schema."""
    group_id: str
    total_children: int
    active_children: int
    total_sessions: int
    total_questions: int
    total_content_consumed: int
    content_by_type: Dict[str, int]
    session_duration_avg: float
    topics: List[str]
    daily_activity: List[Dict[str, Any]]

class EventsResponse(BaseModel):
    """Events response schema."""
    events: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int

class NotificationSettings(BaseModel):
    """Notification settings schema."""
    id: str
    user_id: str
    email_notifications: bool
    push_notifications: bool
    session_alerts: bool
    content_updates: bool
    sync_notifications: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class NotificationSettingsUpdate(BaseModel):
    """Notification settings update schema."""
    email_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    session_alerts: Optional[bool] = None
    content_updates: Optional[bool] = None
    sync_notifications: Optional[bool] = None
