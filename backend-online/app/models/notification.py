from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base
from app.models.models import User
from app.db.models.models import EventType

class NotificationType(PyEnum):
    UPLOAD_COMPLETE = "upload_complete"
    CONTENT_FLAGGED = "content_flagged"
    GROUP_ACTIVITY = "group_activity"
    SYSTEM_ALERT = "system_alert"

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    event = relationship("Event")

class NotificationSettings(Base):
    __tablename__ = "notification_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    email_on_upload_complete = Column(Boolean, default=True)
    email_on_flags = Column(Boolean, default=True)
    sms_alerts = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="notification_settings") 