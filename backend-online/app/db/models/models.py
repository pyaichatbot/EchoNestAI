from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text, Enum, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
import uuid
from datetime import datetime, timedelta
import json
from typing import Dict, Any
from sqlalchemy.dialects.postgresql import JSONB

from app.db.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class UserRole(str, enum.Enum):
    PARENT = "parent"
    TEACHER = "teacher"
    ADMIN = "admin"

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    language = Column(String, default="en")
    timezone = Column(String, default="UTC")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    children = relationship("Child", back_populates="parent")
    groups = relationship("Group", back_populates="teacher")
    password_reset_tokens = relationship("PasswordResetToken", back_populates="user", cascade="all, delete-orphan")
    email_verification_tokens = relationship("EmailVerificationToken", back_populates="user", cascade="all, delete-orphan")
    notification_settings = relationship(
        "NotificationSetting", 
        back_populates="user", 
        uselist=False,  # One-to-one relationship
        cascade="all, delete-orphan"
    )

class Child(Base):
    __tablename__ = "children"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    language = Column(String, default="en")
    avatar = Column(String)
    parent_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    parent = relationship("User", back_populates="children")
    devices = relationship("Device", back_populates="child")
    content_assignments = relationship("ContentAssignment", back_populates="child")
    
class Group(Base):
    __tablename__ = "groups"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    age_range = Column(String)
    location = Column(String)
    teacher_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    teacher = relationship("User", back_populates="groups")
    devices = relationship("Device", back_populates="group")
    content_assignments = relationship("ContentAssignment", back_populates="group")
    activities = relationship("GroupActivity", back_populates="group", cascade="all, delete-orphan")

class ContentType(str, enum.Enum):
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"

class Content(Base):
    __tablename__ = "content"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    type = Column(Enum(ContentType), nullable=False)
    language = Column(String, default="en")
    file_path = Column(String, nullable=False)
    size_mb = Column(Integer, nullable=False)
    sync_offline = Column(Boolean, default=False)
    archived = Column(Boolean, default=False)
    status = Column(String, default="pending", nullable=False)
    version = Column(Integer, default=1)
    created_by = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    assignments = relationship("ContentAssignment", back_populates="content")
    embeddings = relationship("ContentEmbedding", back_populates="content")

class ContentAssignment(Base):
    __tablename__ = "content_assignments"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    content_id = Column(String, ForeignKey("content.id"), nullable=False)
    child_id = Column(String, ForeignKey("children.id"), nullable=True)
    group_id = Column(String, ForeignKey("groups.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    content = relationship("Content", back_populates="assignments")
    child = relationship("Child", back_populates="content_assignments")
    group = relationship("Group", back_populates="content_assignments")

class ContentEmbedding(Base):
    __tablename__ = "content_embeddings"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    content_id = Column(String, ForeignKey("content.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding_vector_id = Column(String, nullable=False)  # ID in vector DB
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    content = relationship("Content", back_populates="embeddings")

class DeviceMode(str, enum.Enum):
    HOME = "home"
    SCHOOL = "school"

class Device(Base):
    __tablename__ = "devices"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    device_id = Column(String, unique=True, nullable=False)
    firmware_version = Column(String, nullable=False)
    mode = Column(Enum(DeviceMode), nullable=False)
    child_id = Column(String, ForeignKey("children.id"), nullable=True)
    group_id = Column(String, ForeignKey("groups.id"), nullable=True)
    last_sync = Column(DateTime(timezone=True), nullable=True)
    last_seen = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    child = relationship("Child", back_populates="devices")
    group = relationship("Group", back_populates="devices")
    sync_status = relationship("DeviceSyncStatus", back_populates="device", uselist=False)

class DeviceSyncStatus(Base):
    __tablename__ = "device_sync_status"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    device_id = Column(String, ForeignKey("devices.id"), unique=True, nullable=False)
    used_mb = Column(Integer, default=0)
    doc_count = Column(Integer, default=0)
    battery_status = Column(String, nullable=True)
    connectivity = Column(String, nullable=True)
    last_error = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    device = relationship("Device", back_populates="sync_status")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    child_id = Column(String, ForeignKey("children.id"), nullable=True)
    group_id = Column(String, ForeignKey("groups.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    is_user = Column(Boolean, default=True)
    content = Column(Text, nullable=False)
    source_documents = Column(Text, nullable=True)  # JSON list of document IDs
    confidence = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    feedback = relationship("ChatFeedback", back_populates="message", uselist=False)

class ChatFeedback(Base):
    __tablename__ = "chat_feedback"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    message_id = Column(String, ForeignKey("chat_messages.id"), unique=True, nullable=False)
    rating = Column(Integer, nullable=True)
    comment = Column(Text, nullable=True)
    flagged = Column(Boolean, default=False)
    flag_reason = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    tags = Column(JSONB, default=list)
    response_quality = Column(String, nullable=True)
    
    # Relationships
    message = relationship("ChatMessage", back_populates="feedback")

class OTAUpdate(Base):
    __tablename__ = "ota_updates"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    version = Column(String, nullable=False, unique=True)
    url = Column(String, nullable=False)
    size_mb = Column(Integer, nullable=False)
    sha256 = Column(String, nullable=False)
    release_notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class EmailVerificationToken(Base):
    __tablename__ = "email_verification_tokens"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="password_reset_tokens")

class NotificationSetting(Base):
    __tablename__ = "notification_settings"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)
    email_on_upload_complete = Column(Boolean, default=True)
    email_on_flags = Column(Boolean, default=True)
    sms_alerts = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="notification_settings")

class EventType(str, enum.Enum):
    UPLOAD_COMPLETE = "upload_complete"
    SYNC_FAILED = "sync_failed"
    FLAGGED_RESPONSE = "flagged_response"

class Event(Base):
    __tablename__ = "events"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    event_type = Column(Enum(EventType), nullable=False)
    target_id = Column(String, nullable=False)
    user_role = Column(Enum(UserRole), nullable=False)
    event_meta = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    def __init__(self, **kwargs):
        if "event_meta" in kwargs:
            meta = kwargs["event_meta"]
            if isinstance(meta, dict):
                kwargs["event_meta"] = json.dumps(meta)
            elif isinstance(meta, str):
                try:
                    # Validate that the string is valid JSON
                    json.loads(meta)
                    kwargs["event_meta"] = meta
                except json.JSONDecodeError:
                    raise ValueError("event_meta must be a valid JSON string or dictionary")
            else:
                raise ValueError("event_meta must be a dictionary or valid JSON string")
        super().__init__(**kwargs)

    @property
    def meta_dict(self) -> Dict[str, Any]:
        """
        Returns the event metadata as a dictionary.
        If event_meta is None or empty, returns an empty dict.
        Raises ValueError if event_meta is not valid JSON.
        """
        if not self.event_meta:
            return {}
        try:
            return json.loads(self.event_meta)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in event_meta")

    def update_meta(self, new_meta: Dict[str, Any]) -> None:
        """
        Updates the event metadata with new values.
        Merges with existing metadata if present.
        
        Args:
            new_meta: Dictionary containing new metadata values
        """
        current_meta = self.meta_dict
        current_meta.update(new_meta)
        self.event_meta = json.dumps(current_meta)

    def set_meta(self, meta: Dict[str, Any]) -> None:
        """
        Sets the event metadata, replacing any existing metadata.
        
        Args:
            meta: Dictionary containing metadata values
        """
        self.event_meta = json.dumps(meta)

class GroupActivity(Base):
    __tablename__ = "group_activities"

    id = Column(String, primary_key=True, default=generate_uuid)
    group_id = Column(String, ForeignKey("groups.id"), nullable=False)
    activity_type = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    duration = Column(Integer, nullable=False)
    participants = Column(Integer, nullable=False)
    completion_rate = Column(Float, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    group = relationship("Group", back_populates="activities")

