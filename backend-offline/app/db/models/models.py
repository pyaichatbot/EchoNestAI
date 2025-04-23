from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Enum, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
import uuid
from datetime import datetime
from typing import Optional, List

Base = declarative_base()

def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())

class UserRole(str, enum.Enum):
    """User role enumeration."""
    ADMIN = "admin"
    PARENT = "parent"
    TEACHER = "teacher"

class ContentType(str, enum.Enum):
    """Content type enumeration."""
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"

class DeviceMode(str, enum.Enum):
    """Device mode enumeration."""
    HOME = "home"
    SCHOOL = "school"
    HYBRID = "hybrid"

class User(Base):
    """User model for authentication and profile information."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default=UserRole.PARENT)
    language = Column(String, default="en")
    timezone = Column(String, default="UTC")
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Child(Base):
    """Child profile model."""
    __tablename__ = "children"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    age = Column(Integer)
    grade = Column(String)
    parent_id = Column(String, ForeignKey("users.id"))
    language = Column(String, default="en")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Group(Base):
    """Group model for classrooms or other child groups."""
    __tablename__ = "groups"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(String)
    teacher_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class GroupMembership(Base):
    """Group membership model for child-group relationships."""
    __tablename__ = "group_memberships"

    id = Column(String, primary_key=True, default=generate_uuid)
    child_id = Column(String, ForeignKey("children.id"))
    group_id = Column(String, ForeignKey("groups.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

class Content(Base):
    """Content model for educational materials."""
    __tablename__ = "content"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    type = Column(String, nullable=False)
    language = Column(String, default="en")
    file_path = Column(String, nullable=False)
    size_mb = Column(Float, default=0.0)
    sync_offline = Column(Boolean, default=False)
    archived = Column(Boolean, default=False)
    version = Column(Integer, default=1)
    created_by = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

class ContentAssignment(Base):
    """Content assignment model for content-child/group relationships."""
    __tablename__ = "content_assignments"

    id = Column(String, primary_key=True, default=generate_uuid)
    content_id = Column(String, ForeignKey("content.id"))
    child_id = Column(String, ForeignKey("children.id"), nullable=True)
    group_id = Column(String, ForeignKey("groups.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ContentEmbedding(Base):
    """Content embedding model for vector search."""
    __tablename__ = "content_embeddings"

    id = Column(String, primary_key=True, default=generate_uuid)
    content_id = Column(String, ForeignKey("content.id"))
    chunk_index = Column(Integer)
    chunk_text = Column(Text)
    embedding_vector = Column(Text)  # Stored as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

class Device(Base):
    """Device model for EchoNest hardware devices."""
    __tablename__ = "devices"

    id = Column(String, primary_key=True, default=generate_uuid)
    device_id = Column(String, unique=True, index=True, nullable=False)
    firmware_version = Column(String)
    mode = Column(String, default=DeviceMode.HOME)
    child_id = Column(String, ForeignKey("children.id"), nullable=True)
    group_id = Column(String, ForeignKey("groups.id"), nullable=True)
    last_seen = Column(DateTime, default=datetime.utcnow)
    last_sync = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DeviceSyncStatus(Base):
    """Device sync status model."""
    __tablename__ = "device_sync_status"

    id = Column(String, primary_key=True, default=generate_uuid)
    device_id = Column(String, ForeignKey("devices.id"), unique=True)
    used_mb = Column(Float, default=0.0)
    doc_count = Column(Integer, default=0)
    last_sync_status = Column(String, default="success")
    last_sync_message = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatSession(Base):
    """Chat session model."""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    child_id = Column(String, ForeignKey("children.id"), nullable=True)
    group_id = Column(String, ForeignKey("groups.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatMessage(Base):
    """Chat message model."""
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    is_user = Column(Boolean, default=True)
    content = Column(Text, nullable=False)
    source_documents = Column(Text)  # Stored as JSON string
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatFeedback(Base):
    """Chat feedback model."""
    __tablename__ = "chat_feedback"

    id = Column(String, primary_key=True, default=generate_uuid)
    message_id = Column(String, ForeignKey("chat_messages.id"), unique=True)
    rating = Column(Integer)
    comment = Column(String)
    flagged = Column(Boolean, default=False)
    flag_reason = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class OTAUpdate(Base):
    """OTA update model."""
    __tablename__ = "ota_updates"

    id = Column(String, primary_key=True, default=generate_uuid)
    version = Column(String, nullable=False)
    url = Column(String, nullable=False)
    size_mb = Column(Float, default=0.0)
    sha256 = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SyncLog(Base):
    """Sync log model for tracking sync operations."""
    __tablename__ = "sync_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    device_id = Column(String, ForeignKey("devices.id"))
    sync_type = Column(String)  # "pull" or "push"
    status = Column(String)
    message = Column(String)
    details = Column(Text)  # Stored as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
