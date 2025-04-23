from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
import uuid
from datetime import datetime

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
