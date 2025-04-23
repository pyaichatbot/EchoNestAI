from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union, Dict, Any
from datetime import datetime
import enum

class ChatRequest(BaseModel):
    """Chat request schema."""
    input: str
    session_id: Optional[str] = None
    child_id: Optional[str] = None
    group_id: Optional[str] = None
    document_scope: Optional[List[str]] = None
    language: str = "en"

class ChatResponse(BaseModel):
    """Chat response schema."""
    response: str
    source_documents: List[str] = []
    confidence: float
    used_docs: int = 0

class ChatSession(BaseModel):
    """Chat session schema."""
    id: str
    child_id: Optional[str] = None
    group_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ChatMessage(BaseModel):
    """Chat message schema."""
    id: str
    session_id: str
    is_user: bool
    content: str
    source_documents: Optional[str] = None
    confidence: Optional[float] = None
    created_at: datetime

    class Config:
        orm_mode = True

class ChatMessageCreate(BaseModel):
    """Chat message creation schema."""
    session_id: str
    is_user: bool
    content: str
    source_documents: Optional[str] = None
    confidence: Optional[float] = None

class ChatFeedback(BaseModel):
    """Chat feedback schema."""
    message_id: str
    rating: Optional[int] = None
    comment: Optional[str] = None
    flagged: bool = False
    flag_reason: Optional[str] = None

class ChatFeedbackCreate(ChatFeedback):
    """Chat feedback creation schema."""
    pass

class SupportedLanguage(BaseModel):
    """Supported language schema."""
    code: str
    name: str
    supported_features: List[str]
    voice_model: str
    embedding_model: str

class LanguageDetectionRequest(BaseModel):
    """Language detection request schema."""
    text: str

class LanguageDetectionResponse(BaseModel):
    """Language detection response schema."""
    detected_language: str
    confidence: float
    supported: bool

class VoiceChatRequest(BaseModel):
    """Voice chat request schema."""
    audio_data: bytes
    session_id: Optional[str] = None
    child_id: Optional[str] = None
    group_id: Optional[str] = None
    language: Optional[str] = None
    document_scope: Optional[List[str]] = None
