from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatBase(BaseModel):
    input: str = Field(..., alias="input")
    child_id: Optional[str] = Field(None, alias="childId")
    group_id: Optional[str] = Field(None, alias="groupId")
    document_scope: Optional[List[str]] = Field(None, alias="documentScope")
    language: str = Field("en", alias="language")
    session_id: Optional[str] = Field(None, alias="sessionId")
    context: Optional[str] = Field(None, alias="context")  # Optional context from frontend

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True

class ChatRequest(ChatBase):
    pass

class VoiceChatRequest(ChatBase):
    """Request schema for voice chat interactions"""
    audio_format: str = Field(default="wav", description="Audio format of the input file")
    sample_rate: int = Field(default=16000, description="Sample rate of the audio in Hz")
    voice_id: Optional[str] = Field(default=None, description="Optional voice ID for response synthesis")
    return_audio: bool = Field(default=True, description="Whether to return synthesized audio response")

class ChatResponse(BaseModel):
    response: str
    source_documents: List[str] = []
    confidence: float
    used_docs: int

class ChatMessageBase(BaseModel):
    session_id: str
    is_user: bool = True
    content: str
    source_documents: Optional[str] = None
    confidence: Optional[float] = None

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessageInDB(ChatMessageBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatMessage(ChatMessageInDB):
    pass

class MessagePersistRequest(BaseModel):
    """Request schema for persisting messages via /chat/messages endpoint"""
    message_id: str = Field(..., description="Unique ID of the message")
    sender_id: str = Field(..., description="User ID of the sender")
    text: str = Field(..., description="Content of the message")
    timestamp: int = Field(..., description="Unix timestamp of the message")
    child_id: Optional[str] = Field(None, description="Child ID associated with the chat")
    group_id: Optional[str] = Field(None, description="Group ID associated with the chat")
    session_id: str = Field(..., description="Chat session ID")
    source_documents: Optional[List[str]] = Field(None, description="Referenced document IDs")
    confidence: Optional[float] = Field(None, description="AI confidence score")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "message_id": "msg_12345",
                "sender_id": "user_12345",
                "text": "Hello, how are you?",
                "timestamp": 1626982200000,
                "session_id": "session_12345",
                "source_documents": ["doc_1", "doc_2"],
                "confidence": 0.95
            }
        }

class ChatSessionBase(BaseModel):
    child_id: Optional[str] = None
    group_id: Optional[str] = None

class ChatSessionCreate(ChatSessionBase):
    pass

class ChatSessionInDB(ChatSessionBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatSession(ChatSessionInDB):
    messages: List[ChatMessage] = []

class ChatFeedbackBase(BaseModel):
    message_id: str
    rating: Optional[int] = None
    comment: Optional[str] = None
    flagged: bool = False
    flag_reason: Optional[str] = None

class ChatFeedbackCreate(ChatFeedbackBase):
    pass

class ChatFeedbackInDB(ChatFeedbackBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatFeedback(ChatFeedbackInDB):
    pass

class FeedbackRequest(BaseModel):
    input: str
    response: str
    child_id: Optional[str] = None
    group_id: Optional[str] = None
    rating: int
    comment: Optional[str] = None

class FlagRequest(BaseModel):
    content_id: str
    reason: str
    flagged_by: str
    notes: Optional[str] = None

class StreamToken(BaseModel):
    token: str
    final: bool = False

class SupportedLanguage(BaseModel):
    code: str
    name: str
    supported_features: List[str]
    voice_model: str
    embedding_model: str

class LanguageDetectionRequest(BaseModel):
    text: str

class LanguageDetectionResponse(BaseModel):
    detected_language: str
    confidence: float
    supported: bool
