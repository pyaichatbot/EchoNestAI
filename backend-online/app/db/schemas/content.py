from pydantic import BaseModel, Field, constr
from typing import Optional, List, Dict, Any, Annotated, Literal
from datetime import datetime
from app.db.models.models import ContentType

class ChildBase(BaseModel):
    name: str
    age: int
    language: Optional[str] = "en"
    avatar: Optional[str] = None

class ChildCreate(ChildBase):
    pass

class ChildUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    language: Optional[str] = None
    avatar: Optional[str] = None

class ChildInDB(ChildBase):
    id: str
    parent_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class Child(ChildInDB):
    pass

class GroupBase(BaseModel):
    name: str
    age_range: Optional[str] = None
    location: Optional[str] = None

class GroupCreate(GroupBase):
    pass

class GroupUpdate(BaseModel):
    name: Optional[str] = None
    age_range: Optional[str] = None
    location: Optional[str] = None

class GroupInDB(GroupBase):
    id: str
    teacher_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class Group(GroupInDB):
    pass

class ContentBase(BaseModel):
    title: str
    type: ContentType
    language: Optional[str] = "en"
    sync_offline: Optional[bool] = False
    status: str = "pending"

class ContentCreate(ContentBase):
    assign_to: List[str] = []  # List of child_id or group_id

class ContentUpdate(BaseModel):
    title: Optional[str] = None
    sync_offline: Optional[bool] = None
    archived: Optional[bool] = None
    expires_at: Optional[datetime] = None

class ContentInDB(ContentBase):
    id: str
    file_path: str
    size_mb: float
    archived: bool
    version: int
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class Content(ContentInDB):
    pass

class ContentAssignmentBase(BaseModel):
    content_id: str
    child_id: Optional[str] = None
    group_id: Optional[str] = None

class ContentAssignmentCreate(ContentAssignmentBase):
    pass

class ContentAssignmentInDB(ContentAssignmentBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class ContentAssignment(ContentAssignmentInDB):
    pass

class ContentEmbeddingBase(BaseModel):
    content_id: str
    chunk_index: int
    chunk_text: str
    embedding_vector_id: str

class ContentEmbeddingCreate(ContentEmbeddingBase):
    pass

class ContentEmbeddingInDB(ContentEmbeddingBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class ContentEmbedding(ContentEmbeddingInDB):
    pass

class RelevantDocument(BaseModel):
    """Schema for relevant document contexts returned from search"""
    document_id: str = Field(..., description="ID of the source document")
    title: str = Field(..., description="Title of the document")
    content_type: str = Field(..., description="Type of the document (document, audio, video)")
    context: str = Field(..., description="Relevant text context from the document")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score between 0 and 1")
    page_number: Optional[int] = Field(None, description="Page number for document content")
    timestamp: Optional[float] = Field(None, description="Timestamp for audio/video content in seconds")
    language: str = Field(..., description="Language of the document content")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata about the document context")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc123",
                "title": "Introduction to AI",
                "content_type": "document",
                "context": "Artificial Intelligence (AI) is the simulation of human intelligence by machines...",
                "relevance_score": 0.95,
                "page_number": 1,
                "timestamp": None,
                "language": "en",
                "metadata": {
                    "section": "Chapter 1",
                    "author": "John Doe"
                }
            }
        }

class RelevantDocumentItem(BaseModel):
    """Schema for a single relevant document in the search results"""
    id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Relevant content excerpt")
    relevance_score: float = Field(..., ge=0, le=1, description="Similarity/relevance score (0-1)")
    language: str = Field(..., description="Document language")
    type: ContentType = Field(..., description="Document type (document/audio/video)")

class RelevantDocumentsRequest(BaseModel):
    """Schema for relevant documents search request"""
    query: Annotated[str, Field(min_length=1, max_length=1000, description="The user's search query text")]
    document_ids: Optional[List[str]] = Field(None, description="Optional array of specific document IDs to search within")
    language: str = Field("en", min_length=2, max_length=5, description="The language code for the search")
    limit: int = Field(5, ge=1, le=20, description="Maximum number of documents to return")

class RelevantDocumentsResponse(BaseModel):
    """Schema for relevant documents search response"""
    documents: List[RelevantDocumentItem] = Field(..., description="Array of relevant documents")
    total_found: int = Field(..., description="Total number of matching documents")

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "id": "doc123",
                        "title": "Introduction to AI",
                        "content": "Artificial Intelligence (AI) is the simulation of human intelligence by machines...",
                        "relevance_score": 0.95,
                        "language": "en",
                        "type": "document"
                    }
                ],
                "total_found": 1
            }
        }

class GroupActivityBase(BaseModel):
    group_id: str
    activity_type: str
    timestamp: datetime
    duration: int
    participants: int
    completion_rate: float
    description: Optional[str] = None

class GroupActivityCreate(GroupActivityBase):
    pass

class GroupActivityUpdate(BaseModel):
    activity_type: Optional[str] = None
    timestamp: Optional[datetime] = None
    duration: Optional[int] = None
    participants: Optional[int] = None
    completion_rate: Optional[float] = None
    description: Optional[str] = None

class GroupActivityInDB(GroupActivityBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class GroupActivity(GroupActivityInDB):
    pass

class ProgressByCategory(BaseModel):
    category: str
    progress: float

class LearningProgress(BaseModel):
    id: str
    user_id: str
    content_id: str
    progress: float
    score: Optional[float] = None
    time_spent: int
    last_accessed: int
    completed_sections: List[str]
    mastery_level: Optional[Literal['beginner', 'intermediate', 'advanced', 'expert']] = None

class LearningProgressSummary(BaseModel):
    average_progress: float
    completion_rate: float
    time_spent: int
    strengths: List[str]
    areas_for_improvement: List[str]
    recent_progress: List[LearningProgress]
    progress_by_category: List[ProgressByCategory]

class DashboardMetric(BaseModel):
    title: str
    value: int
    description: str
    change: Optional[Dict[str, Any]] = None

class ActivityItem(BaseModel):
    id: str
    text: str
    timestamp: str
    type: str
    status: str

class SystemStatus(BaseModel):
    label: str
    value: str
    status: str

class TeacherDashboardOverview(BaseModel):
    metrics: List[DashboardMetric]
    recent_activity: List[ActivityItem]
    system_status: List[SystemStatus]
