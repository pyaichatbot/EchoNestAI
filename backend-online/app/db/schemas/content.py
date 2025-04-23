from pydantic import BaseModel, Field
from typing import Optional, List
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
