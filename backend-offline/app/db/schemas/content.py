from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union, Dict, Any
from datetime import datetime
import enum

class ContentType(str, enum.Enum):
    """Content type enumeration."""
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"

class ContentBase(BaseModel):
    """Base content schema."""
    title: str
    type: ContentType
    language: str = "en"
    sync_offline: bool = False

class ContentCreate(ContentBase):
    """Content creation schema."""
    assign_to: List[str] = []

class ContentUpdate(BaseModel):
    """Content update schema."""
    title: Optional[str] = None
    sync_offline: Optional[bool] = None
    archived: Optional[bool] = None
    expires_at: Optional[datetime] = None

class Content(ContentBase):
    """Content schema."""
    id: str
    file_path: str
    size_mb: float
    archived: bool
    version: int
    created_by: str
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class ContentAssignmentBase(BaseModel):
    """Base content assignment schema."""
    content_id: str
    child_id: Optional[str] = None
    group_id: Optional[str] = None

    @validator('child_id', 'group_id')
    def validate_assignment(cls, v, values):
        """Validate that either child_id or group_id is provided, but not both."""
        if 'child_id' in values and values['child_id'] and 'group_id' in values and values['group_id']:
            raise ValueError('Cannot assign to both child and group')
        if ('child_id' not in values or not values['child_id']) and ('group_id' not in values or not values['group_id']):
            raise ValueError('Must assign to either child or group')
        return v

class ContentAssignmentCreate(ContentAssignmentBase):
    """Content assignment creation schema."""
    pass

class ContentAssignment(ContentAssignmentBase):
    """Content assignment schema."""
    id: str
    created_at: datetime

    class Config:
        orm_mode = True

class Child(BaseModel):
    """Child schema."""
    id: str
    name: str
    age: Optional[int] = None
    grade: Optional[str] = None
    parent_id: str
    language: str = "en"
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class Group(BaseModel):
    """Group schema."""
    id: str
    name: str
    description: Optional[str] = None
    teacher_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
