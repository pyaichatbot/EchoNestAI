from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union, Dict, Any
from datetime import datetime
import enum

class DeviceMode(str, enum.Enum):
    """Device mode enumeration."""
    HOME = "home"
    SCHOOL = "school"
    HYBRID = "hybrid"

class DeviceBase(BaseModel):
    """Base device schema."""
    device_id: str
    firmware_version: str
    mode: DeviceMode = DeviceMode.HOME
    child_id: Optional[str] = None
    group_id: Optional[str] = None

    @validator('child_id', 'group_id')
    def validate_assignment(cls, v, values):
        """Validate that either child_id or group_id is provided, but not both."""
        if 'child_id' in values and values['child_id'] and 'group_id' in values and values['group_id']:
            raise ValueError('Cannot assign to both child and group')
        return v

class DeviceCreate(DeviceBase):
    """Device creation schema."""
    pass

class DeviceUpdate(BaseModel):
    """Device update schema."""
    firmware_version: Optional[str] = None
    mode: Optional[DeviceMode] = None
    child_id: Optional[str] = None
    group_id: Optional[str] = None

class Device(DeviceBase):
    """Device schema."""
    id: str
    last_seen: datetime
    last_sync: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class DeviceSyncStatusBase(BaseModel):
    """Base device sync status schema."""
    used_mb: float = 0.0
    doc_count: int = 0
    last_sync_status: str = "success"
    last_sync_message: Optional[str] = None

class DeviceSyncStatusCreate(DeviceSyncStatusBase):
    """Device sync status creation schema."""
    device_id: str

class DeviceSyncStatusUpdate(DeviceSyncStatusBase):
    """Device sync status update schema."""
    pass

class DeviceSyncStatus(DeviceSyncStatusBase):
    """Device sync status schema."""
    id: str
    device_id: str
    updated_at: datetime

    class Config:
        orm_mode = True

class DocumentInfo(BaseModel):
    """Document info schema for sync manifest."""
    id: str
    title: str
    type: str
    language: str
    size_mb: float
    version: int
    priority: bool
    url: str
    embedding_meta: Dict[str, Any]
    assigned_to: Dict[str, List[str]]
    created_at: datetime
    expires_at: Optional[datetime] = None

class EmbeddingMeta(BaseModel):
    """Embedding metadata schema."""
    chunk_count: int
    embedding_dim: int

class AssignedTo(BaseModel):
    """Assigned to schema."""
    child_ids: List[str] = []
    group_ids: List[str] = []

class SyncProfile(BaseModel):
    """Sync profile schema."""
    mode: str
    child_id: Optional[str] = None
    group_id: Optional[str] = None
    sync_limit_mb: float
    max_documents: int

class SyncManifestRequest(BaseModel):
    """Sync manifest request schema."""
    child_id: Optional[str] = None
    group_id: Optional[str] = None
    mode: str = "home"
    current_cache_size: float = 0.0
    current_doc_ids: List[str] = []

class SyncManifestResponse(BaseModel):
    """Sync manifest response schema."""
    device_id: str
    sync_profile: SyncProfile
    documents: List[DocumentInfo]
    sync_required: bool
    timestamp: datetime

class OTAManifestResponse(BaseModel):
    """OTA manifest response schema."""
    available: bool
    version: Optional[str] = None
    url: Optional[str] = None
    size_mb: Optional[float] = None
    sha256: Optional[str] = None
