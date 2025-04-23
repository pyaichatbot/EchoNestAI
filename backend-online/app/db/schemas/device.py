from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.db.models.models import DeviceMode

class DeviceBase(BaseModel):
    device_id: str
    firmware_version: str
    mode: DeviceMode
    child_id: Optional[str] = None
    group_id: Optional[str] = None

class DeviceCreate(DeviceBase):
    pass

class DeviceUpdate(BaseModel):
    firmware_version: Optional[str] = None
    mode: Optional[DeviceMode] = None
    child_id: Optional[str] = None
    group_id: Optional[str] = None

class DeviceInDB(DeviceBase):
    id: str
    last_sync: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True

class Device(DeviceInDB):
    pass

class DeviceSyncStatusBase(BaseModel):
    device_id: str
    used_mb: int
    doc_count: int
    battery_status: Optional[str] = None
    connectivity: Optional[str] = None
    last_error: Optional[str] = None

class DeviceSyncStatusCreate(DeviceSyncStatusBase):
    pass

class DeviceSyncStatusUpdate(BaseModel):
    used_mb: Optional[int] = None
    doc_count: Optional[int] = None
    battery_status: Optional[str] = None
    connectivity: Optional[str] = None
    last_error: Optional[str] = None

class DeviceSyncStatusInDB(DeviceSyncStatusBase):
    id: str
    updated_at: datetime

    class Config:
        from_attributes = True

class DeviceSyncStatus(DeviceSyncStatusInDB):
    pass

class DeviceDiagnostics(BaseModel):
    cpu_load: float
    memory_used_mb: float
    disk_used_mb: float
    sync_status: str
    last_error: Optional[str] = None
    battery_state: Optional[str] = None
    connectivity: str
    cached_documents: List[str]

class SyncManifestRequest(BaseModel):
    group_id: Optional[str] = None
    child_id: Optional[str] = None
    mode: DeviceMode
    current_cache_size: float
    current_doc_ids: List[str] = []

class EmbeddingMeta(BaseModel):
    chunk_count: int
    embedding_dim: int

class AssignedTo(BaseModel):
    group_ids: List[str] = []
    child_ids: List[str] = []

class DocumentInfo(BaseModel):
    id: str
    title: str
    type: str
    language: str
    size_mb: float
    version: int
    priority: bool
    url: str
    embedding_meta: EmbeddingMeta
    assigned_to: AssignedTo
    created_at: datetime
    expires_at: Optional[datetime] = None

class SyncProfile(BaseModel):
    mode: DeviceMode
    group_id: Optional[str] = None
    child_id: Optional[str] = None
    sync_limit_mb: float
    max_documents: int

class SyncManifestResponse(BaseModel):
    device_id: str
    sync_profile: SyncProfile
    documents: List[DocumentInfo]
    sync_required: bool
    timestamp: datetime

class OTAUpdateBase(BaseModel):
    version: str
    url: str
    size_mb: float
    sha256: str
    release_notes: Optional[str] = None
    is_active: bool = True

class OTAUpdateCreate(OTAUpdateBase):
    pass

class OTAUpdateInDB(OTAUpdateBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class OTAUpdate(OTAUpdateInDB):
    pass

class OTAManifestResponse(BaseModel):
    available: bool
    version: Optional[str] = None
    url: Optional[str] = None
    size_mb: Optional[float] = None
    sha256: Optional[str] = None
