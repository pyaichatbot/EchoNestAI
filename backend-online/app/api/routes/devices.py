from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.schemas.device import (
    Device, DeviceCreate, DeviceUpdate, DeviceSyncStatus,
    SyncManifestRequest, SyncManifestResponse, OTAManifestResponse
)
from app.api.deps.auth import get_current_active_user, get_device_auth
from app.db.crud.device import (
    create_device, get_device, get_devices,
    update_device, update_device_sync_status
)
from app.services.sync_service import generate_sync_manifest
from app.services.ota_service import get_ota_update

router = APIRouter(tags=["devices"])

@router.post("/devices/register", response_model=Device, status_code=status.HTTP_201_CREATED)
async def register_device(
    device_in: DeviceCreate,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Register a new device.
    """
    # Check if device already exists
    existing_device = await get_device(db, device_id=device_in.device_id)
    if existing_device:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Device already registered"
        )
    
    return await create_device(db, obj_in=device_in)

@router.get("/devices/list", response_model=List[Device])
async def list_devices(
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get list of devices with optional filters.
    """
    return await get_devices(
        db, 
        child_id=child_id, 
        group_id=group_id
    )

@router.get("/devices/{device_id}", response_model=Device)
async def get_device_by_id(
    device_id: str,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get device by ID.
    """
    device = await get_device(db, device_id=device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    return device

@router.patch("/devices/{device_id}", response_model=Device)
async def update_device_info(
    device_id: str,
    device_update: DeviceUpdate,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update device information.
    """
    device = await get_device(db, device_id=device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    return await update_device(db, db_obj=device, obj_in=device_update)

@router.post("/devices/sync/manifest", response_model=SyncManifestResponse)
async def get_sync_manifest(
    manifest_request: SyncManifestRequest,
    device: Any = Depends(get_device_auth),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get content sync manifest for device.
    """
    return await generate_sync_manifest(
        db,
        device_id=device.device_id,
        child_id=manifest_request.child_id,
        group_id=manifest_request.group_id,
        mode=manifest_request.mode,
        current_cache_size=manifest_request.current_cache_size,
        current_doc_ids=manifest_request.current_doc_ids
    )

@router.post("/devices/sync/status", response_model=DeviceSyncStatus)
async def update_sync_status(
    sync_status: DeviceSyncStatus,
    device: Any = Depends(get_device_auth),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update device sync status.
    """
    return await update_device_sync_status(db, device_id=device.id, obj_in=sync_status)

@router.get("/devices/ota/check", response_model=OTAManifestResponse)
async def check_ota_update(
    current_version: str,
    device: Any = Depends(get_device_auth),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Check for OTA updates.
    """
    return await get_ota_update(db, current_version=current_version)
