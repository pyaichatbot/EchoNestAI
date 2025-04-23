from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.db.database import get_db
from app.db.schemas.device import (
    Device, DeviceCreate, DeviceUpdate, DeviceSyncStatus,
    DeviceSyncStatusUpdate, SyncManifestRequest, SyncManifestResponse,
    OTAManifestResponse
)
from app.api.deps.auth import get_current_active_user, get_current_admin_user, get_device_auth
from app.db.crud.device import (
    get_device, get_devices, create_device, update_device,
    update_device_last_seen, update_device_last_sync,
    get_device_sync_status, update_device_sync_status, delete_device
)
from app.db.crud.content import get_content_for_device
from app.core.security import create_device_token
from app.services.sync_service import generate_sync_manifest
from app.services.ota_service import check_for_updates

router = APIRouter(tags=["devices"])

@router.get("/", response_model=List[Device])
async def list_devices(
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get list of devices with optional filters.
    """
    devices = await get_devices(
        db, 
        child_id=child_id,
        group_id=group_id,
        skip=skip, 
        limit=limit
    )
    return devices

@router.post("/", response_model=Device)
async def register_device(
    device_in: DeviceCreate,
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Register a new device.
    """
    device = await create_device(db, obj_in=device_in)
    return device

@router.post("/token")
async def create_device_auth_token(
    device_id: str,
    current_user: Any = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Create authentication token for a device (admin only).
    """
    device = await get_device(db, device_id=device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    token = create_device_token(device.id)
    return {"token": token, "expires_in": 365 * 24 * 60 * 60}  # 1 year in seconds

@router.get("/{device_id}", response_model=Device)
async def get_device_by_id(
    device_id: str = Path(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get device by ID.
    """
    device = await get_device(db, id=device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    return device

@router.put("/{device_id}", response_model=Device)
async def update_device_by_id(
    device_in: DeviceUpdate,
    device_id: str = Path(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update device by ID.
    """
    device = await get_device(db, id=device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    device = await update_device(db, db_obj=device, obj_in=device_in)
    return device

@router.delete("/{device_id}")
async def delete_device_by_id(
    device_id: str = Path(...),
    current_user: Any = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Delete device by ID (admin only).
    """
    device = await get_device(db, id=device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    await delete_device(db, id=device_id)
    return {"message": "Device deleted successfully"}

@router.get("/{device_id}/sync-status", response_model=DeviceSyncStatus)
async def get_device_sync_status_by_id(
    device_id: str = Path(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get device sync status by device ID.
    """
    device = await get_device(db, id=device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    sync_status = await get_device_sync_status(db, device_id=device_id)
    if not sync_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sync status not found"
        )
    
    return sync_status

@router.put("/{device_id}/sync-status", response_model=DeviceSyncStatus)
async def update_device_sync_status_by_id(
    status_in: DeviceSyncStatusUpdate,
    device_id: str = Path(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update device sync status by device ID.
    """
    device = await get_device(db, id=device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    sync_status = await update_device_sync_status(db, device_id=device_id, obj_in=status_in)
    return sync_status

@router.post("/sync/manifest", response_model=SyncManifestResponse)
async def get_sync_manifest(
    request: SyncManifestRequest,
    device: Any = Depends(get_device_auth),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get sync manifest for a device.
    """
    # Update last seen timestamp
    await update_device_last_seen(db, device_id=device.id)
    
    # Generate sync manifest
    manifest = await generate_sync_manifest(
        db,
        device=device,
        child_id=request.child_id or device.child_id,
        group_id=request.group_id or device.group_id,
        mode=request.mode,
        current_cache_size=request.current_cache_size,
        current_doc_ids=request.current_doc_ids
    )
    
    # If sync is required, update last sync timestamp
    if manifest.sync_required:
        await update_device_last_sync(db, device_id=device.id)
    
    return manifest

@router.get("/ota/check", response_model=OTAManifestResponse)
async def check_ota_updates(
    current_version: str,
    device: Any = Depends(get_device_auth),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Check for OTA updates.
    """
    # Update last seen timestamp
    await update_device_last_seen(db, device_id=device.id)
    
    # Check for updates
    update_info = await check_for_updates(db, current_version=current_version)
    return update_info
