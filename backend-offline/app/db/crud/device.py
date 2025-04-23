from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete, or_
from typing import Any, Dict, Optional, List, Union
from datetime import datetime

from app.db.models.models import Device, DeviceMode, DeviceSyncStatus
from app.db.schemas.device import DeviceCreate, DeviceUpdate, DeviceSyncStatusCreate, DeviceSyncStatusUpdate

async def get_device(db: AsyncSession, id: Optional[str] = None, device_id: Optional[str] = None) -> Optional[Device]:
    """
    Get device by ID or device_id.
    """
    if id:
        result = await db.execute(select(Device).filter(Device.id == id))
    elif device_id:
        result = await db.execute(select(Device).filter(Device.device_id == device_id))
    else:
        return None
    
    return result.scalars().first()

async def get_devices(
    db: AsyncSession, 
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    skip: int = 0, 
    limit: int = 100
) -> List[Device]:
    """
    Get a list of devices with optional filters.
    """
    query = select(Device).offset(skip).limit(limit)
    
    if child_id:
        query = query.filter(Device.child_id == child_id)
    
    if group_id:
        query = query.filter(Device.group_id == group_id)
    
    result = await db.execute(query)
    return result.scalars().all()

async def create_device(db: AsyncSession, obj_in: DeviceCreate) -> Device:
    """
    Register a new device.
    """
    db_obj = Device(
        device_id=obj_in.device_id,
        firmware_version=obj_in.firmware_version,
        mode=obj_in.mode,
        child_id=obj_in.child_id,
        group_id=obj_in.group_id
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    
    # Initialize sync status
    sync_status = DeviceSyncStatus(
        device_id=db_obj.id,
        used_mb=0,
        doc_count=0
    )
    db.add(sync_status)
    await db.commit()
    
    return db_obj

async def update_device(
    db: AsyncSession, 
    db_obj: Device, 
    obj_in: Union[DeviceUpdate, Dict[str, Any]]
) -> Device:
    """
    Update device information.
    """
    if isinstance(obj_in, dict):
        update_data = obj_in
    else:
        update_data = obj_in.dict(exclude_unset=True)
    
    stmt = (
        update(Device)
        .where(Device.id == db_obj.id)
        .values(**update_data)
    )
    await db.execute(stmt)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def update_device_last_seen(
    db: AsyncSession,
    device_id: str
) -> Optional[Device]:
    """
    Update device last_seen timestamp.
    """
    device = await get_device(db, id=device_id)
    if not device:
        return None
    
    stmt = (
        update(Device)
        .where(Device.id == device_id)
        .values(last_seen=datetime.utcnow())
    )
    await db.execute(stmt)
    await db.commit()
    await db.refresh(device)
    return device

async def update_device_last_sync(
    db: AsyncSession,
    device_id: str
) -> Optional[Device]:
    """
    Update device last_sync timestamp.
    """
    device = await get_device(db, id=device_id)
    if not device:
        return None
    
    stmt = (
        update(Device)
        .where(Device.id == device_id)
        .values(last_sync=datetime.utcnow())
    )
    await db.execute(stmt)
    await db.commit()
    await db.refresh(device)
    return device

async def get_device_sync_status(
    db: AsyncSession,
    device_id: str
) -> Optional[DeviceSyncStatus]:
    """
    Get device sync status.
    """
    result = await db.execute(
        select(DeviceSyncStatus).filter(DeviceSyncStatus.device_id == device_id)
    )
    return result.scalars().first()

async def update_device_sync_status(
    db: AsyncSession,
    device_id: str,
    obj_in: Union[DeviceSyncStatusUpdate, Dict[str, Any]]
) -> Optional[DeviceSyncStatus]:
    """
    Update device sync status.
    """
    sync_status = await get_device_sync_status(db, device_id=device_id)
    
    if not sync_status:
        # Create if not exists
        if isinstance(obj_in, dict):
            create_data = obj_in
        else:
            create_data = obj_in.dict(exclude_unset=True)
        
        create_data["device_id"] = device_id
        db_obj = DeviceSyncStatus(**create_data)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    # Update existing
    if isinstance(obj_in, dict):
        update_data = obj_in
    else:
        update_data = obj_in.dict(exclude_unset=True)
    
    stmt = (
        update(DeviceSyncStatus)
        .where(DeviceSyncStatus.device_id == device_id)
        .values(**update_data)
    )
    await db.execute(stmt)
    await db.commit()
    await db.refresh(sync_status)
    return sync_status

async def delete_device(db: AsyncSession, id: str) -> bool:
    """
    Delete a device and its sync status.
    """
    # Delete sync status first
    await db.execute(delete(DeviceSyncStatus).where(DeviceSyncStatus.device_id == id))
    
    # Delete device
    await db.execute(delete(Device).where(Device.id == id))
    
    await db.commit()
    return True
