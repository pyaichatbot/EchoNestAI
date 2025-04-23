from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.schemas.device import OTAManifestResponse
from app.db.crud.device import update_device_last_seen
from app.db.models.models import OTAUpdate
from sqlalchemy.future import select

async def get_ota_update(
    db: AsyncSession,
    current_version: str
) -> OTAManifestResponse:
    """
    Check if an OTA update is available for a device.
    
    Args:
        db: Database session
        current_version: Current firmware version of the device
        
    Returns:
        OTA manifest response with update information if available
    """
    # Update device last seen time
    await update_device_last_seen(db, device_id=device_id)
    
    # Get latest active OTA update
    result = await db.execute(
        select(OTAUpdate)
        .filter(OTAUpdate.is_active == True)
        .order_by(OTAUpdate.created_at.desc())
        .limit(1)
    )
    latest_update = result.scalars().first()
    
    # Check if update is available
    if not latest_update or latest_update.version <= current_version:
        return OTAManifestResponse(available=False)
    
    # Return update information
    return OTAManifestResponse(
        available=True,
        version=latest_update.version,
        url=latest_update.url,
        size_mb=latest_update.size_mb,
        sha256=latest_update.sha256
    )
