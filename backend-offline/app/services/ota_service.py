import os
import json
import aiofiles
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.models import Device, OTAUpdate
from app.db.schemas.device import OTAManifestResponse
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("ota_service")

async def check_for_updates(
    db: AsyncSession, 
    current_version: str
) -> OTAManifestResponse:
    """
    Check if there are OTA updates available for the device.
    
    Args:
        db: Database session
        current_version: Current firmware version of the device
        
    Returns:
        OTA manifest response with update information
    """
    # Get latest OTA update from database
    from sqlalchemy import select
    from sqlalchemy.sql import desc
    
    query = select(OTAUpdate).filter(
        OTAUpdate.is_active == True
    ).order_by(desc(OTAUpdate.created_at)).limit(1)
    
    result = await db.execute(query)
    latest_update = result.scalars().first()
    
    if not latest_update:
        logger.info(f"No active OTA updates found")
        return OTAManifestResponse(available=False)
    
    # Compare versions
    if latest_update.version == current_version:
        logger.info(f"Device already has the latest version: {current_version}")
        return OTAManifestResponse(available=False)
    
    # Check if the update is newer (simple string comparison, in production would use semver)
    if latest_update.version > current_version:
        logger.info(f"Update available: {current_version} -> {latest_update.version}")
        return OTAManifestResponse(
            available=True,
            version=latest_update.version,
            url=latest_update.url,
            size_mb=latest_update.size_mb,
            sha256=latest_update.sha256
        )
    
    # Current version is newer than latest update (unusual case)
    logger.warning(f"Device version {current_version} is newer than latest update {latest_update.version}")
    return OTAManifestResponse(available=False)

async def register_ota_update(
    db: AsyncSession,
    version: str,
    url: str,
    size_mb: float,
    sha256: str
) -> OTAUpdate:
    """
    Register a new OTA update in the database.
    
    Args:
        db: Database session
        version: Firmware version
        url: Download URL
        size_mb: Size in MB
        sha256: SHA256 hash for verification
        
    Returns:
        Created OTA update record
    """
    # Create new OTA update
    from app.db.models.models import OTAUpdate
    
    # Deactivate previous updates
    from sqlalchemy import update
    
    stmt = update(OTAUpdate).values(is_active=False)
    await db.execute(stmt)
    
    # Create new update
    ota_update = OTAUpdate(
        version=version,
        url=url,
        size_mb=size_mb,
        sha256=sha256,
        is_active=True
    )
    
    db.add(ota_update)
    await db.commit()
    await db.refresh(ota_update)
    
    logger.info(f"Registered new OTA update: {version}")
    return ota_update

async def verify_update_integrity(file_path: str, expected_sha256: str) -> bool:
    """
    Verify the integrity of an update file using SHA256.
    
    Args:
        file_path: Path to the update file
        expected_sha256: Expected SHA256 hash
        
    Returns:
        True if the file integrity is verified, False otherwise
    """
    import hashlib
    
    if not os.path.exists(file_path):
        logger.error(f"Update file not found: {file_path}")
        return False
    
    # Calculate SHA256 hash
    sha256_hash = hashlib.sha256()
    
    async with aiofiles.open(file_path, 'rb') as f:
        # Read and update hash in chunks to handle large files
        while chunk := await f.read(8192):
            sha256_hash.update(chunk)
    
    calculated_hash = sha256_hash.hexdigest()
    
    if calculated_hash != expected_sha256:
        logger.error(f"Hash mismatch: expected {expected_sha256}, got {calculated_hash}")
        return False
    
    logger.info(f"Update file integrity verified: {file_path}")
    return True
