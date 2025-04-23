from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.schemas.user import User
from app.api.deps.auth import get_current_active_user
from app.core.logging import setup_logging

logger = setup_logging("auth_verification")

router = APIRouter()

@router.get("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Verify email using token from email.
    """
    # Verify token
    from app.db.crud.user import verify_email_token, mark_email_verified
    
    user_id = await verify_email_token(db, token=token)
    if not user_id:
        logger.warning(f"Invalid email verification token: {token[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token"
        )
    
    # Mark email as verified
    await mark_email_verified(db, user_id=user_id)
    
    # Invalidate token
    from app.db.models.models import EmailVerificationToken
    from sqlalchemy import delete
    
    stmt = delete(EmailVerificationToken).where(
        EmailVerificationToken.token == token
    )
    await db.execute(stmt)
    await db.commit()
    
    logger.info(f"Email verification successful for user: {user_id}")
    return {"message": "Email has been verified successfully"}
