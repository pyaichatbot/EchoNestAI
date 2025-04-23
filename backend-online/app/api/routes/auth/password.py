from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import EmailStr

from app.db.database import get_db
from app.db.schemas.user import ForgotPassword, PasswordReset
from app.db.crud.user import get_user_by_email, create_password_reset_token, verify_password_reset_token, update_user_password
from app.core.security import get_password_hash
from app.core.logging import setup_logging
from app.api.routes.auth.email import send_password_reset_email

logger = setup_logging("auth_password")

router = APIRouter()

@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(
    background_tasks: BackgroundTasks,
    email_in: ForgotPassword,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Password recovery email sent.
    """
    user = await get_user_by_email(db, email=email_in.email)
    if not user:
        # Don't reveal that the user doesn't exist
        logger.info(f"Password reset requested for non-existent email: {email_in.email}")
        return {"message": "If the email exists, a password reset link has been sent"}
    
    # Generate reset token
    reset_token = await create_password_reset_token(db, user_id=user.id)
    
    # Send password reset email in background
    background_tasks.add_task(
        send_password_reset_email,
        email=user.email,
        token=reset_token
    )
    
    logger.info(f"Password reset token generated for user: {user.id}")
    return {"message": "If the email exists, a password reset link has been sent"}

@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(
    reset_in: PasswordReset,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Reset password using token from email.
    """
    # Verify token
    user_id = await verify_password_reset_token(db, token=reset_in.token)
    if not user_id:
        logger.warning(f"Invalid password reset token: {reset_in.token[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token"
        )
    
    # Update password
    hashed_password = get_password_hash(reset_in.new_password)
    await update_user_password(db, user_id=user_id, hashed_password=hashed_password)
    
    # Invalidate token
    from app.db.models.models import PasswordResetToken
    from sqlalchemy import update
    
    stmt = update(PasswordResetToken).where(
        PasswordResetToken.token == reset_in.token
    ).values(
        used=True
    )
    await db.execute(stmt)
    await db.commit()
    
    logger.info(f"Password reset successful for user: {user_id}")
    return {"message": "Password has been reset successfully"}
