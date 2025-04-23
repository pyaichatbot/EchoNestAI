from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
import secrets
import datetime
from pydantic import EmailStr

from app.db.database import get_db
from app.db.schemas.user import UserCreate, User
from app.db.crud.user import create_user, get_user_by_email, verify_email_token, mark_email_verified
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes.auth.email import send_verification_email

logger = setup_logging("auth_registration")

router = APIRouter()

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(
    background_tasks: BackgroundTasks,
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Register a new user.
    """
    user = await get_user_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    new_user = await create_user(db, obj_in=user_in)
    
    # Generate verification token
    verification_token = secrets.token_urlsafe(32)
    
    # Store token in database with expiration
    from app.db.models.models import EmailVerificationToken
    from sqlalchemy import insert
    
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    
    stmt = insert(EmailVerificationToken).values(
        user_id=new_user.id,
        token=verification_token,
        expires_at=expiration
    )
    await db.execute(stmt)
    await db.commit()
    
    # Send verification email in background
    background_tasks.add_task(
        send_verification_email,
        email=new_user.email,
        token=verification_token
    )
    
    return new_user

@router.get("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Verify email using token from email.
    """
    # Verify token
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
