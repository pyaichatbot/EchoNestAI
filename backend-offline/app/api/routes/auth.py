from fastapi import APIRouter, Depends, HTTPException, status, Form, Body, BackgroundTasks
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import secrets
import datetime
from pydantic import EmailStr

from app.db.database import get_db
from app.db.schemas.user import (
    User, UserCreate, UserUpdate, Token, 
    ForgotPassword, PasswordReset, AssignRole
)
from app.db.crud.user import (
    get_user, get_user_by_email, create_user, 
    update_user, authenticate_user, assign_user_role
)
from app.core.security import create_access_token, get_password_hash
from app.api.deps.auth import get_current_active_user, get_current_admin_user
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("auth_routes")

router = APIRouter(tags=["auth"])

@router.post("/register", response_model=User)
async def register_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Register a new user.
    """
    # Check if user already exists
    user = await get_user_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user = await create_user(db, obj_in=user_in)
    return user

@router.post("/login", response_model=Token)
async def login(
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Login and get access token.
    """
    user = await authenticate_user(db, email=email, password=password)
    if not user:
        logger.warning(f"Failed login attempt for email: {email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        logger.warning(f"Login attempt for inactive user: {email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Create access token
    access_token = create_access_token(
        subject=user.id,
        role=user.role
    )
    
    # Update last login timestamp
    from app.db.models.models import User as UserModel
    from sqlalchemy import update
    
    stmt = update(UserModel).where(UserModel.id == user.id).values(
        last_login=datetime.datetime.utcnow()
    )
    await db.execute(stmt)
    await db.commit()
    
    logger.info(f"Successful login for user: {user.id} ({user.email})")
    
    return {
        "token": access_token,
        "expires_in": 60 * 24 * 7,  # 7 days
        "role": user.role
    }

@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get current user information.
    """
    return current_user

@router.put("/me", response_model=User)
async def update_current_user_info(
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update current user information.
    """
    user = await update_user(db, db_obj=current_user, obj_in=user_in)
    return user

@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(
    background_tasks: BackgroundTasks,
    request: ForgotPassword,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Request password reset.
    """
    user = await get_user_by_email(db, email=request.email)
    if not user:
        # Don't reveal that the user doesn't exist
        logger.info(f"Password reset requested for non-existent email: {request.email}")
        return {"message": "If the email exists, a password reset link has been sent"}
    
    # Generate reset token
    reset_token = secrets.token_urlsafe(32)
    
    # Store token in database with expiration
    from app.db.models.models import PasswordResetToken
    from sqlalchemy import insert
    
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    
    stmt = insert(PasswordResetToken).values(
        user_id=user.id,
        token=reset_token,
        expires_at=expiration,
        used=False
    )
    await db.execute(stmt)
    await db.commit()
    
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
    request: PasswordReset,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Reset password using token.
    """
    # Verify token
    from app.db.models.models import PasswordResetToken
    from sqlalchemy import select
    
    # Get token from database
    query = select(PasswordResetToken).filter(
        PasswordResetToken.token == request.token,
        PasswordResetToken.used == False,
        PasswordResetToken.expires_at > datetime.datetime.utcnow()
    )
    result = await db.execute(query)
    token_record = result.scalars().first()
    
    if not token_record:
        logger.warning(f"Invalid password reset token: {request.token[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token"
        )
    
    # Update password
    user_id = token_record.user_id
    hashed_password = get_password_hash(request.new_password)
    
    from app.db.models.models import User as UserModel
    from sqlalchemy import update
    
    stmt = update(UserModel).where(UserModel.id == user_id).values(
        hashed_password=hashed_password
    )
    await db.execute(stmt)
    
    # Mark token as used
    stmt = update(PasswordResetToken).where(
        PasswordResetToken.token == request.token
    ).values(
        used=True
    )
    await db.execute(stmt)
    await db.commit()
    
    logger.info(f"Password reset successful for user: {user_id}")
    return {"message": "Password has been reset successfully"}

@router.post("/assign-role", response_model=User)
async def assign_role(
    request: AssignRole,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Assign a role to a user (admin only).
    """
    user = await assign_user_role(db, user_id=request.user_id, role=request.role)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    logger.info(f"Role assignment: user {request.user_id} assigned role {request.role} by admin {current_user.id}")
    return user

async def send_password_reset_email(email: EmailStr, token: str) -> None:
    """
    Send password reset email to user.
    
    Args:
        email: User email
        token: Password reset token
    """
    from email.message import EmailMessage
    import aiosmtplib
    import ssl
    
    # Create reset URL
    reset_url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
    
    # Create email message
    message = EmailMessage()
    message["From"] = settings.SMTP_SENDER
    message["To"] = email
    message["Subject"] = "Reset your EchoNest AI password"
    
    # Email content
    content = f"""
    Hello,
    
    You have requested to reset your EchoNest AI password. Please click the link below to reset your password:
    
    {reset_url}
    
    This link will expire in 1 hour.
    
    If you did not request a password reset, please ignore this email.
    
    Best regards,
    The EchoNest AI Team
    """
    
    message.set_content(content)
    
    try:
        # Create SSL context
        context = ssl.create_default_context()
        
        # Send email
        await aiosmtplib.send(
            message,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USERNAME,
            password=settings.SMTP_PASSWORD,
            use_tls=settings.SMTP_TLS,
            tls_context=context
        )
        
        logger.info(f"Password reset email sent to: {email}")
    except Exception as e:
        logger.error(f"Failed to send password reset email to {email}: {str(e)}")
