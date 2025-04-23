from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
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
from app.api.deps.auth import get_current_active_user, get_current_admin_user
from app.db.crud.user import (
    create_user, authenticate_user, get_user_by_email,
    update_user, assign_user_role, create_password_reset_token,
    verify_password_reset_token, update_user_password,
    verify_email_token, mark_email_verified
)
from app.core.security import create_access_token, get_password_hash
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("auth_routes")

router = APIRouter(prefix="/auth", tags=["authentication"])

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

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    user = await authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for email: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        logger.warning(f"Login attempt for inactive user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user"
        )
    
    # Log successful login
    logger.info(f"Successful login for user: {user.id} ({user.email})")
    
    # Create access token
    access_token_expires = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    token = create_access_token(
        subject=user.id, 
        role=user.role, 
        expires_minutes=access_token_expires
    )
    
    # Update last login timestamp
    from app.db.models.models import User as UserModel
    from sqlalchemy import update
    
    stmt = update(UserModel).where(UserModel.id == user.id).values(
        last_login=datetime.datetime.utcnow()
    )
    await db.execute(stmt)
    await db.commit()
    
    return {
        "token": token,
        "expires_in": access_token_expires * 60,
        "role": user.role
    }

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

@router.post("/assign-role", response_model=User)
async def assign_role(
    role_in: AssignRole,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Assign a role to a user (admin only).
    """
    logger.info(f"Role assignment: user {role_in.user_id} assigned role {role_in.role} by admin {current_user.id}")
    return await assign_user_role(db, user_id=role_in.user_id, role=role_in.role)

async def send_verification_email(email: EmailStr, token: str) -> None:
    """
    Send verification email to user.
    
    Args:
        email: User email
        token: Verification token
    """
    from email.message import EmailMessage
    import aiosmtplib
    import ssl
    
    # Create verification URL
    verification_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
    
    # Create email message
    message = EmailMessage()
    message["From"] = settings.SMTP_SENDER
    message["To"] = email
    message["Subject"] = "Verify your EchoNest AI account"
    
    # Email content
    content = f"""
    Hello,
    
    Thank you for registering with EchoNest AI. Please verify your email address by clicking the link below:
    
    {verification_url}
    
    This link will expire in 24 hours.
    
    If you did not register for an EchoNest AI account, please ignore this email.
    
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
        
        logger.info(f"Verification email sent to: {email}")
    except Exception as e:
        logger.error(f"Failed to send verification email to {email}: {str(e)}")

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
