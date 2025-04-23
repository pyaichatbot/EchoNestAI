from typing import Any
from pydantic import EmailStr
import ssl
import aiosmtplib
from email.message import EmailMessage

from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("auth_email")

async def send_verification_email(email: EmailStr, token: str) -> None:
    """
    Send verification email to user.
    
    Args:
        email: User email
        token: Verification token
    """
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
