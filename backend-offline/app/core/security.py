from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from typing import Any, Optional

from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash a password.
    """
    return pwd_context.hash(password)

def create_access_token(subject: str, role: str, expires_minutes: Optional[int] = None) -> str:
    """
    Create a JWT access token.
    """
    if expires_minutes is None:
        expires_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode = {"exp": expire, "sub": str(subject), "role": role}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_device_token(device_id: str, expires_days: Optional[int] = None) -> str:
    """
    Create a JWT token for device authentication.
    """
    if expires_days is None:
        expires_days = settings.DEVICE_TOKEN_EXPIRE_DAYS
    
    expire = datetime.utcnow() + timedelta(days=expires_days)
    to_encode = {"exp": expire, "sub": str(device_id), "role": "device"}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt
