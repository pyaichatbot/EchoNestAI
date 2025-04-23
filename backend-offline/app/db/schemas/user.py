from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Union, Dict, Any
from datetime import datetime
import enum

class UserRole(str, enum.Enum):
    """User role enumeration."""
    ADMIN = "admin"
    PARENT = "parent"
    TEACHER = "teacher"

class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    full_name: str
    role: UserRole = UserRole.PARENT
    language: str = "en"
    timezone: str = "UTC"

class UserCreate(UserBase):
    """User creation schema."""
    password: str

class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    role: Optional[UserRole] = None
    language: Optional[str] = None
    timezone: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None

class User(UserBase):
    """User schema."""
    id: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    """Token schema."""
    token: str
    expires_in: int
    role: UserRole

class TokenPayload(BaseModel):
    """Token payload schema."""
    sub: str
    role: str
    exp: Optional[int] = None

class ForgotPassword(BaseModel):
    """Forgot password schema."""
    email: EmailStr

class PasswordReset(BaseModel):
    """Password reset schema."""
    token: str
    password: str

class AssignRole(BaseModel):
    """Assign role schema."""
    user_id: str
    role: UserRole
