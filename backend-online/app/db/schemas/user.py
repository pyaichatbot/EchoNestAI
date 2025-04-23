from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from app.db.models.models import UserRole

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: UserRole
    language: Optional[str] = "en"
    timezone: Optional[str] = "UTC"

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None

class UserInDB(UserBase):
    id: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class User(UserInDB):
    pass

class Token(BaseModel):
    token: str
    expires_in: int
    role: UserRole

class TokenPayload(BaseModel):
    sub: str
    role: str
    exp: Optional[int] = None

class PasswordReset(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)

class ForgotPassword(BaseModel):
    email: EmailStr

class AssignRole(BaseModel):
    user_id: str
    role: UserRole
