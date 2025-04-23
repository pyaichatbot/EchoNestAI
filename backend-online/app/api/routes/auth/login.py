from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
import datetime

from app.db.database import get_db
from app.db.schemas.user import User, Token
from app.api.deps.auth import get_current_active_user
from app.db.crud.user import authenticate_user
from app.core.security import create_access_token
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("auth_login")

router = APIRouter()

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
