from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete
from typing import Any, Dict, Optional, List, Union

from app.db.models.models import User, UserRole
from app.db.schemas.user import UserCreate, UserUpdate
from app.core.security import get_password_hash, verify_password

async def get_user(db: AsyncSession, id: str) -> Optional[User]:
    """
    Get a user by ID.
    """
    result = await db.execute(select(User).filter(User.id == id))
    return result.scalars().first()

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """
    Get a user by email.
    """
    result = await db.execute(select(User).filter(User.email == email))
    return result.scalars().first()

async def get_users(
    db: AsyncSession, 
    role: Optional[UserRole] = None,
    skip: int = 0, 
    limit: int = 100
) -> List[User]:
    """
    Get a list of users with optional role filter.
    """
    query = select(User).offset(skip).limit(limit)
    if role:
        query = query.filter(User.role == role)
    result = await db.execute(query)
    return result.scalars().all()

async def create_user(db: AsyncSession, obj_in: UserCreate) -> User:
    """
    Create a new user.
    """
    db_obj = User(
        email=obj_in.email,
        full_name=obj_in.full_name,
        hashed_password=get_password_hash(obj_in.password),
        role=obj_in.role,
        language=obj_in.language,
        timezone=obj_in.timezone,
        is_active=True,
        is_verified=False
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def update_user(
    db: AsyncSession, 
    db_obj: User, 
    obj_in: Union[UserUpdate, Dict[str, Any]]
) -> User:
    """
    Update a user.
    """
    if isinstance(obj_in, dict):
        update_data = obj_in
    else:
        update_data = obj_in.dict(exclude_unset=True)
    
    if "password" in update_data:
        hashed_password = get_password_hash(update_data["password"])
        del update_data["password"]
        update_data["hashed_password"] = hashed_password
    
    stmt = (
        update(User)
        .where(User.id == db_obj.id)
        .values(**update_data)
    )
    await db.execute(stmt)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def authenticate_user(
    db: AsyncSession, 
    email: str, 
    password: str
) -> Optional[User]:
    """
    Authenticate a user by email and password.
    """
    user = await get_user_by_email(db, email=email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

async def assign_user_role(
    db: AsyncSession, 
    user_id: str, 
    role: UserRole
) -> Optional[User]:
    """
    Assign a role to a user.
    """
    user = await get_user(db, id=user_id)
    if not user:
        return None
    
    stmt = (
        update(User)
        .where(User.id == user_id)
        .values(role=role)
    )
    await db.execute(stmt)
    await db.commit()
    await db.refresh(user)
    return user

async def delete_user(db: AsyncSession, id: str) -> bool:
    """
    Delete a user.
    """
    stmt = delete(User).where(User.id == id)
    await db.execute(stmt)
    await db.commit()
    return True
