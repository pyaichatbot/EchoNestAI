from fastapi import APIRouter, Depends
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.schemas.user import User, AssignRole
from app.api.deps.auth import get_current_admin_user
from app.db.crud.user import assign_user_role
from app.core.logging import setup_logging

logger = setup_logging("auth_admin")

router = APIRouter()

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
