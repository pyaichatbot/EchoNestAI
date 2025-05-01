from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.db.database import get_db
from app.db.schemas.content import Child, ChildCreate, ChildUpdate
from app.api.deps.auth import get_current_active_user
from app.db.crud.children import (
    get_children_for_parent,
    get_child_by_id,
    create_child,
    update_child,
    delete_child,
)

router = APIRouter(tags=["children"])

@router.get("/children", response_model=List[Child])
async def list_children(
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List all children for the current parent user."""
    return await get_children_for_parent(db, parent_id=current_user.id)

@router.post("/children", response_model=Child, status_code=status.HTTP_201_CREATED)
async def create_child_api(
    child_in: ChildCreate,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new child profile for the current parent user."""
    return await create_child(db, parent_id=current_user.id, obj_in=child_in)

@router.get("/children/{child_id}", response_model=Child)
async def get_child_api(
    child_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a child profile by ID (must belong to current parent)."""
    child = await get_child_by_id(db, child_id=child_id, parent_id=current_user.id)
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")
    return child

@router.patch("/children/{child_id}", response_model=Child)
async def update_child_api(
    child_id: str,
    child_in: ChildUpdate,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a child profile (must belong to current parent)."""
    child = await get_child_by_id(db, child_id=child_id, parent_id=current_user.id)
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")
    return await update_child(db, db_obj=child, obj_in=child_in)

@router.delete("/children/{child_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_child_api(
    child_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a child profile (must belong to current parent)."""
    child = await get_child_by_id(db, child_id=child_id, parent_id=current_user.id)
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")
    await delete_child(db, db_obj=child)
    return None 