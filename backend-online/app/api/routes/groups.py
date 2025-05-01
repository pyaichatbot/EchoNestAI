from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.db.database import get_db
from app.db.schemas.content import GroupActivity, GroupActivityCreate, Group, GroupCreate, Content, Child
from app.api.deps.auth import get_current_active_user
from app.db.crud.groups import (
    list_group_activities,
    get_group_activity,
    create_group_activity,
    get_groups_for_teacher,
    create_group,
    get_group_by_id,
    get_students_in_group,
    get_content_for_group,
)

router = APIRouter(tags=["groups"])

# ... existing group endpoints ...

@router.get("/groups/{group_id}/activities", response_model=List[GroupActivity])
async def api_list_group_activities(
    group_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        return await list_group_activities(db, group_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/groups/{group_id}/activities", response_model=GroupActivity, status_code=201)
async def api_create_group_activity(
    group_id: str,
    activity_in: GroupActivityCreate,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    if activity_in.group_id != group_id:
        raise HTTPException(status_code=400, detail="group_id mismatch")
    try:
        return await create_group_activity(db, activity_in, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/groups/{group_id}/activities/{activity_id}", response_model=GroupActivity)
async def api_get_group_activity(
    group_id: str,
    activity_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        activity = await get_group_activity(db, activity_id, current_user.id)
        if not activity or activity.group_id != group_id:
            raise HTTPException(status_code=404, detail="Activity not found")
        return activity
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/groups", response_model=List[Group])
async def api_list_groups(
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    return await get_groups_for_teacher(db, current_user.id)

@router.post("/groups", response_model=Group, status_code=201)
async def api_create_group(
    group_in: GroupCreate,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    return await create_group(db, current_user.id, group_in)

@router.get("/groups/{group_id}", response_model=Group)
async def api_get_group(
    group_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    group = await get_group_by_id(db, group_id, current_user.id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return group

@router.get("/groups/{group_id}/students", response_model=List[Child])
async def api_get_group_students(
    group_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    group = await get_group_by_id(db, group_id, current_user.id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return await get_students_in_group(db, group_id)

@router.get("/groups/{group_id}/content", response_model=List[Content])
async def api_get_group_content(
    group_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    group = await get_group_by_id(db, group_id, current_user.id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return await get_content_for_group(db, group_id)

# ... existing code ... 