from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.db.database import get_db
from app.db.schemas.content import Child, ChildCreate, ChildUpdate, Group
from app.api.deps.auth import get_current_active_user
from app.db.crud.students import (
    list_students_for_teacher,
    create_student,
    get_student_by_id,
    update_student,
    delete_student,
    get_groups_for_student,
    assign_student_to_group,
    unassign_student_from_group,
)

router = APIRouter(tags=["students"])

@router.get("/students", response_model=List[Child])
async def api_list_students(
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    return await list_students_for_teacher(db, current_user.id)

@router.post("/students", response_model=Child, status_code=201)
async def api_create_student(
    student_in: ChildCreate,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        return await create_student(db, student_in, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/students/{student_id}", response_model=Child)
async def api_update_student(
    student_id: str,
    student_in: ChildUpdate,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    db_obj = await get_student_by_id(db, student_id, current_user.id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Student not found")
    return await update_student(db, db_obj, student_in)

@router.delete("/students/{student_id}", status_code=204)
async def api_delete_student(
    student_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    db_obj = await get_student_by_id(db, student_id, current_user.id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Student not found")
    await delete_student(db, db_obj)
    return None

@router.get("/students/{student_id}/groups", response_model=List[Group])
async def api_get_student_groups(
    student_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    return await get_groups_for_student(db, student_id, current_user.id)

@router.post("/students/group/{group_id}", status_code=200)
async def api_assign_student_to_group(
    group_id: str,
    student_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        await assign_student_to_group(db, student_id, group_id, current_user.id)
        return {"detail": "Student assigned to group"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/students/group/{group_id}/{student_id}", status_code=200)
async def api_unassign_student_from_group(
    group_id: str,
    student_id: str,
    current_user: any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        await unassign_student_from_group(db, student_id, group_id, current_user.id)
        return {"detail": "Student removed from group"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) 