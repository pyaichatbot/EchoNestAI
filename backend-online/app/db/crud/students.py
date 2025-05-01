from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete, and_
from typing import List, Optional, Any, Dict, Union
from datetime import datetime
import uuid

from app.db.models.models import Child, Group
from app.db.schemas.content import ChildCreate, ChildUpdate

async def list_students_for_teacher(db: AsyncSession, teacher_id: str) -> List[Child]:
    # List all students (children) for groups owned by this teacher
    result = await db.execute(
        select(Child).join(Group, Child.group_id == Group.id).filter(Group.teacher_id == teacher_id)
    )
    return result.scalars().all()

async def create_student(db: AsyncSession, obj_in: ChildCreate, teacher_id: str) -> Child:
    # Student must be assigned to a group owned by this teacher
    group_id = getattr(obj_in, 'group_id', None)
    if group_id:
        group = await db.get(Group, group_id)
        if not group or group.teacher_id != teacher_id:
            raise ValueError("Group not found or access denied")
    db_obj = Child(
        id=str(uuid.uuid4()),
        name=obj_in.name,
        age=obj_in.age,
        language=obj_in.language,
        avatar=obj_in.avatar,
        group_id=group_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_student_by_id(db: AsyncSession, student_id: str, teacher_id: str) -> Optional[Child]:
    result = await db.execute(
        select(Child).join(Group, Child.group_id == Group.id).filter(Child.id == student_id, Group.teacher_id == teacher_id)
    )
    return result.scalars().first()

async def update_student(db: AsyncSession, db_obj: Child, obj_in: Union[ChildUpdate, Dict[str, Any]]) -> Child:
    update_data = obj_in.dict(exclude_unset=True) if not isinstance(obj_in, dict) else obj_in
    update_data["updated_at"] = datetime.utcnow()
    stmt = (
        update(Child)
        .where(Child.id == db_obj.id)
        .values(**update_data)
    )
    await db.execute(stmt)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def delete_student(db: AsyncSession, db_obj: Child) -> None:
    stmt = delete(Child).where(Child.id == db_obj.id)
    await db.execute(stmt)
    await db.commit()

async def get_groups_for_student(db: AsyncSession, student_id: str, teacher_id: str) -> List[Group]:
    # Return the group for this student if owned by teacher
    result = await db.execute(
        select(Group).join(Child, Child.group_id == Group.id).filter(Child.id == student_id, Group.teacher_id == teacher_id)
    )
    return result.scalars().all()

async def assign_student_to_group(db: AsyncSession, student_id: str, group_id: str, teacher_id: str) -> None:
    group = await db.get(Group, group_id)
    if not group or group.teacher_id != teacher_id:
        raise ValueError("Group not found or access denied")
    stmt = update(Child).where(Child.id == student_id).values(group_id=group_id, updated_at=datetime.utcnow())
    await db.execute(stmt)
    await db.commit()

async def unassign_student_from_group(db: AsyncSession, student_id: str, group_id: str, teacher_id: str) -> None:
    group = await db.get(Group, group_id)
    if not group or group.teacher_id != teacher_id:
        raise ValueError("Group not found or access denied")
    stmt = update(Child).where(and_(Child.id == student_id, Child.group_id == group_id)).values(group_id=None, updated_at=datetime.utcnow())
    await db.execute(stmt)
    await db.commit() 