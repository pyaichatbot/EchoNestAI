from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete, and_, or_
from typing import List, Optional, Any, Dict, Union
from datetime import datetime
import uuid

from app.db.models.models import Group, Child, Content, ContentAssignment, GroupActivity
from app.db.schemas.content import GroupCreate, GroupUpdate, GroupActivityCreate

# CRUD for Groups
async def get_groups_for_teacher(db: AsyncSession, teacher_id: str) -> List[Group]:
    result = await db.execute(select(Group).filter(Group.teacher_id == teacher_id))
    return result.scalars().all()

async def get_group_by_id(db: AsyncSession, group_id: str, teacher_id: str) -> Optional[Group]:
    result = await db.execute(select(Group).filter(Group.id == group_id, Group.teacher_id == teacher_id))
    return result.scalars().first()

async def create_group(db: AsyncSession, teacher_id: str, obj_in: GroupCreate) -> Group:
    group_id = str(uuid.uuid4())
    db_obj = Group(
        id=group_id,
        name=obj_in.name,
        age_range=obj_in.age_range,
        location=obj_in.location,
        teacher_id=teacher_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def update_group(db: AsyncSession, db_obj: Group, obj_in: Union[GroupUpdate, Dict[str, Any]]) -> Group:
    update_data = obj_in.dict(exclude_unset=True) if not isinstance(obj_in, dict) else obj_in
    update_data["updated_at"] = datetime.utcnow()
    stmt = (
        update(Group)
        .where(Group.id == db_obj.id)
        .values(**update_data)
    )
    await db.execute(stmt)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def delete_group(db: AsyncSession, db_obj: Group) -> None:
    stmt = delete(Group).where(Group.id == db_obj.id)
    await db.execute(stmt)
    await db.commit()

# Group Membership (Students)
async def get_students_in_group(db: AsyncSession, group_id: str) -> List[Child]:
    result = await db.execute(select(Child).filter(Child.group_id == group_id))
    return result.scalars().all()

async def assign_student_to_group(db: AsyncSession, child_id: str, group_id: str) -> None:
    stmt = update(Child).where(Child.id == child_id).values(group_id=group_id, updated_at=datetime.utcnow())
    await db.execute(stmt)
    await db.commit()

async def unassign_student_from_group(db: AsyncSession, child_id: str, group_id: str) -> None:
    stmt = update(Child).where(and_(Child.id == child_id, Child.group_id == group_id)).values(group_id=None, updated_at=datetime.utcnow())
    await db.execute(stmt)
    await db.commit()

# Content Assignment
async def get_content_for_group(db: AsyncSession, group_id: str) -> List[Content]:
    result = await db.execute(
        select(Content).join(ContentAssignment).filter(ContentAssignment.group_id == group_id)
    )
    return result.scalars().all()

async def assign_content_to_group(db: AsyncSession, content_id: str, group_id: str) -> None:
    assignment = ContentAssignment(
        id=str(uuid.uuid4()),
        content_id=content_id,
        group_id=group_id,
        created_at=datetime.utcnow()
    )
    db.add(assignment)
    await db.commit()

async def unassign_content_from_group(db: AsyncSession, content_id: str, group_id: str) -> None:
    stmt = delete(ContentAssignment).where(and_(ContentAssignment.content_id == content_id, ContentAssignment.group_id == group_id))
    await db.execute(stmt)
    await db.commit()

async def list_group_activities(db: AsyncSession, group_id: str, teacher_id: str) -> List[GroupActivity]:
    # Only allow teacher to list activities for their group
    group = await db.get(Group, group_id)
    if not group or group.teacher_id != teacher_id:
        raise ValueError("Group not found or access denied")
    result = await db.execute(select(GroupActivity).filter(GroupActivity.group_id == group_id))
    return result.scalars().all()

async def get_group_activity(db: AsyncSession, activity_id: str, teacher_id: str) -> Optional[GroupActivity]:
    activity = await db.get(GroupActivity, activity_id)
    if not activity:
        return None
    group = await db.get(Group, activity.group_id)
    if not group or group.teacher_id != teacher_id:
        raise ValueError("Group not found or access denied")
    return activity

async def create_group_activity(db: AsyncSession, obj_in: GroupActivityCreate, teacher_id: str) -> GroupActivity:
    group = await db.get(Group, obj_in.group_id)
    if not group or group.teacher_id != teacher_id:
        raise ValueError("Group not found or access denied")
    db_obj = GroupActivity(**obj_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj 