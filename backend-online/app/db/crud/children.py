from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete
from typing import List, Optional, Any, Dict, Union
from datetime import datetime
import uuid

from app.db.models.models import Child
from app.db.schemas.content import ChildCreate, ChildUpdate

async def get_children_for_parent(db: AsyncSession, parent_id: str) -> List[Child]:
    result = await db.execute(select(Child).filter(Child.parent_id == parent_id))
    return result.scalars().all()

async def get_child_by_id(db: AsyncSession, child_id: str, parent_id: str) -> Optional[Child]:
    result = await db.execute(
        select(Child).filter(Child.id == child_id, Child.parent_id == parent_id)
    )
    return result.scalars().first()

async def create_child(db: AsyncSession, parent_id: str, obj_in: ChildCreate) -> Child:
    child_id = str(uuid.uuid4())
    db_obj = Child(
        id=child_id,
        name=obj_in.name,
        age=obj_in.age,
        language=obj_in.language,
        avatar=obj_in.avatar or "",
        parent_id=parent_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def update_child(
    db: AsyncSession, db_obj: Child, obj_in: Union[ChildUpdate, Dict[str, Any]]
) -> Child:
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

async def delete_child(db: AsyncSession, db_obj: Child) -> None:
    stmt = delete(Child).where(Child.id == db_obj.id)
    await db.execute(stmt)
    await db.commit() 