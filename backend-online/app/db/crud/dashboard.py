from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from app.db.models.models import Child, Group, Content, ContentAssignment, Device, DeviceSyncStatus, GroupActivity, Event, LearningProgress, MasteryLevel
from app.db.schemas.content import (
    LearningProgressSummary,
    LearningProgress,
    ProgressByCategory,
    TeacherDashboardOverview,
    DashboardMetric,
    ActivityItem,
    SystemStatus,
)

async def get_child_progress_summary(
    db: AsyncSession, 
    child_id: str, 
    parent_id: str, 
    filters: Dict[str, Any] = {}
) -> LearningProgressSummary:
    # Verify parent owns this child
    child = await db.get(Child, child_id)
    if not child or child.parent_id != parent_id:
        raise ValueError("Child not found or access denied")

    # Query all progress records for this child
    result = await db.execute(
        select(LearningProgress).where(LearningProgress.user_id == child_id)
    )
    progress_records = result.scalars().all()
    if not progress_records:
        return LearningProgressSummary(
            average_progress=0.0,
            completion_rate=0.0,
            time_spent=0,
            strengths=[],
            areas_for_improvement=[],
            recent_progress=[],
            progress_by_category=[]
        )

    # Compute metrics
    total_progress = sum(p.progress for p in progress_records)
    average_progress = total_progress / len(progress_records)
    completed = [p for p in progress_records if p.progress >= 100.0]
    completion_rate = (len(completed) / len(progress_records)) * 100.0
    time_spent = sum(p.time_spent for p in progress_records)
    # Category progress
    category_map = {}
    for p in progress_records:
        if p.category:
            category_map.setdefault(p.category, []).append(p.progress)
    progress_by_category = [
        ProgressByCategory(category=cat, progress=sum(vals)/len(vals))
        for cat, vals in category_map.items()
    ]
    # Strengths/areas: top/bottom categories
    sorted_cats = sorted(progress_by_category, key=lambda x: x.progress, reverse=True)
    strengths = [c.category for c in sorted_cats[:3]]
    areas_for_improvement = [c.category for c in sorted_cats[-3:]]
    # Recent progress (last 5 by last_accessed)
    recent_progress = sorted(progress_records, key=lambda x: x.last_accessed, reverse=True)[:5]
    # Convert to schema
    recent_progress_schema = [
        LearningProgress(
            id=p.id,
            user_id=p.user_id,
            content_id=p.content_id,
            progress=p.progress,
            score=p.score,
            time_spent=p.time_spent,
            last_accessed=int(p.last_accessed.timestamp()),
            completed_sections=p.completed_sections or [],
            mastery_level=p.mastery_level.value if p.mastery_level else None
        ) for p in recent_progress
    ]
    return LearningProgressSummary(
        average_progress=average_progress,
        completion_rate=completion_rate,
        time_spent=time_spent,
        strengths=strengths,
        areas_for_improvement=areas_for_improvement,
        recent_progress=recent_progress_schema,
        progress_by_category=progress_by_category
    )

async def get_group_progress_summary(
    db: AsyncSession, 
    group_id: str, 
    teacher_id: str, 
    filters: Dict[str, Any] = {}
) -> LearningProgressSummary:
    # Verify teacher owns this group
    group = await db.get(Group, group_id)
    if not group or group.teacher_id != teacher_id:
        raise ValueError("Group not found or access denied")

    # Query all progress records for this group
    result = await db.execute(
        select(LearningProgress).where(LearningProgress.group_id == group_id)
    )
    progress_records = result.scalars().all()
    if not progress_records:
        return LearningProgressSummary(
            average_progress=0.0,
            completion_rate=0.0,
            time_spent=0,
            strengths=[],
            areas_for_improvement=[],
            recent_progress=[],
            progress_by_category=[]
        )

    # Compute metrics
    total_progress = sum(p.progress for p in progress_records)
    average_progress = total_progress / len(progress_records)
    completed = [p for p in progress_records if p.progress >= 100.0]
    completion_rate = (len(completed) / len(progress_records)) * 100.0
    time_spent = sum(p.time_spent for p in progress_records)
    # Category progress
    category_map = {}
    for p in progress_records:
        if p.category:
            category_map.setdefault(p.category, []).append(p.progress)
    progress_by_category = [
        ProgressByCategory(category=cat, progress=sum(vals)/len(vals))
        for cat, vals in category_map.items()
    ]
    # Strengths/areas: top/bottom categories
    sorted_cats = sorted(progress_by_category, key=lambda x: x.progress, reverse=True)
    strengths = [c.category for c in sorted_cats[:3]]
    areas_for_improvement = [c.category for c in sorted_cats[-3:]]
    # Recent progress (last 5 by last_accessed)
    recent_progress = sorted(progress_records, key=lambda x: x.last_accessed, reverse=True)[:5]
    # Convert to schema
    recent_progress_schema = [
        LearningProgress(
            id=p.id,
            user_id=p.user_id,
            content_id=p.content_id,
            progress=p.progress,
            score=p.score,
            time_spent=p.time_spent,
            last_accessed=int(p.last_accessed.timestamp()),
            completed_sections=p.completed_sections or [],
            mastery_level=p.mastery_level.value if p.mastery_level else None
        ) for p in recent_progress
    ]
    return LearningProgressSummary(
        average_progress=average_progress,
        completion_rate=completion_rate,
        time_spent=time_spent,
        strengths=strengths,
        areas_for_improvement=areas_for_improvement,
        recent_progress=recent_progress_schema,
        progress_by_category=progress_by_category
    )

async def get_group_activity_summary(
    db: AsyncSession, 
    group_id: str, 
    teacher_id: str, 
    filters: Dict[str, Any] = {}
) -> List[GroupActivity]:
    # Verify teacher owns this group
    group = await db.get(Group, group_id)
    if not group or group.teacher_id != teacher_id:
        raise ValueError("Group not found or access denied")

    # Return real group activities
    result = await db.execute(
        select(GroupActivity).filter(GroupActivity.group_id == group_id)
    )
    return result.scalars().all()

async def get_teacher_overview(
    db: AsyncSession, 
    teacher_id: str
) -> TeacherDashboardOverview:
    # Real counts
    groups_count = await db.scalar(select(func.count(Group.id)).filter(Group.teacher_id == teacher_id))
    students_count = await db.scalar(
        select(func.count(Child.id)).join(Group, Child.group_id == Group.id).filter(Group.teacher_id == teacher_id)
    )
    content_count = await db.scalar(
        select(func.count(Content.id)).filter(Content.created_by == teacher_id)
    )
    device_count = await db.scalar(
        select(func.count(Device.id)).join(Group, Device.group_id == Group.id).filter(Group.teacher_id == teacher_id)
    )

    metrics = [
        DashboardMetric(title="Active Groups", value=groups_count or 0, description="Learning groups"),
        DashboardMetric(title="Total Students", value=students_count or 0, description="Across all groups"),
        DashboardMetric(title="Content Items", value=content_count or 0, description="Learning materials"),
        DashboardMetric(title="Active Devices", value=device_count or 0, description="Connected devices"),
    ]

    # Recent activity: last 5 group activities
    recent_activities = []
    result = await db.execute(
        select(GroupActivity).join(Group, GroupActivity.group_id == Group.id)
        .filter(Group.teacher_id == teacher_id)
        .order_by(GroupActivity.timestamp.desc())
        .limit(5)
    )
    for act in result.scalars().all():
        recent_activities.append(ActivityItem(
            id=act.id,
            text=f"{act.activity_type} ({act.description or ''})",
            timestamp=act.timestamp.isoformat(),
            type="activity",
            status="success"
        ))

    # System status: device sync status for teacher's groups
    system_status = []
    device_result = await db.execute(
        select(Device).join(Group, Device.group_id == Group.id).filter(Group.teacher_id == teacher_id)
    )
    for device in device_result.scalars().all():
        sync_status = device.sync_status
        if sync_status:
            system_status.append(SystemStatus(
                label=f"Device {device.device_id} Sync",
                value=f"{sync_status.used_mb}MB, {sync_status.doc_count} docs",
                status="success" if not sync_status.last_error else "warning"
            ))
        else:
            system_status.append(SystemStatus(
                label=f"Device {device.device_id} Sync",
                value="No data",
                status="warning"
            ))

    return TeacherDashboardOverview(
        metrics=metrics,
        recent_activity=recent_activities,
        system_status=system_status
    ) 