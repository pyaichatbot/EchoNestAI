from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.notification import Notification
from app.db.models.models import Event, NotificationSetting
from app.schemas.notification import NotificationCreate, NotificationSettingsUpdate

class CRUDNotification:
    def create(self, db: Session, *, event_id: str, user_id: str) -> Notification:
        db_obj = Notification(
            user_id=user_id,
            event_id=event_id
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_user_notifications(
        self, db: Session, *, user_id: str, skip: int = 0, limit: int = 50
    ) -> List[Notification]:
        return (
            db.query(Notification)
            .filter(Notification.user_id == user_id)
            .order_by(desc(Notification.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_unread_count(self, db: Session, *, user_id: str) -> int:
        return (
            db.query(Notification)
            .filter(Notification.user_id == user_id, Notification.read == False)
            .count()
        )

    def mark_as_read(self, db: Session, *, notification_id: str, user_id: str) -> Optional[Notification]:
        notification = (
            db.query(Notification)
            .filter(Notification.id == notification_id, Notification.user_id == user_id)
            .first()
        )
        if notification:
            notification.read = True
            db.commit()
            db.refresh(notification)
        return notification

class CRUDNotificationSettings:
    def get_by_user_id(self, db: Session, *, user_id: str) -> Optional[NotificationSetting]:
        return (
            db.query(NotificationSetting)
            .filter(NotificationSetting.user_id == user_id)
            .first()
        )

    def create(self, db: Session, *, user_id: str) -> NotificationSetting:
        db_obj = NotificationSetting(user_id=user_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self, db: Session, *, db_obj: NotificationSetting, obj_in: NotificationSettingsUpdate
    ) -> NotificationSetting:
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db.commit()
        db.refresh(db_obj)
        return db_obj

notification = CRUDNotification()
notification_settings = CRUDNotificationSettings() 