from typing import Optional
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.personalization_setting import PersonalizationSetting
from app.schemas.personalization_setting import PersonalizationSettingCreate, PersonalizationSettingUpdate


class CRUDPersonalizationSetting(CRUDBase[PersonalizationSetting, PersonalizationSettingCreate, PersonalizationSettingUpdate]):
    def get_by_user_and_chapter(self, db: Session, *, user_id: int, chapter_id: int) -> Optional[PersonalizationSetting]:
        return db.query(PersonalizationSetting).filter(
            PersonalizationSetting.user_id == user_id,
            PersonalizationSetting.chapter_id == chapter_id
        ).first()

    def get_by_user(self, db: Session, *, user_id: int):
        return db.query(PersonalizationSetting).filter(PersonalizationSetting.user_id == user_id).all()

    def get_by_chapter(self, db: Session, *, chapter_id: int):
        return db.query(PersonalizationSetting).filter(PersonalizationSetting.chapter_id == chapter_id).all()


personalization_setting = CRUDPersonalizationSetting(PersonalizationSetting)