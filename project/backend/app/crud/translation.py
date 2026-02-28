from typing import Optional
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.translation import Translation
from app.schemas.translation import TranslationCreate, TranslationUpdate


class CRUDTranslation(CRUDBase[Translation, TranslationCreate, TranslationUpdate]):
    def get_by_chapter_and_language(self, db: Session, *, chapter_id: int, language_code: str) -> Optional[Translation]:
        return db.query(Translation).filter(
            Translation.chapter_id == chapter_id,
            Translation.language_code == language_code
        ).first()

    def get_by_chapter(self, db: Session, *, chapter_id: int):
        return db.query(Translation).filter(Translation.chapter_id == chapter_id).all()

    def get_by_language(self, db: Session, *, language_code: str):
        return db.query(Translation).filter(Translation.language_code == language_code).all()

    def get_approved_translations(self, db: Session, *, language_code: str):
        return db.query(Translation).filter(
            Translation.language_code == language_code,
            Translation.is_approved == True
        ).all()


translation = CRUDTranslation(Translation)