from typing import Optional
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.chapter import Chapter
from app.schemas.chapter import ChapterCreate, ChapterUpdate


class CRUDChapter(CRUDBase[Chapter, ChapterCreate, ChapterUpdate]):
    def get_by_slug(self, db: Session, *, slug: str) -> Optional[Chapter]:
        return db.query(Chapter).filter(Chapter.slug == slug).first()

    def get_by_section(self, db: Session, *, section: str):
        return db.query(Chapter).filter(Chapter.section == section).order_by(Chapter.order_num).all()

    def get_published(self, db: Session):
        return db.query(Chapter).filter(Chapter.is_published == True).order_by(Chapter.order_num).all()


chapter = CRUDChapter(Chapter)