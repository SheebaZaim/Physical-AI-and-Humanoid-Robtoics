from typing import Generator
from sqlalchemy.orm import Session

from app.database.session import SessionLocal
from app.core.security import get_current_active_user
from app.models.user import User


def get_db() -> Generator[Session, None, None]:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()