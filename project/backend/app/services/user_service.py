from typing import Optional
from sqlalchemy.orm import Session

from app.models.user import User
from app.database.session import SessionLocal


class UserService:
    def __init__(self, db: Optional[Session] = None):
        self.db = db or SessionLocal()

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get a user by username"""
        try:
            user = self.db.query(User).filter(User.username == username).first()
            return user
        except Exception:
            return None

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get a user by email"""
        try:
            user = self.db.query(User).filter(User.email == email).first()
            return user
        except Exception:
            return None

    def close(self):
        """Close the database session"""
        self.db.close()