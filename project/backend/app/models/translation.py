from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.base_class import Base


class Translation(Base):
    __tablename__ = "translations"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))  # Who requested the translation
    language_code = Column(String(10), nullable=False)  # e.g., 'ur' for Urdu
    translated_content = Column(Text, nullable=False)  # Translated content
    is_approved = Column(Boolean, default=False)  # Whether translation is approved
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    chapter = relationship("Chapter", back_populates="translations")
    user = relationship("User", back_populates="translations")