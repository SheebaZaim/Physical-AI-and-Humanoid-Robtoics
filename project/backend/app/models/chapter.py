from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.base_class import Base


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    slug = Column(String, unique=True, index=True, nullable=False)
    content = Column(Text, nullable=False)  # Markdown content
    content_html = Column(Text)  # Rendered HTML content
    order_num = Column(Integer, nullable=False)  # Order in the book
    section = Column(String, nullable=False)  # Which section/part of the book
    is_published = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    author_id = Column(Integer, ForeignKey("users.id"))
    author = relationship("User", back_populates="authored_chapters")

    # Related data
    personalization_settings = relationship("PersonalizationSetting", back_populates="chapter")
    translations = relationship("Translation", back_populates="chapter")


# Add the relationship to User model in a separate file or append to user model