from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.base_class import Base


class PersonalizationSetting(Base):
    __tablename__ = "personalization_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=False)
    depth_level = Column(String, default="intermediate")  # beginner, intermediate, advanced
    adapted_content = Column(Text)  # Personalized version of the chapter content
    hardware_assumptions = Column(String, default="simulation")  # simulation, real_hardware, both
    examples_modified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="personalization_settings")
    chapter = relationship("Chapter", back_populates="personalization_settings")