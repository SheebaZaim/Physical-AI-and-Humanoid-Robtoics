from sqlalchemy import Boolean, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.base_class import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)

    # Software and hardware background collected during registration
    software_background = Column(Text)
    hardware_background = Column(Text)

    # Personalization settings
    preferred_depth = Column(String, default="intermediate")  # beginner, intermediate, advanced
    preferred_hardware_assumptions = Column(String, default="simulation")  # simulation, real_hardware, both

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    authored_chapters = relationship("Chapter", back_populates="author")
    personalization_settings = relationship("PersonalizationSetting", back_populates="user")
    translations = relationship("Translation", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")