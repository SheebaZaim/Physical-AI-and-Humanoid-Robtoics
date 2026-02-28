from typing import Optional
from pydantic import BaseModel
from datetime import datetime


# Shared properties
class TranslationBase(BaseModel):
    chapter_id: int
    language_code: str  # e.g., 'ur' for Urdu
    translated_content: str
    is_approved: bool = False


# Properties to receive via API on creation
class TranslationCreate(TranslationBase):
    chapter_id: int
    language_code: str
    translated_content: str


# Properties to receive via API on update
class TranslationUpdate(TranslationBase):
    translated_content: Optional[str] = None
    is_approved: Optional[bool] = None


class TranslationInDBBase(TranslationBase):
    id: int
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Additional properties to return via API
class Translation(TranslationInDBBase):
    pass


# Additional properties stored in DB
class TranslationInDB(TranslationInDBBase):
    pass