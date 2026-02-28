from typing import Optional
from pydantic import BaseModel
from datetime import datetime


# Shared properties
class ChapterBase(BaseModel):
    title: str
    slug: str
    content: str
    order_num: int
    section: str
    is_published: bool = True


# Properties to receive via API on creation
class ChapterCreate(ChapterBase):
    title: str
    slug: str
    content: str
    order_num: int
    section: str


# Properties to receive via API on update
class ChapterUpdate(ChapterBase):
    title: Optional[str] = None
    slug: Optional[str] = None
    content: Optional[str] = None
    order_num: Optional[int] = None
    section: Optional[str] = None
    is_published: Optional[bool] = None


class ChapterInDBBase(ChapterBase):
    id: int
    content_html: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    author_id: Optional[int] = None

    class Config:
        from_attributes = True


# Additional properties to return via API
class Chapter(ChapterInDBBase):
    pass


# Additional properties stored in DB
class ChapterInDB(ChapterInDBBase):
    content_html: str