from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime


# Shared properties
class ChatSessionBase(BaseModel):
    session_token: str
    title: Optional[str] = None
    is_active: bool = True


# Properties to receive via API on creation
class ChatSessionCreate(ChatSessionBase):
    session_token: str
    title: Optional[str] = None


# Properties to receive via API on update
class ChatSessionUpdate(ChatSessionBase):
    title: Optional[str] = None
    is_active: Optional[bool] = None


class ChatSessionInDBBase(ChatSessionBase):
    id: int
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Additional properties to return via API
class ChatSession(ChatSessionInDBBase):
    messages: List['ChatMessage'] = []


# Additional properties stored in DB
class ChatSessionInDB(ChatSessionInDBBase):
    pass


# Chat Message schemas
class ChatMessageBase(BaseModel):
    session_id: int
    role: str  # 'user' or 'assistant'
    content: str


class ChatMessageCreate(ChatMessageBase):
    session_id: int
    role: str
    content: str


class ChatMessageUpdate(ChatMessageBase):
    content: Optional[str] = None


class ChatMessageInDBBase(ChatMessageBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ChatMessage(ChatMessageInDBBase):
    pass


class ChatMessageInDB(ChatMessageInDBBase):
    pass


# Update forward references
ChatSession.update_forward_refs()