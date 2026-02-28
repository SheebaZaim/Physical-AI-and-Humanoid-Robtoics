from typing import Optional
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.chat_session import ChatSession, ChatMessage
from app.schemas.chat_session import ChatSessionCreate, ChatSessionUpdate, ChatMessageCreate, ChatMessageUpdate


class CRUDChatSession(CRUDBase[ChatSession, ChatSessionCreate, ChatSessionUpdate]):
    def get_by_session_token(self, db: Session, *, session_token: str) -> Optional[ChatSession]:
        return db.query(ChatSession).filter(ChatSession.session_token == session_token).first()

    def get_by_user(self, db: Session, *, user_id: int):
        return db.query(ChatSession).filter(ChatSession.user_id == user_id).all()


class CRUDChatMessage(CRUDBase[ChatMessage, ChatMessageCreate, ChatMessageUpdate]):
    def get_by_session(self, db: Session, *, session_id: int):
        return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at).all()

    def get_messages_by_session_and_role(self, db: Session, *, session_id: int, role: str):
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id,
            ChatMessage.role == role
        ).order_by(ChatMessage.created_at).all()


chat_session = CRUDChatSession(ChatSession)
chat_message = CRUDChatMessage(ChatMessage)