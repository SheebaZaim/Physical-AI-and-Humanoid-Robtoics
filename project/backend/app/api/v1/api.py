from fastapi import APIRouter

from app.api.v1.endpoints import users, chapters, chat, personalization, translation, ai_setup

api_router = APIRouter()
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(chapters.router, prefix="/chapters", tags=["chapters"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(personalization.router, prefix="/personalization", tags=["personalization"])
api_router.include_router(translation.router, prefix="/translation", tags=["translation"])
api_router.include_router(ai_setup.router, prefix="/ai", tags=["ai"])