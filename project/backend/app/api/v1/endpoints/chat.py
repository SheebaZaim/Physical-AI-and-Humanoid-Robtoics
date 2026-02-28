from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from app import crud, models, schemas
from app.api import deps
from app.services.ai_service import AIService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/session", response_model=schemas.ChatSession)
def create_chat_session(
    *,
    db: Session = Depends(deps.get_db),
    session_in: schemas.ChatSessionCreate,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Create new chat session.
    """
    # Set the user for the session
    session_in_data = session_in.dict()
    session_in_data['user_id'] = current_user.id

    session = crud.chat_session.create(db, obj_in=schemas.ChatSessionCreate(**session_in_data))
    return session


@router.get("/session/{session_id}", response_model=schemas.ChatSession)
def read_chat_session_by_id(
    session_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db)
):
    """
    Get a specific chat session by id.
    """
    session = crud.chat_session.get(db, id=session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found",
        )
    # Ensure the user can only access their own sessions
    if session.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=400,
            detail="Not authorized to access this chat session",
        )
    return session


@router.post("/message", response_model=schemas.ChatMessage)
def create_chat_message(
    *,
    db: Session = Depends(deps.get_db),
    message_in: schemas.ChatMessageCreate,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Create new chat message.
    """
    # Verify that the user has access to this session
    session = crud.chat_session.get(db, id=message_in.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found",
        )
    if session.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=400,
            detail="Not authorized to add messages to this chat session",
        )

    message = crud.chat_message.create(db, obj_in=message_in)
    return message


@router.get("/session/{session_id}/messages", response_model=List[schemas.ChatMessage])
def read_chat_messages_by_session(
    session_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db)
):
    """
    Get all messages for a specific chat session.
    """
    session = crud.chat_session.get(db, id=session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found",
        )
    # Ensure the user can only access their own sessions
    if session.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=400,
            detail="Not authorized to access this chat session",
        )

    messages = crud.chat_message.get_by_session(db, session_id=session_id)
    return messages


@router.get("/debug-ai", response_model=Dict[str, Any])
async def debug_ai_status():
    """Check AI service configuration and test search (no auth required)."""
    from app.core.config import settings
    result = {
        "gemini_key_set": bool(settings.GEMINI_API_KEY),
        "qdrant_url_set": bool(settings.QDRANT_URL),
        "qdrant_api_key_set": bool(settings.QDRANT_API_KEY),
        "openrouter_key_set": bool(settings.OPENROUTER_API_KEY),
        "openrouter_model": settings.OPENROUTER_MODEL,
    }
    try:
        ai_service = AIService()
        search = await ai_service.search_similar_content("humanoid robotics", limit=2)
        result["search_results_count"] = len(search)
        result["use_gemini"] = ai_service.ai_orchestrator.embedding_engine.use_gemini
        result["search_sample"] = search[0]["content"][:100] if search else "NO RESULTS"
    except Exception as e:
        result["search_error"] = str(e)
    return result


@router.post("/public-ask", response_model=Dict[str, Any])
async def public_ask_question(
    *,
    question: str,
    session_id: str = None,
    chapter_id: str = None,
    selected_text: str = "",
):
    """
    Ask a question to the AI assistant (no authentication required).
    """
    try:
        ai_service = AIService()

        if selected_text:
            response = await ai_service.get_contextual_response(
                query=question,
                selected_text=selected_text,
                session_id=session_id
            )
        elif chapter_id:
            response = await ai_service.get_chapter_specific_response(
                query=question,
                chapter_id=chapter_id,
                session_id=session_id
            )
        else:
            response = await ai_service.get_chat_response(
                query=question,
                session_id=session_id
            )

        return response
    except Exception as e:
        logger.error(f"Error processing public AI request: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error processing your request with the AI assistant"
        )


@router.post("/ask", response_model=Dict[str, Any])
async def ask_question(
    *,
    question: str,
    session_id: str = None,
    chapter_id: str = None,
    selected_text: str = "",
    current_user: models.User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db)
):
    """
    Ask a question to the AI assistant.
    """
    try:
        ai_service = AIService()

        if selected_text:
            # If specific text is selected, use contextual response
            response = await ai_service.get_contextual_response(
                query=question,
                selected_text=selected_text,
                session_id=session_id
            )
        elif chapter_id:
            # If a specific chapter is targeted, use chapter-specific response
            response = await ai_service.get_chapter_specific_response(
                query=question,
                chapter_id=chapter_id,
                session_id=session_id
            )
        else:
            # Otherwise, use general chat response with user preferences
            response = await ai_service.get_chat_response(
                query=question,
                session_id=session_id,
                user_id=current_user.id,
                db=db
            )

        return response
    except Exception as e:
        logger.error(f"Error processing AI request: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error processing your request with the AI assistant"
        )


@router.post("/translate", response_model=Dict[str, str])
async def translate_content(
    *,
    text: str,
    target_language: str,
    source_language: str = "English",
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Translate content to the target language.
    """
    try:
        ai_service = AIService()

        translated = await ai_service.translate_content(
            text=text,
            target_language=target_language,
            source_language=source_language
        )

        return {"translated_text": translated}
    except Exception as e:
        logger.error(f"Error translating content: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error translating content"
        )


@router.get("/content-stats", response_model=Dict[str, Any])
async def get_content_statistics():
    """
    Get statistics about the indexed content.
    """
    try:
        ai_service = AIService()

        stats = await ai_service.get_content_index_stats()

        return stats
    except Exception as e:
        logger.error(f"Error getting content stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving content statistics"
        )


@router.post("/search", response_model=List[Dict[str, Any]])
async def search_content(
    *,
    query: str,
    limit: int = 5,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Search for similar content in the book.
    """
    try:
        ai_service = AIService()

        results = await ai_service.search_similar_content(
            query=query,
            limit=limit
        )

        return results
    except Exception as e:
        logger.error(f"Error searching content: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error searching content"
        )