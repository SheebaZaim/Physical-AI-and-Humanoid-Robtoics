from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging
from app.api import deps
from app.models.user import User
from app.services.ai_service import AIService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/initialize-content-index", response_model=dict)
async def initialize_content_index(
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db)
):
    """
    Initialize or re-index all book content in the vector database.
    This endpoint should typically only be called by admin users.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superusers can initialize content index"
        )

    try:
        ai_service = AIService()

        result = await ai_service.index_all_chapters(db)

        logger.info(f"Content indexing completed: {result}")

        return {
            "message": "Content indexing completed successfully",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error initializing content index: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing content index: {str(e)}"
        )


@router.post("/index-chapter/{chapter_id}", response_model=dict)
async def index_single_chapter(
    chapter_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db)
):
    """
    Index a single chapter in the vector database.
    """
    if not current_user.is_superuser and not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only active users can index chapters"
        )

    try:
        # Get the chapter from the database
        from app.crud.chapter import chapter
        chapter_obj = chapter.get(db, id=chapter_id)

        if not chapter_obj:
            raise HTTPException(
                status_code=404,
                detail="Chapter not found"
            )

        ai_service = AIService()

        result = await ai_service.index_chapter_content(chapter_obj)

        logger.info(f"Chapter {chapter_id} indexing completed: {result}")

        return {
            "message": f"Chapter {chapter_id} indexing completed successfully",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error indexing chapter {chapter_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing chapter: {str(e)}"
        )