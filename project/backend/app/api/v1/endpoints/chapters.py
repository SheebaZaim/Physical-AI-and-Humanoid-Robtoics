from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from app import crud, models, schemas
from app.api import deps
from app.services.content_indexing_service import ContentIndexingService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=List[schemas.Chapter])
def read_chapters(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    Retrieve chapters.
    """
    chapters = crud.chapter.get_multi(db, skip=skip, limit=limit)
    return chapters


@router.post("/", response_model=schemas.Chapter)
async def create_chapter(
    *,
    db: Session = Depends(deps.get_db),
    chapter_in: schemas.ChapterCreate,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Create new chapter and index its content.
    """
    # Set the author to the current user
    chapter_in_data = chapter_in.dict()
    chapter_in_data['author_id'] = current_user.id

    chapter = crud.chapter.create(db, obj_in=schemas.ChapterCreate(**chapter_in_data))

    # Index the content in the vector database
    indexing_service = ContentIndexingService()
    try:
        await indexing_service.index_new_chapter(chapter)
    except Exception as e:
        logger.error(f"Failed to index chapter {chapter.id}: {e}")
        # Still return the chapter even if indexing failed
        pass

    return chapter


@router.get("/{chapter_id}", response_model=schemas.Chapter)
def read_chapter_by_id(
    chapter_id: int,
    db: Session = Depends(deps.get_db)
):
    """
    Get a specific chapter by id.
    """
    chapter = crud.chapter.get(db, id=chapter_id)
    if not chapter:
        raise HTTPException(
            status_code=404,
            detail="Chapter not found",
        )
    return chapter


@router.put("/{chapter_id}", response_model=schemas.Chapter)
async def update_chapter(
    *,
    db: Session = Depends(deps.get_db),
    chapter_id: int,
    chapter_in: schemas.ChapterUpdate,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Update a chapter and re-index its content.
    """
    chapter = crud.chapter.get(db, id=chapter_id)
    if not chapter:
        raise HTTPException(
            status_code=404,
            detail="Chapter not found",
        )

    # Check if the user is authorized to update this chapter
    if chapter.author_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to update this chapter"
        )

    chapter = crud.chapter.update(db, db_obj=chapter, obj_in=chapter_in)

    # Re-index the updated content
    indexing_service = ContentIndexingService()
    try:
        await indexing_service.update_indexed_content(chapter)
    except Exception as e:
        logger.error(f"Failed to re-index chapter {chapter.id}: {e}")
        # Still return the chapter even if re-indexing failed
        pass

    return chapter


@router.delete("/{chapter_id}", response_model=schemas.Chapter)
async def delete_chapter(
    *,
    db: Session = Depends(deps.get_db),
    chapter_id: int,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Delete a chapter and remove it from the index.
    """
    chapter = crud.chapter.get(db, id=chapter_id)
    if not chapter:
        raise HTTPException(
            status_code=404,
            detail="Chapter not found",
        )

    # Check if the user is authorized to delete this chapter
    if chapter.author_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to delete this chapter"
        )

    # Remove from the vector index
    indexing_service = ContentIndexingService()
    try:
        await indexing_service.remove_from_index(chapter_id)
    except Exception as e:
        logger.error(f"Failed to remove chapter {chapter_id} from index: {e}")
        # Continue with deletion even if removal from index failed
        pass

    chapter = crud.chapter.remove(db, id=chapter_id)
    return chapter