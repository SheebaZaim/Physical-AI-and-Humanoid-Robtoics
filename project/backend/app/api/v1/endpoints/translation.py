from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app import crud, models, schemas
from app.api import deps
from app.core.config import settings
from ai.translation.translation_service import TranslationService

router = APIRouter()


class TranslateRequest(BaseModel):
    text: str
    target_language: str = "urdu"
    source_language: str = "English"


@router.post("/translate")
async def translate_text(request: TranslateRequest):
    """
    Translate text to target language using AI (no authentication required for demo).
    """
    try:
        # Initialize translation service
        translation_service = TranslationService(
            openai_api_key=settings.OPENAI_API_KEY,
            openrouter_api_key=settings.OPENROUTER_API_KEY,
            openrouter_base_url=settings.OPENROUTER_API_BASE,
            openrouter_model=settings.OPENROUTER_MODEL
        )

        # Translate the text
        translated_text = await translation_service.translate_text(
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language
        )

        return {
            "original_text": request.text,
            "translated_text": translated_text,
            "target_language": request.target_language,
            "source_language": request.source_language
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation error: {str(e)}"
        )


@router.get("/", response_model=List[schemas.Translation])
def read_translations(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    Retrieve translations.
    """
    translations = crud.translation.get_multi(db, skip=skip, limit=limit)
    return translations


@router.post("/", response_model=schemas.Translation)
def create_translation(
    *,
    db: Session = Depends(deps.get_db),
    translation_in: schemas.TranslationCreate,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Create new translation.
    """
    # Set the user who is requesting the translation
    translation_in_data = translation_in.dict()
    translation_in_data['user_id'] = current_user.id

    translation = crud.translation.create(db, obj_in=schemas.TranslationCreate(**translation_in_data))
    return translation


@router.get("/{translation_id}", response_model=schemas.Translation)
def read_translation_by_id(
    translation_id: int,
    db: Session = Depends(deps.get_db)
):
    """
    Get a specific translation by id.
    """
    translation = crud.translation.get(db, id=translation_id)
    if not translation:
        raise HTTPException(
            status_code=404,
            detail="Translation not found",
        )
    return translation


@router.put("/{translation_id}", response_model=schemas.Translation)
def update_translation(
    *,
    db: Session = Depends(deps.get_db),
    translation_id: int,
    translation_in: schemas.TranslationUpdate,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Update a translation.
    """
    translation = crud.translation.get(db, id=translation_id)
    if not translation:
        raise HTTPException(
            status_code=404,
            detail="Translation not found",
        )
    # Only allow updates if the user is a superuser or the creator
    if not current_user.is_superuser and translation.user_id != current_user.id:
        raise HTTPException(
            status_code=400,
            detail="Not authorized to update this translation",
        )
    translation = crud.translation.update(db, db_obj=translation, obj_in=translation_in)
    return translation


@router.delete("/{translation_id}", response_model=schemas.Translation)
def delete_translation(
    *,
    db: Session = Depends(deps.get_db),
    translation_id: int,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Delete a translation.
    """
    translation = crud.translation.get(db, id=translation_id)
    if not translation:
        raise HTTPException(
            status_code=404,
            detail="Translation not found",
        )
    # Only allow deletion if the user is a superuser or the creator
    if not current_user.is_superuser and translation.user_id != current_user.id:
        raise HTTPException(
            status_code=400,
            detail="Not authorized to delete this translation",
        )
    translation = crud.translation.remove(db, id=translation_id)
    return translation