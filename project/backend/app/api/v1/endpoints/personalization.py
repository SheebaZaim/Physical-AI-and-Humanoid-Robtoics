from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()


@router.get("/", response_model=List[schemas.PersonalizationSetting])
def read_personalization_settings(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    Retrieve personalization settings.
    """
    personalization_settings = crud.personalization_setting.get_multi(db, skip=skip, limit=limit)
    return personalization_settings


@router.post("/", response_model=schemas.PersonalizationSetting)
def create_personalization_setting(
    *,
    db: Session = Depends(deps.get_db),
    personalization_in: schemas.PersonalizationSettingCreate,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Create new personalization setting.
    """
    # Ensure the user can only create settings for themselves
    if personalization_in.user_id != current_user.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot create personalization settings for another user",
        )

    personalization = crud.personalization_setting.create(db, obj_in=personalization_in)
    return personalization


@router.get("/{personalization_id}", response_model=schemas.PersonalizationSetting)
def read_personalization_setting_by_id(
    personalization_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db)
):
    """
    Get a specific personalization setting by id.
    """
    personalization = crud.personalization_setting.get(db, id=personalization_id)
    if not personalization:
        raise HTTPException(
            status_code=404,
            detail="Personalization setting not found",
        )
    # Ensure the user can only access their own settings
    if personalization.user_id != current_user.id:
        raise HTTPException(
            status_code=400,
            detail="Not authorized to access this personalization setting",
        )
    return personalization


@router.put("/{personalization_id}", response_model=schemas.PersonalizationSetting)
def update_personalization_setting(
    *,
    db: Session = Depends(deps.get_db),
    personalization_id: int,
    personalization_in: schemas.PersonalizationSettingUpdate,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Update a personalization setting.
    """
    personalization = crud.personalization_setting.get(db, id=personalization_id)
    if not personalization:
        raise HTTPException(
            status_code=404,
            detail="Personalization setting not found",
        )
    # Ensure the user can only update their own settings
    if personalization.user_id != current_user.id:
        raise HTTPException(
            status_code=400,
            detail="Not authorized to update this personalization setting",
        )
    personalization = crud.personalization_setting.update(db, db_obj=personalization, obj_in=personalization_in)
    return personalization


@router.delete("/{personalization_id}", response_model=schemas.PersonalizationSetting)
def delete_personalization_setting(
    *,
    db: Session = Depends(deps.get_db),
    personalization_id: int,
    current_user: models.User = Depends(deps.get_current_active_user)
):
    """
    Delete a personalization setting.
    """
    personalization = crud.personalization_setting.get(db, id=personalization_id)
    if not personalization:
        raise HTTPException(
            status_code=404,
            detail="Personalization setting not found",
        )
    # Ensure the user can only delete their own settings
    if personalization.user_id != current_user.id:
        raise HTTPException(
            status_code=400,
            detail="Not authorized to delete this personalization setting",
        )
    personalization = crud.personalization_setting.remove(db, id=personalization_id)
    return personalization