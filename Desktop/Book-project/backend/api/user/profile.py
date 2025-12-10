"""
API routes for user profile management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import asyncpg
import os
from ...auth import get_current_user  # Assuming we have an auth module

router = APIRouter(prefix="/api/user", tags=["user"])

# Pydantic models
class UserProfileUpdate(BaseModel):
    software_background: Optional[str] = None
    hardware_background: Optional[str] = None

class UserProfileResponse(BaseModel):
    id: str
    user_id: str
    software_background: Optional[str] = None
    hardware_background: Optional[str] = None
    created_at: str
    updated_at: str

class UserPreferenceUpdate(BaseModel):
    preferred_language: Optional[str] = None
    personalization_enabled: Optional[bool] = None
    dark_mode_enabled: Optional[bool] = None

# Database connection function
async def get_db_connection():
    """Get database connection to Neon Postgres"""
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER", "user"),
        password=os.getenv("DB_PASSWORD", "password"),
        database=os.getenv("DB_NAME", "physical_ai_book")
    )
    return conn

@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """
    Get the current user's profile information
    """
    try:
        conn = await get_db_connection()
        try:
            query = """
                SELECT id, user_id, software_background, hardware_background,
                       created_at, updated_at
                FROM user_profiles
                WHERE user_id = $1
            """
            record = await conn.fetchrow(query, current_user["id"])

            if not record:
                # Create a default profile if it doesn't exist
                insert_query = """
                    INSERT INTO user_profiles (user_id)
                    VALUES ($1)
                    RETURNING id, user_id, software_background, hardware_background,
                             created_at, updated_at
                """
                record = await conn.fetchrow(insert_query, current_user["id"])

            return UserProfileResponse(
                id=record["id"],
                user_id=record["user_id"],
                software_background=record["software_background"],
                hardware_background=record["hardware_background"],
                created_at=record["created_at"].isoformat() if record["created_at"] else None,
                updated_at=record["updated_at"].isoformat() if record["updated_at"] else None
            )
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching user profile: {str(e)}"
        )

@router.put("/profile", response_model=UserProfileResponse)
async def update_user_profile(
    profile_update: UserProfileUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user profile information including software and hardware background
    """
    try:
        conn = await get_db_connection()
        try:
            # Check if profile exists, create if not
            query = """
                SELECT id FROM user_profiles WHERE user_id = $1
            """
            existing_profile = await conn.fetchrow(query, current_user["id"])

            if existing_profile:
                # Update existing profile
                update_query = """
                    UPDATE user_profiles
                    SET software_background = COALESCE($1, software_background),
                        hardware_background = COALESCE($2, hardware_background),
                        updated_at = NOW()
                    WHERE user_id = $3
                    RETURNING id, user_id, software_background, hardware_background,
                              created_at, updated_at
                """
                record = await conn.fetchrow(
                    update_query,
                    profile_update.software_background,
                    profile_update.hardware_background,
                    current_user["id"]
                )
            else:
                # Create new profile
                insert_query = """
                    INSERT INTO user_profiles (user_id, software_background, hardware_background)
                    VALUES ($1, $2, $3)
                    RETURNING id, user_id, software_background, hardware_background,
                              created_at, updated_at
                """
                record = await conn.fetchrow(
                    insert_query,
                    current_user["id"],
                    profile_update.software_background,
                    profile_update.hardware_background
                )

            if not record:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to update user profile"
                )

            return UserProfileResponse(
                id=record["id"],
                user_id=record["user_id"],
                software_background=record["software_background"],
                hardware_background=record["hardware_background"],
                created_at=record["created_at"].isoformat() if record["created_at"] else None,
                updated_at=record["updated_at"].isoformat() if record["updated_at"] else None
            )
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user profile: {str(e)}"
        )

@router.get("/preferences")
async def get_user_preferences(current_user: dict = Depends(get_current_user)):
    """
    Get user preferences
    """
    try:
        conn = await get_db_connection()
        try:
            query = """
                SELECT preferred_language, personalization_enabled, dark_mode_enabled
                FROM user_preferences
                WHERE user_id = $1
            """
            record = await conn.fetchrow(query, current_user["id"])

            if not record:
                # Create default preferences
                insert_query = """
                    INSERT INTO user_preferences (user_id)
                    VALUES ($1)
                    RETURNING preferred_language, personalization_enabled, dark_mode_enabled
                """
                record = await conn.fetchrow(insert_query, current_user["id"])

            return {
                "preferred_language": record["preferred_language"],
                "personalization_enabled": record["personalization_enabled"],
                "dark_mode_enabled": record["dark_mode_enabled"]
            }
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching user preferences: {str(e)}"
        )

@router.put("/preferences")
async def update_user_preferences(
    preferences_update: UserPreferenceUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user preferences
    """
    try:
        conn = await get_db_connection()
        try:
            # Check if preferences exist, create if not
            query = """
                SELECT user_id FROM user_preferences WHERE user_id = $1
            """
            existing_preferences = await conn.fetchrow(query, current_user["id"])

            if existing_preferences:
                # Update existing preferences
                update_query = """
                    UPDATE user_preferences
                    SET preferred_language = COALESCE($1, preferred_language),
                        personalization_enabled = COALESCE($2, personalization_enabled),
                        dark_mode_enabled = COALESCE($3, dark_mode_enabled)
                    WHERE user_id = $4
                    RETURNING preferred_language, personalization_enabled, dark_mode_enabled
                """
                record = await conn.fetchrow(
                    update_query,
                    preferences_update.preferred_language,
                    preferences_update.personalization_enabled,
                    preferences_update.dark_mode_enabled,
                    current_user["id"]
                )
            else:
                # Create new preferences
                insert_query = """
                    INSERT INTO user_preferences (
                        user_id,
                        preferred_language,
                        personalization_enabled,
                        dark_mode_enabled
                    )
                    VALUES ($1, $2, $3, $4)
                    RETURNING preferred_language, personalization_enabled, dark_mode_enabled
                """
                record = await conn.fetchrow(
                    insert_query,
                    current_user["id"],
                    preferences_update.preferred_language,
                    preferences_update.personalization_enabled,
                    preferences_update.dark_mode_enabled
                )

            if not record:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to update user preferences"
                )

            return {
                "preferred_language": record["preferred_language"],
                "personalization_enabled": record["personalization_enabled"],
                "dark_mode_enabled": record["dark_mode_enabled"]
            }
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user preferences: {str(e)}"
        )

# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check for user API
    """
    return {"status": "ok", "service": "user-profile-api"}