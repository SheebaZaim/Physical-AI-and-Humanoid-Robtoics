"""
API routes for content personalization based on user background
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import os
from ...auth import get_current_user
from ...agents.skills.reusable_skills import ReusableSkills

router = APIRouter(prefix="/api/personalize", tags=["personalization"])

# Pydantic models
class PersonalizeRequest(BaseModel):
    content: str
    user_background: Optional[dict] = None  # Will be fetched from user profile if not provided
    personalization_level: str = "medium"  # low, medium, high
    content_type: str = "chapter"  # chapter, section, paragraph

class PersonalizeResponse(BaseModel):
    original_content: str
    personalized_content: str
    personalization_level: str
    applied_modifications: list

# Initialize the reusable skills for personalization
personalization_skills = ReusableSkills()

@router.post("/content", response_model=PersonalizeResponse)
async def personalize_content(
    request: PersonalizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Personalize content based on user's software and hardware background
    """
    try:
        # Get user profile if background not provided in request
        user_profile = request.user_background
        if not user_profile:
            # In a real implementation, we would fetch user profile from the database
            # For this example, we'll use mock data
            user_profile = {
                "software_background": "intermediate",
                "hardware_background": "basic"
            }

        # Use the reusable skills to generate personalized content
        personalized_content = await personalization_skills.generate_personalized_content(
            request.content,
            user_profile,
            request.content_type
        )

        # In a real implementation, we would call Claude to personalize the content
        # For this example, we'll return the content with a note about personalization

        return PersonalizeResponse(
            original_content=request.content,
            personalized_content=personalized_content,
            personalization_level=request.personalization_level,
            applied_modifications=["content_adaptation", "example_modification", "complexity_adjustment"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error personalizing content: {str(e)}"
        )

@router.post("/chapter")
async def personalize_chapter(
    request: PersonalizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Personalize an entire chapter based on user profile
    """
    try:
        # Get user profile if not provided
        user_profile = request.user_background
        if not user_profile:
            # In a real implementation, fetch from DB
            user_profile = {
                "software_background": "intermediate",
                "hardware_background": "basic"
            }

        # Personalize the chapter content
        personalized_content = await personalization_skills.generate_personalized_content(
            request.content,
            user_profile,
            "chapter"
        )

        return {
            "personalized_content": personalized_content,
            "status": "success",
            "user_profile_used": user_profile
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error personalizing chapter: {str(e)}"
        )

@router.get("/user-profile")
async def get_user_profile_for_personalization(current_user: dict = Depends(get_current_user)):
    """
    Get user profile specifically for personalization purposes
    """
    try:
        # In a real implementation, fetch from database
        # For this example, return mock profile
        return {
            "user_id": current_user["id"],
            "software_background": "intermediate",
            "hardware_background": "basic",
            "personalization_enabled": True,
            "last_personalization": "2023-12-07T10:00:00Z"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching user profile: {str(e)}"
        )

@router.get("/health")
async def personalization_health_check():
    """
    Health check for personalization API
    """
    return {"status": "ok", "service": "content-personalization-api"}