"""
API routes for content translation (Urdu, Roman Urdu, Arabic, German)
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import os
from ...auth import get_current_user  # Assuming we have an auth module
from ...agents.subagents.translation_helper_agent import TranslationHelper

router = APIRouter(prefix="/api/translate", tags=["translation"])

# Pydantic models
class TranslationRequest(BaseModel):
    text: str
    target_language: str  # en, ur, ru, ar, de
    source_language: str = "en"
    preserve_formatting: bool = True

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    translation_quality: Optional[str] = None

# Initialize the translation helper
translation_helper = TranslationHelper()

@router.post("/", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Translate text between supported languages
    """
    try:
        # Validate target language
        supported_languages = ["en", "ur", "ru", "ar", "de"]
        if request.target_language not in supported_languages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target language '{request.target_language}' not supported. Supported languages: {supported_languages}"
            )

        if request.source_language not in supported_languages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Source language '{request.source_language}' not supported. Supported languages: {supported_languages}"
            )

        # Perform translation using the TranslationHelper agent
        if request.target_language == "ur":
            translated_text = await translation_helper.translate_to_urdu(
                request.text,
                request.preserve_formatting
            )
        elif request.target_language == "ru":  # Roman Urdu
            translated_text = await translation_helper.translate_to_roman_urdu(
                request.text,
                request.preserve_formatting
            )
        elif request.target_language == "ar":
            translated_text = await translation_helper.translate_to_arabic(
                request.text,
                request.preserve_formatting
            )
        elif request.target_language == "de":
            translated_text = await translation_helper.translate_to_german(
                request.text,
                request.preserve_formatting
            )
        else:  # Default to English
            translated_text = request.text

        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            translation_quality="auto_generated"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating text: {str(e)}"
        )

@router.post("/chapter")
async def translate_chapter(
    request: TranslationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Translate an entire chapter with proper handling of structure
    """
    try:
        # Validate target language
        supported_languages = ["en", "ur", "ru", "ar", "de"]
        if request.target_language not in supported_languages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target language '{request.target_language}' not supported"
            )

        # Use the translation helper to translate the chapter
        translated_content = await translation_helper.translate_chapter(
            request.text,
            request.target_language
        )

        return {
            "translated_content": translated_content,
            "target_language": request.target_language,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating chapter: {str(e)}"
        )

@router.get("/supported-languages")
async def get_supported_languages(current_user: dict = Depends(get_current_user)):
    """
    Get list of supported languages for translation
    """
    return {
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "ur", "name": "Urdu"},
            {"code": "ru", "name": "Roman Urdu"},
            {"code": "ar", "name": "Arabic"},
            {"code": "de", "name": "German"}
        ],
        "total_supported": 5
    }

@router.get("/health")
async def translation_health_check():
    """
    Health check for translation API
    """
    return {"status": "ok", "service": "content-translation-api"}