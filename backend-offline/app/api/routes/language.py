from typing import Dict, List, Any, Optional
import json
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.schemas.chat import (
    LanguageDetectionRequest, LanguageDetectionResponse,
    SupportedLanguage
)
from app.api.deps.auth import get_current_active_user
from app.services.language_service import detect_language, get_supported_languages

router = APIRouter(tags=["language"])

@router.get("/supported", response_model=List[SupportedLanguage])
async def list_supported_languages() -> Any:
    """
    Get list of supported languages with their capabilities.
    """
    return get_supported_languages()

@router.post("/detect", response_model=LanguageDetectionResponse)
async def detect_text_language(
    request: LanguageDetectionRequest
) -> Any:
    """
    Detect language of input text.
    """
    return await detect_language(request.text)
