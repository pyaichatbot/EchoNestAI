from fastapi import APIRouter, Depends, HTTPException, status, Form
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import os
import tempfile
import pyttsx3

from app.db.database import get_db
from app.api.deps.auth import get_current_active_user
from app.services.language_service import get_language_model
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("voice_tts")

router = APIRouter()

@router.post("/voice/tts")
async def text_to_speech_endpoint(
    text: str = Form(...),
    language: str = Form("en"),
    voice_id: Optional[str] = Form(None),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Convert text to speech in the specified language.
    
    Args:
        text: Text to convert to speech
        language: Language code
        voice_id: Optional voice ID
        
    Returns:
        Path to generated audio file
    """
    try:
        # Ensure output directory exists
        audio_output_folder = os.path.join(settings.MEDIA_ROOT, "audio")
        os.makedirs(audio_output_folder, exist_ok=True)
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=audio_output_folder) as temp_file:
            output_path = temp_file.name
        
        # Get appropriate TTS model for language
        language_models = get_language_model(language)
        
        # Use pyttsx3 for basic TTS
        engine = pyttsx3.init()
        
        # Set voice properties
        voices = engine.getProperty('voices')
        
        # Select voice based on language
        if language == "en":
            engine.setProperty('voice', voices[0].id)  # English voice
        elif language in ["hi", "te", "ta"]:
            # Try to find an Indian voice, fallback to default
            indian_voice = next((v for v in voices if "indian" in v.name.lower()), None)
            if indian_voice:
                engine.setProperty('voice', indian_voice.id)
        elif language == "de":
            # Try to find a German voice, fallback to default
            german_voice = next((v for v in voices if "german" in v.name.lower()), None)
            if german_voice:
                engine.setProperty('voice', german_voice.id)
        
        # Set other properties
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Generate speech
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        logger.info(f"Generated speech for text in {language}: {text[:50]}...")
        
        # Get relative path for response
        relative_path = os.path.relpath(output_path, settings.MEDIA_ROOT)
        media_url = f"/media/{relative_path}"
        
        return {
            "status": "success",
            "audio_url": media_url,
            "language": language,
            "text": text
        }
    
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating speech: {str(e)}"
        )
