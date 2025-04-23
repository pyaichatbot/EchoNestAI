from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import os
import tempfile
import json

from app.db.database import get_db
from app.db.schemas.chat import (
    LanguageDetectionRequest, LanguageDetectionResponse,
    SupportedLanguage
)
from app.api.deps.auth import get_current_active_user, get_device_auth
from app.services.language_service import detect_language, get_supported_languages
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("language_routes")

router = APIRouter(tags=["language"])

@router.get("/languages/supported", response_model=List[SupportedLanguage])
async def list_supported_languages() -> Any:
    """
    Get list of supported languages with their capabilities.
    """
    return get_supported_languages()

@router.post("/languages/detect", response_model=LanguageDetectionResponse)
async def detect_text_language(
    request: LanguageDetectionRequest
) -> Any:
    """
    Detect language of input text.
    """
    return await detect_language(request.text)

@router.post("/languages/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Transcribe audio to text in the specified language.
    If language is not provided, it will be auto-detected.
    """
    # Create temporary file to store uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as temp_file:
        temp_path = temp_file.name
        
        # Write uploaded file to temporary file
        content = await audio.read()
        temp_file.write(content)
    
    try:
        # Process audio file
        import whisper
        
        # Ensure models directory exists
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
        
        # Set environment variable for model download location
        os.environ["WHISPER_MODELS_DIR"] = settings.MODELS_FOLDER
        
        # Load model
        model = whisper.load_model("base")
        
        # Transcribe audio
        if language:
            # Use specified language
            result = model.transcribe(
                temp_path,
                language=language,
                fp16=False
            )
        else:
            # Auto-detect language
            result = model.transcribe(
                temp_path,
                fp16=False
            )
        
        # Extract results
        text = result["text"]
        detected_language = result.get("language", "en")
        
        # If no text was transcribed, return empty result
        if not text.strip():
            logger.warning(f"No speech detected in audio file: {audio.filename}")
            return {
                "text": "",
                "detected_language": detected_language,
                "confidence": 0.0
            }
        
        # If language wasn't provided, verify with our language detector
        if not language:
            language_detection = await detect_language(text)
            detected_language = language_detection["detected_language"]
            confidence = language_detection["confidence"]
        else:
            confidence = 0.9  # Default confidence when language is specified
        
        logger.info(f"Transcribed audio in {detected_language}: {text[:50]}...")
        
        return {
            "text": text,
            "detected_language": detected_language,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/languages/tts")
async def text_to_speech(
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
        from app.services.language_service import get_language_model
        language_models = get_language_model(language)
        
        # Use pyttsx3 for basic TTS
        import pyttsx3
        
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

@router.post("/languages/translate")
async def translate_text(
    text: str = Form(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Translate text from source language to target language.
    
    Args:
        text: Text to translate
        source_language: Source language code
        target_language: Target language code
        
    Returns:
        Translated text
    """
    try:
        # Import translation library
        from transformers import MarianMTModel, MarianTokenizer
        
        # Map language codes to model names
        model_map = {
            ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
            ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
            ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
            ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
            # For Tamil and Telugu, we might need to use a more general model
            ("en", "ta"): "Helsinki-NLP/opus-mt-en-mul",
            ("ta", "en"): "Helsinki-NLP/opus-mt-mul-en",
            ("en", "te"): "Helsinki-NLP/opus-mt-en-mul",
            ("te", "en"): "Helsinki-NLP/opus-mt-mul-en"
        }
        
        # Get model name
        model_name = model_map.get((source_language, target_language))
        
        if not model_name:
            logger.warning(f"No translation model found for {source_language} to {target_language}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Translation from {source_language} to {target_language} is not supported"
            )
        
        # Load model and tokenizer
        model_path = os.path.join(settings.MODELS_FOLDER, model_name.split("/")[-1])
        
        if os.path.exists(model_path):
            # Load from local path if available
            tokenizer = MarianTokenizer.from_pretrained(model_path)
            model = MarianMTModel.from_pretrained(model_path)
        else:
            # Download model if not available locally
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            # Save model for future use
            os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
        
        # Translate text
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Translated text from {source_language} to {target_language}")
        
        return {
            "translated_text": translated_text, 
            "source": source_language, 
            "target": target_language,
            "original_text": text
        }
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating text: {str(e)}"
        )
