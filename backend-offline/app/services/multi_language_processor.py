from typing import Dict, List, Any, Optional
import os
import tempfile
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.language_service import detect_language, get_language_model

logger = setup_logging("multi_language_processor")

async def process_audio(
    audio_path: str,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process audio file for speech recognition.
    
    Args:
        audio_path: Path to audio file
        language: Optional language code (auto-detected if not provided)
        
    Returns:
        Dictionary with transcribed text, detected language, and confidence
    """
    import whisper
    
    # Ensure models directory exists
    os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
    
    # Set environment variable for model download location
    os.environ["WHISPER_MODELS_DIR"] = settings.MODELS_FOLDER
    
    try:
        # Load model
        model = whisper.load_model("base")
        
        # Transcribe audio
        if language:
            # Use specified language
            result = model.transcribe(
                audio_path,
                language=language,
                fp16=False
            )
        else:
            # Auto-detect language
            result = model.transcribe(
                audio_path,
                fp16=False
            )
        
        # Extract results
        text = result["text"]
        detected_language = result.get("language", "en")
        
        # If no text was transcribed, return empty result
        if not text.strip():
            logger.warning(f"No speech detected in audio file: {audio_path}")
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
        return {
            "text": "",
            "detected_language": language or "en",
            "confidence": 0.0,
            "error": str(e)
        }

async def text_to_speech(
    text: str,
    language: str = "en",
    voice_id: Optional[str] = None
) -> str:
    """
    Convert text to speech.
    
    Args:
        text: Text to convert to speech
        language: Language code
        voice_id: Optional voice ID
        
    Returns:
        Path to generated audio file
    """
    # Ensure output directory exists
    os.makedirs(settings.AUDIO_OUTPUT_FOLDER, exist_ok=True)
    
    # Create temporary file for output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.AUDIO_OUTPUT_FOLDER) as temp_file:
        output_path = temp_file.name
    
    try:
        # Get appropriate TTS model for language
        language_models = get_language_model(language)
        
        # Use pyttsx3 for basic TTS (in production, would use a more advanced model)
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
        
        return output_path
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        
        # If there was an error, return a path to a default audio file
        default_audio_path = os.path.join(settings.AUDIO_OUTPUT_FOLDER, "error.wav")
        
        # Create a simple error audio file if it doesn't exist
        if not os.path.exists(default_audio_path):
            import wave
            import struct
            
            # Create a simple beep sound
            with wave.open(default_audio_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                
                # Generate a simple beep
                for i in range(44100):
                    value = 32767 * np.sin(2 * np.pi * 440 * i / 44100)
                    data = struct.pack('<h', int(value))
                    wf.writeframes(data)
        
        return default_audio_path

async def translate_text(
    text: str,
    source_language: str,
    target_language: str
) -> str:
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
            return text
        
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
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
        
        # Translate text
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Translated text from {source_language} to {target_language}")
        
        return translated_text
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return text  # Return original text on error
