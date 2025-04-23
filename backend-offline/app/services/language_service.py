from typing import Dict, List, Any, Optional
import json
import langdetect
from langdetect import DetectorFactory
import numpy as np

from app.core.config import settings
from app.core.logging import setup_logging

# Set seed for deterministic language detection
DetectorFactory.seed = 0

logger = setup_logging("language_service")

# Define supported languages with their capabilities
SUPPORTED_LANGUAGES = {
    "en": {
        "code": "en",
        "name": "English",
        "supported_features": ["chat", "voice", "rag", "tts"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "llama-2-7b-chat-q4_0"
    },
    "te": {
        "code": "te",
        "name": "Telugu",
        "supported_features": ["chat", "voice", "rag", "tts"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "llm_model": "llama-2-7b-chat-q4_0"
    },
    "ta": {
        "code": "ta",
        "name": "Tamil",
        "supported_features": ["chat", "voice", "rag", "tts"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "llm_model": "llama-2-7b-chat-q4_0"
    },
    "de": {
        "code": "de",
        "name": "German",
        "supported_features": ["chat", "voice", "rag", "tts"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "llm_model": "llama-2-7b-chat-q4_0"
    },
    "hi": {
        "code": "hi",
        "name": "Hindi",
        "supported_features": ["chat", "voice", "rag", "tts"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "llm_model": "llama-2-7b-chat-q4_0"
    }
}

def is_language_supported(language_code: str) -> bool:
    """
    Check if a language is supported.
    
    Args:
        language_code: Language code to check
        
    Returns:
        True if language is supported, False otherwise
    """
    return language_code in SUPPORTED_LANGUAGES

def get_language_model(language_code: str) -> Dict[str, Any]:
    """
    Get language model information for a language.
    
    Args:
        language_code: Language code
        
    Returns:
        Dictionary with language model information
    """
    if not is_language_supported(language_code):
        language_code = settings.DEFAULT_LANGUAGE
    
    return SUPPORTED_LANGUAGES[language_code]

def get_supported_languages() -> List[Dict[str, Any]]:
    """
    Get list of supported languages.
    
    Returns:
        List of supported languages with their capabilities
    """
    return list(SUPPORTED_LANGUAGES.values())

async def detect_language(text: str) -> Dict[str, Any]:
    """
    Detect language of input text.
    
    Args:
        text: Text to detect language for
        
    Returns:
        Dictionary with detected language, confidence, and support status
    """
    try:
        # Use langdetect for language detection
        detection = langdetect.detect_langs(text)
        
        # Get most likely language
        detected_lang = detection[0].lang
        confidence = detection[0].prob
        
        # Map to our language codes if needed
        # langdetect uses ISO 639-1 codes which match our codes
        
        # Check if language is supported
        supported = is_language_supported(detected_lang)
        
        # If not supported, use default language
        if not supported:
            logger.info(f"Detected unsupported language: {detected_lang}, falling back to {settings.DEFAULT_LANGUAGE}")
            detected_lang = settings.DEFAULT_LANGUAGE
            
        return {
            "detected_language": detected_lang,
            "confidence": confidence,
            "supported": supported or detected_lang == settings.DEFAULT_LANGUAGE
        }
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        # Fall back to default language
        return {
            "detected_language": settings.DEFAULT_LANGUAGE,
            "confidence": 1.0,
            "supported": True
        }

def get_prompt(prompt_key: str, language: str) -> str:
    """
    Get a language-specific prompt.
    
    Args:
        prompt_key: Key for the prompt
        language: Language code
        
    Returns:
        Language-specific prompt text
    """
    from app.llm.language_prompts import get_prompt as get_prompt_impl
    return get_prompt_impl(prompt_key, language)

def get_system_prompt(language: str) -> str:
    """
    Get the system prompt for the LLM in the specified language.
    
    Args:
        language: Language code
        
    Returns:
        System prompt for the LLM
    """
    from app.llm.language_prompts import get_system_prompt as get_system_prompt_impl
    return get_system_prompt_impl(language)

def get_rag_prompt_template(language: str) -> str:
    """
    Get the RAG prompt template for the specified language.
    
    Args:
        language: Language code
        
    Returns:
        RAG prompt template
    """
    from app.llm.language_prompts import get_rag_prompt_template as get_rag_prompt_template_impl
    return get_rag_prompt_template_impl(language)
