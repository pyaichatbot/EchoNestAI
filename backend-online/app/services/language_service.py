from typing import List, Dict, Any, Optional
from app.db.schemas.chat import SupportedLanguage, LanguageDetectionResponse
import asyncio
from fastapi import HTTPException
import os
import json
import numpy as np

from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("language_service")

# Define supported languages with their capabilities
SUPPORTED_LANGUAGES = {
    "en": {
        "code": "en",
        "name": "English",
        "supported_features": ["chat", "voice", "rag", "transcription"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    "te": {
        "code": "te",
        "name": "Telugu",
        "supported_features": ["chat", "voice", "rag", "transcription"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "multilingual-e5-large",
        "llm_model": "ai4bharat/indic-llm-7b-base"
    },
    "ta": {
        "code": "ta",
        "name": "Tamil",
        "supported_features": ["chat", "voice", "rag", "transcription"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "multilingual-e5-large",
        "llm_model": "ai4bharat/indic-llm-7b-base"
    },
    "de": {
        "code": "de",
        "name": "German",
        "supported_features": ["chat", "voice", "rag", "transcription"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "multilingual-e5-large",
        "llm_model": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    "hi": {
        "code": "hi",
        "name": "Hindi",
        "supported_features": ["chat", "voice", "rag", "transcription"],
        "voice_model": "whisper-large-v3",
        "embedding_model": "multilingual-e5-large",
        "llm_model": "ai4bharat/indic-llm-7b-base"
    }
}

def get_supported_languages() -> List[SupportedLanguage]:
    """
    Get list of supported languages with their capabilities.
    """
    return [
        SupportedLanguage(
            code=lang_data["code"],
            name=lang_data["name"],
            supported_features=lang_data["supported_features"],
            voice_model=lang_data["voice_model"],
            embedding_model=lang_data["embedding_model"]
        )
        for lang_code, lang_data in SUPPORTED_LANGUAGES.items()
    ]

async def detect_language(text: str) -> LanguageDetectionResponse:
    """
    Detect the language of input text using a language detection model.
    """
    try:
        # Use fasttext for language detection
        import fasttext
        
        # Ensure models directory exists
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
        
        # Path to fasttext model
        model_path = os.path.join(settings.MODELS_FOLDER, "lid.176.bin")
        
        # Download model if not exists
        if not os.path.exists(model_path):
            import urllib.request
            logger.info("Downloading fasttext language detection model...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                model_path
            )
        
        # Load model
        model = fasttext.load_model(model_path)
        
        # Predict language
        predictions = model.predict(text, k=3)
        
        # Extract results
        languages = [lang.replace("__label__", "") for lang in predictions[0]]
        confidences = [float(conf) for conf in predictions[1]]
        
        # Map ISO codes to our supported languages
        iso_to_supported = {
            "en": "en",
            "te": "te",
            "ta": "ta",
            "de": "de",
            "hi": "hi"
        }
        
        # Find the first supported language in the predictions
        detected_lang = None
        confidence = 0.0
        
        for lang, conf in zip(languages, confidences):
            # Check if this language is in our mapping
            if lang in iso_to_supported:
                detected_lang = iso_to_supported[lang]
                confidence = conf
                break
        
        # If no supported language found, use the highest confidence language
        if not detected_lang:
            # Check if the language is close to any of our supported languages
            # For example, if we detect 'en-us', map it to 'en'
            for lang, conf in zip(languages, confidences):
                for supported_prefix in iso_to_supported.keys():
                    if lang.startswith(supported_prefix):
                        detected_lang = iso_to_supported[supported_prefix]
                        confidence = conf
                        break
                if detected_lang:
                    break
        
        # If still no match, default to English
        if not detected_lang:
            detected_lang = "en"
            confidence = 0.5
            logger.warning(f"Unsupported language detected: {languages[0]}, defaulting to English")
        
        # Check if language is supported
        supported = detected_lang in SUPPORTED_LANGUAGES
        
        logger.info(f"Detected language: {detected_lang} with confidence {confidence:.2f}")
        
        return LanguageDetectionResponse(
            detected_language=detected_lang,
            confidence=confidence,
            supported=supported
        )
    
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        
        # Fallback to rule-based detection
        return rule_based_language_detection(text)

def rule_based_language_detection(text: str) -> LanguageDetectionResponse:
    """
    Fallback language detection using rule-based approach.
    """
    text = text.lower()
    
    # Simple detection based on common words and character sets
    if any(word in text for word in ["the", "is", "and", "hello", "how", "what"]):
        return LanguageDetectionResponse(
            detected_language="en",
            confidence=0.8,
            supported=True
        )
    elif any(word in text for word in ["నమస్కారం", "తెలుగు", "ధన్యవాదాలు"]) or \
         any(char in "అఆఇఈఉఊఋఌఎఏఐఒఓఔ" for char in text):
        return LanguageDetectionResponse(
            detected_language="te",
            confidence=0.8,
            supported=True
        )
    elif any(word in text for word in ["வணக்கம்", "தமிழ்", "நன்றி"]) or \
         any(char in "அஆஇஈஉஊஎஏஐஒஓஔ" for char in text):
        return LanguageDetectionResponse(
            detected_language="ta",
            confidence=0.8,
            supported=True
        )
    elif any(word in text for word in ["hallo", "guten", "danke", "wie", "und", "ich"]):
        return LanguageDetectionResponse(
            detected_language="de",
            confidence=0.8,
            supported=True
        )
    elif any(word in text for word in ["नमस्ते", "हिंदी", "धन्यवाद"]) or \
         any(char in "अआइईउऊएऐओऔ" for char in text):
        return LanguageDetectionResponse(
            detected_language="hi",
            confidence=0.8,
            supported=True
        )
    else:
        # Default to English if no clear match
        return LanguageDetectionResponse(
            detected_language="en",
            confidence=0.6,
            supported=True
        )

def get_language_model(language_code: str) -> Dict[str, str]:
    """
    Get the appropriate language model configuration for a given language.
    """
    if language_code not in SUPPORTED_LANGUAGES:
        logger.warning(f"Language '{language_code}' is not supported, falling back to English")
        language_code = "en"
    
    return {
        "voice_model": SUPPORTED_LANGUAGES[language_code]["voice_model"],
        "embedding_model": SUPPORTED_LANGUAGES[language_code]["embedding_model"],
        "llm_model": SUPPORTED_LANGUAGES[language_code]["llm_model"]
    }

def is_language_supported(language_code: str) -> bool:
    """
    Check if a language is supported.
    """
    return language_code in SUPPORTED_LANGUAGES

def get_language_name(language_code: str) -> str:
    """
    Get the human-readable name of a language.
    """
    if language_code not in SUPPORTED_LANGUAGES:
        return language_code
    
    return SUPPORTED_LANGUAGES[language_code]["name"]

async def translate_text(text: str, source_language: str, target_language: str) -> str:
    """
    Translate text from source language to target language.
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
            os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
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
