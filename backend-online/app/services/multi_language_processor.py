from typing import Dict, List, Any, Optional
import os
import tempfile
import json
import asyncio
import numpy as np
from fastapi import HTTPException

from app.services.language_service import is_language_supported, get_language_model
from app.llm.language_prompts import get_prompt, get_system_prompt, get_rag_prompt_template
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("multi_language_processor")

class MultiLanguageProcessor:
    """
    Processor for handling multi-language voice and text processing.
    This class provides methods for transcription, translation, and text-to-speech
    with support for multiple languages.
    """
    
    def __init__(self):
        # Initialize supported languages and models
        self.default_language = "en"
        self.model_cache = {}
        
        # Ensure models directory exists
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
    
    async def transcribe_audio(self, audio_data: bytes, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio data to text in the specified language.
        
        Args:
            audio_data: Raw audio data bytes
            language: Language code (auto-detect if None)
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)
            
            # Import whisper
            import whisper
            
            # Set environment variable for model download location
            os.environ["WHISPER_MODELS_DIR"] = settings.MODELS_FOLDER
            
            # Load model
            model_key = "whisper_base"
            if model_key not in self.model_cache:
                logger.info(f"Loading Whisper model for transcription")
                self.model_cache[model_key] = whisper.load_model("base")
            
            model = self.model_cache[model_key]
            
            # Transcribe audio
            if language and is_language_supported(language):
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
            detected_language = result.get("language", language or "en")
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Validate detected language
            if not is_language_supported(detected_language):
                detected_language = self.default_language
            
            # Get appropriate model for language
            language_models = get_language_model(detected_language)
            voice_model = language_models["voice_model"]
            
            logger.info(f"Transcribed audio in {detected_language}: {text[:50]}...")
            
            return {
                "text": text,
                "detected_language": detected_language,
                "confidence": 0.95,  # Whisper doesn't provide confidence scores
                "model_used": voice_model
            }
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
    
    async def text_to_speech(self, text: str, language: str) -> bytes:
        """
        Convert text to speech in the specified language.
        
        Args:
            text: Text to convert to speech
            language: Language code
            
        Returns:
            Audio data as bytes
        """
        try:
            # Validate language
            if not is_language_supported(language):
                language = self.default_language
            
            # Get appropriate model for language
            language_models = get_language_model(language)
            
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_path = temp_file.name
            
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
            
            # Read audio data
            with open(output_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            os.unlink(output_path)
            
            logger.info(f"Generated speech for text in {language}: {text[:50]}...")
            
            return audio_data
        
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")
    
    async def translate_text(self, text: str, source_language: str, target_language: str) -> str:
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
            # Validate languages
            if not is_language_supported(source_language):
                source_language = self.default_language
            
            if not is_language_supported(target_language):
                target_language = self.default_language
            
            # If source and target are the same, return original text
            if source_language == target_language:
                return text
            
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
            
            # Create cache key
            model_key = f"translation_{source_language}_{target_language}"
            
            # Load model and tokenizer from cache or initialize
            if model_key not in self.model_cache:
                logger.info(f"Loading translation model for {source_language} to {target_language}")
                
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
                
                # Store in cache
                self.model_cache[model_key] = (model, tokenizer)
            
            # Get model and tokenizer from cache
            model, tokenizer = self.model_cache[model_key]
            
            # Translate text
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            outputs = model.generate(**inputs)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Translated text from {source_language} to {target_language}")
            
            return translated_text
        
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            # Return original text on error
            return text
    
    def get_language_specific_prompt(self, prompt_key: str, language: str) -> str:
        """
        Get a language-specific prompt.
        
        Args:
            prompt_key: Key for the prompt
            language: Language code
            
        Returns:
            Language-specific prompt text
        """
        # Validate language
        if not is_language_supported(language):
            language = self.default_language
        
        return get_prompt(prompt_key, language)
    
    def get_language_specific_system_prompt(self, language: str) -> str:
        """
        Get the system prompt for the LLM in the specified language.
        
        Args:
            language: Language code
            
        Returns:
            System prompt for the LLM
        """
        # Validate language
        if not is_language_supported(language):
            language = self.default_language
        
        return get_system_prompt(language)
    
    def get_language_specific_rag_template(self, language: str) -> str:
        """
        Get the RAG prompt template for the specified language.
        
        Args:
            language: Language code
            
        Returns:
            RAG prompt template
        """
        # Validate language
        if not is_language_supported(language):
            language = self.default_language
        
        return get_rag_prompt_template(language)
    
    async def detect_language_from_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Detect language from audio data.
        
        Args:
            audio_data: Raw audio data bytes
            
        Returns:
            Dictionary with detected language and confidence
        """
        # Transcribe audio without specifying language to auto-detect
        result = await self.transcribe_audio(audio_data)
        
        return {
            "detected_language": result["detected_language"],
            "confidence": result["confidence"],
            "text": result["text"]
        }
    
    async def generate_embeddings(self, texts: List[str], language: str) -> np.ndarray:
        """
        Generate embeddings for texts in the specified language.
        
        Args:
            texts: List of texts to embed
            language: Language code
            
        Returns:
            Array of embeddings
        """
        try:
            # Validate language
            if not is_language_supported(language):
                language = self.default_language
            
            # Get appropriate model for language
            language_models = get_language_model(language)
            model_name = language_models["embedding_model"]
            
            # Create cache key
            model_key = f"embedding_{language}"
            
            # Load model from cache or initialize
            if model_key not in self.model_cache:
                logger.info(f"Loading embedding model for {language}")
                
                from sentence_transformers import SentenceTransformer
                
                # Load model (with caching)
                model_path = os.path.join(settings.MODELS_FOLDER, model_name)
                if os.path.exists(model_path):
                    model = SentenceTransformer(model_path)
                else:
                    model = SentenceTransformer(model_name)
                    # Save model for future use
                    os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
                    model.save(model_path)
                
                # Store in cache
                self.model_cache[model_key] = model
            
            # Get model from cache
            model = self.model_cache[model_key]
            
            # Generate embeddings
            embeddings = model.encode(texts, convert_to_numpy=True)
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

# Create a global instance
multi_language_processor = MultiLanguageProcessor()
