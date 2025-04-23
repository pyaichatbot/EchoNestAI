import os
import json
import logging
import time
import threading
import queue
from pathlib import Path

logger = logging.getLogger("echonest_device.language_manager")

class LanguageManager:
    """
    Manages language-specific functionality for the EchoNest AI device.
    Handles language detection, translation, and voice processing.
    """
    
    def __init__(self, models_path, cache_path, api_client=None):
        """
        Initialize the language manager.
        
        Args:
            models_path: Path to models directory
            cache_path: Path to cache directory
            api_client: Optional API client for fetching language resources
        """
        self.models_path = Path(models_path)
        self.cache_path = Path(cache_path)
        self.api_client = api_client
        
        # Create language-specific directories
        self.language_models_path = self.models_path / "languages"
        self.language_models_path.mkdir(parents=True, exist_ok=True)
        
        # Define supported languages
        self.supported_languages = {
            "en": {
                "name": "English",
                "models": {
                    "stt": "whisper-small-en",
                    "tts": "piper-en",
                    "embedding": "all-MiniLM-L6-v2",
                    "translation": "m2m100-418M"
                }
            },
            "hi": {
                "name": "Hindi",
                "models": {
                    "stt": "whisper-small-hi",
                    "tts": "piper-hi",
                    "embedding": "multilingual-e5-small",
                    "translation": "m2m100-418M"
                }
            },
            "te": {
                "name": "Telugu",
                "models": {
                    "stt": "whisper-small-te",
                    "tts": "piper-te",
                    "embedding": "multilingual-e5-small",
                    "translation": "m2m100-418M"
                }
            },
            "ta": {
                "name": "Tamil",
                "models": {
                    "stt": "whisper-small-ta",
                    "tts": "piper-ta",
                    "embedding": "multilingual-e5-small",
                    "translation": "m2m100-418M"
                }
            },
            "de": {
                "name": "German",
                "models": {
                    "stt": "whisper-small-de",
                    "tts": "piper-de",
                    "embedding": "multilingual-e5-small",
                    "translation": "m2m100-418M"
                }
            }
        }
        
        # Initialize model cache
        self.model_cache = {}
        self.model_cache_lock = threading.Lock()
        
        # Initialize language detection model
        self._init_language_detection()
    
    def _init_language_detection(self):
        """
        Initialize the language detection model.
        """
        try:
            import fasttext
            
            # Check if language detection model exists
            model_path = self.language_models_path / "lid.176.bin"
            
            if not model_path.exists():
                logger.info("Language detection model not found, downloading...")
                
                # Create directory if it doesn't exist
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download model
                import urllib.request
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                    model_path
                )
            
            # Load model
            self.lang_detect_model = fasttext.load_model(str(model_path))
            logger.info("Language detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing language detection model: {e}")
            self.lang_detect_model = None
    
    def detect_language(self, text):
        """
        Detect the language of the given text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            dict: Detection result with language code and confidence
        """
        if not self.lang_detect_model:
            logger.warning("Language detection model not initialized")
            return {"detected_language": "en", "confidence": 0.0}
        
        try:
            # Predict language
            predictions = self.lang_detect_model.predict(text, k=3)
            labels, confidences = predictions
            
            # Extract language code from label (format: __label__en)
            top_lang = labels[0].replace("__label__", "")
            top_confidence = confidences[0]
            
            # Check if detected language is supported
            if top_lang in self.supported_languages:
                detected_lang = top_lang
            else:
                # Default to English if language not supported
                detected_lang = "en"
                logger.info(f"Detected language {top_lang} not supported, defaulting to English")
            
            return {
                "detected_language": detected_lang,
                "confidence": float(top_confidence),
                "all_languages": [
                    {
                        "language": label.replace("__label__", ""),
                        "confidence": float(conf)
                    }
                    for label, conf in zip(labels, confidences)
                ]
            }
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return {"detected_language": "en", "confidence": 0.0}
    
    def translate_text(self, text, source_lang=None, target_lang="en"):
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (auto-detected if None)
            target_lang: Target language code
            
        Returns:
            dict: Translation result
        """
        # Detect source language if not provided
        if not source_lang:
            detection = self.detect_language(text)
            source_lang = detection["detected_language"]
        
        # No translation needed if source and target are the same
        if source_lang == target_lang:
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang
            }
        
        try:
            # Load translation model
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
            
            model_name = self.supported_languages[target_lang]["models"]["translation"]
            model_path = self.language_models_path / "translation" / model_name
            
            # Check if model exists locally
            if not model_path.exists():
                logger.info(f"Translation model {model_name} not found locally, using remote model")
                model_path = f"facebook/{model_name}"
            
            # Get or load model and tokenizer
            with self.model_cache_lock:
                cache_key = f"translation_{model_name}"
                
                if cache_key not in self.model_cache:
                    tokenizer = M2M100Tokenizer.from_pretrained(model_path)
                    model = M2M100ForConditionalGeneration.from_pretrained(model_path)
                    self.model_cache[cache_key] = (model, tokenizer)
                else:
                    model, tokenizer = self.model_cache[cache_key]
            
            # Translate text
            tokenizer.src_lang = source_lang
            encoded = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(target_lang)
            )
            translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            return {
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang
            }
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "error": str(e)
            }
    
    def speech_to_text(self, audio_path, language=None):
        """
        Convert speech to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (auto-detected if None)
            
        Returns:
            dict: Transcription result
        """
        try:
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            # Determine language to use
            if not language:
                # We'll use a multilingual model and let it detect the language
                language = None
                model_name = "whisper-small"
            else:
                # Use language-specific model if available
                model_name = self.supported_languages[language]["models"]["stt"]
            
            model_path = self.language_models_path / "stt" / model_name
            
            # Check if model exists locally
            if not model_path.exists():
                logger.info(f"STT model {model_name} not found locally, using remote model")
                model_path = f"openai/{model_name}"
            
            # Get or load model and processor
            with self.model_cache_lock:
                cache_key = f"stt_{model_name}"
                
                if cache_key not in self.model_cache:
                    processor = WhisperProcessor.from_pretrained(model_path)
                    model = WhisperForConditionalGeneration.from_pretrained(model_path)
                    self.model_cache[cache_key] = (model, processor)
                else:
                    model, processor = self.model_cache[cache_key]
            
            # Load audio
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio
            input_features = processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features
            
            # Generate transcription
            forced_decoder_ids = None
            if language:
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
            
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Detect language if not provided
            detected_language = language
            if not detected_language:
                detection = self.detect_language(transcription)
                detected_language = detection["detected_language"]
            
            return {
                "text": transcription,
                "language": detected_language
            }
        except Exception as e:
            logger.error(f"Error in speech to text conversion: {e}")
            return {
                "text": "",
                "language": language or "en",
                "error": str(e)
            }
    
    def text_to_speech(self, text, language="en", voice=None):
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            language: Language code
            voice: Optional voice name
            
        Returns:
            dict: Path to generated audio file and metadata
        """
        try:
            # Determine model to use
            model_name = self.supported_languages[language]["models"]["tts"]
            model_path = self.language_models_path / "tts" / model_name
            
            # Generate output path
            output_dir = self.cache_path / "tts"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"tts_{int(time.time())}.wav"
            
            # Check if we should use local model or fallback to pyttsx3
            if model_path.exists():
                # Use Piper TTS model
                from piper import PiperVoice
                
                # Get available voices
                voices_json = model_path / "voices.json"
                with open(voices_json, 'r') as f:
                    voices = json.load(f)
                
                # Select voice
                if not voice:
                    # Use first voice for the language
                    voice = next((v for v in voices if v["language"] == language), voices[0])["name"]
                
                # Initialize Piper
                piper = PiperVoice.load(model_path / f"{voice}.onnx", model_path / f"{voice}.json")
                
                # Generate speech
                piper.synthesize(text, output_path)
            else:
                # Fallback to pyttsx3
                import pyttsx3
                
                engine = pyttsx3.init()
                
                # Set properties
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                
                # Set voice based on language
                voices = engine.getProperty('voices')
                for v in voices:
                    if language in v.languages:
                        engine.setProperty('voice', v.id)
                        break
                
                # Generate speech
                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
            
            return {
                "audio_path": str(output_path),
                "language": language,
                "voice": voice,
                "text": text
            }
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            return {
                "audio_path": None,
                "language": language,
                "text": text,
                "error": str(e)
            }
    
    def get_language_resources(self, language):
        """
        Get language-specific resources.
        
        Args:
            language: Language code
            
        Returns:
            dict: Language resources
        """
        if language not in self.supported_languages:
            logger.warning(f"Language {language} not supported")
            return {}
        
        # Try to get resources from API if available
        if self.api_client:
            try:
                resources = self.api_client.get_language_resources(language)
                if resources.get('success'):
                    return resources.get('resources', {})
            except Exception as e:
                logger.error(f"Error getting language resources from API: {e}")
        
        # Return local resources
        return {
            "language": language,
            "name": self.supported_languages[language]["name"],
            "models": self.supported_languages[language]["models"],
            "translations": self._get_local_translations(language)
        }
    
    def _get_local_translations(self, language):
        """
        Get local translations for a language.
        
        Args:
            language: Language code
            
        Returns:
            dict: Translations
        """
        translations_path = self.language_models_path / "translations" / f"{language}.json"
        
        if translations_path.exists():
            try:
                with open(translations_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading translations for {language}: {e}")
        
        # Return empty translations if file doesn't exist or error occurs
        return {}
