"""
Production-Grade Speech-to-Text and Text-to-Speech System Architecture
Optimized hybrid approach with cost-effective scaling
"""

import os
import time
import json
import uuid
import torch
import logging
import whisper
import tempfile
import numpy as np
import soundfile as sf
import requests
import base64
import hashlib
import asyncio
import traceback
from typing import Dict, List, Optional, Union, Callable, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
from langdetect import detect, DetectorFactory, LangDetectException
import requests as pyrequests
from transformers import AutoTokenizer, AutoFeatureExtractor

# Import configuration
from config import (
    TTS_MODEL_ID,
    TTS_MAX_LOAD_ATTEMPTS,
    TTS_LOAD_RETRY_DELAY,
    TTS_DEFAULT_SAMPLING_RATE,
    TTS_FORCE_DOWNLOAD,
    TTS_USE_CUDA,
    TTS_MODEL_CACHE_DIRS
)

# STT
from elevenlabs.client import ElevenLabs

# For TTS
from TTS.api import TTS

# Redis for caching and job queue
import redis
from redis.exceptions import RedisError

# For REST API
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Depends, Request, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

# Rate limiting
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# WebSocket support
import websockets
from websocket_service import WebSocketAudioStreamingService

# Configure logging
logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv("LOG_FILE", "speech_service.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from parler_tts import ParlerTTSForConditionalGeneration
    PARLER_TTS_AVAILABLE = True
except ImportError:
    logger.warning("parler-tts library not found. AI4BharatTTSProvider will be unavailable.")
    PARLER_TTS_AVAILABLE = False
    class ParlerTTSForConditionalGeneration: pass

# Environment variables with defaults
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = int(os.getenv("REDIS_DB", 0))
CACHE_TTL = int(os.getenv("CACHE_TTL", 86400))  # 24 hours by default
RATE_LIMIT_TIER1 = int(os.getenv("RATE_LIMIT_TIER1", 100))  # requests per minute
RATE_LIMIT_TIER2 = int(os.getenv("RATE_LIMIT_TIER2", 300))  # requests per minute

# API keys for third-party services
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")

# Cost configuration
MAX_DAILY_COST = float(os.getenv("MAX_DAILY_COST", "10.0"))  # Maximum daily cost in USD
COST_TIER_RATIO = float(os.getenv("COST_TIER_RATIO", "0.2"))  # Percent of paid API usage (0.0-1.0)

# Model cache directories (always use env, fallback to Docker default)
WHISPER_CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", "/app/models/whisper")
TTS_CACHE_DIR = os.getenv("TORCH_HOME", "/app/models/coqui")

# Ensure Coqui TTS uses the correct cache path
os.environ["TTS_CACHE_PATH"] = os.getenv("TTS_CACHE_PATH", TTS_CACHE_DIR)

ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", "")

# Redis connection with error handling
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True
    )
    redis_client.ping()  # Test connection
    logger.info("Redis connection established")
except RedisError as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
    # Fallback to in-memory cache
    class InMemoryCache:
        def __init__(self):
            self.cache = {}
            self.expiry = {}
        
        def ping(self):
            return True
            
        def get(self, key):
            if key in self.cache and (key not in self.expiry or self.expiry[key] > datetime.now()):
                return self.cache[key]
            return None
            
        def set(self, key, value):
            self.cache[key] = value
            return True
            
        def setex(self, key, ttl, value):
            self.cache[key] = value
            self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
            return True
            
        def incr(self, key):
            if key not in self.cache:
                self.cache[key] = 1
            else:
                self.cache[key] = int(self.cache[key]) + 1
            return self.cache[key]
            
        def expire(self, key, ttl):
            if key in self.cache:
                self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
                return True
            return False
            
        def delete(self, key):
            if key in self.cache:
                del self.cache[key]
                if key in self.expiry:
                    del self.expiry[key]
                return True
            return False
    
    redis_client = InMemoryCache()

# Monitoring and metrics dictionary
metrics = {
    "api_calls": 0,
    "stt_requests": 0,
    "tts_requests": 0,
    "provider_usage": {},
    "latency": [],
    "errors": {},
    "cost": 0.0,
    "daily_costs": {}
}

# Tier classifications for model/provider selection
class Tier(str, Enum):
    ECONOMY = "economy"    # Fastest, least expensive, lower quality
    STANDARD = "standard"  # Good balance of speed, cost, and quality
    PREMIUM = "premium"    # Best quality, more expensive


# --- Utility Functions ---

def calculate_md5(data):
    """Calculate MD5 hash of data (string or bytes)"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.md5(data).hexdigest()

def convert_audio_format(input_path, output_path, format="wav", sample_rate=16000):
    """Convert audio to specified format using ffmpeg"""
    try:
        import subprocess
        command = [
            "ffmpeg", "-y", "-i", input_path, 
            "-ar", str(sample_rate), 
            "-ac", "1",  # Mono
            output_path
        ]
        subprocess.run(command, check=True, capture_output=True)
        return output_path
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return input_path  # Return original path on failure

def audio_to_base64(file_path):
    """Convert audio file to base64 string"""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def update_cost_tracker(provider_name, operation_type, duration_seconds=0, characters=0):
    """
    Track cost for different providers
    
    Args:
        provider_name: Name of the provider used
        operation_type: 'stt' or 'tts'
        duration_seconds: For STT operations, duration in seconds
        characters: For TTS operations, number of characters
    """
    # Cost estimates per provider per operation
    cost_table = {
        "whisper_tiny": {"stt": 0.0001 * duration_seconds},   # Very low cost, self-hosted
        "whisper_base": {"stt": 0.0002 * duration_seconds},   # Low cost, self-hosted
        "whisper_small": {"stt": 0.0004 * duration_seconds},  # Medium cost, self-hosted
        "whisper_medium": {"stt": 0.0008 * duration_seconds}, # Higher cost, self-hosted
        "whisper_large": {"stt": 0.0015 * duration_seconds},  # Highest cost, self-hosted
        
        "coqui_tacotron2": {"tts": 0.0001 * characters},      # Low cost, self-hosted 
        "coqui_vits": {"tts": 0.0002 * characters},           # Medium cost, self-hosted
        
        "google_stt": {"stt": 0.006 * (duration_seconds / 15)},  # $0.006 per 15 seconds
        "google_tts_standard": {"tts": 0.000004 * characters},   # $4.00 per million chars
        "google_tts_wavenet": {"tts": 0.000016 * characters},    # $16.00 per million chars
        
        "azure_stt": {"stt": 0.0002778 * duration_seconds},      # $1.00 per hour
        "azure_tts_standard": {"tts": 0.000004 * characters},    # $4.00 per million chars
        "azure_tts_neural": {"tts": 0.000016 * characters},      # $16.00 per million chars
        
        "aws_transcribe": {"stt": 0.0004 * duration_seconds},    # $0.024 per minute
        "aws_polly_standard": {"tts": 0.000004 * characters},    # $4.00 per million chars
        "aws_polly_neural": {"tts": 0.000016 * characters}       # $16.00 per million chars
    }
    
    # Get today's date string
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Calculate cost if provider exists in cost table
        cost = 0.0
        if provider_name in cost_table and operation_type in cost_table[provider_name]:
            cost = cost_table[provider_name][operation_type]
        
        # Update metrics
        with metrics_lock:
            # Update provider usage count
            if provider_name not in metrics["provider_usage"]:
                metrics["provider_usage"][provider_name] = 0
            metrics["provider_usage"][provider_name] += 1
            
            # Update cost
            metrics["cost"] += cost
            
            # Update daily cost
            if today not in metrics["daily_costs"]:
                metrics["daily_costs"][today] = 0.0
            metrics["daily_costs"][today] += cost
        
        # Also track in Redis for persistence
        try:
            redis_key = f"cost:{today}"
            redis_client.incrbyfloat(redis_key, cost)
            redis_client.expire(redis_key, 60 * 60 * 24 * 30)  # 30 days expiry
        except Exception as e:
            logger.warning(f"Failed to update cost in Redis: {e}")
        
        return cost
    except Exception as e:
        logger.error(f"Error updating cost tracker: {e}")
        return 0.0

def check_cost_limit():
    """
    Check if we've exceeded the daily cost limit
    
    Returns:
        tuple: (exceeded, current_cost, limit)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Try to get cost from Redis first (more accurate across instances)
        current_cost = 0.0
        try:
            redis_cost = redis_client.get(f"cost:{today}")
            if redis_cost:
                current_cost = float(redis_cost)
        except Exception:
            # Fall back to in-memory metrics if Redis fails
            if today in metrics["daily_costs"]:
                current_cost = metrics["daily_costs"][today]
        
        return (current_cost > MAX_DAILY_COST, current_cost, MAX_DAILY_COST)
    except Exception as e:
        logger.error(f"Error checking cost limit: {e}")
        return (False, 0.0, MAX_DAILY_COST)  # Assume not exceeded on error

# Create lock for thread-safe metrics updates
import threading
metrics_lock = threading.Lock()

# --- Base Abstract Classes ---

class BaseEngine(ABC):
    """Base class for speech engines"""
    
    @abstractmethod
    def initialize(self):
        """Initialize the engine resources"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Release resources properly"""
        pass
    
    @property
    def providers_by_tier(self) -> Dict[Tier, List]:
        """Group providers by tier"""
        pass


class BaseProvider(ABC):
    """Abstract class defining provider interface"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return provider name"""
        pass
    
    @property
    @abstractmethod
    def tier(self) -> Tier:
        """Return provider tier (economy, standard, premium)"""
        pass
    
    @property
    @abstractmethod
    def cost_per_unit(self) -> float:
        """Return cost per unit (second for STT, character for TTS)"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available for use"""
        pass
    
    @property
    def is_paid_service(self) -> bool:
        """Return True if this is a paid third-party service"""
        return False


# --- STT Implementation ---

@dataclass
class TranscriptionResult:
    """Structured transcription result"""
    text: str
    confidence: float
    segments: List[Dict] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
    provider: str = ""
    processing_time: float = 0.0
    cost: float = 0.0
    word_timestamps: bool = False


class STTProvider(BaseProvider):
    """Base class for STT providers"""
    
    @abstractmethod
    def transcribe(self, 
                  audio_path: str, 
                  language: Optional[str] = None,
                  **kwargs) -> TranscriptionResult:
        """Transcribe audio file to text"""
        pass
    
    @property
    @abstractmethod
    def supported_languages(self) -> Set[str]:
        """Return set of supported language codes"""
        pass
    
    @property
    @abstractmethod
    def supports_word_timestamps(self) -> bool:
        """Return True if provider supports word-level timestamps"""
        pass

class AssembkyAISTTProivder(STTProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ASSEMBLYAI_API_KEY
        self._supported_languages = {"en", "hi", "de", "fr", "es", "it", "ja", "ko", "pt", "ru", "zh", "kn", "ml", "ta", "te"}  # Update if assemblyai supports more
        self.base_url = "https://api.assemblyai.com"
        self._endpoint = self.base_url + "/v2/transcript"
        
    @property
    def name(self) -> str:
        return "assemblyai_stt"

    @property
    def tier(self) -> Tier:
        return Tier.PREMIUM

    @property
    def cost_per_unit(self) -> float:
        # Set the real cost per second if known
        return 0.001  # Example: $0.001 per second

    @property
    def is_paid_service(self) -> bool:
        return True

    @property
    def supported_languages(self) -> Set[str]:
        return self._supported_languages

    @property
    def supports_word_timestamps(self) -> bool:
        # Set to True if AssemblyAI supports word-level timestamps
        return False

    def is_available(self) -> bool:
        return bool(self.api_key)

    def transcribe(self, audio_path: str, language: Optional[str] = None, **kwargs) -> TranscriptionResult:
        """Transcribe using AssemblyAI Speech-to-Text API"""

        if not self.is_available():
            raise RuntimeError("AssemblyAI API key not configured")

        start_time = time.time()

        try:
            audio_data = None
            # Read audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()
                
            if audio_data is None:
                raise RuntimeError("Audio data is None")
            
            if len(audio_data) == 0:
                raise RuntimeError("Audio data is empty")
            
            if not isinstance(audio_data, bytes):
                raise RuntimeError("Audio data is not bytes")
            
            headers = {
                "authorization": self.api_key
            }
            response = requests.post(self.base_url + "/v2/upload", headers=headers, data=audio_data) 
            audio_url = response.json()["upload_url"]

            data = {
                "audio_url": audio_url,
                "speech_model": "universal"
            }

            url = self.base_url + "/v2/transcript"
            response = requests.post(url, json=data, headers=headers)

            transcript_id = response.json()['id']
            polling_endpoint = self.base_url + "/v2/transcript/" + transcript_id

            while True:
                transcription_result = requests.get(polling_endpoint, headers=headers).json()
                transcript_text = transcription_result['text']

                if transcription_result['status'] == 'completed':
                    print(f"Transcript Text:", transcript_text)
                    break

                elif transcription_result['status'] == 'error':
                    raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

                else:
                    time.sleep(3)
            
            data = {
                "language": language or "en"
            }

            # Parse result (adapt to actual ElevenLabs API response)
            transcript = transcript_text
            confidence = 1.0
            segments = []
            duration = 0.0

            processing_time = time.time() - start_time

            cost = update_cost_tracker(self.name, "stt", duration_seconds=duration)

            return TranscriptionResult(
                text=transcript,
                confidence=confidence,
                segments=segments,
                language=language or "en",
                duration=duration,
                provider=self.name,
                processing_time=processing_time,
                cost=cost,
                word_timestamps=False
            )
        except Exception as e:
            logger.error(f"AssemblyAI transcription error: {e}")
            raise RuntimeError(f"AssemblyAI transcription failed: {str(e)}")

class ElevenLabsProvider(STTProvider):
    """ElevenLabs Speech-to-Text API provider"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ELEVENLABS_API_KEY
        self._supported_languages = {"en"}  # Update if ElevenLabs supports more
        self._endpoint = "https://api.elevenlabs.io/v1/speech-to-text"  # Example endpoint

    @property
    def name(self) -> str:
        return "elevenlabs_stt"

    @property
    def tier(self) -> Tier:
        return Tier.PREMIUM

    @property
    def cost_per_unit(self) -> float:
        # Set the real cost per second if known
        return 0.001  # Example: $0.001 per second

    @property
    def is_paid_service(self) -> bool:
        return True

    @property
    def supported_languages(self) -> Set[str]:
        return self._supported_languages

    @property
    def supports_word_timestamps(self) -> bool:
        # Set to True if ElevenLabs supports word-level timestamps
        return False

    def is_available(self) -> bool:
        return bool(self.api_key)

    def transcribe(self, audio_path: str, language: Optional[str] = None, **kwargs) -> TranscriptionResult:
        """Transcribe using ElevenLabs Speech-to-Text API"""

        if not self.is_available():
            raise RuntimeError("ElevenLabs API key not configured")

        start_time = time.time()

        try:
            audio_data = None
            # Read audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            headers = {
                "xi-api-key": self.api_key,
                "Accept": "application/json"
            }
            files = {
                "audio": (os.path.basename(audio_path), audio_data, "audio/wav")
            }
            data = {
                "language": language or "en"
            }

            response = requests.post(self._endpoint, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()

            # Parse result (adapt to actual ElevenLabs API response)
            transcript = result.get("text", "")
            confidence = result.get("confidence", 1.0)
            segments = result.get("segments", [])
            duration = result.get("duration", 0.0)

            processing_time = time.time() - start_time

            cost = update_cost_tracker(self.name, "stt", duration_seconds=duration)

            return TranscriptionResult(
                text=transcript,
                confidence=confidence,
                segments=segments,
                language=language or "en",
                duration=duration,
                provider=self.name,
                processing_time=processing_time,
                cost=cost,
                word_timestamps=False
            )
        except Exception as e:
            logger.error(f"ElevenLabs transcription error: {e}")
            raise RuntimeError(f"ElevenLabs transcription failed: {str(e)}")

class WhisperProvider(STTProvider):
    """Whisper-based STT provider"""
    
    _TIER_MAPPING = {
        "tiny": Tier.ECONOMY,
        "base": Tier.ECONOMY,
        "small": Tier.STANDARD,
        "medium": Tier.STANDARD,
        "large": Tier.PREMIUM
    }
    
    _COST_MAPPING = {
        "tiny": 0.0001,    # Cost per second
        "base": 0.0002,    # Cost per second
        "small": 0.0004,   # Cost per second
        "medium": 0.0008,  # Cost per second
        "large": 0.0015    # Cost per second
    }
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._supported_languages = set()
        self._load_attempts = 0
        self._max_load_attempts = 3
        self._last_load_attempt = 0
        
    @property
    def name(self) -> str:
        return f"whisper_{self.model_name}"
    
    @property
    def tier(self) -> Tier:
        return self._TIER_MAPPING.get(self.model_name, Tier.STANDARD)
    
    @property
    def cost_per_unit(self) -> float:
        """Cost per second of audio"""
        return self._COST_MAPPING.get(self.model_name, 0.0004)
    
    @property
    def supported_languages(self) -> Set[str]:
        """Return set of supported language codes"""
        # Whisper supports many languages, but we don't need to list them all
        # as it has automatic language detection
        return self._supported_languages or {"en", "es", "fr", "de", "it", "ja", "zh", "ru", "pt"}
    
    @property
    def supports_word_timestamps(self) -> bool:
        """Return True if provider supports word-level timestamps"""
        return True
    
    def is_available(self) -> bool:
        # Throttle reload attempts to prevent constant retries
        current_time = time.time()
        if (self.model is None and 
            self._load_attempts >= self._max_load_attempts and 
            current_time - self._last_load_attempt < 300):  # 5 minutes
            return False
            
        try:
            if self.model is None:
                self._last_load_attempt = current_time
                self._load_attempts += 1
                
                # Use pre-downloaded models from local cache
                model_cache_dir = os.getenv("WHISPER_CACHE_DIR", "/app/models/whisper")
                model_path = os.path.join(model_cache_dir, self.model_name)
                if not os.path.exists(model_path):
                    logger.warning(f"Model not found in cache: {model_path}, falling back to download")
                    model_path = None
                
                self.model = whisper.load_model(
                    self.model_name,
                    device=self.device,
                    download_root=model_cache_dir
                )
                
                # Reset counter on success
                self._load_attempts = 0
                
                # Get list of supported languages from Whisper
                if hasattr(self.model, "tokenizer") and hasattr(self.model.tokenizer, "language_codes"):
                    self._supported_languages = set(self.model.tokenizer.language_codes)
                
            return True
        except Exception as e:
            logger.exception(f"Whisper model {self.model_name} is not available: {e}")
            return False
    
    def transcribe(self, 
                  audio_path: str, 
                  language: Optional[str] = None,
                  word_timestamps: bool = False,
                  **kwargs) -> TranscriptionResult:
        """Transcribe using Whisper model"""
        
        if self.model is None and not self.is_available():
            raise RuntimeError(f"Whisper model {self.model_name} is not available")
        
        start_time = time.time()
        
        try:
            # Process audio with ffmpeg if needed
            audio_duration = 0
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                audio_duration = info.duration
                
                # Convert if not already 16kHz mono WAV
                if info.samplerate != 16000 or info.channels > 1:
                    temp_path = f"{audio_path}.converted.wav"
                    audio_path = convert_audio_format(audio_path, temp_path)
            except Exception as e:
                logger.warning(f"Could not get audio info, using original file: {e}")
            
            # Set up transcription options
            options = {
                "language": language,
                "word_timestamps": word_timestamps,
                **kwargs
            }
            
            # Filter out None values
            options = {k: v for k, v in options.items() if v is not None}
            
            # Transcribe
            result = self.model.transcribe(audio_path, **options)
            
            processing_time = time.time() - start_time
            
            # Calculate cost
            if audio_duration == 0 and "duration" in result:
                audio_duration = result["duration"]
            
            cost = update_cost_tracker(self.name, "stt", duration_seconds=audio_duration)
            
            return TranscriptionResult(
                text=result["text"],
                confidence=self._calculate_confidence(result),
                segments=result.get("segments", []),
                language=result.get("language", language or ""),
                duration=audio_duration or result.get("duration", 0.0),
                provider=self.name,
                processing_time=processing_time,
                cost=cost,
                word_timestamps=word_timestamps and "words" in result.get("segments", [{}])[0] if result.get("segments") else False
            )
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            # Attempt to reload the model on error
            try:
                self.model = None
                self.is_available()
            except:
                pass
            raise RuntimeError(f"Whisper transcription failed: {str(e)}")
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate overall confidence score from segments"""
        if not result.get("segments"):
            return 0.0
        
        confidences = [seg.get("confidence", 0.0) for seg in result["segments"]]
        return sum(confidences) / len(confidences) if confidences else 0.0


class GoogleSTTProvider(STTProvider):
    """Google Speech-to-Text API provider"""
    
    def __init__(self, api_key: Optional[str] = None, use_enhanced: bool = False):
        self.api_key = api_key or GOOGLE_API_KEY
        self.use_enhanced = use_enhanced
        self._supported_languages = set()
    
    @property
    def name(self) -> str:
        return f"google_stt{'_enhanced' if self.use_enhanced else ''}"
    
    @property
    def tier(self) -> Tier:
        return Tier.PREMIUM if self.use_enhanced else Tier.STANDARD
    
    @property
    def cost_per_unit(self) -> float:
        """Cost per second of audio"""
        return 0.0006 if self.use_enhanced else 0.0004  # $0.009 or $0.006 per 15 seconds
    
    @property
    def is_paid_service(self) -> bool:
        return True
    
    @property
    def supported_languages(self) -> Set[str]:
        """Return set of supported language codes"""
        if not self._supported_languages:
            # Common languages supported by Google STT
            self._supported_languages = {
                "en-US", "en-GB", "es-ES", "es-US", "fr-FR", "de-DE", 
                "it-IT", "ja-JP", "ko-KR", "pt-BR", "zh-CN", "ru-RU"
            }
        return self._supported_languages
    
    @property
    def supports_word_timestamps(self) -> bool:
        """Return True if provider supports word-level timestamps"""
        return True
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def transcribe(self, 
                  audio_path: str, 
                  language: Optional[str] = None,
                  word_timestamps: bool = False,
                  **kwargs) -> TranscriptionResult:
        """Transcribe using Google Speech-to-Text API"""
        
        if not self.is_available():
            raise RuntimeError("Google Speech API key not configured")
        
        start_time = time.time()
        
        try:
            # Convert audio to base64
            audio_content = audio_to_base64(audio_path)
            
            # Get audio duration
            audio_duration = 0
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                audio_duration = info.duration
            except Exception as e:
                logger.warning(f"Could not get audio duration: {e}")
            
            # Prepare request
            url = f"https://speech.googleapis.com/v1p1beta1/speech:recognize?key={self.api_key}"
            
            # Handle language code
            if language and "-" not in language and len(language) == 2:
                # Convert ISO 639-1 to BCP-47
                language_mapping = {
                    "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
                    "it": "it-IT", "ja": "ja-JP", "ko": "ko-KR", "pt": "pt-BR",
                    "zh": "zh-CN", "ru": "ru-RU"
                }
                language = language_mapping.get(language, f"{language}-{language.upper()}")
            
            data = {
                "config": {
                    "encoding": "LINEAR16",
                    "sampleRateHertz": 16000,
                    "languageCode": language or "en-US",
                    "enableWordTimeOffsets": word_timestamps,
                    "enableAutomaticPunctuation": True,
                    "model": "latest_long" if self.use_enhanced else "latest_short",
                    "useEnhanced": self.use_enhanced
                },
                "audio": {
                    "content": audio_content
                }
            }
            
            # Make API request
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Process results
            transcript = ""
            confidence = 0.0
            segments = []
            
            if "results" in result:
                for res in result["results"]:
                    if "alternatives" in res and res["alternatives"]:
                        alt = res["alternatives"][0]
                        transcript += alt["transcript"] + " "
                        confidence += alt.get("confidence", 0.0)
                        
                        # Process word timestamps if available
                        if word_timestamps and "words" in alt:
                            # Convert Google's word timing format to Whisper-like format
                            words = []
                            for word in alt["words"]:
                                word_start = float(word["startTime"].replace("s", ""))
                                word_end = float(word["endTime"].replace("s", ""))
                                words.append({
                                    "word": word["word"],
                                    "start": word_start,
                                    "end": word_end,
                                    "confidence": alt.get("confidence", 0.0)
                                })
                            
                            # Create a segment
                            segments.append({
                                "text": alt["transcript"],
                                "start": words[0]["start"] if words else 0,
                                "end": words[-1]["end"] if words else 0,
                                "confidence": alt.get("confidence", 0.0),
                                "words": words
                            })
                        else:
                            # Create a simple segment without word timestamps
                            segments.append({
                                "text": alt["transcript"],
                                "confidence": alt.get("confidence", 0.0)
                            })
            
            # Average confidence across all segments
            if result.get("results", []):
                confidence = confidence / len(result["results"])
            
            processing_time = time.time() - start_time
            
            # Calculate cost
            cost = update_cost_tracker(self.name, "stt", duration_seconds=audio_duration)
            
            return TranscriptionResult(
                text=transcript.strip(),
                confidence=confidence,
                segments=segments,
                language=language or "en-US",
                duration=audio_duration,
                provider=self.name,
                processing_time=processing_time,
                cost=cost,
                word_timestamps=word_timestamps and any("words" in seg for seg in segments)
            )
        except Exception as e:
            logger.error(f"Google STT API error: {e}")
            raise RuntimeError(f"Google transcription failed: {str(e)}")

class AzureSTTProvider(STTProvider):
    """Azure Speech-to-Text API provider"""
    
    def __init__(self, api_key: Optional[str] = None, region: Optional[str] = None):
        self.api_key = api_key or AZURE_SPEECH_KEY
        self.region = region or AZURE_SPEECH_REGION
        self._supported_languages = set()
    
    @property
    def name(self) -> str:
        return "azure_stt"
    
    @property
    def tier(self) -> Tier:
        return Tier.PREMIUM
    
    @property
    def cost_per_unit(self) -> float:
        """Cost per second of audio"""
        return 0.0002778  # $1.00 per hour
    
    @property
    def is_paid_service(self) -> bool:
        return True
    
    @property
    def supported_languages(self) -> Set[str]:
        """Return set of supported language codes"""
        if not self._supported_languages:
            # Common languages supported by Azure STT
            self._supported_languages = {
                "en-US", "en-GB", "es-ES", "fr-FR", "de-DE", 
                "it-IT", "ja-JP", "ko-KR", "pt-BR", "zh-CN", "ru-RU"
            }
        return self._supported_languages
    
    @property
    def supports_word_timestamps(self) -> bool:
        """Return True if provider supports word-level timestamps"""
        return True
    
    def is_available(self) -> bool:
        return bool(self.api_key) and bool(self.region)
    
    def transcribe(self, 
                  audio_path: str, 
                  language: Optional[str] = None,
                  word_timestamps: bool = False,
                  **kwargs) -> TranscriptionResult:
        """Transcribe using Azure Speech-to-Text API"""
        
        if not self.is_available():
            raise RuntimeError("Azure Speech API key or region not configured")
        
        start_time = time.time()
        
        try:
            # Convert audio to base64
            audio_content = audio_to_base64(audio_path)
            
            # Get audio duration
            audio_duration = 0
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                audio_duration = info.duration
            except Exception as e:
                logger.warning(f"Could not get audio duration: {e}")
            
            # Handle language code
            if language and "-" not in language and len(language) == 2:
                # Convert ISO 639-1 to BCP-47
                language_mapping = {
                    "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
                    "it": "it-IT", "ja": "ja-JP", "ko": "ko-KR", "pt": "pt-BR",
                    "zh": "zh-CN", "ru": "ru-RU"
                }
                language = language_mapping.get(language, f"{language}-{language.upper()}")
            
            # Prepare request
            url = f"https://{self.region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            
            params = {
                "language": language or "en-US",
                "format": "detailed",
                "profanity": "masked"
            }
            
            headers = {
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                "Accept": "application/json"
            }
            
            # Read audio file
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Make API request
            response = requests.post(url, params=params, headers=headers, data=audio_data)
            response.raise_for_status()
            result = response.json()
            
            # Process results
            transcript = result.get("DisplayText", "")
            confidence = result.get("Confidence", 0.0)
            
            # Create segments
            segments = [{
                "text": transcript,
                "confidence": confidence
            }]
            
            # Process word timestamps if available and requested
            if word_timestamps and "Words" in result:
                words = []
                for word_info in result["Words"]:
                    words.append({
                        "word": word_info["Word"],
                        "start": word_info["Offset"] / 10000000,  # Convert from 100-nanosecond units to seconds
                        "end": (word_info["Offset"] + word_info["Duration"]) / 10000000,
                        "confidence": word_info.get("Confidence", 0.0)
                    })
                
                # Add words to the segment
                if words:
                    segments[0]["words"] = words
                    segments[0]["start"] = words[0]["start"]
                    segments[0]["end"] = words[-1]["end"]
            
            processing_time = time.time() - start_time
            
            # Calculate cost
            cost = update_cost_tracker(self.name, "stt", duration_seconds=audio_duration)
            
            return TranscriptionResult(
                text=transcript,
                confidence=confidence,
                segments=segments,
                language=language or "en-US",
                duration=audio_duration,
                provider=self.name,
                processing_time=processing_time,
                cost=cost,
                word_timestamps=word_timestamps and "Words" in result
            )
        except Exception as e:
            logger.error(f"Azure STT API error: {e}")
            raise RuntimeError(f"Azure transcription failed: {str(e)}")

class AWSSTTProvider(STTProvider):
    """AWS Transcribe API provider"""
    
    def __init__(self, access_key: Optional[str] = None, 
                 secret_key: Optional[str] = None,
                 region: Optional[str] = None):
        self.access_key = access_key or AWS_ACCESS_KEY
        self.secret_key = secret_key or AWS_SECRET_KEY
        self.region = region or AWS_REGION
        self._supported_languages = set()
        self.client = None
    
    @property
    def name(self) -> str:
        return "aws_transcribe"
    
    @property
    def tier(self) -> Tier:
        return Tier.PREMIUM
    
    @property
    def cost_per_unit(self) -> float:
        """Cost per second of audio"""
        return 0.0004  # $0.024 per minute
    
    @property
    def is_paid_service(self) -> bool:
        return True
    
    @property
    def supported_languages(self) -> Set[str]:
        """Return set of supported language codes"""
        if not self._supported_languages:
            # Common languages supported by AWS Transcribe
            self._supported_languages = {
                "en-US", "en-GB", "es-ES", "fr-FR", "de-DE", 
                "it-IT", "ja-JP", "ko-KR", "pt-BR", "zh-CN", "ru-RU"
            }
        return self._supported_languages
    
    @property
    def supports_word_timestamps(self) -> bool:
        """Return True if provider supports word-level timestamps"""
        return True
    
    def is_available(self) -> bool:
        if not (self.access_key and self.secret_key and self.region):
            return False
            
        try:
            if self.client is None:
                import boto3
                self.client = boto3.client(
                    'transcribe',
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name=self.region
                )
            return True
        except Exception as e:
            logger.exception(f"AWS Transcribe client initialization failed: {e}")
            return False
    
    def transcribe(self, 
                  audio_path: str, 
                  language: Optional[str] = None,
                  word_timestamps: bool = False,
                  **kwargs) -> TranscriptionResult:
        """Transcribe using AWS Transcribe API"""
        
        if not self.is_available():
            raise RuntimeError("AWS Transcribe credentials not configured")
        
        start_time = time.time()
        
        try:
            import boto3
            import uuid
            
            # Generate a unique job name
            job_name = f"transcribe-{str(uuid.uuid4())}"
            
            # Get S3 bucket from env or use default
            s3_bucket = os.getenv("AWS_S3_BUCKET")
            
            if not s3_bucket:
                raise RuntimeError("AWS S3 bucket not configured")
            
            # Get audio duration
            audio_duration = 0
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                audio_duration = info.duration
            except Exception as e:
                logger.warning(f"Could not get audio duration: {e}")
            
            # Handle language code
            if language and "-" not in language and len(language) == 2:
                # Convert ISO 639-1 to BCP-47
                language_mapping = {
                    "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
                    "it": "it-IT", "ja": "ja-JP", "ko": "ko-KR", "pt": "pt-BR",
                    "zh": "zh-CN", "ru": "ru-RU"
                }
                language = language_mapping.get(language, f"{language}-{language.upper()}")
            
            # Upload audio to S3
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            
            s3_file_key = f"transcribe_inputs/{job_name}.wav"
            s3_client.upload_file(audio_path, s3_bucket, s3_file_key)
            
            # Create transcription job
            media_uri = f"s3://{s3_bucket}/{s3_file_key}"
            
            transcribe_args = {
                'TranscriptionJobName': job_name,
                'Media': {'MediaFileUri': media_uri},
                'MediaFormat': 'wav',
                'LanguageCode': language or 'en-US',
                'Settings': {
                    'ShowAlternatives': False,
                    'MaxAlternatives': 1
                }
            }
            
            # Add word timestamps if requested
            if word_timestamps:
                transcribe_args['Settings']['ShowSpeakerLabels'] = False
                transcribe_args['Settings']['EnableWordTimeOffsets'] = True
            
            # Start transcription job
            self.client.start_transcription_job(**transcribe_args)
            
            # Wait for completion with timeout
            max_wait = 300  # 5 minutes timeout
            wait_time = 0
            while wait_time < max_wait:
                job_status = self.client.get_transcription_job(TranscriptionJobName=job_name)
                status = job_status['TranscriptionJob']['TranscriptionJobStatus']
                
                if status in ['COMPLETED', 'FAILED']:
                    break
                    
                time.sleep(5)
                wait_time += 5
            
            if status == 'FAILED':
                failure_reason = job_status['TranscriptionJob'].get('FailureReason', 'Unknown error')
                raise RuntimeError(f"AWS Transcribe job failed: {failure_reason}")
                
            if status != 'COMPLETED':
                raise RuntimeError("AWS Transcribe job timed out")
            
            # Get result
            transcript_url = job_status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcript_response = requests.get(transcript_url)
            transcript_response.raise_for_status()
            result = transcript_response.json()
            
            # Parse results
            transcript = result['results']['transcripts'][0]['transcript']
            
            # Process segments with word timestamps if available
            segments = []
            words = []
            confidence_sum = 0
            confidence_count = 0
            
            if word_timestamps and 'items' in result['results']:
                current_segment = {
                    "text": "",
                    "words": [],
                    "start": 0,
                    "end": 0,
                    "confidence": 0
                }
                
                for item in result['results']['items']:
                    # Skip non-word items like punctuation
                    if item['type'] != 'pronunciation':
                        continue
                        
                    word_confidence = float(item['alternatives'][0]['confidence'])
                    confidence_sum += word_confidence
                    confidence_count += 1
                    
                    word_info = {
                        "word": item['alternatives'][0]['content'],
                        "start": float(item['start_time']),
                        "end": float(item['end_time']),
                        "confidence": word_confidence
                    }
                    
                    words.append(word_info)
                
                # Create a single segment with all words
                if words:
                    segments.append({
                        "text": transcript,
                        "words": words,
                        "start": words[0]["start"],
                        "end": words[-1]["end"],
                        "confidence": confidence_sum / confidence_count if confidence_count > 0 else 0
                    })
            else:
                # Create a simple segment without word timestamps
                segments.append({
                    "text": transcript,
                    "confidence": 0.0  # AWS doesn't provide an overall confidence
                })
            
            # Calculate average confidence
            confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
            
            processing_time = time.time() - start_time
            
            # Clean up resources
            try:
                # Delete S3 file
                s3_client.delete_object(Bucket=s3_bucket, Key=s3_file_key)
                
                # Delete transcription job
                self.client.delete_transcription_job(TranscriptionJobName=job_name)
            except Exception as e:
                logger.warning(f"Clean-up error: {e}")
            
            # Calculate cost
            cost = update_cost_tracker(self.name, "stt", duration_seconds=audio_duration)
            
            return TranscriptionResult(
                text=transcript,
                confidence=confidence,
                segments=segments,
                language=language or "en-US",
                duration=audio_duration,
                provider=self.name,
                processing_time=processing_time,
                cost=cost,
                word_timestamps=bool(words)
            )
        except Exception as e:
            logger.error(f"AWS Transcribe error: {e}")
            raise RuntimeError(f"AWS transcription failed: {str(e)}")

class STTEngine(BaseEngine):
    """Production-grade STT engine with multiple provider support"""
    
    def __init__(self, providers: Optional[List[STTProvider]] = None):
        self.providers = providers or []
        self.executor = None
        self.cache_enabled = True
    
    def initialize(self):
        """Initialize resources"""
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # Verify provider availability
        available_providers = []
        for provider in self.providers:
            if provider.is_available():
                available_providers.append(provider)
            else:
                logger.warning(f"Provider {provider.name} is not available")
        
        self.providers = available_providers
        
        if not self.providers:
            logger.error("No STT providers are available")
            raise RuntimeError("No STT providers are available")
        
        logger.info(f"STT Engine initialized with providers: {[p.name for p in self.providers]}")
    
    def shutdown(self):
        """Release resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def add_provider(self, provider: STTProvider) -> None:
        """Add a new provider to the engine"""
        if provider.is_available():
            self.providers.append(provider)
            logger.info(f"Added provider: {provider.name}")
        else:
            logger.warning(f"Provider {provider.name} is not available and will not be added")
    
    def transcribe(self, 
                  audio_path: str, 
                  language: Optional[str] = None,
                  use_cache: bool = True,
                  provider_name: Optional[str] = None,
                  fallback: bool = True,
                  **kwargs) -> TranscriptionResult:
        """
        Transcribe audio with specified provider or use fallback strategy
        
        Args:
            audio_path: Path to audio file
            language: Language code
            use_cache: Whether to use Redis cache
            provider_name: Specific provider to use
            fallback: Whether to try other providers if the specified one fails
            **kwargs: Additional parameters
            
        Returns:
            TranscriptionResult object
        """
        # Check if audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Generate cache key based on file content hash and parameters
        cache_key = None
        if use_cache and self.cache_enabled:
            import hashlib
            with open(audio_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            param_str = json.dumps({
                "language": language,
                "provider": provider_name,
                **kwargs
            }, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()
            cache_key = f"stt:{file_hash}:{param_hash}"
            
            # Check cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {cache_key}")
                return TranscriptionResult(**json.loads(cached_result))
        
        # Select providers
        selected_providers = []
        if provider_name:
            # Find the specific provider
            for provider in self.providers:
                if provider.name == provider_name:
                    selected_providers.append(provider)
                    break
            
            if not selected_providers and fallback:
                # If specified provider not found and fallback is enabled
                selected_providers = self.providers
            elif not selected_providers:
                raise ValueError(f"Provider {provider_name} not found or not available")
        else:
            # Use all providers
            selected_providers = self.providers
        
        # Try providers in order
        last_error = None
        for provider in selected_providers:
            try:
                result = provider.transcribe(
                    audio_path=audio_path,
                    language=language,
                    **kwargs
                )
                
                # Cache result
                if cache_key:
                    redis_client.setex(
                        cache_key,
                        CACHE_TTL,
                        json.dumps(asdict(result))
                    )
                
                return result
            
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {str(e)}")
                last_error = e
        
        # If all providers failed
        if last_error:
            logger.error(f"All providers failed to transcribe: {str(last_error)}")
            raise RuntimeError(f"Transcription failed with all providers: {str(last_error)}")
        
        raise RuntimeError("No providers available for transcription")


# --- TTS Implementation ---

@dataclass
class SynthesisResult:
    """Structured synthesis result"""
    audio_path: Optional[str]
    audio_data: Optional[np.ndarray]
    sample_rate: int
    duration: float
    provider: str
    processing_time: float

class TTSProvider(BaseProvider):
    """Base class for TTS providers"""
    
    @abstractmethod
    def synthesize(self, 
                  text: str,
                  voice: Optional[str] = None,
                  language: Optional[str] = None,
                  output_path: Optional[str] = None,
                  **kwargs) -> SynthesisResult:
        """Synthesize text to speech"""
        pass

class CoquiTTSProvider(TTSProvider):
    """Coqui TTS-based provider"""
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", 
                 use_cuda: bool = True):
        self.model_name = model_name
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.tts = None
    
    @property
    def name(self) -> str:
        return f"coqui_{self.model_name.split('/')[-1]}"
    
    @property
    def tier(self) -> Tier:
        return Tier.PREMIUM

    @property
    def cost_per_unit(self) -> float:
        return 0.001  # Example: $0.001 per second
    
    def is_available(self) -> bool:
        try:
            if self.tts is None:
                # Use pre-downloaded models from local cache
                cache_dir = os.getenv("TORCH_HOME", "/app/models/coqui")
                if not os.path.exists(cache_dir):
                    logger.warning(f"Cache directory not found: {cache_dir}, falling back to download")
                    cache_dir = None
                
                self.tts = TTS(
                    model_name=self.model_name,
                    progress_bar=False,
                    gpu=self.use_cuda
                )
            return True
        except Exception as e:
            logger.exception(f"Coqui TTS model {self.model_name} is not available: {e}")
            return False

    async def synthesize(self,
                        text: str,
                        voice: Optional[str] = None,
                        language: Optional[str] = None,
                        output_path: Optional[str] = None,
                        **kwargs) -> SynthesisResult:
        """
        Synthesize speech from text using Coqui TTS.
        
        Args:
            text: The text to synthesize
            voice: The voice to use (not used for Coqui TTS)
            language: The language of the text (not used for Coqui TTS)
            output_path: Path to save the audio file (optional)
            **kwargs: Additional arguments for model generation
            
        Returns:
            SynthesisResult: The synthesis result containing audio data and metadata
            
        Raises:
            RuntimeError: If the model is not available or synthesis fails
        """
        if not self.is_available():
            raise RuntimeError(f"Coqui TTS model {self.model_name} is not available")

        start_time = time.time()

        try:
            # Generate audio
            audio_array = self.tts.tts(text=text, **kwargs)
            processing_time = time.time() - start_time
            duration = len(audio_array) / self.tts.synthesizer.output_sample_rate

            # Save audio if output path is provided
            saved_path = None
            if output_path:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    sf.write(output_path, audio_array, self.tts.synthesizer.output_sample_rate)
                    saved_path = output_path
                    logger.info(f"Audio saved to: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save audio to {output_path}: {e}")

            # Update cost tracking
            cost = update_cost_tracker(self.name, "tts", characters=len(text))

            return SynthesisResult(
                audio_path=saved_path,
                audio_data=audio_array if not saved_path else None,
                sample_rate=self.tts.synthesizer.output_sample_rate,
                duration=duration,
                provider=self.name,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Coqui TTS synthesis error: {e}")
            raise RuntimeError(f"Coqui TTS synthesis failed: {str(e)}")

class AI4BharatTTSProvider(TTSProvider):
    """
    TTS provider using AI4Bharat models (like Indic Parler-TTS) via Hugging Face.
    Production-grade implementation with robust model loading and caching.
    """
    SPEAKER_DESCRIPTIONS = {
        "rohit": "Rohit speaks with a clear voice at a normal pace.",
        "divya": "Divya's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
        "aditi": "Aditi speaks with a slightly higher pitch in a close-sounding environment. Her voice is clear, with subtle emotional depth and a normal pace, all captured in high-quality recording.",
        "sita": "Sita speaks at a fast pace with a slightly low-pitched voice, captured clearly in a close-sounding environment with excellent recording quality.",
        "karan": "Karan's high-pitched, engaging voice is captured in a clear, close-sounding recording. His slightly slower delivery conveys a positive tone.",
        "sunita": "Sunita speaks with a high pitch in a close environment. Her voice is clear, with slight dynamic changes, and the recording is of excellent quality.",
        "jaya": "Jaya speaks with a clear, moderate-pitched voice.",
        "prakash": "Prakash speaks with a moderate pace and clear articulation.",
        "suresh": "Suresh has a standard male voice with good clarity.",
        "anjali": "Anjali speaks with a high pitch at a normal pace in a clear, close-sounding environment.",
        "yash": "Yash speaks clearly with a standard male pitch.",
        "default_female": "A female speaker with a slightly low-pitched voice speaks in a very monotonous way, with a close recording quality.",
        "default_male": "A male speaker with a low-pitched voice speaks in a very monotonous way, with a close recording quality."
    }

    def __init__(self, model_id: str = TTS_MODEL_ID, device: Optional[str] = None):
        """
        Initialize the AI4Bharat TTS provider.
        
        Args:
            model_id: The Hugging Face model ID to use
            device: The device to run the model on (cuda/cpu)
        """
        if not PARLER_TTS_AVAILABLE:
            raise RuntimeError("parler-tts library is required but not installed.")

        self.model_id = model_id
        self.device = device or ("cuda" if TTS_USE_CUDA and torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        self.sampling_rate = TTS_DEFAULT_SAMPLING_RATE
        self._load_attempts = 0
        self._max_load_attempts = TTS_MAX_LOAD_ATTEMPTS
        self._last_load_attempt = 0
        self._model_lock = asyncio.Lock()  # For thread-safe model loading
        self._is_loaded = False

    @property
    def name(self) -> str:
        return f"ai4bharat_{self.model_id.split('/')[-1]}"

    @property
    def tier(self) -> Tier:
        return Tier.STANDARD

    @property
    def cost_per_unit(self) -> float:
        return 0.0

    def _get_cache_dir(self) -> Optional[str]:
        """
        Get the first available cache directory.
        
        Returns:
            Optional[str]: Path to the first available cache directory, or None if none are available
        """
        for cache_dir in TTS_MODEL_CACHE_DIRS:
            if cache_dir and os.path.exists(cache_dir):
                logger.info(f"Using cache directory: {cache_dir}")
                return cache_dir
        logger.warning("No cache directory found, will use default huggingface cache")
        return None

    async def _load_model(self) -> bool:
        """
        Load the model and tokenizers asynchronously.
        
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        async with self._model_lock:
            if self.model is not None:
                return True

            current_time = time.time()
            if (self._load_attempts >= self._max_load_attempts and 
                current_time - self._last_load_attempt < TTS_LOAD_RETRY_DELAY):
                return False

            try:
                logger.info(f"Attempting to load AI4Bharat model: {self.model_id} to {self.device}")
                self._last_load_attempt = current_time
                self._load_attempts += 1

                cache_dir = self._get_cache_dir()
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)

                # Load model and tokenizers
                self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                    self.model_id,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    local_files_only=not TTS_FORCE_DOWNLOAD
                ).to(self.device)

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    cache_dir=cache_dir,
                    local_files_only=not TTS_FORCE_DOWNLOAD
                )

                desc_tokenizer_name = getattr(self.model.config, "text_encoder", {}).get("_name_or_path", "google/flan-t5-large")
                self.description_tokenizer = AutoTokenizer.from_pretrained(
                    desc_tokenizer_name,
                    cache_dir=cache_dir,
                    local_files_only=not TTS_FORCE_DOWNLOAD
                )

                # Set sampling rate from model config if available
                if hasattr(self.model.config, "sampling_rate"):
                    self.sampling_rate = self.model.config.sampling_rate
                elif hasattr(self.model.config, "audio_encoder") and hasattr(self.model.config.audio_encoder, "sampling_rate"):
                    self.sampling_rate = self.model.config.audio_encoder.sampling_rate
                else:
                    logger.warning(f"Could not determine sampling rate from model config for {self.model_id}, using default: {self.sampling_rate}Hz")

                self._load_attempts = 0
                logger.info(f"Successfully loaded AI4Bharat model: {self.model_id}")
                self._is_loaded = True
                return True

            except Exception as e:
                logger.exception(f"Failed to load AI4Bharat model {self.model_id}: {e}")
                self.model = None
                self.tokenizer = None
                self.description_tokenizer = None
                self._is_loaded = False
                return False

    def is_available(self) -> bool:
        """
        Synchronous check: returns True if model is already loaded, False otherwise.
        """
        return self.model is not None and self._is_loaded

    async def async_is_available(self) -> bool:
        """
        Asynchronous check: loads model if needed, returns True if available.
        """
        return await self._load_model()

    async def synthesize(self,
                        text: str,
                        voice: Optional[str] = None,
                        language: Optional[str] = None,
                        output_path: Optional[str] = None,
                        **kwargs) -> SynthesisResult:
        """
        Synthesize speech from text using the AI4Bharat model.
        
        Args:
            text: The text to synthesize
            voice: The voice to use (must be one of SPEAKER_DESCRIPTIONS keys)
            language: The language of the text (optional)
            output_path: Path to save the audio file (optional)
            **kwargs: Additional arguments for model generation
            
        Returns:
            SynthesisResult: The synthesis result containing audio data and metadata
            
        Raises:
            RuntimeError: If the model is not available or synthesis fails
        """
        if not await self._load_model():
            raise RuntimeError(f"AI4Bharat model {self.model_id} is not available")

        start_time = time.time()

        try:
            # Get voice description
            voice_key = str(voice).lower() if voice else "default_female"
            description_text = self.SPEAKER_DESCRIPTIONS.get(
                voice_key,
                self.SPEAKER_DESCRIPTIONS["default_female"]
            )
            logger.info(f"Using description for voice '{voice}': {description_text}")

            # Prepare inputs
            prompt_input_ids = self.tokenizer(text, return_tensors="pt").to(self.device)
            description_input_ids = self.description_tokenizer(description_text, return_tensors="pt").to(self.device)

            # Generate audio
            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask,
                    **kwargs
                ).cpu()

            audio_array = generation.squeeze().numpy()
            processing_time = time.time() - start_time
            duration = len(audio_array) / self.sampling_rate

            # Save audio if output path is provided
            saved_path = None
            if output_path:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    sf.write(output_path, audio_array, self.sampling_rate)
                    saved_path = output_path
                    logger.info(f"Audio saved to: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save audio to {output_path}: {e}")

            # Update cost tracking
            cost = update_cost_tracker(self.name, "tts", characters=len(text))

            return SynthesisResult(
                audio_path=saved_path,
                audio_data=audio_array if not saved_path else None,
                sample_rate=self.sampling_rate,
                duration=duration,
                provider=self.name,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"AI4Bharat TTS synthesis error: {e}")
            raise RuntimeError(f"AI4Bharat synthesis failed: {str(e)}")

class TTSEngine(BaseEngine):
    """Production-grade TTS engine with multiple provider support"""
    
    def __init__(self, providers: Optional[List[TTSProvider]] = None):
        self.providers = providers or []
        self.executor = None
        self.cache_enabled = True
        
        # Map providers to their supported languages
        self.provider_language_map = {
            "ai4bharat_indic-parler-tts": {"hi", "bn", "as", "doi", "mr", "ta", "te", "kn", "ml", "gu"},
            "coqui_tacotron2-DDC": {"en"},
            "coqui_vits": {"en"},
            "coqui_tacotron2-DCA": {"de"}
        }
    
    def initialize(self):
        """Initialize resources"""
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # Verify provider availability
        available_providers = []
        for provider in self.providers:
            # Use sync check for now; async check will be used in health check
            if provider.is_available():
                available_providers.append(provider)
            else:
                logger.warning(f"Provider {provider.name} is not available")
        
        self.providers = available_providers
        
        if not self.providers:
            logger.error("No TTS providers are available")
            raise RuntimeError("No TTS providers are available")
        
        logger.info(f"TTS Engine initialized with providers: {[p.name for p in self.providers]}")
        
    def shutdown(self):
        """Release resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def add_provider(self, provider: TTSProvider) -> None:
        """Add a new provider to the engine"""
        if provider.is_available():
            self.providers.append(provider)
            logger.info(f"Added TTS provider: {provider.name}")
        else:
            logger.warning(f"Provider {provider.name} is not available and will not be added")
    
    def _get_provider_for_language(self, language: Optional[str]) -> Optional[TTSProvider]:
        """Get the best provider for the given language"""
        if not language:
            return self.providers[0] if self.providers else None
            
        for provider in self.providers:
            if language in self.provider_language_map.get(provider.name, set()):
                return provider
        return None
    
    async def synthesize(self, 
                        text: str,
                        voice: Optional[str] = None,
                        language: Optional[str] = None,
                        output_path: Optional[str] = None,
                        use_cache: bool = True,
                        provider_name: Optional[str] = None,
                        fallback: bool = True,
                        **kwargs) -> SynthesisResult:
        """
        Synthesize speech from text using the best available provider.
        
        Args:
            text: The text to synthesize
            voice: The voice to use
            language: The language of the text
            output_path: Path to save the audio file
            use_cache: Whether to use cached results
            provider_name: Specific provider to use
            fallback: Whether to fall back to other providers if the preferred one fails
            **kwargs: Additional arguments for model generation
            
        Returns:
            SynthesisResult: The synthesis result
            
        Raises:
            RuntimeError: If no provider is available or synthesis fails
        """
        if not self.providers:
            raise RuntimeError("No TTS providers are available")

        # Try to get the requested provider
        provider = None
        if provider_name:
            provider = next((p for p in self.providers if p.name == provider_name), None)
            if not provider:
                raise RuntimeError(f"Requested provider {provider_name} is not available")

        # If no specific provider requested, try to find one for the language
        if not provider and language:
            provider = self._get_provider_for_language(language)

        # If still no provider, use the first available one
        if not provider:
            provider = self.providers[0]

        try:
            # Check cache if enabled
            if use_cache and self.cache_enabled:
                cache_key = f"tts:{provider.name}:{hashlib.md5(text.encode()).hexdigest()}"
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    try:
                        result_dict = json.loads(cached_result)
                        return SynthesisResult(**result_dict)
                    except Exception as e:
                        logger.warning(f"Failed to load cached result: {e}")

            # Perform synthesis
            result = await provider.synthesize(
                text=text,
                voice=voice,
                language=language,
                output_path=output_path,
                **kwargs
            )

            # Cache the result if enabled
            if use_cache and self.cache_enabled:
                try:
                    result_dict = asdict(result)
                    redis_client.setex(
                        cache_key,
                        CACHE_TTL,
                        json.dumps(result_dict)
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")

            return result

        except Exception as e:
            logger.error(f"Synthesis failed with provider {provider.name}: {e}")
            
            if fallback and len(self.providers) > 1:
                # Try other providers
                for fallback_provider in self.providers:
                    if fallback_provider != provider:
                        try:
                            logger.info(f"Trying fallback provider: {fallback_provider.name}")
                            result = await fallback_provider.synthesize(
                                text=text,
                                voice=voice,
                                language=language,
                                output_path=output_path,
                                **kwargs
                            )
                            return result
                        except Exception as fallback_error:
                            logger.error(f"Fallback provider {fallback_provider.name} failed: {fallback_error}")
                            continue
            
            raise RuntimeError(f"All TTS providers failed: {str(e)}")


# --- REST API Implementation ---

app = FastAPI(
    title="Speech Processing API",
    description="Production-grade STT and TTS API",
    version="1.0.0"
)

# --- CORS Middleware (allow all origins for demo, restrict in prod) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TranscriptionRequest(BaseModel):
    language: Optional[str] = None
    provider: Optional[str] = None
    use_cache: bool = True
    fallback: bool = True

class TranscriptionResponse(BaseModel):
    id: str
    text: str
    confidence: float
    language: str
    duration: float
    provider: str
    processing_time: float
    segments: List[Dict] = Field(default_factory=list)

class SynthesisRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    provider: Optional[str] = None
    use_cache: bool = True
    fallback: bool = True

class SynthesisResponse(BaseModel):
    id: str
    duration: float
    provider: str
    processing_time: float

# Create engines
stt_engine = STTEngine([
    WhisperProvider(model_name="small"),
    WhisperProvider(model_name="base")
])

# Create TTS engine with available providers
tts_providers = []

# Add AI4Bharat provider if parler-tts is available
if PARLER_TTS_AVAILABLE:
    try:
        ai4b_provider = AI4BharatTTSProvider(
            model_id="ai4bharat/indic-parler-tts",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        tts_providers.append(ai4b_provider)
        logger.info("Added AI4Bharat TTS provider")
    except Exception as e:
        logger.warning(f"Failed to initialize AI4Bharat TTS provider: {e}")

# Add Coqui providers
try:
    coqui_providers = [
        CoquiTTSProvider(model_name="tts_models/en/ljspeech/tacotron2-DDC"),
        CoquiTTSProvider(model_name="tts_models/en/vctk/vits"),
        CoquiTTSProvider(model_name="tts_models/de/thorsten/tacotron2-DCA")
    ]
    tts_providers.extend(coqui_providers)
    logger.info("Added Coqui TTS providers")
except Exception as e:
    logger.warning(f"Failed to initialize Coqui TTS providers: {e}")

# Ensure we have at least one TTS provider
if not tts_providers:
    logger.error("No TTS providers available")
    raise RuntimeError("No TTS providers available")

tts_engine = TTSEngine(tts_providers)

# Initialize engines on startup
@app.on_event("startup")
async def startup_event():
    try:
        stt_engine.initialize()
        tts_engine.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize engines: {e}")
        # Let the application start, but endpoints will fail

# Shutdown engines on application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    stt_engine.shutdown()
    tts_engine.shutdown()

# Helper: LLM-based language detection (Anthropic)
def detect_language_llm(text):
    """Detect language using Anthropic LLM as fallback."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        prompt = f"What is the ISO 639-1 language code for the following text? Only return the code.\nText: {text}"
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 5,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        resp = pyrequests.post(url, headers=headers, json=data, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        # Extract the code from the response
        code = result["content"][0]["text"].strip().split()[0]
        return code
    except Exception as e:
        logger.error(f"Anthropic LLM language detection failed: {e}")
        return None

# STT endpoint
@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    request: TranscriptionRequest = Depends(),
    file: UploadFile = File(...)
):
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save uploaded file to temp directory
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, f"{request_id}_{file.filename}")
    
    try:
        # Save the file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Always convert to 16kHz mono WAV for backend robustness
        converted_path = temp_file_path + ".converted.wav"
        convert_audio_format(temp_file_path, converted_path, format="wav", sample_rate=16000)
        
        # Process transcription using the converted file
        result = stt_engine.transcribe(
            audio_path=converted_path,
            language=request.language,
            use_cache=request.use_cache,
            provider_name=request.provider,
            fallback=request.fallback
        )
        
        # Language detection logic
        detected_lang = None
        try:
            detected_lang = detect(result.text)
        except LangDetectException:
            detected_lang = None
            
        # Fallback to Anthropic LLM if detection failed
        if not detected_lang:
            detected_lang = detect_language_llm(result.text)
            
        # If user provided language, check for mismatch
        if request.language:
            if detected_lang and detected_lang != request.language:
                background_tasks.add_task(lambda: os.remove(temp_file_path))
                background_tasks.add_task(lambda: os.remove(converted_path))
                background_tasks.add_task(lambda: os.rmdir(temp_dir))
                logger.error(f"Language mismatch: selected {request.language}, detected {detected_lang}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Language mismatch: you selected '{request.language}' but spoke '{detected_lang}'."
                )
            language_out = request.language
        else:
            if not detected_lang:
                background_tasks.add_task(lambda: os.remove(temp_file_path))
                background_tasks.add_task(lambda: os.remove(converted_path))
                background_tasks.add_task(lambda: os.rmdir(temp_dir))
                raise HTTPException(status_code=400, detail="Could not detect language.")
            language_out = detected_lang
            
        # Clean up temporary files
        background_tasks.add_task(lambda: os.remove(temp_file_path))
        background_tasks.add_task(lambda: os.remove(converted_path))
        background_tasks.add_task(lambda: os.rmdir(temp_dir))
        
        return TranscriptionResponse(
            id=request_id,
            text=result.text,
            confidence=result.confidence,
            language=language_out,
            duration=result.duration,
            provider=result.provider,
            processing_time=result.processing_time,
            segments=result.segments
        )
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        converted_path = temp_file_path + ".converted.wav"
        if os.path.exists(converted_path):
            os.remove(converted_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# TTS endpoint
@app.post("/api/synthesize", response_model=SynthesisResponse)
async def synthesize_speech(request: SynthesisRequest):
    request_id = str(uuid.uuid4())
    output_dir = os.path.join("output", "audio")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{request_id}.wav")
    # If language is not provided, try to detect it
    language = request.language
    if not language:
        from langdetect import detect, LangDetectException
        try:
            language = detect(request.text)
        except LangDetectException:
            language = None
        if not language:
            language = detect_language_llm(request.text)
        if not language:
            logger.error("Could not detect language for TTS synthesis.")
            if os.path.exists(output_path):
                os.remove(output_path)
            raise HTTPException(status_code=400, detail="Could not detect language for synthesis.")
    try:
        result = tts_engine.synthesize(
            text=request.text,
            voice=request.voice,
            language=language,
            output_path=output_path,
            use_cache=request.use_cache,
            provider_name=request.provider,
            fallback=request.fallback
        )
        return SynthesisResponse(
            id=request_id,
            duration=result.duration,
            provider=result.provider,
            processing_time=result.processing_time
        )
    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=str(e))

# Get synthesized audio
@app.get("/api/audio/{audio_id}")
async def get_audio(audio_id: str):
    file_path = os.path.join("output", "audio", f"{audio_id}.wav")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=f"{audio_id}.wav"
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    async def check_provider(provider):
        # Use async_is_available if available, else fallback to sync
        if hasattr(provider, "async_is_available"):
            try:
                return provider.name if await provider.async_is_available() else None
            except Exception as e:
                logger.warning(f"Provider {provider.name} async check failed: {e}")
                return None
        else:
            try:
                return provider.name if provider.is_available() else None
            except Exception as e:
                logger.warning(f"Provider {provider.name} sync check failed: {e}")
                return None

    stt_names = [name for name in await asyncio.gather(*[check_provider(p) for p in stt_engine.providers]) if name]
    tts_names = [name for name in await asyncio.gather(*[check_provider(p) for p in tts_engine.providers]) if name]

    health = {
        "status": "ok",
        "stt_providers": stt_names,
        "tts_providers": tts_names
    }
    return JSONResponse(content=health)

# --- New endpoint: Get available AI4Bharat speakers ---
@app.get("/api/ai4b_speakers")
def get_ai4b_speakers():
    # Find the AI4BharatTTSProvider in tts_engine
    ai4b_provider = None
    for provider in tts_engine.providers:
        if hasattr(provider, '__class__') and provider.__class__.__name__ == "AI4BharatTTSProvider":
            ai4b_provider = provider
            break
    
    if ai4b_provider is None:
        return JSONResponse(
            content={"error": "AI4BharatTTSProvider not available"}, 
            status_code=404
        )
    
    # Return the speaker keys and descriptions
    speakers = [
        {"key": k, "description": v}
        for k, v in getattr(ai4b_provider, "SPEAKER_DESCRIPTIONS", {}).items()
    ]
    return JSONResponse(content=speakers)

# --- Example Usage ---

def main():
    # Initialize engines
    stt = STTEngine([
        WhisperProvider(model_name="base")
    ])
    stt.initialize()
    
    tts = TTSEngine([
        CoquiTTSProvider()
    ])
    tts.initialize()
    
    try:
        # Example STT usage
        result = stt.transcribe("sample.wav", language="en")
        print(f"Transcription: {result.text}")
        print(f"Confidence: {result.confidence}")
        
        # Example TTS usage
        result = tts.synthesize(
            "This is a test of the text to speech system.",
            output_path="output.wav"
        )
        print(f"Audio generated: {result.audio_path}")
        print(f"Duration: {result.duration}s")
    
    finally:
        # Clean up resources
        stt.shutdown()
        tts.shutdown()

# Initialize WebSocket service
websocket_service = WebSocketAudioStreamingService(
    stt_engine=stt_engine, 
    tts_engine=tts_engine,
    redis_client=redis_client
)

# WebSocket endpoint
@app.websocket("/ws/audio/{session_id}")
async def websocket_audio_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for full-duplex audio streaming."""
    await websocket.accept()
    
    # Extract parameters from query string
    user_id = websocket.query_params.get("user_id")
    language = websocket.query_params.get("language", "en")
    
    if not user_id:
        await websocket.close(code=1008, reason="Missing user_id parameter")
        return
    
    # Handle WebSocket connection
    await websocket_service.handle_websocket_connection(websocket, f"/ws/audio/{session_id}?user_id={user_id}&language={language}")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_websocket_client():
    """Serve the WebSocket client HTML page."""
    return FileResponse("static/websocket_client.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)