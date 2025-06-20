"""
Configuration module for the Voice Service API.
Centralizes all environment variables and settings.
"""

import os
from typing import Optional, List

# Server configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "speech_service.log")

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = int(os.getenv("REDIS_DB", 0))
CACHE_TTL = int(os.getenv("CACHE_TTL", 86400))  # 24 hours by default

# Rate limiting
RATE_LIMIT_TIER1 = int(os.getenv("RATE_LIMIT_TIER1", 100))  # requests per minute
RATE_LIMIT_TIER2 = int(os.getenv("RATE_LIMIT_TIER2", 300))  # requests per minute

# API keys for third-party services
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Cost configuration
MAX_DAILY_COST = float(os.getenv("MAX_DAILY_COST", "10.0"))  # Maximum daily cost in USD
COST_TIER_RATIO = float(os.getenv("COST_TIER_RATIO", "0.2"))  # Percent of paid API usage (0.0-1.0)

# Application directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model cache directories
WHISPER_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
TTS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")

# TTS Configuration
TTS_MODEL_CACHE_DIRS: List[str] = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "parler_models"),
    os.path.join(os.path.expanduser("~"), ".cache", "parler_models"),
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    "/root/.cache/parler_models",
    "/root/.cache/huggingface"
]

# TTS Model Settings
TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "ai4bharat/indic-parler-tts")
TTS_MAX_LOAD_ATTEMPTS = int(os.getenv("TTS_MAX_LOAD_ATTEMPTS", "3"))
TTS_LOAD_RETRY_DELAY = int(os.getenv("TTS_LOAD_RETRY_DELAY", "300"))  # seconds
TTS_DEFAULT_SAMPLING_RATE = int(os.getenv("TTS_DEFAULT_SAMPLING_RATE", "24000"))
TTS_FORCE_DOWNLOAD = os.getenv("TTS_FORCE_DOWNLOAD", "false").lower() == "true"
TTS_USE_CUDA = os.getenv("TTS_USE_CUDA", "true").lower() == "true"

# WebSocket Configuration
WEBSOCKET_RATE_LIMIT = int(os.getenv("WEBSOCKET_RATE_LIMIT", "100"))  # requests per minute
VAD_SILENCE_THRESHOLD = float(os.getenv("VAD_SILENCE_THRESHOLD", "0.5"))  # seconds
TURN_TIMEOUT = float(os.getenv("TURN_TIMEOUT", "2.0"))  # seconds

# Audio Streaming Configuration
AUDIO_CHUNK_SIZE = int(os.getenv("AUDIO_CHUNK_SIZE", "3200"))  # bytes (200ms at 16kHz)
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
RATE_LIMIT_REQUESTS_PER_HOUR = int(os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR", "1000"))
RATE_LIMIT_REQUESTS_PER_DAY = int(os.getenv("RATE_LIMIT_REQUESTS_PER_DAY", "10000"))
RATE_LIMIT_BURST_LIMIT = int(os.getenv("RATE_LIMIT_BURST_LIMIT", "10"))
RATE_LIMIT_WINDOW_SIZE = int(os.getenv("RATE_LIMIT_WINDOW_SIZE", "60"))
RATE_LIMIT_COST_BASED = os.getenv("RATE_LIMIT_COST_BASED", "false").lower() == "true"
RATE_LIMIT_COST_PER_REQUEST = float(os.getenv("RATE_LIMIT_COST_PER_REQUEST", "1.0"))
RATE_LIMIT_MAX_DAILY_COST = float(os.getenv("RATE_LIMIT_MAX_DAILY_COST", "100.0"))

# LLM Configuration
LLM_OPENAI_MODEL = os.getenv("LLM_OPENAI_MODEL", "claude-3-haiku-20240307")
LLM_ANTHROPIC_MODEL = os.getenv("LLM_ANTHROPIC_MODEL", "claude-3-haiku-20240307")
LLM_AZURE_MODEL = os.getenv("LLM_AZURE_MODEL", "gpt-35-turbo")
LLM_AWS_MODEL = os.getenv("LLM_AWS_MODEL", "anthropic.claude-v2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0"))
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.0"))

# Intent Detection Configuration
INTENT_DETECTION_MODEL = os.getenv("INTENT_DETECTION_MODEL", "claude-3-haiku-20240307")
INTENT_DETECTION_TEMPERATURE = float(os.getenv("INTENT_DETECTION_TEMPERATURE", "0.1"))
INTENT_DETECTION_MAX_TOKENS = int(os.getenv("INTENT_DETECTION_MAX_TOKENS", "150"))
INTENT_DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_DETECTION_CONFIDENCE_THRESHOLD", "0.7"))
INTENT_DETECTION_LANGUAGE_DETECTION = os.getenv("INTENT_DETECTION_LANGUAGE_DETECTION", "true").lower() == "true"
INTENT_DETECTION_ENTITY_EXTRACTION = os.getenv("INTENT_DETECTION_ENTITY_EXTRACTION", "true").lower() == "true"
INTENT_DETECTION_SLOT_FILLING = os.getenv("INTENT_DETECTION_SLOT_FILLING", "true").lower() == "true"

# Guard Rails Configuration
GUARD_RAILS_SAFETY_THRESHOLD = float(os.getenv("GUARD_RAILS_SAFETY_THRESHOLD", "0.7"))
GUARD_RAILS_ENABLE_CONTENT_FILTERING = os.getenv("GUARD_RAILS_ENABLE_CONTENT_FILTERING", "true").lower() == "true"
GUARD_RAILS_ENABLE_TOXICITY_DETECTION = os.getenv("GUARD_RAILS_ENABLE_TOXICITY_DETECTION", "true").lower() == "true"
GUARD_RAILS_ENABLE_BIAS_DETECTION = os.getenv("GUARD_RAILS_ENABLE_BIAS_DETECTION", "true").lower() == "true"
GUARD_RAILS_ENABLE_PII_DETECTION = os.getenv("GUARD_RAILS_ENABLE_PII_DETECTION", "true").lower() == "true"
GUARD_RAILS_ENABLE_HARMFUL_CONTENT_DETECTION = os.getenv("GUARD_RAILS_ENABLE_HARMFUL_CONTENT_DETECTION", "true").lower() == "true"
GUARD_RAILS_MAX_CONTENT_LENGTH = int(os.getenv("GUARD_RAILS_MAX_CONTENT_LENGTH", "10000"))
GUARD_RAILS_ENABLE_LLM_MODERATION = os.getenv("GUARD_RAILS_ENABLE_LLM_MODERATION", "true").lower() == "true"
GUARD_RAILS_FALLBACK_TO_RULES = os.getenv("GUARD_RAILS_FALLBACK_TO_RULES", "true").lower() == "true"

# Blocked keywords for content filtering
BLOCKED_KEYWORDS = os.getenv("BLOCKED_KEYWORDS", "").split(",") if os.getenv("BLOCKED_KEYWORDS") else []

# RAG Configuration (placeholder for future implementation)
RAG_ENABLED = os.getenv("RAG_ENABLED", "false").lower() == "true"
RAG_VECTOR_DB_URL = os.getenv("RAG_VECTOR_DB_URL", "")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-ada-002")
RAG_MAX_RESULTS = int(os.getenv("RAG_MAX_RESULTS", "5"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))

# Create cache directories
for cache_dir in TTS_MODEL_CACHE_DIRS:
    if cache_dir:
        if not cache_dir.startswith(BASE_DIR) or cache_dir == os.path.join(BASE_DIR, "parler_models"):
             try:
                os.makedirs(cache_dir, exist_ok=True)
             except PermissionError:
                print(f"Warning: Could not create cache directory {cache_dir} due to PermissionError. This might be expected if it's a root-owned build-time cache.")
             except Exception as e:
                print(f"Warning: Could not create cache directory {cache_dir}. Error: {e}") 