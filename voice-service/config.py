"""
Configuration module for the Voice Service API.
Centralizes all environment variables and settings.
"""

import os
from typing import Optional

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

# Cost configuration
MAX_DAILY_COST = float(os.getenv("MAX_DAILY_COST", "10.0"))  # Maximum daily cost in USD
COST_TIER_RATIO = float(os.getenv("COST_TIER_RATIO", "0.2"))  # Percent of paid API usage (0.0-1.0)

# Application directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model cache directories
WHISPER_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
TTS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub") 