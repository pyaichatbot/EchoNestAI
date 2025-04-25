from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from dotenv import load_dotenv
import json
from pydantic import field_validator, model_validator
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Base directory
    BASE_DIR: Path = Path(__file__).parent.parent.parent.resolve()
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "EchoNest AI Backend Server"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://echonest:echonest_password@postgres:5432/echonest")
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    REDIS_URL: str = "redis://redis:6379/0"
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour in seconds
    
    # Qdrant settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "echonest_documents")
    
    # CORS settings
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    
    # Content settings
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "./echonest_uploads")
    MAX_CONTENT_SIZE_MB: int = int(os.getenv("MAX_CONTENT_SIZE_MB", "100"))
    MAX_DOCUMENTS_PER_DEVICE: int = int(os.getenv("MAX_DOCUMENTS_PER_DEVICE", "10"))
    MODELS_FOLDER: str = os.getenv("MODELS_FOLDER", "./models")
    CONTENT_FOLDER: str = os.getenv("CONTENT_FOLDER", "./content")
    
    # Device settings
    DEVICE_TOKEN_EXPIRE_DAYS: int = int(os.getenv("DEVICE_TOKEN_EXPIRE_DAYS", "365"))
    
    # OTA settings
    OTA_FOLDER: str = os.getenv("OTA_FOLDER", "/tmp/echonest_ota")
    
    # Reflection settings
    ENABLE_REFLECTION: bool = True
    REFLECTION_THRESHOLD: float = 0.7
    
    # Prometheus settings
    PROMETHEUS_GATEWAY_URL: Optional[str] = None  # Set in production
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
