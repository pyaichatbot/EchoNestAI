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
    DESCRIPTION: str = "EchoNest AI Backend Server"
    
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    
    # Database settings
    DATABASE_HOST: str = os.getenv("DATABASE_HOST", "localhost")
    DATABASE_PORT: int = int(os.getenv("DATABASE_PORT", "5432"))
    DATABASE_USER: str = os.getenv("DATABASE_USER", "echonest")
    DATABASE_PASSWORD: str = os.getenv("DATABASE_PASSWORD", "echonest_password")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "echonest_db")
    @property
    def DATABASE_URL(self) -> str:
        """Build Database URL based on host"""        
        # Use localhost for development, postgres for production/docker
        host = "localhost" if self.DATABASE_HOST == "localhost" else "postgres"
        driver = "postgresql+psycopg2" if self.DATABASE_HOST == "localhost" else "postgresql+asyncpg"
        return f"{driver}://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}@{host}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    @property
    def REDIS_URL(self) -> str:
        """Build Redis URL from components"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour in seconds
    
    # Qdrant settings
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "echonest_documents")
    QDRANT_API_KEY: Optional[str] = None
    
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
    
    # Document embedding settings
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    EMBEDDING_DIMENSION: int = 384
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
