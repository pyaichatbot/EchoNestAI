import os
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseSettings, validator, PostgresDsn, AnyHttpUrl, EmailStr

class Settings(BaseSettings):
    """
    Application settings for the EchoNest AI Offline Backend Server.
    
    These settings can be configured via environment variables.
    """
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "EchoNest AI Offline Backend"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development_secret_key_change_in_production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    DEVICE_TOKEN_EXPIRE_DAYS: int = 365  # 1 year
    
    # Database settings
    SQLITE_DATABASE_URL: str = "sqlite:///./echonest_offline.db"
    
    # File storage settings
    UPLOAD_FOLDER: str = "./uploads"
    CONTENT_FOLDER: str = "./content"
    MODELS_FOLDER: str = "./models"
    
    # Sync settings
    SYNC_INTERVAL_MINUTES: int = 60  # 1 hour
    MAX_CONTENT_SIZE_MB: float = 1000.0  # 1 GB
    MAX_DOCUMENTS_PER_DEVICE: int = 100
    
    # Language settings
    DEFAULT_LANGUAGE: str = "en"
    SUPPORTED_LANGUAGES: List[str] = ["en", "te", "ta", "de", "hi"]
    
    # LLM settings
    DEFAULT_LLM_MODEL: str = "llama-2-7b-chat-q4_0"
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Device settings
    DEVICE_ID: Optional[str] = None
    DEVICE_MODE: str = "offline"
    
    # Online server settings
    ONLINE_SERVER_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings object
settings = Settings()
