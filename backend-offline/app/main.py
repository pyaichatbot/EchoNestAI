from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from contextlib import asynccontextmanager

from app.api.api import api_router
from app.core.config import settings
from app.db.database import init_db
from app.core.logging import setup_logging

logger = setup_logging("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Initialize database
    logger.info("Initializing database...")
    await init_db()
    
    # Create necessary directories
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(settings.CONTENT_FOLDER, exist_ok=True)
    os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
    
    logger.info("Application startup complete")
    yield
    # Shutdown: Clean up resources
    logger.info("Application shutdown")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """
    Root endpoint for health check.
    """
    return {"status": "online", "message": "EchoNest AI Offline Backend is running"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.DEVICE_MODE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
