from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.database import init_db, close_db
from app.api.api import api_router
from app.middleware.rate_limit import auth_limiter, chat_limiter, default_limiter
from app.middleware.error_handler import ErrorHandler, RequestValidator, RequestLogger

logger = setup_logging("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create necessary directories
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(settings.CONTENT_FOLDER, exist_ok=True)
    os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
    
    # Startup: Initialize connections and resources
    logger.info("Starting up EchoNest AI Backend Server")
    await init_db()
    yield
    # Shutdown: Close connections and free resources
    logger.info("Shutting down EchoNest AI Backend Server")
    await close_db()

def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Set up CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middlewares
    application.middleware("http")(RequestLogger())
    application.middleware("http")(RequestValidator())
    application.middleware("http")(ErrorHandler())

    # Add rate limiting for specific endpoints
    application.middleware("http")(auth_limiter)
    application.middleware("http")(chat_limiter)
    application.middleware("http")(default_limiter)

    # Include API router with versioning
    application.include_router(api_router, prefix=settings.API_V1_STR)

    # Add health check endpoint
    @application.get("/health")
    async def health_check():
        """
        Health check endpoint.
        Returns:
            dict: Status information including version
        """
        return {
            "status": "healthy",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT
        }

    return application

app = create_application()