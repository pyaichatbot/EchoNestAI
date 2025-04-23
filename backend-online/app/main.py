from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import auth, content, chat, feedback, devices, dashboard, events
from app.db.database import init_db, close_db

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
    )

    # Set up CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    application.include_router(auth.router, prefix=settings.API_V1_STR)
    application.include_router(content.router, prefix=settings.API_V1_STR)
    application.include_router(chat.router, prefix=settings.API_V1_STR)
    application.include_router(feedback.router, prefix=settings.API_V1_STR)
    application.include_router(devices.router, prefix=settings.API_V1_STR)
    application.include_router(dashboard.router, prefix=settings.API_V1_STR)
    application.include_router(events.router, prefix=settings.API_V1_STR)

    return application

app = create_application()
