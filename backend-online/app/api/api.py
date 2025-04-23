from fastapi import APIRouter

from app.api.routes import auth, content, chat, devices, dashboard, events, language, voice

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth")
api_router.include_router(content.router, prefix="/content")
api_router.include_router(chat.router, prefix="/chat")
api_router.include_router(devices.router, prefix="/devices")
api_router.include_router(dashboard.router, prefix="/dashboard")
api_router.include_router(events.router, prefix="/events")
api_router.include_router(language.router, prefix="/languages")
api_router.include_router(voice.router, prefix="/voice")
