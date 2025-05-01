from fastapi import APIRouter
from app.api.routes import auth, content, chat, devices, dashboard, events, language, voice, feedback, children, groups, students
from app.core.config import settings

# Create versioned router
api_router = APIRouter()

# Include all route modules with proper prefixes and tags
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

api_router.include_router(
    content.router,
    prefix="/content",
    tags=["Content Management"]
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"]
)

api_router.include_router(
    devices.router,
    prefix="/devices",
    tags=["Device Management"]
)

api_router.include_router(
    dashboard.router,
    prefix="/dashboard",
    tags=["Dashboard"]
)

api_router.include_router(
    events.router,
    prefix="/events",
    tags=["Events"]
)

api_router.include_router(
    language.router,
    prefix="/languages",
    tags=["Language Support"]
)

api_router.include_router(
    voice.router,
    prefix="/voice",
    tags=["Voice Processing"]
)

api_router.include_router(
    feedback.router,
    prefix="/feedback",
    tags=["Feedback"]
)

api_router.include_router(
    children.router,
    prefix="/children",
    tags=["Children"]
)

api_router.include_router(
    groups.router,
    prefix="/groups",
    tags=["Groups"]
)

api_router.include_router(
    students.router,
    prefix="/students",
    tags=["Students"]
)
