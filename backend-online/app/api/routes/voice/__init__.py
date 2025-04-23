from fastapi import APIRouter
from app.api.routes.voice.transcription import router as transcription_router
from app.api.routes.voice.tts import router as tts_router

router = APIRouter(tags=["voice"])

# Include all voice-related routers
router.include_router(transcription_router)
router.include_router(tts_router)
