from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
import os
import aiofiles
import tempfile

from app.db.database import get_db
from app.api.deps.auth import get_current_active_user, get_device_auth
from app.services.multi_language_processor import process_audio, text_to_speech
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("voice_routes")

router = APIRouter(tags=["voice"])

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Transcribe audio file to text.
    
    Args:
        file: Audio file to transcribe
        language: Optional language code (auto-detected if not provided)
        
    Returns:
        Transcription result
    """
    # Create temporary file to store uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_path = temp_file.name
        
        # Write uploaded file to temporary file
        content = await file.read()
        temp_file.write(content)
    
    try:
        # Process audio file
        result = await process_audio(temp_path, language)
        
        # Return transcription result
        return {
            "text": result["text"],
            "detected_language": result["detected_language"],
            "confidence": result["confidence"]
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/tts")
async def text_to_speech_endpoint(
    text: str = Form(...),
    language: str = Form("en"),
    voice_id: Optional[str] = Form(None),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Convert text to speech.
    
    Args:
        text: Text to convert to speech
        language: Language code
        voice_id: Optional voice ID
        
    Returns:
        Path to generated audio file
    """
    # Generate audio file
    audio_path = await text_to_speech(text, language, voice_id)
    
    # Return path to audio file
    return {
        "audio_path": audio_path,
        "language": language
    }

@router.post("/chat-voice")
async def voice_chat(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    child_id: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Process voice chat query.
    
    Args:
        file: Audio file with query
        session_id: Optional chat session ID
        child_id: Optional child ID
        group_id: Optional group ID
        language: Optional language code
        
    Returns:
        Chat response with text and audio
    """
    from app.llm.chat_service import process_chat_query
    
    # Create temporary file to store uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_path = temp_file.name
        
        # Write uploaded file to temporary file
        content = await file.read()
        temp_file.write(content)
    
    try:
        # Process audio file to get text
        audio_result = await process_audio(temp_path, language)
        query_text = audio_result["text"]
        detected_language = audio_result["detected_language"]
        
        # Use detected language if none provided
        if not language:
            language = detected_language
        
        # Create or get session
        if not session_id:
            from app.db.crud.chat import create_chat_session
            session = await create_chat_session(
                db, 
                child_id=child_id, 
                group_id=group_id
            )
            session_id = session.id
        
        # Process query
        response = await process_chat_query(
            db,
            query=query_text,
            session_id=session_id,
            child_id=child_id,
            group_id=group_id,
            language=language
        )
        
        # Generate audio response
        audio_path = await text_to_speech(response["response"], language)
        
        # Save messages to database
        from app.db.crud.chat import create_chat_message
        
        # Save user message
        await create_chat_message(
            db,
            session_id=session_id,
            is_user=True,
            content=query_text
        )
        
        # Save assistant message
        await create_chat_message(
            db,
            session_id=session_id,
            is_user=False,
            content=response["response"],
            source_documents=str(response["source_documents"]),
            confidence=response["confidence"]
        )
        
        # Return response
        return {
            "session_id": session_id,
            "query_text": query_text,
            "response_text": response["response"],
            "audio_path": audio_path,
            "language": language,
            "source_documents": response["source_documents"],
            "confidence": response["confidence"]
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
