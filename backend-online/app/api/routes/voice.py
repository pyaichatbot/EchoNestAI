from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from typing import Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse
import asyncio
import os
import tempfile
import aiofiles

from app.db.database import get_db
from app.db.schemas.chat import (
    VoiceChatRequest, ChatResponse, ChatSession, 
    ChatMessage, ChatFeedback, SupportedLanguage
)
from app.api.deps.auth import get_current_active_user, get_device_auth
from app.db.crud.chat import (
    create_chat_session, get_chat_session,
    create_chat_message, create_chat_feedback
)
from app.llm.chat_service import process_chat_query, stream_chat_response
from app.sse.event_manager import chat_stream_manager
from app.services.language_service import detect_language, get_supported_languages
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("voice_routes")

router = APIRouter(tags=["voice"])

@router.post("/voice/chat", response_model=ChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    child_id: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    document_scope: Optional[List[str]] = Form([]),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Process a voice chat query and return a text response.
    
    Steps:
    1. Transcribe audio to text
    2. Process the text query
    3. Return text response
    
    If language is not provided, it will be auto-detected.
    """
    # Create temporary file to store uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as temp_file:
        temp_path = temp_file.name
        
        # Write uploaded file to temporary file
        content = await audio.read()
        temp_file.write(content)
    
    try:
        # Process audio file
        import whisper
        
        # Ensure models directory exists
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
        
        # Set environment variable for model download location
        os.environ["WHISPER_MODELS_DIR"] = settings.MODELS_FOLDER
        
        # Load model
        model = whisper.load_model("base")
        
        # Transcribe audio
        if language:
            # Use specified language
            result = model.transcribe(
                temp_path,
                language=language,
                fp16=False
            )
        else:
            # Auto-detect language
            result = model.transcribe(
                temp_path,
                fp16=False
            )
        
        # Extract results
        query_text = result["text"]
        detected_language = result.get("language", "en")
        
        # If no text was transcribed, return error
        if not query_text.strip():
            logger.warning(f"No speech detected in audio file")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No speech detected in audio file"
            )
        
        # If language wasn't provided, verify with our language detector
        if not language:
            language_detection = await detect_language(query_text)
            detected_language = language_detection["detected_language"]
        
        logger.info(f"Transcribed audio in {detected_language}: {query_text[:50]}...")
        
        # Create or get session
        if not session_id:
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
            document_scope=document_scope,
            language=detected_language
        )
        
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
            content=response.response,
            source_documents=response.source_documents,
            confidence=response.confidence
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing voice chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing voice chat: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/voice/chat/stream")
async def voice_chat_stream(
    request: Request,
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    child_id: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    document_scope: Optional[List[str]] = Form([]),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Process a voice chat query and stream the text response.
    
    Steps:
    1. Transcribe audio to text
    2. Process the text query
    3. Stream text response tokens
    
    If language is not provided, it will be auto-detected.
    """
    # Create temporary file to store uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as temp_file:
        temp_path = temp_file.name
        
        # Write uploaded file to temporary file
        content = await audio.read()
        temp_file.write(content)
    
    try:
        # Process audio file
        import whisper
        
        # Ensure models directory exists
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
        
        # Set environment variable for model download location
        os.environ["WHISPER_MODELS_DIR"] = settings.MODELS_FOLDER
        
        # Load model
        model = whisper.load_model("base")
        
        # Transcribe audio
        if language:
            # Use specified language
            result = model.transcribe(
                temp_path,
                language=language,
                fp16=False
            )
        else:
            # Auto-detect language
            result = model.transcribe(
                temp_path,
                fp16=False
            )
        
        # Extract results
        query_text = result["text"]
        detected_language = result.get("language", "en")
        
        # If no text was transcribed, return error
        if not query_text.strip():
            logger.warning(f"No speech detected in audio file")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No speech detected in audio file"
            )
        
        # If language wasn't provided, verify with our language detector
        if not language:
            language_detection = await detect_language(query_text)
            detected_language = language_detection["detected_language"]
        
        logger.info(f"Transcribed audio in {detected_language}: {query_text[:50]}...")
        
        # Create or get session
        if not session_id:
            session = await create_chat_session(
                db, 
                child_id=child_id, 
                group_id=group_id
            )
            session_id = session.id
        
        # Save user message
        await create_chat_message(
            db,
            session_id=session_id,
            is_user=True,
            content=query_text
        )
        
        # Set up streaming
        client_id = f"user_{current_user.id}_session_{session_id}"
        
        async def event_generator():
            try:
                # Start streaming response
                full_response = ""
                async for token in stream_chat_response(
                    query=query_text,
                    session_id=session_id,
                    language=detected_language
                ):
                    if await request.is_disconnected():
                        break
                    
                    full_response += token
                    yield {
                        "event": "token",
                        "data": token
                    }
                
                # Save assistant message
                source_documents, confidence = await save_streamed_response(
                    db,
                    session_id=session_id,
                    query=query_text,
                    response=full_response,
                    language=detected_language,
                    document_scope=document_scope
                )
                
                # Send completion event with metadata
                yield {
                    "event": "complete",
                    "data": {
                        "response": full_response,
                        "source_documents": source_documents,
                        "confidence": confidence
                    }
                }
            except Exception as e:
                logger.error(f"Error streaming response: {str(e)}")
                yield {
                    "event": "error",
                    "data": str(e)
                }
        
        return EventSourceResponse(event_generator())
    
    except Exception as e:
        logger.error(f"Error processing voice chat stream: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing voice chat stream: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

async def save_streamed_response(
    db: AsyncSession,
    session_id: str,
    query: str,
    response: str,
    language: str,
    document_scope: Optional[List[str]] = None
) -> tuple:
    """
    Save streamed response and retrieve source documents and confidence.
    
    Args:
        db: Database session
        session_id: Chat session ID
        query: User query
        response: Generated response
        language: Language code
        document_scope: Optional document scope
        
    Returns:
        Tuple of (source_documents, confidence)
    """
    # Retrieve relevant documents for the query
    from app.llm.chat_service import generate_embedding, retrieve_relevant_documents
    
    # Generate embedding for query
    query_embedding = await generate_embedding(query, language)
    
    # Retrieve relevant documents
    context_passages, source_documents = await retrieve_relevant_documents(
        db, 
        query_embedding, 
        document_scope=document_scope,
        top_k=3
    )
    
    # Calculate confidence based on document relevance
    if context_passages:
        confidence = 0.85  # High confidence when documents are found
    else:
        confidence = 0.6   # Lower confidence when no documents are found
    
    # Save assistant message
    await create_chat_message(
        db,
        session_id=session_id,
        is_user=False,
        content=response,
        source_documents=source_documents,
        confidence=confidence
    )
    
    return source_documents, confidence

@router.post("/voice/tts")
async def text_to_speech_endpoint(
    text: str = Form(...),
    language: str = Form("en"),
    voice_id: Optional[str] = Form(None),
    current_user: Any = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Convert text to speech in the specified language.
    
    Args:
        text: Text to convert to speech
        language: Language code
        voice_id: Optional voice ID
        
    Returns:
        Path to generated audio file
    """
    try:
        # Ensure output directory exists
        audio_output_folder = os.path.join(settings.MEDIA_ROOT, "audio")
        os.makedirs(audio_output_folder, exist_ok=True)
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=audio_output_folder) as temp_file:
            output_path = temp_file.name
        
        # Get appropriate TTS model for language
        from app.services.language_service import get_language_model
        language_models = get_language_model(language)
        
        # Use pyttsx3 for basic TTS
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Set voice properties
        voices = engine.getProperty('voices')
        
        # Select voice based on language
        if language == "en":
            engine.setProperty('voice', voices[0].id)  # English voice
        elif language in ["hi", "te", "ta"]:
            # Try to find an Indian voice, fallback to default
            indian_voice = next((v for v in voices if "indian" in v.name.lower()), None)
            if indian_voice:
                engine.setProperty('voice', indian_voice.id)
        elif language == "de":
            # Try to find a German voice, fallback to default
            german_voice = next((v for v in voices if "german" in v.name.lower()), None)
            if german_voice:
                engine.setProperty('voice', german_voice.id)
        
        # Set other properties
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Generate speech
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        logger.info(f"Generated speech for text in {language}: {text[:50]}...")
        
        # Get relative path for response
        relative_path = os.path.relpath(output_path, settings.MEDIA_ROOT)
        media_url = f"/media/{relative_path}"
        
        return {
            "status": "success",
            "audio_url": media_url,
            "language": language,
            "text": text
        }
    
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating speech: {str(e)}"
        )
