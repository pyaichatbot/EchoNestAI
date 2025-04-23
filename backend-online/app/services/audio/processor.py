from typing import Dict, List, Optional, Any
import json
import os
import datetime
import whisper

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.content.embeddings import generate_embeddings, store_embeddings, split_text_into_chunks

logger = setup_logging("audio_processor")

async def process_audio(
    content_id: str,
    file_path: str,
    language: str,
    event_manager: Any
) -> None:
    """
    Process an audio file:
    1. Transcribe audio
    2. Split into chunks
    3. Generate embeddings
    4. Store in vector database
    """
    # Update: Transcription (30%)
    await event_manager.broadcast(
        {
            "content_id": content_id,
            "status": "processing",
            "progress": 30,
            "message": f"Transcribing audio in {language}"
        },
        client_filter=content_id
    )
    
    try:
        # Transcribe audio
        transcription_result = await transcribe_audio(file_path, language)
        transcribed_text = transcription_result["text"]
        detected_language = transcription_result["language"]
        
        # Update: Chunking (50%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 50,
                "message": "Splitting transcript into chunks"
            },
            client_filter=content_id
        )
        
        # Split text into chunks
        chunks = split_text_into_chunks(transcribed_text)
        
        # Update: Embedding generation (80%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 80,
                "message": "Generating embeddings"
            },
            client_filter=content_id
        )
        
        # Generate embeddings for each chunk
        embeddings = await generate_embeddings(chunks, detected_language)
        
        # Update: Vector storage (95%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 95,
                "message": "Storing in vector database"
            },
            client_filter=content_id
        )
        
        # Store embeddings in database
        await store_embeddings(content_id, chunks, embeddings)
        
        # Store audio metadata
        await store_audio_metadata(content_id, detected_language, transcribed_text)
        
    except Exception as e:
        logger.error(f"Error processing audio {content_id}: {str(e)}")
        raise

async def transcribe_audio(file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper.
    
    Args:
        file_path: Path to audio file
        language: Optional language code (auto-detected if not provided)
        
    Returns:
        Dictionary with transcribed text and detected language
    """
    try:
        import asyncio
        
        # Ensure models directory exists
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
        
        # Set environment variable for model download location
        os.environ["WHISPER_MODELS_DIR"] = settings.MODELS_FOLDER
        
        # Load model (this is CPU-intensive, so run in a thread pool)
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, lambda: whisper.load_model("base"))
        
        # Transcribe audio
        if language:
            # Use specified language
            result = await loop.run_in_executor(
                None,
                lambda: model.transcribe(
                    file_path,
                    language=language,
                    fp16=False
                )
            )
        else:
            # Auto-detect language
            result = await loop.run_in_executor(
                None,
                lambda: model.transcribe(
                    file_path,
                    fp16=False
                )
            )
        
        # Extract text and detected language
        transcribed_text = result["text"]
        detected_language = result.get("language", language or "en")
        
        return {
            "text": transcribed_text,
            "language": detected_language
        }
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise

async def store_audio_metadata(content_id: str, language: str, transcript: str) -> None:
    """Store audio metadata in database."""
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker
        from app.db.models.models import Content
        from sqlalchemy import update
        
        # Create async engine
        engine = create_async_engine(settings.DATABASE_URI)
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with async_session() as session:
            # Update content record with metadata
            stmt = update(Content).where(Content.id == content_id).values(
                metadata=json.dumps({
                    "language": language,
                    "transcript": transcript,
                    "processed_at": datetime.datetime.utcnow().isoformat()
                })
            )
            
            await session.execute(stmt)
            await session.commit()
    except Exception as e:
        logger.error(f"Error storing audio metadata: {str(e)}")
        raise
