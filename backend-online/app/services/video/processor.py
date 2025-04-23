from typing import Dict, List, Optional, Any
import json
import os
import asyncio
import subprocess
from pathlib import Path

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.content.embeddings import generate_embeddings, store_embeddings, split_text_into_chunks
from app.services.audio.processor import transcribe_audio

logger = setup_logging("video_processor")

async def process_video(
    content_id: str,
    file_path: str,
    language: str,
    event_manager: Any
) -> None:
    """
    Process a video file:
    1. Extract audio
    2. Transcribe audio
    3. Split into chunks
    4. Generate embeddings
    5. Store in vector database
    """
    # Update: Audio extraction (20%)
    await event_manager.broadcast(
        {
            "content_id": content_id,
            "status": "processing",
            "progress": 20,
            "message": "Extracting audio from video"
        },
        client_filter=content_id
    )
    
    try:
        # Extract audio from video
        audio_path = await extract_audio_from_video(file_path)
        
        # Update: Transcription (40%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 40,
                "message": f"Transcribing audio in {language}"
            },
            client_filter=content_id
        )
        
        # Transcribe audio
        transcription_result = await transcribe_audio(audio_path, language)
        transcribed_text = transcription_result["text"]
        detected_language = transcription_result["language"]
        
        # Update: Chunking (60%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 60,
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
        
        # Store video metadata
        await store_video_metadata(content_id, detected_language, transcribed_text)
        
        # Clean up temporary audio file
        try:
            os.remove(audio_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary audio file {audio_path}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing video {content_id}: {str(e)}")
        raise

async def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to extracted audio file
    """
    try:
        # Create output path
        video_filename = Path(video_path).stem
        audio_path = os.path.join(
            settings.TEMP_FOLDER, 
            f"{video_filename}_{os.urandom(4).hex()}.mp3"
        )
        
        # Ensure temp directory exists
        os.makedirs(settings.TEMP_FOLDER, exist_ok=True)
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            "-f", "mp3",
            audio_path
        ]
        
        # Run ffmpeg command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_message = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Failed to extract audio: {error_message}")
        
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio from video: {str(e)}")
        raise

async def store_video_metadata(content_id: str, language: str, transcript: str) -> None:
    """Store video metadata in database."""
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker
        from app.db.models.models import Content
        from sqlalchemy import update
        import datetime
        
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
        logger.error(f"Error storing video metadata: {str(e)}")
        raise
