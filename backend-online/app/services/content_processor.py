from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
import aiofiles
from fastapi import UploadFile, BackgroundTasks
import asyncio
import uuid
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import setup_logging
from app.sse.event_manager import content_upload_manager

logger = setup_logging("content_processor")

async def process_content_async(
    content_id: str,
    file_path: str,
    content_type: str,
    language: str,
    event_manager: Any
) -> None:
    """
    Process uploaded content asynchronously.
    This includes:
    - For documents: text extraction, chunking, embedding generation
    - For audio: transcription
    - For video: audio extraction and transcription
    
    Updates are sent via SSE to the client.
    """
    try:
        # Send initial status
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 0,
                "message": "Starting content processing"
            },
            client_filter=content_id
        )
        
        # Process based on content type
        if content_type == "document":
            await process_document(content_id, file_path, language, event_manager)
        elif content_type == "audio":
            await process_audio(content_id, file_path, language, event_manager)
        elif content_type == "video":
            await process_video(content_id, file_path, language, event_manager)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Send completion status
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "completed",
                "progress": 100,
                "message": "Content processing completed"
            },
            client_filter=content_id
        )
        
    except Exception as e:
        logger.error(f"Error processing content {content_id}: {str(e)}")
        # Send error status
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "error",
                "progress": 0,
                "message": f"Error processing content: {str(e)}"
            },
            client_filter=content_id
        )

async def process_document(
    content_id: str,
    file_path: str,
    language: str,
    event_manager: Any
) -> None:
    """
    Process a document file:
    1. Extract text
    2. Split into chunks
    3. Generate embeddings
    4. Store in vector database
    """
    # Update: Text extraction (20%)
    await event_manager.broadcast(
        {
            "content_id": content_id,
            "status": "processing",
            "progress": 20,
            "message": "Extracting text from document"
        },
        client_filter=content_id
    )
    
    # Extract text based on file type
    file_extension = Path(file_path).suffix.lower()
    extracted_text = ""
    
    try:
        if file_extension == ".pdf":
            extracted_text = await extract_text_from_pdf(file_path)
        elif file_extension in [".docx", ".doc"]:
            extracted_text = await extract_text_from_docx(file_path)
        elif file_extension in [".txt", ".md"]:
            extracted_text = await extract_text_from_text_file(file_path)
        else:
            # Try generic text extraction
            extracted_text = await extract_text_generic(file_path)
        
        if not extracted_text:
            raise ValueError(f"Could not extract text from {file_path}")
        
        # Update: Chunking (40%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 40,
                "message": "Splitting document into chunks"
            },
            client_filter=content_id
        )
        
        # Split text into chunks
        chunks = split_text_into_chunks(extracted_text)
        
        # Update: Embedding generation (70%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 70,
                "message": "Generating embeddings"
            },
            client_filter=content_id
        )
        
        # Generate embeddings for each chunk
        embeddings = await generate_embeddings(chunks, language)
        
        # Update: Vector storage (90%)
        await event_manager.broadcast(
            {
                "content_id": content_id,
                "status": "processing",
                "progress": 90,
                "message": "Storing in vector database"
            },
            client_filter=content_id
        )
        
        # Store embeddings in database
        await store_embeddings(content_id, chunks, embeddings)
        
    except Exception as e:
        logger.error(f"Error processing document {content_id}: {str(e)}")
        raise

async def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        import PyPDF2
        
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

async def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        import docx
        
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise

async def extract_text_from_text_file(file_path: str) -> str:
    """Extract text from plain text file."""
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
            text = await file.read()
        return text
    except UnicodeDecodeError:
        # Try different encodings
        encodings = ["latin-1", "cp1252", "iso-8859-1"]
        for encoding in encodings:
            try:
                async with aiofiles.open(file_path, "r", encoding=encoding) as file:
                    text = await file.read()
                return text
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode text file with any encoding")

async def extract_text_generic(file_path: str) -> str:
    """Try to extract text using a generic method."""
    try:
        import textract
        
        # textract is not async, so run in a thread pool
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, lambda: textract.process(file_path).decode("utf-8")
        )
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text with textract: {str(e)}")
        raise

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    
    current_chunk = ""
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep some overlap
            current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
        
        current_chunk += paragraph + "\n\n"
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If no chunks were created (e.g., very short text), use the whole text
    if not chunks:
        chunks = [text.strip()]
    
    return chunks

async def generate_embeddings(chunks: List[str], language: str) -> List[np.ndarray]:
    """Generate embeddings for text chunks."""
    try:
        from sentence_transformers import SentenceTransformer
        from app.services.language_service import get_language_model
        
        # Get appropriate model for language
        language_models = get_language_model(language)
        model_name = language_models["embedding_model"]
        
        # Load model (with caching)
        model_path = os.path.join(settings.MODELS_FOLDER, model_name)
        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer(model_name)
            # Save model for future use
            os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
            model.save(model_path)
        
        # Generate embeddings
        embeddings = model.encode(chunks, convert_to_numpy=True)
        
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

async def store_embeddings(content_id: str, chunks: List[str], embeddings: List[np.ndarray]) -> None:
    """Store embeddings in database."""
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker
        from app.db.models.models import ContentEmbedding
        from sqlalchemy import insert
        
        # Create async engine
        engine = create_async_engine(settings.DATABASE_URI)
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with async_session() as session:
            # Store each chunk and its embedding
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Convert embedding to JSON-serializable format
                embedding_json = json.dumps(embedding.tolist())
                
                # Create embedding record
                stmt = insert(ContentEmbedding).values(
                    content_id=content_id,
                    chunk_index=i,
                    chunk_text=chunk,
                    embedding_vector=embedding_json
                )
                
                await session.execute(stmt)
            
            # Commit the transaction
            await session.commit()
    except Exception as e:
        logger.error(f"Error storing embeddings: {str(e)}")
        raise

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
                file_path,
                language=language,
                fp16=False
            )
        else:
            # Auto-detect language
            result = model.transcribe(
                file_path,
                fp16=False
            )
        
        # Extract text
        transcribed_text = result["text"]
        detected_language = result.get("language", language or "en")
        
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
                audio_path,
                language=language,
                fp16=False
            )
        else:
            # Auto-detect language
            result = model.transcribe(
                audio_path,
                fp16=False
            )
        
        # Extract text
        transcribed_text = result["text"]
        detected_language = result.get("language", language or "en")
        
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
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        
    except Exception as e:
        logger.error(f"Error processing video {content_id}: {str(e)}")
        raise

async def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video file."""
    try:
        import subprocess
        
        # Create output directory if it doesn't exist
        os.makedirs(settings.TEMP_FOLDER, exist_ok=True)
        
        # Generate output path
        audio_path = os.path.join(settings.TEMP_FOLDER, f"{uuid.uuid4()}.wav")
        
        # Extract audio using ffmpeg
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
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
            logger.error(f"Error extracting audio: {stderr.decode()}")
            raise ValueError(f"Failed to extract audio from video: {stderr.decode()}")
        
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

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to the upload folder.
    Returns the file path.
    """
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    
    # Generate unique filename
    file_extension = Path(upload_file.filename).suffix
    content_id = str(uuid.uuid4())
    unique_filename = f"{content_id}_{upload_file.filename}"
    file_path = os.path.join(settings.UPLOAD_FOLDER, unique_filename)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return file_path

async def search_content(
    db: AsyncSession, 
    query: str, 
    language: str = "en",
    document_scope: Optional[List[str]] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search for relevant content based on query.
    
    Args:
        db: Database session
        query: Search query
        language: Language code
        document_scope: Optional list of document IDs to search within
        top_k: Number of top results to return
        
    Returns:
        List of search results with content ID, chunk text, and similarity score
    """
    try:
        # Generate embedding for query
        query_embedding = (await generate_embeddings([query], language))[0]
        
        # Get content embeddings from database
        from sqlalchemy import select
        from app.db.models.models import ContentEmbedding, Content, ContentAssignment
        
        # Build query for content embeddings
        query = select(ContentEmbedding)
        
        # Apply document scope filter if provided
        if document_scope:
            query = query.filter(ContentEmbedding.content_id.in_(document_scope))
        
        # Execute query
        result = await db.execute(query)
        embeddings = result.scalars().all()
        
        # Calculate similarity scores
        search_results = []
        
        for emb in embeddings:
            # Parse embedding vector from JSON
            vector = np.array(json.loads(emb.embedding_vector))
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, vector)
            
            search_results.append({
                'content_id': emb.content_id,
                'chunk_index': emb.chunk_index,
                'chunk_text': emb.chunk_text,
                'similarity': float(similarity)
            })
        
        # Sort by similarity (descending) and take top_k
        search_results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = search_results[:top_k]
        
        return top_results
    except Exception as e:
        logger.error(f"Error searching content: {str(e)}")
        raise

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
