import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
import aiofiles
from datetime import datetime

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.models.models import Content, ContentEmbedding

logger = setup_logging("content_processor")

class ContentProcessor:
    """
    Content processor for handling document processing, embedding generation,
    and content indexing for search.
    """
    
    def __init__(self):
        """Initialize the content processor."""
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.chunk_size = 512     # Default chunk size for text splitting
        self.chunk_overlap = 50   # Default chunk overlap
        
        # Ensure content directories exist
        os.makedirs(settings.CONTENT_FOLDER, exist_ok=True)
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
    
    async def process_document(self, db: AsyncSession, content_id: str) -> bool:
        """
        Process a document for search indexing.
        
        Args:
            db: Database session
            content_id: Content ID to process
            
        Returns:
            True if processing was successful, False otherwise
        """
        from sqlalchemy import select
        
        # Get content from database
        query = select(Content).filter(Content.id == content_id)
        result = await db.execute(query)
        content = result.scalars().first()
        
        if not content:
            logger.error(f"Content not found: {content_id}")
            return False
        
        # Check if file exists
        if not os.path.exists(content.file_path):
            logger.error(f"Content file not found: {content.file_path}")
            return False
        
        try:
            # Extract text based on content type
            if content.type == "document":
                text = await self._extract_text_from_document(content.file_path)
            elif content.type == "audio":
                text = await self._extract_text_from_audio(content.file_path, content.language)
            elif content.type == "video":
                text = await self._extract_text_from_video(content.file_path, content.language)
            else:
                logger.error(f"Unsupported content type: {content.type}")
                return False
            
            # Split text into chunks
            chunks = self._split_text(text)
            
            # Generate embeddings for each chunk
            embeddings = await self._generate_embeddings(chunks, content.language)
            
            # Store embeddings in database
            await self._store_embeddings(db, content_id, chunks, embeddings)
            
            logger.info(f"Successfully processed content: {content_id}, generated {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing content {content_id}: {str(e)}")
            return False
    
    async def _extract_text_from_document(self, file_path: str) -> str:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text
        """
        # Determine file type from extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.txt':
            # Simple text file
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        
        elif ext in ['.pdf', '.docx', '.doc', '.rtf']:
            # For PDF and Office documents, use appropriate libraries
            # This is a simplified implementation
            if ext == '.pdf':
                import PyPDF2
                
                text = ""
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text() + "\n"
                return text
            
            elif ext in ['.docx', '.doc']:
                # For a production implementation, use python-docx or similar
                # This is a simplified placeholder
                import subprocess
                
                result = subprocess.run(['catdoc', file_path], capture_output=True, text=True)
                return result.stdout
        
        else:
            logger.warning(f"Unsupported document format: {ext}")
            return ""
    
    async def _extract_text_from_audio(self, file_path: str, language: str) -> str:
        """
        Extract text from an audio file using speech recognition.
        
        Args:
            file_path: Path to the audio file
            language: Language code
            
        Returns:
            Transcribed text
        """
        # In a production environment, this would use a proper speech recognition model
        # For this implementation, we'll use a basic approach with SpeechRecognition
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            
            try:
                # Convert language code to SpeechRecognition format
                lang_code = self._map_language_to_speech_recognition(language)
                text = recognizer.recognize_google(audio_data, language=lang_code)
                return text
            except sr.UnknownValueError:
                logger.warning(f"Speech recognition could not understand audio: {file_path}")
                return ""
            except sr.RequestError as e:
                logger.error(f"Could not request results from speech recognition service: {str(e)}")
                return ""
    
    async def _extract_text_from_video(self, file_path: str, language: str) -> str:
        """
        Extract text from a video file.
        
        Args:
            file_path: Path to the video file
            language: Language code
            
        Returns:
            Extracted text
        """
        # Extract audio from video
        import tempfile
        import subprocess
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Extract audio using ffmpeg
            subprocess.run([
                'ffmpeg', '-i', file_path, '-q:a', '0', '-map', 'a', temp_audio_path
            ], check=True, capture_output=True)
            
            # Process the audio file
            text = await self._extract_text_from_audio(temp_audio_path, language)
            return text
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting audio from video: {str(e)}")
            return ""
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Simple splitting by character count
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > self.chunk_size:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk, with overlap from previous chunk
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_size = 0
                    overlap_words = []
                    
                    # Add words from the end of the previous chunk for overlap
                    for w in reversed(current_chunk):
                        if overlap_size + len(w) + 1 > self.chunk_overlap:
                            break
                        
                        overlap_words.insert(0, w)
                        overlap_size += len(w) + 1
                    
                    current_chunk = overlap_words
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(word)
            current_size += word_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[str], language: str) -> List[np.ndarray]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            language: Language code
            
        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []
        
        # Load the embedding model
        from sentence_transformers import SentenceTransformer
        
        # Select appropriate model based on language
        model_name = self._get_embedding_model_for_language(language)
        
        # Load model (with caching)
        model_path = os.path.join(settings.MODELS_FOLDER, model_name)
        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer(model_name)
            # Save model for future use
            model.save(model_path)
        
        # Generate embeddings
        embeddings = model.encode(chunks, convert_to_numpy=True)
        return embeddings
    
    async def _store_embeddings(
        self, 
        db: AsyncSession, 
        content_id: str, 
        chunks: List[str], 
        embeddings: List[np.ndarray]
    ) -> None:
        """
        Store embeddings in the database.
        
        Args:
            db: Database session
            content_id: Content ID
            chunks: List of text chunks
            embeddings: List of embedding vectors
        """
        # Delete existing embeddings for this content
        from sqlalchemy import delete
        
        stmt = delete(ContentEmbedding).where(ContentEmbedding.content_id == content_id)
        await db.execute(stmt)
        
        # Store new embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Convert numpy array to list and then to JSON string
            embedding_json = json.dumps(embedding.tolist())
            
            # Create embedding record
            embedding_record = ContentEmbedding(
                content_id=content_id,
                chunk_index=i,
                chunk_text=chunk,
                embedding_vector=embedding_json
            )
            
            db.add(embedding_record)
        
        # Commit changes
        await db.commit()
    
    def _map_language_to_speech_recognition(self, language: str) -> str:
        """
        Map language code to speech recognition format.
        
        Args:
            language: Language code (e.g., 'en', 'hi')
            
        Returns:
            Language code in speech recognition format (e.g., 'en-US', 'hi-IN')
        """
        language_map = {
            'en': 'en-US',
            'te': 'te-IN',
            'ta': 'ta-IN',
            'de': 'de-DE',
            'hi': 'hi-IN'
        }
        
        return language_map.get(language, 'en-US')
    
    def _get_embedding_model_for_language(self, language: str) -> str:
        """
        Get appropriate embedding model for language.
        
        Args:
            language: Language code
            
        Returns:
            Model name
        """
        # Map language to appropriate model
        language_model_map = {
            'en': 'all-MiniLM-L6-v2',
            'de': 'paraphrase-multilingual-MiniLM-L12-v2',
            'hi': 'paraphrase-multilingual-MiniLM-L12-v2',
            'ta': 'paraphrase-multilingual-MiniLM-L12-v2',
            'te': 'paraphrase-multilingual-MiniLM-L12-v2'
        }
        
        return language_model_map.get(language, 'paraphrase-multilingual-MiniLM-L12-v2')
    
    async def search_content(
        self, 
        db: AsyncSession, 
        query: str, 
        language: str,
        document_scope: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for content using vector similarity.
        
        Args:
            db: Database session
            query: Search query
            language: Language code
            document_scope: Optional list of document IDs to search within
            top_k: Number of results to return
            
        Returns:
            List of search results with content ID, chunk text, and similarity score
        """
        # Generate embedding for query
        query_embedding = (await self._generate_embeddings([query], language))[0]
        
        # Get all embeddings from database
        from sqlalchemy import select
        from sqlalchemy.sql import func
        
        query = select(ContentEmbedding)
        
        if document_scope:
            query = query.filter(ContentEmbedding.content_id.in_(document_scope))
        
        result = await db.execute(query)
        embeddings = result.scalars().all()
        
        # Calculate similarity scores
        search_results = []
        
        for emb in embeddings:
            # Parse embedding vector from JSON
            vector = np.array(json.loads(emb.embedding_vector))
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, vector)
            
            search_results.append({
                'content_id': emb.content_id,
                'chunk_index': emb.chunk_index,
                'chunk_text': emb.chunk_text,
                'similarity': float(similarity)
            })
        
        # Sort by similarity (descending) and take top_k
        search_results.sort(key=lambda x: x['similarity'], reverse=True)
        return search_results[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Create global instance
content_processor = ContentProcessor()
