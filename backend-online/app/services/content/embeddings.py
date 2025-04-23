from typing import Dict, List, Optional, Any
import json
import os
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.language_service import get_language_model

logger = setup_logging("content_embeddings")

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

async def search_embeddings(
    db: AsyncSession, 
    query_embedding: np.ndarray, 
    document_scope: Optional[List[str]] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search for similar content using vector similarity.
    
    Args:
        db: Database session
        query_embedding: Query embedding vector
        document_scope: Optional list of document IDs to search within
        top_k: Number of results to return
        
    Returns:
        List of search results with content chunks and metadata
    """
    try:
        from sqlalchemy import text
        import json
        
        # Convert query embedding to JSON string
        query_embedding_json = json.dumps(query_embedding.tolist())
        
        # Build SQL query with vector similarity search
        # This uses PostgreSQL with pgvector extension
        sql = """
        WITH vector_matches AS (
            SELECT 
                ce.content_id,
                ce.chunk_index,
                ce.chunk_text,
                c.title,
                c.content_type,
                c.metadata,
                1 - (ce.embedding_vector <=> :query_vector) as similarity
            FROM 
                content_embeddings ce
            JOIN
                contents c ON ce.content_id = c.id
            WHERE 
                1=1
        """
        
        # Add document scope filter if provided
        if document_scope:
            placeholders = ", ".join([f":doc_id_{i}" for i in range(len(document_scope))])
            sql += f" AND ce.content_id IN ({placeholders})"
        
        # Complete the query
        sql += """
            ORDER BY 
                similarity DESC
            LIMIT :top_k
        )
        SELECT * FROM vector_matches
        """
        
        # Prepare parameters
        params = {"query_vector": query_embedding_json, "top_k": top_k}
        
        # Add document scope parameters if provided
        if document_scope:
            for i, doc_id in enumerate(document_scope):
                params[f"doc_id_{i}"] = doc_id
        
        # Execute query
        result = await db.execute(text(sql), params)
        rows = result.fetchall()
        
        # Format results
        search_results = []
        for row in rows:
            search_results.append({
                "content_id": row.content_id,
                "chunk_index": row.chunk_index,
                "chunk_text": row.chunk_text,
                "title": row.title,
                "content_type": row.content_type,
                "metadata": json.loads(row.metadata) if row.metadata else {},
                "similarity": float(row.similarity)
            })
        
        return search_results
    except Exception as e:
        logger.error(f"Error searching embeddings: {str(e)}")
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
