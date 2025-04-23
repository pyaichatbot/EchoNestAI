from typing import List, Dict, Any, Optional, Tuple
import json
import os
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import setup_logging
from app.db.models.models import ContentEmbedding, Content, ContentAssignment
from app.services.language_service import get_language_model

logger = setup_logging("chat_retrieval")

async def generate_embedding(text: str, language: str) -> np.ndarray:
    """
    Generate embedding for text.
    
    Args:
        text: Text to embed
        language: Language code
        
    Returns:
        Embedding vector
    """
    from app.core.config import settings
    
    # Get appropriate model for language
    language_models = get_language_model(language)
    model_name = language_models["embedding_model"]
    
    # Import embedding model
    from sentence_transformers import SentenceTransformer
    
    # Load model (with caching)
    model_path = os.path.join(settings.MODELS_FOLDER, model_name)
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
    else:
        model = SentenceTransformer(model_name)
        # Save model for future use
        os.makedirs(settings.MODELS_FOLDER, exist_ok=True)
        model.save(model_path)
    
    # Generate embedding
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

async def retrieve_relevant_documents(
    db: AsyncSession,
    query_embedding: np.ndarray,
    document_scope: Optional[List[str]] = None,
    child_id: Optional[str] = None,
    group_id: Optional[str] = None,
    top_k: int = 3
) -> Tuple[List[str], List[str]]:
    """
    Retrieve relevant documents based on query embedding.
    
    Args:
        db: Database session
        query_embedding: Query embedding vector
        document_scope: Optional list of document IDs to search within
        child_id: Optional child ID for filtering
        group_id: Optional group ID for filtering
        top_k: Number of top results to return
        
    Returns:
        Tuple of (context passages, source document IDs)
    """
    from sqlalchemy import select
    
    # Build query for content embeddings
    query = select(ContentEmbedding)
    
    # Apply document scope filter if provided
    if document_scope:
        query = query.filter(ContentEmbedding.content_id.in_(document_scope))
    
    # Apply child/group filters if provided
    if child_id or group_id:
        query = query.join(Content, Content.id == ContentEmbedding.content_id)
        query = query.join(ContentAssignment, ContentAssignment.content_id == Content.id)
        
        if child_id:
            query = query.filter(ContentAssignment.child_id == child_id)
        if group_id:
            query = query.filter(ContentAssignment.group_id == group_id)
    
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
    
    # Extract context passages and source documents
    context_passages = [result['chunk_text'] for result in top_results]
    source_documents = list(set([result['content_id'] for result in top_results]))
    
    return context_passages, source_documents

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
