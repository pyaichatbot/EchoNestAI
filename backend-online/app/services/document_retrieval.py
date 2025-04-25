from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import numpy as np
from datetime import datetime, timedelta

from app.db.models.models import Content, ContentEmbedding, ContentAssignment
from app.services.vector_store import get_vector_store
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.cache import cache

logger = setup_logging("document_retrieval")

class DocumentRetriever:
    def __init__(self):
        self._vector_store = None
        self._cache_ttl = 300  # 5 minutes
    
    async def get_vector_store(self):
        """Get vector store instance"""
        if self._vector_store is None:
            self._vector_store = await get_vector_store()
        return self._vector_store
    
    async def retrieve_documents(
        self,
        db: AsyncSession,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        language: str = "en",
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        limit: int = 5,
        min_score: float = 0.6,
        use_cache: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Retrieve relevant documents based on query
        
        Args:
            db: Database session
            query: Search query text
            query_embedding: Optional pre-computed query embedding
            language: Language code
            document_ids: Optional list of document IDs to filter by
            user_id: Optional user ID for access control
            group_ids: Optional group IDs for access control
            limit: Maximum number of results
            min_score: Minimum similarity score
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (list of documents, total count)
        """
        try:
            # Check cache first
            if use_cache:
                cache_key = f"doc_retrieval:{query}:{language}:{document_ids}:{user_id}:{group_ids}:{limit}"
                cached = await cache.get_cache(cache_key)
                if cached:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached["documents"], cached["total"]
            
            # Get accessible document IDs
            accessible_ids = await self._get_accessible_documents(
                db, user_id, group_ids, document_ids
            )
            
            if not accessible_ids:
                logger.debug("No accessible documents found")
                return [], 0
            
            # Get vector store
            vector_store = await self.get_vector_store()
            
            # Generate embedding if not provided
            if query_embedding is None:
                from app.llm.chat_service import generate_embedding
                query_embedding = await generate_embedding(query, language)
            
            # Prepare filter conditions for Qdrant
            filter_conditions = {
                "must": [
                    {
                        "key": "document_id",
                        "match": {"any": accessible_ids}
                    }
                ]
            }
            
            # Add language filter if specified
            if language:
                filter_conditions["must"].append({
                    "key": "language",
                    "match": {"value": language}
                })
            
            # Perform similarity search
            search_results = await vector_store.search_vectors(
                query_vector=query_embedding,
                filter_conditions=filter_conditions,
                limit=limit,
                score_threshold=min_score
            )
            
            # Format results and fetch additional metadata
            documents = []
            for result in search_results:
                # Get document metadata
                metadata = await self.get_document_metadata(
                    db,
                    result["document_id"]
                )
                
                if metadata:
                    formatted_doc = {
                        "id": result["document_id"],
                        "title": metadata["title"],
                        "content": result.get("text", ""),  # Handle missing text field
                        "relevance_score": float(result["score"]),
                        "language": metadata["language"],
                        "type": metadata["type"],
                        "metadata": {
                            **metadata,
                            "chunk_index": result.get("chunk_index"),
                            "page_number": result.get("page_number")
                        }
                    }
                    documents.append(formatted_doc)
            
            total = len(documents)
            
            # Cache results
            if use_cache and documents:  # Only cache if we have results
                await cache.set_cache(
                    cache_key,
                    {"documents": documents, "total": total},
                    expire=self._cache_ttl
                )
            
            return documents, total
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}", exc_info=True)
            return [], 0
    
    async def _get_accessible_documents(
        self,
        db: AsyncSession,
        user_id: Optional[str],
        group_ids: Optional[List[str]],
        document_ids: Optional[List[str]]
    ) -> List[str]:
        """Get list of document IDs accessible to the user"""
        try:
            # Base query for non-archived content
            query = select(Content.id).filter(Content.archived == False)
            
            # Add document ID filter if provided
            if document_ids:
                query = query.filter(Content.id.in_(document_ids))
            
            # Add access control filters
            if user_id:
                # Documents created by user
                created_filter = (Content.created_by == user_id)
                
                # Documents assigned to user's groups
                group_filter = None
                if group_ids:
                    group_filter = ContentAssignment.group_id.in_(group_ids)
                
                # Combine filters
                if group_filter:
                    query = query.join(ContentAssignment).filter(
                        or_(created_filter, group_filter)
                    )
                else:
                    query = query.filter(created_filter)
            
            result = await db.execute(query)
            document_ids = result.scalars().all()
            
            return list(document_ids)
            
        except Exception as e:
            logger.error(f"Error getting accessible documents: {str(e)}", exc_info=True)
            return []
    
    async def get_document_metadata(
        self,
        db: AsyncSession,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document"""
        try:
            # Try cache first
            cache_key = f"doc_metadata:{document_id}"
            cached = await cache.get_cache(cache_key)
            if cached:
                return cached
            
            # Query database
            query = select(Content).filter(Content.id == document_id)
            result = await db.execute(query)
            document = result.scalars().first()
            
            if not document:
                return None
            
            metadata = {
                "id": document.id,
                "title": document.title,
                "type": document.type,
                "language": document.language,
                "created_by": document.created_by,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat() if document.updated_at else None,
                "size_mb": document.size_mb,
                "version": document.version,
                "status": document.status,
                "source": document.source
            }
            
            # Cache metadata
            await cache.set_cache(
                cache_key,
                metadata,
                expire=self._cache_ttl
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting document metadata: {str(e)}", exc_info=True)
            return None
    
    async def clear_document_cache(self, document_id: Optional[str] = None):
        """Clear document cache entries"""
        try:
            if document_id:
                # Clear specific document cache
                await cache.delete_cache(f"doc_metadata:{document_id}")
                await cache.clear_cache(f"doc_retrieval:*{document_id}*")
            else:
                # Clear all document caches
                await cache.clear_cache("doc_metadata:*")
                await cache.clear_cache("doc_retrieval:*")
            
            logger.info(f"Cleared document cache for ID: {document_id or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing document cache: {str(e)}", exc_info=True)
            return False

# Create global retriever instance
_document_retriever = None

async def get_document_retriever() -> DocumentRetriever:
    """Get the global document retriever instance"""
    global _document_retriever
    if _document_retriever is None:
        _document_retriever = DocumentRetriever()
    return _document_retriever 