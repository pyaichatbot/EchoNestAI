from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import numpy as np
from datetime import datetime

from app.db.models.models import Content, ContentEmbedding, ContentAssignment
from app.services.vector_store import get_vector_store
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.cache import cache

logger = setup_logging("retrieval")

class DocumentRetriever:
    def __init__(self):
        self._vector_store = None
    
    async def get_vector_store(self):
        """Get vector store instance"""
        if self._vector_store is None:
            self._vector_store = await get_vector_store()
        return self._vector_store
    
    async def retrieve_documents(
        self,
        db: AsyncSession,
        query: str,
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
            query: Search query
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
                cache_key = f"retrieval:{query}:{language}:{document_ids}:{user_id}:{group_ids}:{limit}"
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
            
            # Perform similarity search
            search_results = await vector_store.similarity_search_with_score(
                query=query,
                language=language,
                filter_ids=accessible_ids,
                top_k=limit,
                score_threshold=min_score
            )
            
            # Format results
            documents = []
            for doc, score in search_results:
                formatted_doc = {
                    "id": doc["document_id"],
                    "title": doc["title"],
                    "content": doc["context"],
                    "relevance_score": float(score),
                    "language": doc["language"],
                    "type": doc["content_type"],
                    "metadata": doc["metadata"]
                }
                documents.append(formatted_doc)
            
            total = len(documents)
            
            # Cache results
            if use_cache:
                await cache.set_cache(
                    cache_key,
                    {"documents": documents, "total": total},
                    expire=300  # 5 minutes
                )
            
            return documents, total
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}", exc_info=True)
            raise
    
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
            raise
    
    async def get_document_metadata(
        self,
        db: AsyncSession,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document"""
        try:
            query = select(Content).filter(Content.id == document_id)
            result = await db.execute(query)
            document = result.scalars().first()
            
            if not document:
                return None
            
            return {
                "id": document.id,
                "title": document.title,
                "type": document.type,
                "language": document.language,
                "created_by": document.created_by,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat() if document.updated_at else None,
                "size_mb": document.size_mb,
                "version": document.version
            }
            
        except Exception as e:
            logger.error(f"Error getting document metadata: {str(e)}", exc_info=True)
            raise

# Create global retriever instance
document_retriever = DocumentRetriever()

async def get_document_retriever() -> DocumentRetriever:
    """Get the global document retriever instance"""
    return document_retriever 