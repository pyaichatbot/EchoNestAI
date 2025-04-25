from typing import List, Dict, Any, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import asyncio
from functools import lru_cache

from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("vector_store")

class VectorStore:
    def __init__(self):
        self._client = None
        self._collection_name = "document_embeddings"
        self._vector_size = 768  # Default for most transformer models
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize vector store and create collection if needed"""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:  # Double check after acquiring lock
                return
                
            try:
                self._client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=10.0  # 10 second timeout
                )
                
                # Check if collection exists
                collections = self._client.get_collections().collections
                exists = any(c.name == self._collection_name for c in collections)
                
                if not exists:
                    # Create collection with optimal parameters
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=VectorParams(
                            size=self._vector_size,
                            distance=Distance.COSINE
                        ),
                        optimizers_config=models.OptimizersConfigDiff(
                            indexing_threshold=20000,  # Build index after 20k vectors
                            memmap_threshold=50000     # Use memmap after 50k vectors
                        )
                    )
                    logger.info(f"Created vector collection: {self._collection_name}")
                
                self._initialized = True
                logger.info("Vector store initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {str(e)}", exc_info=True)
                raise
    
    async def upsert_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Upsert vectors and metadata to the store.
        
        Args:
            vectors: List of embedding vectors
            metadata: List of metadata dicts
            ids: Optional list of IDs (generated if not provided)
            
        Returns:
            bool: Success status
        """
        try:
            await self.initialize()
            
            # Convert vectors to float32
            vectors = [v.astype(np.float32) for v in vectors]
            
            # Generate IDs if not provided
            if ids is None:
                from uuid import uuid4
                ids = [str(uuid4()) for _ in vectors]
            
            # Create points
            points = [
                models.PointStruct(
                    id=id_,
                    vector=vector.tolist(),
                    payload=meta
                )
                for id_, vector, meta in zip(ids, vectors, metadata)
            ]
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=batch
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {str(e)}", exc_info=True)
            return False
    
    async def search_vectors(
        self,
        query_vector: np.ndarray,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with optional filtering.
        
        Args:
            query_vector: Query embedding vector
            filter_conditions: Optional filter conditions
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of results with scores and metadata
        """
        try:
            await self.initialize()
            
            # Convert query vector to float32
            query_vector = query_vector.astype(np.float32)
            
            # Prepare filter
            filter_query = None
            if filter_conditions:
                filter_query = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=k,
                            match=models.MatchValue(value=v)
                        )
                        for k, v in filter_conditions.items()
                    ]
                )
            
            # Perform search
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector.tolist(),
                query_filter=filter_query,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            formatted_results = []
            for res in results:
                result = {
                    "id": res.id,
                    "score": float(res.score),
                    **res.payload
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}", exc_info=True)
            return []
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        try:
            await self.initialize()
            
            # Delete in batches of 100
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i + batch_size]
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=models.PointIdsList(
                        points=batch
                    )
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}", exc_info=True)
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and info"""
        try:
            await self.initialize()
            
            collection_info = self._client.get_collection(
                collection_name=self._collection_name
            )
            
            return {
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}", exc_info=True)
            return {}

# Global instance
_vector_store = None

async def get_vector_store() -> VectorStore:
    """Get the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        await _vector_store.initialize()
    return _vector_store 