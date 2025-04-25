from typing import Any, Optional, Union, Callable, TypeVar, List
import json
import aioredis
import functools
import inspect
from app.core.config import settings
from app.core.logging import setup_logging
import asyncio

logger = setup_logging("cache")

T = TypeVar('T')

class Cache:
    def __init__(self):
        self._redis = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._connection_retries = 0
        self._max_retries = 3
        self._retry_delay = 1  # seconds
    
    async def get_redis(self) -> aioredis.Redis:
        """Get Redis connection, creating if needed"""
        if not self._redis or not self._connected:
            async with self._lock:
                if not self._redis or not self._connected:
                    try:
                        if self._redis:
                            await self._redis.close()
                            
                        self._redis = aioredis.from_url(
                            settings.REDIS_URL,
                            encoding="utf-8",
                            decode_responses=True,
                            max_connections=10,
                            retry_on_timeout=True,
                            health_check_interval=30
                        )
                        await self._redis.ping()
                        self._connected = True
                        self._connection_retries = 0
                        logger.info("Redis connection established")
                    except Exception as e:
                        self._connected = False
                        self._connection_retries += 1
                        if self._connection_retries >= self._max_retries:
                            logger.error(f"Redis connection failed after {self._max_retries} retries: {str(e)}")
                            raise
                        logger.warning(f"Redis connection attempt {self._connection_retries} failed: {str(e)}")
                        await asyncio.sleep(self._retry_delay * self._connection_retries)
                        return await self.get_redis()  # Retry connection
        return self._redis
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            redis = await self.get_redis()
            value = await redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}", exc_info=True)
            return None
    
    async def set_cache(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional expiration"""
        try:
            redis = await self.get_redis()
            serialized = json.dumps(value)
            if expire:
                await redis.setex(key, expire, serialized)
            else:
                await redis.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}", exc_info=True)
            return False
    
    async def delete_cache(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            redis = await self.get_redis()
            await redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}", exc_info=True)
            return False
    
    async def clear_cache(self, pattern: str = "*") -> bool:
        """Clear all cache entries matching pattern"""
        try:
            redis = await self.get_redis()
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern)
                if keys:
                    await redis.delete(*keys)
                if cursor == 0:
                    break
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}", exc_info=True)
            return False
    
    def cache_decorator(
        self,
        ttl: Optional[int] = None,
        key_prefix: Optional[str] = None,
        skip_kwargs: Optional[List[str]] = None
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Create a caching decorator.
        
        Args:
            ttl: Time to live in seconds
            key_prefix: Optional prefix for cache key
            skip_kwargs: List of kwargs to skip in cache key generation
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            # Get function signature for better key generation
            sig = inspect.signature(func)
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Filter out skipped kwargs
                cache_kwargs = kwargs
                if skip_kwargs:
                    cache_kwargs = {
                        k: v for k, v in kwargs.items()
                        if k not in skip_kwargs
                    }
                
                # Create cache key
                key_parts = [
                    key_prefix or func.__name__,
                    *[str(arg) for arg in args],
                    *[f"{k}:{v}" for k, v in sorted(cache_kwargs.items())]
                ]
                cache_key = ":".join(key_parts)
                
                # Try to get from cache
                cached_value = await self.get_cache(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_value
                
                # Call function if not in cache
                result = await func(*args, **kwargs)
                
                # Cache the result
                if result is not None:
                    await self.set_cache(
                        cache_key,
                        result,
                        expire=ttl or settings.CACHE_TTL
                    )
                
                return result
            
            return wrapper
        return decorator

# Create global cache instance
cache = Cache()

def cached(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    skip_kwargs: Optional[List[str]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Cache decorator for easier usage.
    
    Example:
        @cached(ttl=300)
        async def my_function():
            pass
    """
    return cache.cache_decorator(
        ttl=ttl,
        key_prefix=key_prefix,
        skip_kwargs=skip_kwargs
    )

async def get_cache() -> Cache:
    """Get the global cache instance"""
    return cache

async def clear_cache(pattern: str = "*"):
    """Clear cache entries matching the given pattern."""
    try:
        keys = await redis.keys(pattern)
        if keys:
            await redis.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries matching pattern: {pattern}")
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")

async def get_cache_stats():
    """Get cache statistics."""
    try:
        info = await redis.info()
        return {
            "used_memory": info["used_memory_human"],
            "connected_clients": info["connected_clients"],
            "total_keys": len(await redis.keys("*")),
            "hits": info["keyspace_hits"],
            "misses": info["keyspace_misses"]
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return None 