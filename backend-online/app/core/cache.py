import functools
import json
import pickle
from typing import Any, Callable
import aioredis
import hashlib

from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("cache")

# Initialize Redis connection
redis = aioredis.from_url(settings.REDIS_URL)

def generate_cache_key(*args, **kwargs) -> str:
    """Generate a unique cache key based on function arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def cache(ttl: int = None):
    """
    Cache decorator for async functions using Redis.
    
    Args:
        ttl (int, optional): Time to live in seconds. Defaults to settings.CACHE_TTL.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not settings.ENABLE_CACHE:
                return await func(*args, **kwargs)

            # Generate cache key
            cache_key = f"{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            try:
                # Try to get from cache
                cached_value = await redis.get(cache_key)
                if cached_value:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return pickle.loads(cached_value)

                # If not in cache, execute function
                logger.debug(f"Cache miss for key: {cache_key}")
                result = await func(*args, **kwargs)

                # Store in cache
                cache_ttl = ttl if ttl is not None else settings.CACHE_TTL
                await redis.setex(
                    cache_key,
                    cache_ttl,
                    pickle.dumps(result)
                )
                return result

            except Exception as e:
                logger.error(f"Cache error: {str(e)}")
                # If caching fails, just execute the function
                return await func(*args, **kwargs)

        return wrapper
    return decorator

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