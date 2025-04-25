from fastapi import Request, HTTPException, status
from starlette.responses import Response
from typing import Callable, Awaitable
import time
import hashlib
from app.core.cache import cache
from app.core.config import settings
from app.core.logging import setup_logging

logger = setup_logging("rate_limit")

class RateLimiter:
    def __init__(
        self,
        times: int = 100,  # Number of requests
        seconds: int = 60   # Per time window
    ):
        self.times = times
        self.seconds = seconds
    
    def _get_client_identifier(self, request: Request) -> str:
        """Generate a unique identifier for the client"""
        # Use X-Forwarded-For if behind a proxy, fallback to client host
        client_ip = request.headers.get("X-Forwarded-For", request.client.host)
        # Include user agent to differentiate between different clients from same IP
        user_agent = request.headers.get("User-Agent", "")
        # Get authenticated user ID if available
        user_id = getattr(request.state, "user_id", "anonymous")
        
        # Create unique identifier
        identifier = f"{client_ip}:{user_agent}:{user_id}"
        # Hash the identifier for privacy and consistent length
        return hashlib.sha256(identifier.encode()).hexdigest()
    
    async def _get_cache_key(self, request: Request) -> str:
        """Generate cache key from client identifier and endpoint"""
        client_id = self._get_client_identifier(request)
        endpoint = request.url.path
        return f"rate_limit:{client_id}:{endpoint}"
    
    async def is_rate_limited(self, request: Request) -> bool:
        """Check if request should be rate limited"""
        try:
            cache_key = await self._get_cache_key(request)
            redis = await cache.get_redis()
            
            pipe = redis.pipeline()
            now = int(time.time())
            window_start = now - self.seconds
            
            # Remove old requests outside the window
            pipe.zremrangebyscore(cache_key, "-inf", window_start)
            # Add current request
            pipe.zadd(cache_key, {str(now): now})
            # Count requests in window
            pipe.zcount(cache_key, window_start, "+inf")
            # Set key expiration
            pipe.expire(cache_key, self.seconds)
            
            # Execute pipeline
            results = await pipe.execute()
            request_count = results[2]
            
            # Check if under limit
            return request_count > self.times
            
        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}")
            # On error, allow request
            return False
    
    async def __call__(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]]
    ):
        """Middleware implementation"""
        # Skip rate limiting for certain paths
        if request.url.path.startswith("/docs") or \
           request.url.path.startswith("/redoc") or \
           request.url.path.startswith("/openapi.json") or \
           request.url.path.startswith("/health"):
            return await call_next(request)
        
        is_limited = await self.is_rate_limited(request)
        if is_limited:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests"
            )
        
        return await call_next(request)

# Create rate limiters with different configs
auth_limiter = RateLimiter(times=5, seconds=60)  # 5 requests per minute for auth
chat_limiter = RateLimiter(times=30, seconds=60)  # 30 requests per minute for chat
default_limiter = RateLimiter(times=100, seconds=60)  # 100 requests per minute default 