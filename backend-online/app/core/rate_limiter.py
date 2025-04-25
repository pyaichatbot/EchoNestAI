from fastapi import Request
from datetime import datetime
import aioredis
from app.core.config import settings
import hashlib

class RateLimiter:
    def __init__(self):
        self.redis = None
    
    async def get_redis(self) -> aioredis.Redis:
        """Get or create Redis connection"""
        if self.redis is None:
            self.redis = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
        return self.redis
    
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
    
    async def check_rate_limit(
        self,
        request: Request,
        key_prefix: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """
        Check if the request is within rate limits
        
        Args:
            request: FastAPI request object
            key_prefix: Prefix for the rate limit key
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds
            
        Returns:
            bool: True if request is allowed, False if rate limit exceeded
        """
        redis = await self.get_redis()
        
        # Generate rate limit key
        client_id = self._get_client_identifier(request)
        key = f"rate_limit:{key_prefix}:{client_id}"
        
        pipe = redis.pipeline()
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds
        
        try:
            # Remove old requests outside the window
            pipe.zremrangebyscore(key, "-inf", window_start)
            # Add current request
            pipe.zadd(key, {str(now): now})
            # Count requests in window
            pipe.zcount(key, window_start, "+inf")
            # Set key expiration
            pipe.expire(key, window_seconds)
            
            # Execute pipeline
            results = await pipe.execute()
            request_count = results[2]
            
            # Check if under limit
            return request_count <= max_requests
            
        except Exception as e:
            # Log error but allow request in case of Redis failure
            logger.error(f"Rate limit check failed: {str(e)}")
            return True
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()

# Create global rate limiter instance
rate_limiter = RateLimiter() 