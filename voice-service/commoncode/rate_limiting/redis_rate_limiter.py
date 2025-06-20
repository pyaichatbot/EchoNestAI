"""
Production-Grade Redis Rate Limiter
Generic, reusable rate limiting implementation using Redis
"""

import time
import json
import logging
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size_seconds: int = 60
    retry_after_header: bool = True
    cost_based_limits: bool = False
    cost_per_request: float = 1.0
    max_daily_cost: float = 100.0

@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None
    cost_used: float = 0.0
    daily_cost: float = 0.0
    limit_exceeded: Optional[str] = None

class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters"""
    
    @abstractmethod
    def is_allowed(self, key: str, cost: float = 1.0) -> RateLimitResult:
        """Check if request is allowed"""
        pass
    
    @abstractmethod
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for key"""
        pass
    
    @abstractmethod
    def reset_key(self, key: str) -> bool:
        """Reset rate limit for key"""
        pass

class RedisRateLimiter(BaseRateLimiter):
    """
    Production-grade Redis-based rate limiter with multiple strategies
    """
    
    def __init__(self, 
                 redis_client: redis.Redis,
                 config: RateLimitConfig,
                 prefix: str = "rate_limit"):
        """
        Initialize Redis rate limiter
        
        Args:
            redis_client: Redis client instance
            config: Rate limiting configuration
            prefix: Key prefix for Redis
        """
        self.redis_client = redis_client
        self.config = config
        self.prefix = prefix
        self._validate_config()
    
    def _validate_config(self):
        """Validate rate limit configuration"""
        if self.config.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if self.config.requests_per_hour <= 0:
            raise ValueError("requests_per_hour must be positive")
        if self.config.requests_per_day <= 0:
            raise ValueError("requests_per_day must be positive")
        if self.config.burst_limit <= 0:
            raise ValueError("burst_limit must be positive")
        if self.config.cost_per_request <= 0:
            raise ValueError("cost_per_request must be positive")
        if self.config.max_daily_cost <= 0:
            raise ValueError("max_daily_cost must be positive")
    
    def is_allowed(self, key: str, cost: float = 1.0) -> RateLimitResult:
        """
        Check if request is allowed using sliding window algorithm
        
        Args:
            key: Unique identifier for rate limiting
            cost: Cost of the request (for cost-based limiting)
            
        Returns:
            RateLimitResult with allowance status and metadata
        """
        try:
            current_time = int(time.time())
            window_start = current_time - self.config.window_size_seconds
            
            # Create Redis keys
            minute_key = f"{self.prefix}:minute:{key}"
            hour_key = f"{self.prefix}:hour:{key}"
            day_key = f"{self.prefix}:day:{key}"
            cost_key = f"{self.prefix}:cost:{key}:{current_time // 86400}"
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries from sliding windows
            pipe.zremrangebyscore(minute_key, 0, window_start)
            pipe.zremrangebyscore(hour_key, 0, current_time - 3600)
            pipe.zremrangebyscore(day_key, 0, current_time - 86400)
            
            # Get current counts
            pipe.zcard(minute_key)
            pipe.zcard(hour_key)
            pipe.zcard(day_key)
            
            # Get cost information if enabled
            if self.config.cost_based_limits:
                pipe.get(cost_key)
            
            # Execute pipeline
            results = pipe.execute()
            
            minute_count = results[3]
            hour_count = results[4]
            day_count = results[5]
            daily_cost = float(results[6] or 0) if self.config.cost_based_limits else 0.0
            
            # Check limits
            allowed = True
            limit_exceeded = None
            retry_after = None
            
            # Check minute limit
            if minute_count >= self.config.requests_per_minute:
                allowed = False
                limit_exceeded = "minute"
                retry_after = self._get_retry_after(minute_key, self.config.requests_per_minute)
            
            # Check hour limit
            elif hour_count >= self.config.requests_per_hour:
                allowed = False
                limit_exceeded = "hour"
                retry_after = self._get_retry_after(hour_key, self.config.requests_per_hour)
            
            # Check day limit
            elif day_count >= self.config.requests_per_day:
                allowed = False
                limit_exceeded = "day"
                retry_after = self._get_retry_after(day_key, self.config.requests_per_day)
            
            # Check cost limit
            elif self.config.cost_based_limits:
                request_cost = cost * self.config.cost_per_request
                if daily_cost + request_cost > self.config.max_daily_cost:
                    allowed = False
                    limit_exceeded = "cost"
                    retry_after = 86400 - (current_time % 86400)  # Seconds until next day
            
            # If allowed, add request to tracking
            if allowed:
                pipe = self.redis_client.pipeline()
                pipe.zadd(minute_key, {str(current_time): current_time})
                pipe.zadd(hour_key, {str(current_time): current_time})
                pipe.zadd(day_key, {str(current_time): current_time})
                pipe.expire(minute_key, self.config.window_size_seconds)
                pipe.expire(hour_key, 3600)
                pipe.expire(day_key, 86400)
                
                # Update cost tracking
                if self.config.cost_based_limits:
                    request_cost = cost * self.config.cost_per_request
                    pipe.incrbyfloat(cost_key, request_cost)
                    pipe.expire(cost_key, 86400)
                
                pipe.execute()
                
                # Update counts
                minute_count += 1
                hour_count += 1
                day_count += 1
                daily_cost += request_cost if self.config.cost_based_limits else 0.0
            
            # Calculate remaining requests
            remaining = min(
                self.config.requests_per_minute - minute_count,
                self.config.requests_per_hour - hour_count,
                self.config.requests_per_day - day_count
            )
            
            return RateLimitResult(
                allowed=allowed,
                remaining=max(0, remaining),
                reset_time=current_time + self.config.window_size_seconds,
                retry_after=retry_after,
                cost_used=request_cost if allowed and self.config.cost_based_limits else 0.0,
                daily_cost=daily_cost,
                limit_exceeded=limit_exceeded
            )
            
        except RedisError as e:
            logger.error(f"Redis error in rate limiting: {e}")
            # Allow request on Redis failure (fail open)
            return RateLimitResult(
                allowed=True,
                remaining=999,
                reset_time=int(time.time()) + self.config.window_size_seconds
            )
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request on error (fail open)
            return RateLimitResult(
                allowed=True,
                remaining=999,
                reset_time=int(time.time()) + self.config.window_size_seconds
            )
    
    def _get_retry_after(self, key: str, limit: int) -> Optional[int]:
        """Calculate retry after time for a specific limit"""
        try:
            # Get the oldest entry that would be removed to allow a new request
            oldest_allowed = self.redis_client.zrange(key, limit - 1, limit - 1, withscores=True)
            if oldest_allowed:
                return int(oldest_allowed[0][1]) + self.config.window_size_seconds - int(time.time())
            return None
        except Exception as e:
            logger.error(f"Error calculating retry after: {e}")
            return None
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for key"""
        try:
            current_time = int(time.time())
            window_start = current_time - self.config.window_size_seconds
            
            minute_key = f"{self.prefix}:minute:{key}"
            
            # Remove old entries
            self.redis_client.zremrangebyscore(minute_key, 0, window_start)
            
            # Get current count
            count = self.redis_client.zcard(minute_key)
            return max(0, self.config.requests_per_minute - count)
            
        except Exception as e:
            logger.error(f"Error getting remaining requests: {e}")
            return 999  # Return high number on error
    
    def reset_key(self, key: str) -> bool:
        """Reset rate limit for key"""
        try:
            pipe = self.redis_client.pipeline()
            pipe.delete(f"{self.prefix}:minute:{key}")
            pipe.delete(f"{self.prefix}:hour:{key}")
            pipe.delete(f"{self.prefix}:day:{key}")
            pipe.delete(f"{self.prefix}:cost:{key}:*")
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
            return False
    
    def get_stats(self, key: str) -> Dict[str, Any]:
        """Get detailed statistics for a key"""
        try:
            current_time = int(time.time())
            
            minute_key = f"{self.prefix}:minute:{key}"
            hour_key = f"{self.prefix}:hour:{key}"
            day_key = f"{self.prefix}:day:{key}"
            cost_key = f"{self.prefix}:cost:{key}:{current_time // 86400}"
            
            pipe = self.redis_client.pipeline()
            pipe.zcard(minute_key)
            pipe.zcard(hour_key)
            pipe.zcard(day_key)
            pipe.get(cost_key)
            pipe.ttl(minute_key)
            pipe.ttl(hour_key)
            pipe.ttl(day_key)
            
            results = pipe.execute()
            
            return {
                "minute_count": results[0],
                "hour_count": results[1],
                "day_count": results[2],
                "daily_cost": float(results[3] or 0),
                "minute_ttl": results[4],
                "hour_ttl": results[5],
                "day_ttl": results[6],
                "limits": {
                    "minute": self.config.requests_per_minute,
                    "hour": self.config.requests_per_hour,
                    "day": self.config.requests_per_day,
                    "max_daily_cost": self.config.max_daily_cost
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, rate_limiter: RedisRateLimiter, key_func=None):
        """
        Initialize rate limit middleware
        
        Args:
            rate_limiter: Redis rate limiter instance
            key_func: Function to extract rate limit key from request
        """
        self.rate_limiter = rate_limiter
        self.key_func = key_func or self._default_key_func
    
    def _default_key_func(self, request) -> str:
        """Default function to extract rate limit key from request"""
        # Use client IP as default key
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        return f"{client_ip}:{hash(user_agent) % 1000}"
    
    async def __call__(self, request, call_next):
        """Process request with rate limiting"""
        key = self.key_func(request)
        result = self.rate_limiter.is_allowed(key)
        
        if not result.allowed:
            from fastapi import HTTPException
            from fastapi.responses import JSONResponse
            
            response_data = {
                "error": "Rate limit exceeded",
                "limit_exceeded": result.limit_exceeded,
                "retry_after": result.retry_after
            }
            
            response = JSONResponse(
                content=response_data,
                status_code=429
            )
            
            if result.retry_after:
                response.headers["Retry-After"] = str(result.retry_after)
            
            return response
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_time)
        
        if result.cost_used > 0:
            response.headers["X-RateLimit-Cost"] = str(result.cost_used)
            response.headers["X-RateLimit-DailyCost"] = str(result.daily_cost)
        
        return response 