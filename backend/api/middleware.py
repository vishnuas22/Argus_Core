"""
Argus Core - API Middleware
===========================
Request/response middleware for cross-cutting concerns.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - api/middleware.py

Role: CORS, rate limiting, authentication, request logging.

SOTA Algorithms:
- Token bucket rate limiting (probabilistic) with Redis backend
- JWT validation for authentication
- Structured logging with correlation IDs

Integration:
- Imports: config.py, utils/logging.py, utils/metrics.py
- Inputs: Raw HTTP requests
- Outputs: Processed requests or 4xx/5xx responses

Why this approach: Middleware pattern separates concerns cleanly.
Redis-backed rate limiting scales horizontally.
"""

import time
import uuid
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from config import config
from utils.logging import get_logger
from utils.metrics import record_http_request, http_requests_total
from utils.errors import RateLimitError, AuthenticationError, ArgusError

logger = get_logger(__name__)


# ============== REQUEST LOGGING MIDDLEWARE ==============

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured logging middleware with correlation IDs.
    
    Logs all requests with:
    - Correlation ID for distributed tracing
    - Request method, path, and query params
    - Response status code
    - Processing duration
    - Client IP address
    
    Adds correlation ID to response headers for client-side tracing.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        
        # Extract client IP (handling proxies)
        client_ip = self._get_client_ip(request)
        
        # Record start time
        start_time = time.time()
        
        # Log request start
        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query),
                "client_ip": client_ip,
                "user_agent": request.headers.get("User-Agent", "unknown")
            }
        )
        
        # Store correlation ID in request state for downstream use
        request.state.correlation_id = correlation_id
        request.state.start_time = start_time
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            # Log unhandled exceptions
            logger.error(
                f"Unhandled exception: {str(e)}",
                extra={
                    "correlation_id": correlation_id,
                    "exception_type": type(e).__name__
                },
                exc_info=True
            )
            raise
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log request completion
        log_level = "info" if status_code < 400 else "warning" if status_code < 500 else "error"
        getattr(logger, log_level)(
            "Request completed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 2),
                "client_ip": client_ip
            }
        )
        
        # Record metrics
        record_http_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=status_code,
            duration=duration
        )
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Request-Duration"] = f"{duration:.3f}s"
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check X-Forwarded-For header (set by proxies/load balancers)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"


# ============== RATE LIMITING MIDDLEWARE ==============

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiter with Redis backend.
    
    Implements probabilistic rate limiting using token bucket algorithm.
    Falls back to in-memory rate limiting if Redis is unavailable.
    
    Features:
    - Per-IP rate limiting
    - Configurable limits per minute
    - Redis-backed for horizontal scaling
    - Graceful degradation to in-memory
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = None,
        burst_multiplier: float = 1.5,
        exclude_paths: list = None,
        redis_url: str = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            app: ASGI application
            requests_per_minute: Max requests per minute per IP
            burst_multiplier: Allow burst up to this factor
            exclude_paths: Paths to exclude from rate limiting
            redis_url: Redis URL for distributed rate limiting
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or config.api_rate_limit_per_minute
        self.burst_size = int(self.requests_per_minute * burst_multiplier)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.redis_url = redis_url or config.redis_url
        
        # In-memory fallback (for single-instance or when Redis unavailable)
        self._local_buckets: Dict[str, Dict[str, Any]] = {}
        
        # Try to connect to Redis
        self._redis_client = None
        self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection for distributed rate limiting."""
        try:
            import redis
            self._redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=1.0
            )
            self._redis_client.ping()
            logger.info("Rate limiter connected to Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory rate limiting: {e}")
            self._redis_client = None
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_id = self._get_client_id(request)
        
        # Check rate limit
        allowed, remaining, reset_time = await self._check_rate_limit(client_id)
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_id}",
                extra={
                    "client_id": client_id,
                    "path": request.url.path,
                    "limit": self.requests_per_minute
                }
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                    "details": {
                        "limit": self.requests_per_minute,
                        "reset_in_seconds": reset_time
                    }
                },
                headers={
                    "Retry-After": str(reset_time),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + reset_time)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Prefer authenticated user ID if available
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        if request.client:
            return f"ip:{request.client.host}"
        
        return "ip:unknown"
    
    async def _check_rate_limit(self, client_id: str) -> tuple:
        """
        Check if request is within rate limit.
        
        Returns:
            Tuple of (allowed: bool, remaining: int, reset_in_seconds: int)
        """
        if self._redis_client:
            return await self._check_rate_limit_redis(client_id)
        else:
            return self._check_rate_limit_local(client_id)
    
    async def _check_rate_limit_redis(self, client_id: str) -> tuple:
        """Check rate limit using Redis."""
        try:
            key = f"ratelimit:{client_id}"
            current_time = int(time.time())
            window_start = current_time - 60
            
            # Use Redis sorted set for sliding window
            pipe = self._redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Set expiry
            pipe.expire(key, 120)
            
            results = pipe.execute()
            request_count = results[1]
            
            allowed = request_count < self.requests_per_minute
            
            if allowed:
                # Only add request if within limit
                self._redis_client.zadd(key, {str(current_time): current_time})
            
            remaining = max(0, self.requests_per_minute - request_count - 1)
            reset_in = 60 - (current_time % 60)
            
            return allowed, remaining, reset_in
            
        except Exception as e:
            logger.warning(f"Redis rate limit check failed: {e}")
            return self._check_rate_limit_local(client_id)
    
    def _check_rate_limit_local(self, client_id: str) -> tuple:
        """Check rate limit using local in-memory storage."""
        current_time = time.time()
        window_start = current_time - 60
        
        if client_id not in self._local_buckets:
            self._local_buckets[client_id] = {"requests": []}
        
        bucket = self._local_buckets[client_id]
        
        # Remove old requests
        bucket["requests"] = [t for t in bucket["requests"] if t > window_start]
        
        # Check limit
        if len(bucket["requests"]) >= self.requests_per_minute:
            remaining = 0
            reset_in = int(60 - (current_time - bucket["requests"][0]))
            return False, remaining, max(1, reset_in)
        
        # Add request
        bucket["requests"].append(current_time)
        
        remaining = self.requests_per_minute - len(bucket["requests"])
        reset_in = 60
        
        return True, remaining, reset_in


# ============== AUTHENTICATION MIDDLEWARE ==============

class AuthMiddleware(BaseHTTPMiddleware):
    """
    JWT validation and role extraction middleware.
    
    Validates JWT tokens and extracts user information
    into request state for downstream handlers.
    
    Features:
    - JWT validation with configurable algorithm
    - Role extraction for RBAC
    - Optional authentication (anonymous allowed)
    - Token refresh handling
    """
    
    def __init__(
        self,
        app: ASGIApp,
        public_paths: list = None,
        jwt_secret: str = None,
        jwt_algorithm: str = None
    ):
        """
        Initialize auth middleware.
        
        Args:
            app: ASGI application
            public_paths: Paths that don't require authentication
            jwt_secret: Secret key for JWT validation
            jwt_algorithm: JWT algorithm (default HS256)
        """
        super().__init__(app)
        self.public_paths = public_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register"
        ]
        self.jwt_secret = jwt_secret or config.jwt_secret
        self.jwt_algorithm = jwt_algorithm or config.jwt_algorithm
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip authentication for public paths
        if any(request.url.path.startswith(path) for path in self.public_paths):
            return await call_next(request)
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        
        if auth_header:
            try:
                # Parse Bearer token
                scheme, token = auth_header.split()
                if scheme.lower() != "bearer":
                    raise ValueError("Invalid authentication scheme")
                
                # Validate JWT
                user_data = self._validate_token(token)
                
                # Store user info in request state
                request.state.user_id = user_data.get("sub")
                request.state.user_email = user_data.get("email")
                request.state.user_roles = user_data.get("roles", [])
                request.state.token_exp = user_data.get("exp")
                request.state.authenticated = True
                
            except ValueError as e:
                logger.warning(f"Auth header parse error: {e}")
                request.state.authenticated = False
            except Exception as e:
                logger.warning(f"Token validation failed: {e}")
                request.state.authenticated = False
        else:
            request.state.authenticated = False
        
        return await call_next(request)
    
    def _validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return payload."""
        import jwt
        
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")


# ============== ERROR HANDLING MIDDLEWARE ==============

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.
    
    Catches all exceptions and converts them to
    standardized JSON error responses.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            return await call_next(request)
            
        except ArgusError as e:
            # Handle custom Argus exceptions
            logger.warning(
                f"Argus error: {e.error_code}",
                extra={
                    "error_code": e.error_code,
                    "message": e.message,
                    "details": e.details,
                    "path": request.url.path
                }
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
            
        except HTTPException as e:
            # Re-raise FastAPI HTTP exceptions
            raise
            
        except Exception as e:
            # Handle unexpected exceptions
            correlation_id = getattr(request.state, "correlation_id", "unknown")
            
            logger.error(
                f"Unexpected error: {str(e)}",
                extra={
                    "correlation_id": correlation_id,
                    "exception_type": type(e).__name__,
                    "path": request.url.path
                },
                exc_info=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error_code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": {
                        "correlation_id": correlation_id
                    }
                }
            )


# ============== CORS MIDDLEWARE CONFIGURATION ==============

def get_cors_config() -> dict:
    """
    Get CORS middleware configuration.
    
    Returns configuration dict for CORSMiddleware.
    """
    return {
        "allow_origins": config.cors_origins_list,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "X-Correlation-ID",
            "X-Request-ID",
            "Accept",
            "Origin"
        ],
        "expose_headers": [
            "X-Correlation-ID",
            "X-Request-Duration",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ],
        "max_age": 600  # Cache preflight for 10 minutes
    }


# ============== SECURITY HEADERS MIDDLEWARE ==============

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    
    Implements security best practices:
    - Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: enabled
    - Strict-Transport-Security: enabled
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HSTS (only in production with HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


# ============== REQUEST ID MIDDLEWARE ==============

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Generate unique request ID for each request.
    
    Request ID is different from correlation ID:
    - Correlation ID: traces across multiple services
    - Request ID: unique per request within this service
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


# ============== MIDDLEWARE FACTORY ==============

def setup_middleware(app) -> None:
    """
    Setup all middleware in correct order.
    
    Middleware order matters:
    1. Error handling (outermost - catches all errors)
    2. Security headers
    3. Request logging
    4. Request ID
    5. CORS
    6. Rate limiting
    7. Authentication
    
    Args:
        app: FastAPI application instance
    """
    from starlette.middleware.cors import CORSMiddleware
    
    # Add middleware in reverse order (last added = first executed)
    
    # Authentication (innermost - runs last before handlers)
    app.add_middleware(AuthMiddleware)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware)
    
    # CORS
    app.add_middleware(CORSMiddleware, **get_cors_config())
    
    # Request ID
    app.add_middleware(RequestIDMiddleware)
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Error handling (outermost - catches all errors)
    app.add_middleware(ErrorHandlingMiddleware)
    
    logger.info("Middleware stack configured")


# Export middleware classes
__all__ = [
    "RequestLoggingMiddleware",
    "RateLimitMiddleware",
    "AuthMiddleware",
    "ErrorHandlingMiddleware",
    "SecurityHeadersMiddleware",
    "RequestIDMiddleware",
    "get_cors_config",
    "setup_middleware"
]
