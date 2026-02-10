"""
Argus Core - API Dependency Providers
=====================================
FastAPI dependency injection providers for service instances.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - api/deps.py

Role: Create and cache service instances per request or application lifetime.
Centralizes dependency management for easy mocking in tests.

Integration:
- Imports: storage/storage.py, storage/db.py, processing/sanitize.py, core/orchestrator.py
- Inputs: None
- Outputs: Service instances (StorageClient, DatabaseClient, etc.)

Why this approach: Centralized dependency management enables easy mocking 
for tests and consistent resource handling across all API endpoints.
"""

from typing import AsyncGenerator, Optional, TYPE_CHECKING
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import config, get_settings, Settings
from storage.storage import StorageClient, get_storage_client
from storage.db import DatabaseClient, get_db_client, close_db_client
from processing.sanitize import InputSanitizer
from core.engine import InferenceEngine, get_inference_engine
from core.fusion import MultiModalFusion, get_multi_modal_fusion
from core.scorer import TrustScorer, get_trust_scorer
from utils.logging import get_logger
from utils.errors import AuthenticationError

# Import Orchestrator conditionally to avoid circular imports
if TYPE_CHECKING:
    from core.orchestrator import Orchestrator

logger = get_logger(__name__)

# Security scheme for JWT authentication
security = HTTPBearer(auto_error=False)


# ============== CONFIGURATION ==============

def get_config() -> Settings:
    """
    Get application configuration.
    
    Returns cached settings instance from environment variables.
    Use this dependency for access to config values in routes.
    
    Returns:
        Settings instance
    """
    return get_settings()


# ============== DATABASE ==============

async def get_db() -> AsyncGenerator[DatabaseClient, None]:
    """
    Get connection-pooled MongoDB client.
    
    Uses singleton pattern with connection pooling.
    Connection is maintained across requests.
    
    Yields:
        DatabaseClient instance
        
    Note: Connection pool is managed by Motor/PyMongo,
    so we don't close per request.
    """
    db = await get_db_client()
    yield db


# ============== STORAGE ==============

def get_storage() -> StorageClient:
    """
    Get singleton MinIO storage client.
    
    Returns the same client instance for all requests.
    Client handles connection pooling internally.
    
    Returns:
        StorageClient instance
    """
    return get_storage_client()


# ============== SANITIZER ==============

def get_sanitizer(
    defense_level: str = "standard"
) -> InputSanitizer:
    """
    Get per-request input sanitizer instance.
    
    Creates new sanitizer for each request to allow
    per-request configuration of defense level.
    
    Args:
        defense_level: Adversarial defense level (none, standard, aggressive)
        
    Returns:
        InputSanitizer instance configured with defense level
    """
    return InputSanitizer(
        max_size_mb=config.max_file_size_mb,
        defense_level=defense_level
    )


def get_sanitizer_standard() -> InputSanitizer:
    """Get sanitizer with standard defense level."""
    return get_sanitizer("standard")


def get_sanitizer_aggressive() -> InputSanitizer:
    """Get sanitizer with aggressive defense level."""
    return get_sanitizer("aggressive")


def get_sanitizer_none() -> InputSanitizer:
    """Get sanitizer with no defense (faster)."""
    return get_sanitizer("none")


# ============== INFERENCE ENGINE ==============

def get_engine() -> InferenceEngine:
    """
    Get singleton inference engine.
    
    The inference engine manages model loading, VRAM allocation,
    and batch inference execution.
    
    Returns:
        InferenceEngine instance
    """
    return get_inference_engine()


# ============== MULTI-MODAL FUSION ==============

def get_fusion() -> MultiModalFusion:
    """
    Get singleton multi-modal fusion instance.
    
    Combines outputs from all analyzers using attention-weighted
    fusion with uncertainty quantification.
    
    Returns:
        MultiModalFusion instance
    """
    return get_multi_modal_fusion()


# ============== TRUST SCORER ==============

def get_scorer() -> TrustScorer:
    """
    Get singleton trust scorer instance.
    
    Computes calibrated Trust Score and determines verdict
    from aggregated multi-modal results.
    
    Returns:
        TrustScorer instance
    """
    return get_trust_scorer()


# ============== ORCHESTRATOR ==============

_orchestrator: Optional["Orchestrator"] = None


async def get_orchestrator() -> "Orchestrator":
    """
    Get Celery task orchestrator.
    
    Manages job queuing, status tracking, and retry logic
    for the analysis pipeline.
    
    Returns:
        Orchestrator instance
        
    Note: Lazy import to avoid circular dependencies.
    """
    global _orchestrator
    
    if _orchestrator is None:
        # Lazy import to avoid circular dependency
        from core.orchestrator import Orchestrator, get_orchestrator as _get_orch
        _orchestrator = _get_orch()
    
    return _orchestrator


# ============== AUTHENTICATION ==============

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_config)
) -> Optional[dict]:
    """
    Get current user from JWT token (optional).
    
    Returns None if no token provided, allowing anonymous access.
    Used for endpoints that work with or without authentication.
    
    Args:
        credentials: HTTP Bearer credentials
        settings: Application settings
        
    Returns:
        User dict with claims or None if not authenticated
    """
    if credentials is None:
        return None
    
    try:
        import jwt
        
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
            "exp": payload.get("exp")
        }
        
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings: Settings = Depends(get_config)
) -> dict:
    """
    Get current user from JWT token (required).
    
    Raises 401 if token is missing or invalid.
    Used for protected endpoints.
    
    Args:
        credentials: HTTP Bearer credentials
        settings: Application settings
        
    Returns:
        User dict with claims
        
    Raises:
        HTTPException 401: If not authenticated
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        import jwt
        
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
            "exp": payload.get("exp")
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


def require_roles(*required_roles: str):
    """
    Dependency factory for role-based access control.
    
    Creates a dependency that checks if user has any of the required roles.
    
    Args:
        *required_roles: Role names that grant access
        
    Returns:
        Dependency function
        
    Usage:
        @router.get("/admin")
        async def admin_endpoint(user: dict = Depends(require_roles("admin", "superuser"))):
            ...
    """
    async def role_checker(
        user: dict = Depends(get_current_user)
    ) -> dict:
        user_roles = set(user.get("roles", []))
        
        if not user_roles.intersection(required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(required_roles)}"
            )
        
        return user
    
    return role_checker


# ============== REQUEST CONTEXT ==============

def get_correlation_id(request: Request) -> str:
    """
    Get or generate correlation ID for request tracing.
    
    Correlation ID is used for distributed tracing across
    services and in logs for debugging.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Correlation ID string
    """
    # Check for existing correlation ID in headers
    correlation_id = request.headers.get("X-Correlation-ID")
    
    if not correlation_id:
        import uuid
        correlation_id = str(uuid.uuid4())
    
    return correlation_id


def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.
    
    Handles both direct connections and proxied requests.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client IP address
    """
    # Check X-Forwarded-For header (set by proxies)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Get first IP in chain (original client)
        return forwarded.split(",")[0].strip()
    
    # Fall back to direct connection
    if request.client:
        return request.client.host
    
    return "unknown"


# ============== RATE LIMITING ==============

class RateLimiter:
    """
    Simple in-memory rate limiter for dependency injection.
    
    For production, use Redis-backed rate limiting via middleware.
    This is primarily for testing and development.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict = {}
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        import time
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        if key not in self._requests:
            self._requests[key] = []
        
        # Remove old requests outside window
        self._requests[key] = [
            t for t in self._requests[key] if t > window_start
        ]
        
        # Check if under limit
        if len(self._requests[key]) >= self.max_requests:
            return False
        
        # Record request
        self._requests[key].append(current_time)
        return True


_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get singleton rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            max_requests=config.api_rate_limit_per_minute,
            window_seconds=60
        )
    
    return _rate_limiter


async def check_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
) -> None:
    """
    Check rate limit for current request.
    
    Raises 429 if rate limit exceeded.
    
    Args:
        request: FastAPI request object
        rate_limiter: Rate limiter instance
        
    Raises:
        HTTPException 429: If rate limit exceeded
    """
    client_ip = get_client_ip(request)
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={
                "Retry-After": str(60),
                "X-RateLimit-Limit": str(config.api_rate_limit_per_minute),
                "X-RateLimit-Remaining": "0"
            }
        )


# ============== COMBINED DEPENDENCIES ==============

class AnalysisDependencies:
    """
    Combined dependencies for analysis endpoints.
    
    Groups commonly-used dependencies for cleaner endpoint signatures.
    """
    
    def __init__(
        self,
        db: DatabaseClient,
        storage: StorageClient,
        sanitizer: InputSanitizer,
        engine: InferenceEngine,
        fusion: MultiModalFusion,
        scorer: TrustScorer,
        correlation_id: str
    ):
        self.db = db
        self.storage = storage
        self.sanitizer = sanitizer
        self.engine = engine
        self.fusion = fusion
        self.scorer = scorer
        self.correlation_id = correlation_id


async def get_analysis_deps(
    request: Request,
    db: DatabaseClient = Depends(get_db),
    storage: StorageClient = Depends(get_storage),
    sanitizer: InputSanitizer = Depends(get_sanitizer_standard)
) -> AnalysisDependencies:
    """
    Get combined analysis dependencies.
    
    Provides all services needed for running analysis in one dependency.
    
    Returns:
        AnalysisDependencies with all required services
    """
    return AnalysisDependencies(
        db=db,
        storage=storage,
        sanitizer=sanitizer,
        engine=get_engine(),
        fusion=get_fusion(),
        scorer=get_scorer(),
        correlation_id=get_correlation_id(request)
    )


# ============== LIFECYCLE MANAGEMENT ==============

async def startup_dependencies() -> None:
    """
    Initialize dependencies on application startup.
    
    Called during FastAPI startup event.
    """
    logger.info("Initializing dependencies...")
    
    # Connect to database
    try:
        await get_db_client()
        logger.info("Database connected")
    except Exception as e:
        logger.warning(f"Database connection failed (non-critical): {e}")
    
    # Initialize storage (skip if MinIO not available)
    try:
        storage = get_storage_client()
        # Run with timeout to avoid blocking if MinIO is down
        try:
            await asyncio.wait_for(storage.ensure_default_buckets(), timeout=3.0)
            logger.info("Storage initialized")
        except asyncio.TimeoutError:
            logger.warning("Storage initialization timed out (MinIO unavailable)")
    except Exception as e:
        logger.warning(f"Storage initialization failed (non-critical): {e}")
    
    # Initialize inference engine (lazy - doesn't load models yet)
    try:
        get_inference_engine()
        logger.info("Inference engine initialized")
    except Exception as e:
        logger.warning(f"Inference engine initialization failed (non-critical): {e}")
    
    logger.info("All dependencies initialized")


async def shutdown_dependencies() -> None:
    """
    Cleanup dependencies on application shutdown.
    
    Called during FastAPI shutdown event.
    """
    logger.info("Shutting down dependencies...")
    
    # Close database connection
    await close_db_client()
    logger.info("Database disconnected")
    
    # Cleanup inference engine
    engine = get_inference_engine()
    await engine.cleanup()
    logger.info("Inference engine cleaned up")
    
    logger.info("All dependencies shut down")


# Export commonly used dependencies
__all__ = [
    # Configuration
    "get_config",
    
    # Core services
    "get_db",
    "get_storage",
    "get_sanitizer",
    "get_sanitizer_standard",
    "get_sanitizer_aggressive",
    "get_sanitizer_none",
    
    # ML services
    "get_engine",
    "get_fusion",
    "get_scorer",
    "get_orchestrator",
    
    # Authentication
    "get_current_user",
    "get_current_user_optional",
    "require_roles",
    
    # Request context
    "get_correlation_id",
    "get_client_ip",
    
    # Rate limiting
    "get_rate_limiter",
    "check_rate_limit",
    
    # Combined dependencies
    "AnalysisDependencies",
    "get_analysis_deps",
    
    # Lifecycle
    "startup_dependencies",
    "shutdown_dependencies",
]
