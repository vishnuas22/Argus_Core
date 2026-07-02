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
import subprocess
import signal
import os
import shutil

from fastapi import Depends, Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import config, get_settings, Settings
from storage.storage import StorageClient, get_storage_client
from storage.db import DatabaseClient, get_db_client, close_db_client
from processing.sanitize import InputSanitizer
from core.engine import InferenceEngine, get_inference_engine
from core.fusion import MultiModalFusion, get_multi_modal_fusion
from core.scorer import TrustScorer, get_trust_scorer
from models.manager import get_model_manager
from utils.logging import get_logger
from utils.errors import AuthenticationError

# Import Orchestrator conditionally to avoid circular imports
if TYPE_CHECKING:
    from core.orchestrator import Orchestrator

logger = get_logger(__name__)

# Security scheme for JWT authentication
security = HTTPBearer(auto_error=False)

# ============== SERVICE MANAGER ==============

class ServiceManager:
    """
    Manages automatic startup of infrastructure services.
    
    Handles Redis, MinIO, MongoDB, and Celery worker processes.
    Services are started automatically when the application initializes.
    """
    
    _instance: Optional["ServiceManager"] = None
    _celery_process: Optional[subprocess.Popen] = None
    _services_started: bool = False
    
    def __new__(cls) -> "ServiceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _is_docker_available(self) -> bool:
        """Check if Docker is available on the system."""
        return shutil.which("docker") is not None
    
    def _is_service_running(self, container_name: str) -> bool:
        """Check if a Docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={container_name}", "--filter", "status=running", "-q"],
                capture_output=True, text=True, timeout=10
            )
            return bool(result.stdout.strip())
        except Exception:
            return False
    
    def _start_docker_service(self, container_name: str, image: str, ports: list, env_vars: list = None) -> bool:
        """
        Start a Docker service container.
        
        Args:
            container_name: Name for the container
            image: Docker image to use
            ports: List of port mappings (e.g., ["6379:6379"])
            env_vars: List of environment variables (e.g., ["KEY=value"])
            
        Returns:
            True if service started successfully
        """
        try:
            # Check if container exists but is stopped
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={container_name}", "-q"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.stdout.strip():
                # Container exists, start it
                subprocess.run(["docker", "start", container_name], capture_output=True, timeout=30)
                logger.info(f"Started existing container: {container_name}")
                return True
            
            # Create new container
            cmd = ["docker", "run", "-d", "--name", container_name]
            
            for port in ports:
                cmd.extend(["-p", port])
            
            if env_vars:
                for env in env_vars:
                    cmd.extend(["-e", env])
            
            cmd.append(image)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"Created and started container: {container_name}")
                return True
            else:
                logger.error(f"Failed to start container {container_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout starting container: {container_name}")
            return False
        except Exception as e:
            logger.error(f"Error starting container {container_name}: {e}")
            return False
    
    def start_redis(self) -> bool:
        """
        Start Redis service.
        
        Returns:
            True if Redis is available
        """
        container_name = "argus-redis"
        
        if self._is_service_running(container_name):
            logger.info("Redis container already running")
            return True
        
        if self._is_docker_available():
            success = self._start_docker_service(
                container_name=container_name,
                image="redis:7-alpine",
                ports=["6379:6379"]
            )
            if success:
                # Wait for Redis to be ready
                import time
                for _ in range(10):
                    try:
                        result = subprocess.run(
                            ["docker", "exec", container_name, "redis-cli", "ping"],
                            capture_output=True, text=True, timeout=5
                        )
                        if "PONG" in result.stdout:
                            logger.info("Redis service ready")
                            return True
                    except Exception:
                        logger.debug("Redis ping failed, retrying...")
                    time.sleep(1)
            return success
        
        logger.warning("Docker not available, assuming Redis is running externally")
        return True
    
    def start_minio(self) -> bool:
        """
        Start MinIO service.
        
        Returns:
            True if MinIO is available
        """
        container_name = "argus-minio"
        
        if self._is_service_running(container_name):
            logger.info("MinIO container already running")
            return True
        
        if self._is_docker_available():
            success = self._start_docker_service(
                container_name=container_name,
                image="minio/minio",
                ports=["9000:9000", "9001:9001"],
                env_vars=[
                    "MINIO_ROOT_USER=minioadmin",
                    "MINIO_ROOT_PASSWORD=minioadmin"
                ]
            )
            if success:
                # Override the default command for MinIO
                subprocess.run(
                    ["docker", "exec", container_name, "mkdir", "-p", "/data"],
                    capture_output=True, timeout=10
                )
                logger.info("MinIO service ready")
                return True
            return success
        
        logger.warning("Docker not available, assuming MinIO is running externally")
        return True
    
    def start_mongodb(self) -> bool:
        """
        Start MongoDB service.
        
        Returns:
            True if MongoDB is available
        """
        container_name = "argus-mongo"
        
        if self._is_service_running(container_name):
            logger.info("MongoDB container already running")
            return True
        
        if self._is_docker_available():
            success = self._start_docker_service(
                container_name=container_name,
                image="mongo:7",
                ports=["27017:27017"]
            )
            if success:
                logger.info("MongoDB service ready")
                return True
            return success
        
        logger.warning("Docker not available, assuming MongoDB is running externally")
        return True
    
    def start_celery(self) -> bool:
        """
        Start Celery worker process.
        
        Returns:
            True if Celery worker started successfully
        """
        if self._celery_process is not None:
            # Check if process is still running
            if self._celery_process.poll() is None:
                logger.info("Celery worker already running")
                return True
            else:
                self._celery_process = None
        
        try:
            # Get the backend directory path
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(backend_dir)
            
            # Start Celery worker - use core.orchestrator which has the analysis tasks
            self._celery_process = subprocess.Popen(
                [
                    "celery", "-A", "core.orchestrator.celery_app", "worker",
                    "--loglevel=info",
                    "--concurrency=2",
                    "--pool=prefork"
                ],
                cwd=parent_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Give it a moment to start
            import time
            time.sleep(2)
            
            if self._celery_process.poll() is None:
                logger.info(f"Celery worker started (PID: {self._celery_process.pid})")
                return True
            else:
                logger.error("Celery worker failed to start")
                self._celery_process = None
                return False
                
        except Exception as e:
            logger.error(f"Failed to start Celery worker: {e}")
            self._celery_process = None
            return False
    
    def create_minio_buckets(self) -> bool:
        """
        Create required MinIO buckets.
        
        Returns:
            True if buckets were created successfully
        """
        container_name = "argus-minio"
        
        if not self._is_service_running(container_name):
            return False
        
        buckets = ["argus-uploads", "argus-preprocessed", "argus-results"]
        
        try:
            # Configure mc alias
            subprocess.run(
                ["docker", "exec", container_name, "mc", "alias", "set", "local",
                 "http://localhost:9000", "minioadmin", "minioadmin"],
                capture_output=True, timeout=10
            )
            
            for bucket in buckets:
                subprocess.run(
                    ["docker", "exec", container_name, "mc", "mb", f"local/{bucket}"],
                    capture_output=True, timeout=10
                )
            
            logger.info(f"MinIO buckets created: {buckets}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to create MinIO buckets: {e}")
            return False
    
    async def start_all_services(self) -> dict:
        """
        Start all infrastructure services.
        
        Returns:
            Dict with status of each service
        """
        if self._services_started:
            logger.info("Services already started")
            return {"status": "already_started"}
        
        logger.info("="*60)
        logger.info("AUTO-STARTING INFRASTRUCTURE SERVICES")
        logger.info("="*60)
        
        results = {}
        
        # Start services in order (infrastructure first)
        logger.info("Starting Redis...")
        results["redis"] = self.start_redis()
        
        logger.info("Starting MinIO...")
        results["minio"] = self.start_minio()
        
        logger.info("Starting MongoDB...")
        results["mongodb"] = self.start_mongodb()
        
        # Wait for services to be ready
        await asyncio.sleep(2)
        
        # Create MinIO buckets
        if results["minio"]:
            self.create_minio_buckets()
        
        # Start Celery (depends on Redis)
        logger.info("Starting Celery worker...")
        results["celery"] = self.start_celery()
        
        self._services_started = True
        
        logger.info("="*60)
        logger.info("SERVICE STARTUP RESULTS:")
        for service, status in results.items():
            status_icon = "✓" if status else "✗"
            logger.info(f"  {status_icon} {service}: {'running' if status else 'failed'}")
        logger.info("="*60)
        
        return results
    
    def stop_celery(self) -> None:
        """Stop Celery worker process."""
        if self._celery_process is not None:
            try:
                self._celery_process.terminate()
                self._celery_process.wait(timeout=5)
                logger.info("Celery worker stopped")
            except Exception as e:
                logger.warning(f"Error stopping Celery worker: {e}")
                try:
                    self._celery_process.kill()
                except Exception:
                    logger.warning("Failed to kill Celery worker process")
            finally:
                self._celery_process = None


# Global service manager instance
_service_manager: Optional[ServiceManager] = None


def get_service_manager() -> ServiceManager:
    """Get singleton service manager instance."""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager


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
    
    MAX_BUCKETS = 10000  # Maximum number of unique keys to track
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict = {}
        self._access_count: int = 0  # Track total accesses for eviction
    
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
        
        # Periodically evict stale keys (empty windows)
        self._access_count += 1
        if self._access_count % 1000 == 0:
            self._evict_stale_keys(window_start)
        
        return True
    
    def _evict_stale_keys(self, window_start: float) -> None:
        """Remove keys with no recent requests to prevent memory growth."""
        stale_keys = [
            k for k, v in self._requests.items()
            if not v or (v and v[-1] <= window_start)
        ]
        for k in stale_keys:
            del self._requests[k]
        if stale_keys:
            logger.debug(f"Evicted {len(stale_keys)} stale rate limit buckets")


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

async def wait_for_redis(max_retries: int = 10, retry_delay: float = 2.0) -> bool:
    """
    Wait for Redis to be available.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if Redis is available, False otherwise
    """
    import redis.asyncio as aioredis
    
    for attempt in range(1, max_retries + 1):
        try:
            redis_client = aioredis.from_url(config.redis_url, decode_responses=True)
            await redis_client.ping()
            await redis_client.close()
            logger.info(f"✓ Redis is available")
            return True
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Redis not ready (attempt {attempt}/{max_retries}), retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"✗ Redis unavailable after {max_retries} attempts: {e}")
                return False
    return False


async def wait_for_minio(max_retries: int = 10, retry_delay: float = 2.0) -> bool:
    """
    Wait for MinIO to be available.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if MinIO is available (or local fallback is ready), False otherwise
    """
    for attempt in range(1, max_retries + 1):
        try:
            storage = get_storage_client()
            await asyncio.wait_for(storage.ensure_default_buckets(), timeout=10.0)
            # Check if storage is ready (either MinIO or local fallback)
            ready = await storage.wait_for_ready(timeout=5.0)
            if ready:
                mode = "local fallback" if storage._use_local_fallback else "MinIO"
                logger.info(f"✓ Storage ready ({mode}) and buckets initialized")
                return True
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"MinIO not ready (attempt {attempt}/{max_retries}), retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"✗ MinIO unavailable after {max_retries} attempts: {e}")
                # Try local fallback as last resort
                storage = get_storage_client()
                storage._use_local_fallback = True
                try:
                    await storage.ensure_default_buckets()
                    logger.info("✓ Using local storage fallback")
                    return True
                except Exception as fallback_error:
                    logger.error(f"✗ Local fallback also failed: {fallback_error}")
                    return False
    return False


async def startup_dependencies() -> None:
    """
    Initialize dependencies on application startup.
    
    Called during FastAPI startup event.
    Preloads critical models for immediate availability.
    Ensures Redis and MinIO services are ready before proceeding.
    Auto-starts infrastructure services if not running.
    """
    logger.info("="*60)
    logger.info("ARGUS CORE - INITIALIZING SERVICES")
    logger.info("="*60)
    
    # Auto-start infrastructure services
    service_manager = get_service_manager()
    await service_manager.start_all_services()
    
    # Wait for Redis to be available
    logger.info("Checking Redis availability...")
    redis_available = await wait_for_redis()
    if not redis_available:
        logger.error("CRITICAL: Redis is required for application functionality")
        raise RuntimeError("Redis service is not available")
    
    # Wait for MinIO to be available
    logger.info("Checking MinIO availability...")
    minio_available = await wait_for_minio()
    if not minio_available:
        logger.error("CRITICAL: MinIO is required for file storage")
        raise RuntimeError("MinIO service is not available")
    
    # Connect to database
    try:
        await get_db_client()
        logger.info("✓ MongoDB connected")
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        raise RuntimeError(f"MongoDB connection failed: {e}")
    
    logger.info("="*60)
    logger.info("ALL INFRASTRUCTURE SERVICES READY")
    logger.info("="*60)
    
    # Initialize inference engine and warmup critical models
    try:
        engine = get_inference_engine()
        manager = get_model_manager()
        logger.info("Inference engine initialized")
        
        # Skip model warmup if disabled via env
        skip_warmup = os.environ.get("SKIP_MODEL_WARMUP", "").lower() in ("true", "1", "yes")
        if skip_warmup:
            logger.info("SKIP_MODEL_WARMUP=true — skipping model loading")
        else:
            # Warmup critical models for immediate availability
            logger.info("="*60)
            logger.info("DOWNLOADING & LOADING AI MODELS...")
            logger.info("="*60)
            
            critical_models = [
                "deepfake_detector_v3",  # Primary deepfake image detection model
                "retinaface",
                "clip_vit_b16",
                "xclip_temporal"
            ]
            
            warmup_start = asyncio.get_event_loop().time()
            successfully_loaded = 0
            failed_models = []
            
            for model_name in critical_models:
                try:
                    logger.info(f"Loading model: {model_name}...")
                    await engine.warmup_model(model_name)
                    successfully_loaded += 1
                    logger.info(f"  ✓ {model_name} - READY")
                except Exception as e:
                    failed_models.append(model_name)
                    logger.warning(f"  ✗ {model_name} - FAILED: {str(e)[:100]}")
            
            warmup_time = asyncio.get_event_loop().time() - warmup_start
            
            # Log summary
            logger.info("="*60)
            logger.info(f"MODEL LOADING SUMMARY:")
            logger.info(f"  • Successfully loaded: {successfully_loaded}/{len(critical_models)} models")
            logger.info(f"  • Total time: {warmup_time:.2f}s")
            
            if failed_models:
                logger.error(f"  • Failed models: {', '.join(failed_models)}")
                logger.error(f"  • Critical models failed to load - application may not function correctly")
                logger.error(f"  • Ensure model weights are downloaded. Check AUTO_START_CONFIGURATION.md")
            
            # Log VRAM usage
            loaded = manager.get_loaded_models()
            vram_used = manager.get_vram_usage()
            vram_available = manager.get_available_vram()
            
            logger.info(f"VRAM STATUS:")
            logger.info(f"  • Used: {vram_used}MB")
            logger.info(f"  • Available: {vram_available}MB")
            logger.info(f"  • Loaded models: {', '.join(loaded) if loaded else 'None'}")
            logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Inference engine initialization failed: {e}", exc_info=True)
        logger.warning("Application will continue with limited functionality")
    
    logger.info("All dependencies initialized")


async def shutdown_dependencies() -> None:
    """
    Cleanup dependencies on application shutdown.
    
    Called during FastAPI shutdown event.
    """
    logger.info("Shutting down dependencies...")
    
    # Stop Celery worker
    service_manager = get_service_manager()
    service_manager.stop_celery()
    logger.info("Celery worker stopped")
    
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
