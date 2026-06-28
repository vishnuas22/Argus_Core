"""
Argus Core - FastAPI Application Entry Point
=============================================
Main server application for the Multi-Modal Deepfake Detection Platform.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - server.py

Role: FastAPI application entry point. Initializes app, includes routers,
configures middleware, manages lifecycle events.

Integration:
- Imports: config.py, api/router.py, api/middleware.py, storage/db.py
- Inputs: None (entry point)
- Outputs: FastAPI application instance

Why this approach: Single entry point follows 12-factor app principles.
Lifecycle hooks ensure clean startup/shutdown of database connections and model caches.
"""

import sys
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from api.router import router as api_router
from api.websocket import router as ws_router, startup_websocket, shutdown_websocket
from api.middleware import setup_middleware, get_cors_config
from api.deps import startup_dependencies, shutdown_dependencies
from utils.logging import setup_logging, get_logger
from utils.errors import ArgusError

# Setup structured logging
setup_logging(level=config.log_level, json_format=(config.log_format == "json"))

# Initialize Sentry error monitoring (if configured)
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.celery import CeleryIntegration
    
    sentry_dsn = os.environ.get("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                CeleryIntegration(),
            ],
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            profiles_sample_rate=float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
            environment=os.environ.get("ARGUS_ENV", "production"),
        )
        print("[Sentry] Error monitoring initialized")
except ImportError:
    pass
except Exception as exc:
    print(f"[Sentry] Initialization failed: {exc}")

logger = get_logger(__name__)


# ============== LIFESPAN MANAGEMENT ==============

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events:
    - Startup: Database connection, storage initialization, model warmup
    - Shutdown: Clean resource release
    """
    # ===== STARTUP =====
    logger.info("="*60)
    logger.info("ARGUS CORE - Starting up...")
    logger.info("="*60)
    
    try:
        # Initialize core dependencies
        await startup_dependencies()
        logger.info("Core dependencies initialized")
        
        # Bootstrap ML models (download from HuggingFace if missing)
        try:
            from models.bootstrap import ensure_primary_models
            models_ready = await ensure_primary_models()
            if models_ready:
                logger.info("Primary ML models ready")
            else:
                logger.warning("Primary models unavailable - inference will use fallbacks")
        except Exception as model_exc:
            logger.warning(f"Model bootstrap skipped: {model_exc}")
        
        # Start WebSocket Redis listener
        await startup_websocket()
        logger.info("WebSocket manager started")
        
        # Log configuration
        logger.info(f"Configuration:")
        logger.info(f"  - MongoDB: {config.mongo_url.split('@')[-1] if '@' in config.mongo_url else 'localhost'}")
        logger.info(f"  - MinIO: {config.minio_endpoint}")
        logger.info(f"  - Redis: {config.redis_url}")
        logger.info(f"  - GPU Enabled: {config.use_gpu}")
        logger.info(f"  - VRAM Limit: {config.gpu_memory_limit_mb}MB")
        logger.info(f"  - Max File Size: {config.max_file_size_mb}MB")
        
        logger.info("="*60)
        logger.info("ARGUS CORE - Ready to accept requests")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # ===== SHUTDOWN =====
    logger.info("="*60)
    logger.info("ARGUS CORE - Shutting down...")
    logger.info("="*60)
    
    try:
        # Stop WebSocket manager
        await shutdown_websocket()
        logger.info("WebSocket manager stopped")
        
        # Cleanup dependencies
        await shutdown_dependencies()
        logger.info("Dependencies cleaned up")
        
        logger.info("ARGUS CORE - Shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}", exc_info=True)


# ============== APPLICATION FACTORY ==============

def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=config.api_title,
        description=config.api_description,
        version=config.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        
        # OpenAPI metadata
        openapi_tags=[
            {
                "name": "analysis",
                "description": "Deepfake detection and analysis operations"
            },
            {
                "name": "websocket",
                "description": "Real-time progress updates via WebSocket"
            },
            {
                "name": "system",
                "description": "System health and information endpoints"
            }
        ],
        
        # Contact and license
        contact={
            "name": "Argus Core Team",
            "url": "https://github.com/argus-core",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        }
    )
    
    # Configure middleware
    setup_middleware(app)
    
    # Include routers
    app.include_router(api_router)
    app.include_router(ws_router)

    # Include auth router
    from api.auth import auth_router
    app.include_router(auth_router)
    
    # Add exception handlers
    @app.exception_handler(ArgusError)
    async def argus_error_handler(request: Request, exc: ArgusError):
        """Handle custom Argus exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {}
            }
        )
    
    return app


# ============== ROOT ENDPOINTS ==============

# Create application instance
app = create_app()


@app.get("/", tags=["system"])
async def root():
    """
    Root endpoint.
    
    Returns basic API information.
    """
    return {
        "name": config.api_title,
        "version": config.api_version,
        "description": config.api_description,
        "docs_url": "/docs",
        "health_url": "/api/v1/health"
    }


@app.get("/health", tags=["system"])
async def health():
    """
    Basic health check endpoint.
    
    For detailed health with component status, use /api/v1/health.
    """
    return {"status": "healthy"}


@app.get("/metrics", tags=["system"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns application metrics in Prometheus format.
    """
    from utils.metrics import get_prometheus_metrics
    
    try:
        metrics_text = get_prometheus_metrics()
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=metrics_text,
            media_type="text/plain; version=0.0.4"
        )
    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        return {"error": "Metrics unavailable"}


# ============== DEVELOPMENT SERVER ==============

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
        access_log=True,
        workers=1  # Single worker for development
    )
