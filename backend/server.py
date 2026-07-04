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
        
        # LAZY MODEL LOADING (2026-07-02):
        # Models are NOT downloaded or loaded at startup. They are loaded
        # on first inference call via ModelManager.get_model(). This drops
        # startup time from 30-60s to 2-3s.
        #
        # Optional: background warmup of the most-likely-needed models
        # (retinaface + deepfake_detector_v3) AFTER the server starts
        # accepting requests. This makes the first image analysis fast
        # without blocking startup. Controlled by config.warmup_on_startup.
        if config.download_on_startup:
            # Legacy behavior: download primary models at startup (blocks).
            try:
                from models.bootstrap import ensure_primary_models
                models_ready = await ensure_primary_models()
                if models_ready:
                    logger.info("Primary ML models ready (download_on_startup=true)")
                else:
                    logger.warning("Primary models unavailable - inference will use fallbacks")
            except Exception as model_exc:
                logger.warning(f"Model bootstrap skipped: {model_exc}")
        else:
            logger.info(
                "LAZY MODEL LOADING: models will be downloaded/loaded on "
                "first inference call. Startup is fast; first analysis "
                "per modality takes a one-time load hit (~3-10s)."
            )
        
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
        logger.info(f"  - Model download_on_startup: {config.download_on_startup}")
        logger.info(f"  - Model warmup_on_startup: {config.warmup_on_startup}")
        
        logger.info("="*60)
        logger.info("ARGUS CORE - Ready to accept requests")
        logger.info("="*60)

        # Background warmup (non-blocking): pre-load the most-likely-needed
        # models AFTER the server is accepting requests. First image
        # analysis will be fast; first audio/video analysis still takes a
        # one-time load hit for those modality-specific models.
        if config.warmup_on_startup and not config.download_on_startup:
            import asyncio
            import threading

            async def _background_warmup():
                """Pre-load primary models in the background."""
                try:
                    await asyncio.sleep(2.0)  # Let the server settle first
                    logger.info("Background warmup: loading retinaface + deepfake_detector_v3...")
                    from core.engine import get_inference_engine
                    engine = get_inference_engine()
                    for model_name in ["retinaface", "deepfake_detector_v3"]:
                        try:
                            await engine.warmup_model(model_name)
                            logger.info(f"Background warmup: {model_name} loaded")
                        except Exception as e:
                            logger.debug(f"Background warmup: {model_name} failed (will lazy-load on first use): {e}")
                except Exception as e:
                    logger.debug(f"Background warmup skipped: {e}")

            # Fire-and-forget — don't block the event loop
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(_background_warmup())
            except Exception:
                pass  # Warmup is best-effort; never block startup

        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # ===== SHUTDOWN =====
    logger.info("="*60)
    logger.info("ARGUS CORE - Shutting down (draining in-flight requests)...")
    logger.info("="*60)
    
    # Graceful drain: wait up to 25s for in-flight requests to complete.
    # The docker stop_grace_period is 30s, so we leave 5s for cleanup.
    # This prevents losing in-flight analyses on deploy/restart.
    import asyncio
    drain_timeout = 25.0
    drain_start = asyncio.get_event_loop().time()
    
    # Signal new requests to fail fast (503) during drain
    app.state.draining = True
    logger.info("Marked app as draining — new requests will get 503")
    
    # Wait for in-flight requests to complete (best-effort)
    # We don't track active request count precisely, so we just sleep
    # to allow the reverse proxy to stop sending new traffic and for
    # in-flight requests to finish.
    elapsed = 0.0
    while elapsed < drain_timeout:
        await asyncio.sleep(1.0)
        elapsed = asyncio.get_event_loop().time() - drain_start
        if int(elapsed) % 5 == 0:
            logger.info("Draining... %d/%d seconds", int(elapsed), int(drain_timeout))
    
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
    
    # ============== SYSTEM ENDPOINTS ==============

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
        For the full operational dashboard, use /health/detailed.
        """
        return {"status": "healthy"}

    @app.get("/health/detailed", tags=["system"])
    async def health_detailed():
        """
        Iteration 7: Detailed health endpoint surfacing the operational state
        of every Iteration 1-6 subsystem.

        Returns:
            JSON with per-subsystem status:
            - drift: PSI/MMD/severity per modality
            - retrain: last cycle status per modality
            - ab_test: candidate status per modality
            - calibration: ECE/Brier/temperature per modality
            - feedback: buffer size per modality
            - models: which detectors are loaded
            - defenses: adversarial defense counters
            - certified_robustness: certification counters
        """
        import time as _time
        from datetime import datetime, timezone

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "healthy",
            "subsystems": {},
        }

        # --- Drift status ---
        drift_status = {}
        try:
            from monitoring.drift_detector import get_default_drift_detector
            from monitoring.reference_store import get_default_reference_store
            ref = get_default_reference_store()
            for modality in ["image", "audio", "video"]:
                if ref.embeddings is not None and ref.modality == modality:
                    drift_status[modality] = {
                        "reference_loaded": True,
                        "num_reference_samples": ref.num_samples,
                        "reference_created_at": ref.created_at,
                    }
                else:
                    drift_status[modality] = {
                        "reference_loaded": False,
                        "message": "no reference distribution loaded",
                    }
        except Exception as e:
            drift_status["error"] = str(e)
        result["subsystems"]["drift"] = drift_status

        # --- Retrain status ---
        retrain_status = {}
        try:
            from continuous_learning.feedback_buffer import get_default_feedback_buffer
            buf = get_default_feedback_buffer()
            from config import config as _cfg
            for modality in ["image", "audio", "video"]:
                count = buf.count(modality=modality)
                min_samples = getattr(_cfg, "retrain_min_samples", 50)
                retrain_status[modality] = {
                    "feedback_samples": count,
                    "min_samples_for_retrain": min_samples,
                    "ready_for_retrain": count >= min_samples,
                    "schedule_hours": getattr(_cfg, "retrain_schedule_hours", 24.0),
                }
        except Exception as e:
            retrain_status["error"] = str(e)
        result["subsystems"]["retrain"] = retrain_status

        # --- A/B test status ---
        ab_status = {}
        try:
            from continuous_learning.ab_test import get_default_ab_router
            router = get_default_ab_router()
            for modality in ["image", "audio", "video"]:
                evaluation = router.evaluate_candidate(modality)
                ab_status[modality] = evaluation
        except Exception as e:
            ab_status["error"] = str(e)
        result["subsystems"]["ab_test"] = ab_status

        # --- Calibration status ---
        cal_status = {}
        try:
            from config import config as _cfg
            for modality in ["image", "audio", "video"]:
                ts_path = getattr(_cfg, "temperature_scaler_path", "")
                cp_path = getattr(_cfg, "conformal_raps_path", "")
                cal_status[modality] = {
                    "temperature_scaler_loaded": bool(ts_path and os.path.exists(ts_path)),
                    "conformal_raps_loaded": bool(cp_path and os.path.exists(cp_path)),
                    "temperature_scaler_path": ts_path,
                    "conformal_raps_path": cp_path,
                }
        except Exception as e:
            cal_status["error"] = str(e)
        result["subsystems"]["calibration"] = cal_status

        # --- Feedback buffer ---
        feedback_status = {}
        try:
            from continuous_learning.feedback_buffer import get_default_feedback_buffer
            buf = get_default_feedback_buffer()
            feedback_status["total"] = buf.count()
            feedback_status["by_modality"] = {
                m: buf.count(modality=m) for m in ["image", "audio", "video"]
            }
        except Exception as e:
            feedback_status["error"] = str(e)
        result["subsystems"]["feedback"] = feedback_status

        # --- Embedding buffer ---
        emb_status = {}
        try:
            from monitoring.embedding_buffer import get_default_embedding_buffer
            emb = get_default_embedding_buffer()
            if emb:
                emb_status = emb.counts_all()
            else:
                emb_status = {"error": "Redis not connected"}
        except Exception as e:
            emb_status = {"error": str(e)}
        result["subsystems"]["embedding_buffer"] = emb_status

        # --- Model loading status ---
        models_status = {}
        try:
            from models.registry import DEFAULT_MODELS
            sota_keys = [
                "clip_image_detector", "dinov2_image_detector", "siglip_image_detector",
                "aasist3_audio_detector", "wav2vec2_xls_r_audio_detector",
                "ecapa_audio_detector",
                "videomae_video_detector", "altfree_video_detector",
                "timesformer_video_detector",
            ]
            import os as _os
            for key in sota_keys:
                if key in DEFAULT_MODELS:
                    meta = DEFAULT_MODELS[key]
                    # Check if the model directory exists
                    model_path = meta.path
                    path_exists = _os.path.exists(model_path) if model_path else False
                    models_status[key] = {
                        "path": model_path,
                        "path_exists": path_exists,
                        "vram_mb": meta.vram_mb,
                        "category": meta.category.value if hasattr(meta.category, 'value') else str(meta.category),
                        "license": meta.license,
                    }
        except Exception as e:
            models_status["error"] = str(e)
        result["subsystems"]["models"] = models_status

        # --- Defense flags ---
        defense_status = {}
        try:
            from config import config as _cfg
            defense_status["rps_enabled"] = getattr(_cfg, "enable_rps", False)
            defense_status["adversarial_gate_enabled"] = getattr(_cfg, "enable_adversarial_gate", False)
            defense_status["rs_lite_enabled"] = getattr(_cfg, "enable_rs_lite", False)
            defense_status["certified_robustness_enabled"] = getattr(_cfg, "enable_certified_robustness", False)
        except Exception as e:
            defense_status["error"] = str(e)
        result["subsystems"]["defenses"] = defense_status

        # --- Continuous learning config ---
        cl_status = {}
        try:
            from config import config as _cfg
            cl_status["enabled"] = getattr(_cfg, "enable_continuous_learning", False)
            cl_status["retrain_min_samples"] = getattr(_cfg, "retrain_min_samples", 50)
            cl_status["retrain_schedule_hours"] = getattr(_cfg, "retrain_schedule_hours", 24.0)
            cl_status["ab_test_ratio"] = getattr(_cfg, "retrain_ab_test_ratio", 0.1)
        except Exception as e:
            cl_status["error"] = str(e)
        result["subsystems"]["continuous_learning"] = cl_status

        return result

    @app.get("/metrics", tags=["system"])
    async def metrics():
        """
        Prometheus metrics endpoint.

        Returns all application metrics in Prometheus format.
        Both Iteration 6 observability metrics and legacy metrics
        are registered in the default prometheus_client registry.
        """
        from fastapi.responses import PlainTextResponse

        try:
            from prometheus_client import generate_latest
            output = generate_latest()
            return PlainTextResponse(
                content=output.decode() if isinstance(output, bytes) else output,
                media_type="text/plain; version=0.0.4"
            )
        except Exception as e:
            logger.debug(f"Metrics generation failed: {e}")
            return {"error": "Metrics unavailable"}

    return app


# ============== PRODUCTION INSTANCE ==============

app = create_app()


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
