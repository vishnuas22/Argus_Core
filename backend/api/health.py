"""
Argus Core - Health Check Module
=================================
Dedicated health check logic extracted from router.py.
Ensures all component checks are async-safe and use public APIs.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
import asyncio

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

_redis_pool: Optional[Any] = None


async def _get_redis_pool():
    global _redis_pool
    if _redis_pool is None:
        import redis.asyncio as aioredis
        _redis_pool = aioredis.from_url(
            config.redis_url,
            decode_responses=True,
            max_connections=5
        )
    return _redis_pool


async def check_database(db) -> Dict[str, Any]:
    try:
        async with asyncio.timeout(10):
            await db.db.command("ping")
        return {"status": "healthy"}
    except asyncio.TimeoutError:
        logger.warning("Database health check timed out")
        return {"status": "unhealthy: timeout"}
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return {"status": f"unhealthy: {str(e)}"}


async def check_storage(storage) -> Dict[str, Any]:
    try:
        result = await storage.health_check()
        return {
            "status": result.get("status", "healthy"),
            "mode": result.get("mode", "unknown"),
            "latency_ms": result.get("latency_ms"),
            "buckets": result.get("buckets", []),
        }
    except Exception as e:
        logger.warning(f"Storage health check failed: {e}")
        return {"status": f"unhealthy: {str(e)}"}


async def check_redis() -> Dict[str, Any]:
    try:
        client = await _get_redis_pool()
        async with asyncio.timeout(5):
            await client.ping()
        return {"status": "healthy"}
    except asyncio.TimeoutError:
        logger.warning("Redis health check timed out")
        return {"status": "unhealthy: timeout"}
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return {"status": f"unhealthy: {str(e)}"}


async def check_celery() -> Dict[str, Any]:
    try:
        from processing.tasks import celery_app
        loop = asyncio.get_running_loop()
        inspect = celery_app.control.inspect()
        stats = await loop.run_in_executor(None, inspect.stats)
        if stats:
            return {
                "status": "healthy",
                "active_workers": len(stats)
            }
        return {
            "status": "no_workers",
            "active_workers": 0
        }
    except Exception as e:
        logger.warning(f"Celery health check failed: {e}")
        return {"status": f"unhealthy: {str(e)}"}


async def check_models() -> Dict[str, Any]:
    try:
        from models.manager import get_model_manager
        loop = asyncio.get_running_loop()
        manager = get_model_manager()

        loaded_models = await loop.run_in_executor(None, manager.get_loaded_models)
        vram_used = await loop.run_in_executor(None, manager.get_vram_usage)
        vram_available = await loop.run_in_executor(None, manager.get_available_vram)
        model_stats = await loop.run_in_executor(None, manager.get_model_stats)

        if len(loaded_models) >= 3:
            models_status = "healthy"
        elif len(loaded_models) >= 1:
            models_status = "degraded"
        else:
            models_status = "unhealthy"

        details = {}
        for name, stats in model_stats.items():
            details[name] = {
                "vram_mb": stats.get("vram_mb", 0),
                "use_count": stats.get("use_count", 0),
                "age_seconds": round(stats.get("age_seconds", 0), 2),
            }

        return {
            "status": models_status,
            "loaded": len(loaded_models),
            "model_names": list(loaded_models),
            "vram_used_mb": vram_used,
            "vram_available_mb": vram_available,
            "details": details,
        }
    except Exception as e:
        logger.warning(f"Models health check failed: {e}")
        return {"status": f"unhealthy: {str(e)}", "loaded": 0}


async def run_health_check(db, storage) -> Dict[str, Any]:
    db_result = await check_database(db)
    storage_result = await check_storage(storage)
    redis_result = await check_redis()
    celery_result = await check_celery()
    models_result = await check_models()

    components = {
        "database": db_result.pop("status", "unknown"),
        "storage": storage_result,
        "redis": redis_result.pop("status", "unknown"),
        "celery": celery_result,
        "models": models_result,
    }

    # Determine overall status
    unhealthy = []
    for name, status in components.items():
        if isinstance(status, str) and status.startswith("unhealthy"):
            unhealthy.append(name)
        elif isinstance(status, dict) and status.get("status", "").startswith("unhealthy"):
            unhealthy.append(name)

    overall = "degraded" if unhealthy else "healthy"

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": config.api_version,
        "components": components,
        "unhealthy_components": unhealthy if unhealthy else None,
    }
