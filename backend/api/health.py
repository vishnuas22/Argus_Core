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
    """Run all component health checks in parallel for low latency.

    Each check is independent and uses ``asyncio.timeout`` so a single
    slow component cannot stall the overall health endpoint. Results are
    returned verbatim — we do NOT mutate them via ``.pop("status")``
    (the previous implementation did, which silently dropped the
    latency/buckets metadata from storage and the active_workers count
    from celery).
    """
    # Run all checks concurrently. ``asyncio.gather`` returns results in
    # the same order as the input awaitables, so we can unpack positionally.
    db_result, storage_result, redis_result, celery_result, models_result = (
        await asyncio.gather(
            check_database(db),
            check_storage(storage),
            check_redis(),
            check_celery(),
            check_models(),
        )
    )

    components = {
        "database": db_result,
        "storage": storage_result,
        "redis": redis_result,
        "celery": celery_result,
        "models": models_result,
    }

    # Determine overall status: unhealthy if ANY component is unhealthy,
    # degraded if any is "degraded" or "no_workers", otherwise healthy.
    unhealthy = []
    degraded = []
    for name, result in components.items():
        # Status is either a top-level string (rare) or nested in a dict.
        if isinstance(result, str):
            status_str = result
        else:
            status_str = result.get("status", "unknown")
        if status_str.startswith("unhealthy"):
            unhealthy.append(name)
        elif status_str in ("degraded", "no_workers"):
            degraded.append(name)

    if unhealthy:
        overall = "unhealthy"
    elif degraded:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": config.api_version,
        "components": components,
        "unhealthy_components": unhealthy if unhealthy else None,
        "degraded_components": degraded if degraded else None,
    }
