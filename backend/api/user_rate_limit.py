"""
Argus Core — Per-User Rate Limiting
====================================

Prevents a single authenticated user from monopolizing the analysis
pipeline. The generic IP-based RateLimitMiddleware in middleware.py
protects against anonymous abuse; this module adds authenticated-user
limits on the expensive /api/v1/analyze endpoint.

Limits (configurable via env):
  - FREE_TIER:  10 analyses / hour  (default)
  - PRO_TIER:   100 analyses / hour
  - ADMIN:      unlimited

Implementation: Redis sliding window via INCR + EXPIRE.
Fallback: in-memory dict if Redis unavailable (single-instance only).

Usage in router.py:
    from api.user_rate_limit import check_user_rate_limit
    allowed, retry_after = await check_user_rate_limit(user_id, tier="free")
    if not allowed:
        raise HTTPException(429, ...)
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple
from collections import defaultdict
from threading import Lock

from utils.logging import get_logger

logger = get_logger(__name__)


# ============== Configuration ==============
# Read from env so ops can tune without redeploy.
FREE_TIER_LIMIT = int(os.environ.get("RATE_LIMIT_FREE_PER_HOUR", "10"))
PRO_TIER_LIMIT = int(os.environ.get("RATE_LIMIT_PRO_PER_HOUR", "100"))
# Admin is unlimited.

# Window size in seconds (1 hour).
WINDOW_SECONDS = 3600


# ============== Redis Backend ==============
_redis_client = None
_redis_init_attempted = False


def _get_redis():
    """Lazy-init Redis client. Returns None if unavailable."""
    global _redis_client, _redis_init_attempted
    if _redis_init_attempted:
        return _redis_client
    _redis_init_attempted = True
    try:
        import redis.asyncio as aioredis
        from config import config
        _redis_client = aioredis.from_url(
            config.redis_url,
            decode_responses=True,
            max_connections=5,
        )
        logger.info("Per-user rate limiter connected to Redis")
    except Exception as e:
        logger.warning(
            "Per-user rate limiter: Redis unavailable, using in-memory "
            "(single-instance only): %s", e,
        )
        _redis_client = None
    return _redis_client


# ============== In-Memory Fallback ==============
# Sliding window: { user_id: [(timestamp, ...), ...] }
_local_windows: dict = defaultdict(list)
_local_lock = Lock()


def _check_local(user_id: str, limit: int) -> Tuple[bool, int]:
    """In-memory sliding window check. Returns (allowed, retry_after_seconds)."""
    now = time.time()
    cutoff = now - WINDOW_SECONDS
    with _local_lock:
        # Prune old entries
        _local_windows[user_id] = [
            t for t in _local_windows[user_id] if t > cutoff
        ]
        if len(_local_windows[user_id]) >= limit:
            # Retry after the oldest entry expires
            oldest = _local_windows[user_id][0]
            retry_after = int(oldest + WINDOW_SECONDS - now) + 1
            return False, max(1, retry_after)
        _local_windows[user_id].append(now)
        return True, 0


async def _check_redis(user_id: str, limit: int) -> Tuple[bool, int]:
    """Redis sliding window check via INCR + EXPIRE."""
    redis = _get_redis()
    if redis is None:
        return _check_local(user_id, limit)

    key = f"argus:rate_limit:user:{user_id}"
    try:
        # Atomic INCR; first request sets the key with EXPIRE.
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, WINDOW_SECONDS)
        if count > limit:
            ttl = await redis.ttl(key)
            return False, max(1, ttl)
        return True, 0
    except Exception as e:
        logger.warning(
            "Per-user rate limit Redis check failed, falling back to "
            "in-memory: %s", e,
        )
        return _check_local(user_id, limit)


# ============== Public API ==============

def get_user_limit(tier: str) -> Optional[int]:
    """Return the hourly analysis limit for a tier, or None for unlimited."""
    tier = (tier or "free").lower()
    if tier == "admin":
        return None  # unlimited
    if tier in ("pro", "paid", "premium"):
        return PRO_TIER_LIMIT
    return FREE_TIER_LIMIT


async def check_user_rate_limit(
    user_id: str,
    tier: str = "free",
) -> Tuple[bool, int, int]:
    """
    Check if user can submit another analysis.

    Args:
        user_id: Authenticated user identifier (from JWT).
        tier: User tier ("free", "pro", "admin").

    Returns:
        Tuple of (allowed, retry_after_seconds, limit).
        For admin/unlimited, always (True, 0, -1).
    """
    if not user_id:
        # Anonymous users fall back to IP-based limiter in middleware.
        return True, 0, -1

    limit = get_user_limit(tier)
    if limit is None:
        # Unlimited tier (admin)
        return True, 0, -1

    allowed, retry_after = await _check_redis(user_id, limit)
    return allowed, retry_after, limit


async def get_user_usage(user_id: str) -> int:
    """Return current usage count in the window (for UI display)."""
    redis = _get_redis()
    if redis is None:
        now = time.time()
        cutoff = now - WINDOW_SECONDS
        with _local_lock:
            return sum(1 for t in _local_windows[user_id] if t > cutoff)
    try:
        key = f"argus:rate_limit:user:{user_id}"
        count = await redis.get(key)
        return int(count) if count else 0
    except Exception:
        return 0
