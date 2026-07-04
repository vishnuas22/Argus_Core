"""
Argus Core — Stuck Task Reaper
===============================

Background task that finds analyses stuck in STARTED/ANALYZING state
past their time limit and marks them FAILED. Without this, a Celery
worker crash leaves analyses in PENDING/STARTED forever — users see
"analyzing..." indefinitely.

Scheduled via Celery Beat every 5 minutes. Also runnable manually:

    python -m core.stuck_task_reaper

Time limits (configurable via env):
  - PREPROCESSING: 10 min (should take <2 min)
  - ANALYZING:     15 min (should take <6 min, Celery hard limit)
  - AGGREGATING:   10 min (should take <1 min)

The reaper is conservative: it only marks tasks FAILED if they have
been in their current state for >2x the expected duration. This avoids
killing legitimately slow analyses.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import config
from schemas.schemas import AnalysisStatus
from utils.logging import get_logger

logger = get_logger(__name__)


# ============== Configuration ==============
# 2x the Celery hard time limit as the safety margin.
DEFAULT_PREPROCESSING_TIMEOUT_MIN = int(os.environ.get("STUCK_PREPROCESSING_MIN", "20"))
DEFAULT_ANALYZING_TIMEOUT_MIN = int(os.environ.get("STUCK_ANALYZING_MIN", "30"))
DEFAULT_AGGREGATING_TIMEOUT_MIN = int(os.environ.get("STUCK_AGGREGATING_MIN", "20"))

# Only consider tasks updated more than this long ago.
MIN_AGE_MIN = 5


async def reap_stuck_tasks() -> dict:
    """
    Find and fail stuck analyses.

    Returns:
        Summary dict: { "reaped": int, "details": [...] }
    """
    from storage.db import get_db_client

    db = await get_db_client()
    now = datetime.now(timezone.utc)

    reaped = []
    timeout_map = {
        AnalysisStatus.PREPROCESSING.value: DEFAULT_PREPROCESSING_TIMEOUT_MIN,
        AnalysisStatus.ANALYZING.value: DEFAULT_ANALYZING_TIMEOUT_MIN,
        AnalysisStatus.AGGREGATING.value: DEFAULT_AGGREGATING_TIMEOUT_MIN,
    }

    for status_str, timeout_min in timeout_map.items():
        cutoff = now - timedelta(minutes=timeout_min)
        # Find analyses in this state older than the cutoff.
        # `updated_at` is set every time update_status() runs.
        cursor = db._db.analyses.find({
            "status": status_str,
            "updated_at": {"$lt": cutoff},
        })

        async for doc in cursor:
            analysis_id = doc.get("analysis_id", "unknown")
            updated_at = doc.get("updated_at")
            if isinstance(updated_at, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                except ValueError:
                    updated_at = None

            age_min = None
            if updated_at:
                # Make timezone-aware if needed
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                age_min = (now - updated_at).total_seconds() / 60.0

            # Skip if not actually old enough (defensive double-check)
            if age_min is not None and age_min < MIN_AGE_MIN:
                continue

            error_msg = (
                f"Analysis stuck in {status_str} for "
                f"{age_min:.1f} min (limit {timeout_min} min) — "
                f"likely Celery worker crash. Reaped by stuck_task_reaper."
            )

            try:
                await db.update_analysis(analysis_id, {
                    "status": AnalysisStatus.FAILED.value,
                    "error_message": error_msg,
                    "completed_at": now.isoformat(),
                })
                reaped.append({
                    "analysis_id": analysis_id,
                    "was_status": status_str,
                    "age_minutes": round(age_min, 1) if age_min else None,
                })
                logger.warning("Reaped stuck analysis %s: %s", analysis_id, error_msg)
            except Exception as e:
                logger.error(
                    "Failed to reap stuck analysis %s: %s", analysis_id, e,
                )

    summary = {
        "reaped_count": len(reaped),
        "reaped": reaped,
        "checked_at": now.isoformat(),
    }
    if reaped:
        logger.warning("Stuck task reaper: %d analyses failed", len(reaped))
    else:
        logger.debug("Stuck task reaper: no stuck tasks found")
    return summary


# ============== Celery Task Wrapper ==============
# This is what Celery Beat calls on schedule.
def reap_stuck_tasks_task():
    """Sync entry point for Celery Beat."""
    try:
        from core.orchestrator import run_async
        return run_async(reap_stuck_tasks())
    except Exception as e:
        logger.error("Stuck task reaper failed: %s", e)
        return {"error": str(e)}


# ============== Manual Entry Point ==============
if __name__ == "__main__":
    result = asyncio.run(reap_stuck_tasks())
    print(f"Reaped {result['reaped_count']} stuck analyses:")
    for r in result["reaped"]:
        print(f"  - {r['analysis_id']}: was {r['was_status']}, age {r['age_minutes']} min")
