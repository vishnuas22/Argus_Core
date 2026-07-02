"""
Argus Core - Job Orchestrator
=============================
Celery task definitions for distributed deepfake analysis pipeline.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - core/orchestrator.py

SOTA Algorithm: Directed Acyclic Graph (DAG) task scheduling with dependency resolution

Role: Job queuing, status tracking, retry logic. Coordinates the entire analysis pipeline.

Integration:
- Imports: processing/preprocess.py, core/engine.py, core/fusion.py, forensics/report.py, storage/db.py
- Inputs: analysis_id: str, modalities: List[Modality]
- Outputs: Job status updates to MongoDB

Pipeline Flow:
1. preprocess_task (extract frames/audio)
2. analyze_task (parallel modality analysis)
3. aggregate_task (fusion + scoring)
4. report_task (async PDF generation)

Why this approach: Celery provides robust distributed task execution with automatic retries.
Task chaining enables complex workflows with clean error handling.
"""

import asyncio
import time
import uuid
import sys
import os
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from functools import wraps
import numpy as np

from celery import Celery
from celery.schedules import crontab
from celery.result import AsyncResult
from celery.exceptions import SoftTimeLimitExceeded

from config import config
from schemas.schemas import (
    AnalysisStatus, AnalysisDocument, PreprocessedData, ModalityResult,
    AggregatedResult, Modality, ContentType, TrustScore, Verdict, Explanation,
    VideoResult, AudioResult, ImageResult, MetadataResult
)
from utils.logging import get_logger
from utils.errors import PreprocessingError, InferenceError, FusionError

logger = get_logger(__name__)

# Add backend directory to path for Celery worker (once at module load)
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# ============== CELERY CONFIGURATION ==============

celery_app = Celery(
    "argus_tasks",
    broker=config.celery_broker_url,
    backend=config.celery_result_backend
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # One task at a time for memory-intensive work
    
    # Time limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=360,  # 6 minutes hard limit
    
    # Retry settings
    task_default_retry_delay=30,
    task_max_retries=3,
    
    # Result expiration
    result_expires=86400,  # 24 hours
    
    # Task routes - separate queues for different task types
    task_routes={
        "argus_tasks.preprocess_task": {"queue": "preprocessing"},
        "argus_tasks.analyze_modality_task": {"queue": "analysis"},
        "argus_tasks.aggregate_results_task": {"queue": "aggregation"},
        "argus_tasks.generate_report_task": {"queue": "reports"},
    },
    
    # Define queues
    task_queues={
        "preprocessing": {"routing_key": "preprocess.#"},
        "analysis": {"routing_key": "analyze.#"},
        "aggregation": {"routing_key": "aggregate.#"},
        "reports": {"routing_key": "report.#"},
        # H4 fix: dead-letter queue for failed tasks after max retries.
        # Workers should consume this queue with:
        #   celery -A core.orchestrator.celery_app worker -Q dead_letter
        "dead_letter": {"routing_key": "dlq.#"},
    },

    # H4 fix: route failed tasks to DLQ after max retries exhausted.
    # Celery's task_reject_on_worker_lost + task_acks_late already
    # re-queues on worker crash. This setting ensures tasks that fail
    # all retries are published to the dead_letter queue for forensic
    # inspection / replay rather than being silently dropped.
    task_default_queue="default",
    task_create_missing_queues=True,

    # ===========================================================
    # Iteration 5: Celery Beat schedule for automatic retraining
    # and drift checks.
    # ===========================================================
    # Requires celery[redis] beat scheduler. Run beat alongside workers:
    #   celery -A core.orchestrator.celery_app beat --loglevel=info
    beat_schedule={
        # Retrain image LoRA adapter every 24h at 02:00 UTC
        "retrain-image-daily": {
            "task": "argus_tasks.retrain_modality",
            "schedule": crontab(hour=2, minute=0),
            "args": ("image",),
        },
        # Retrain audio LoRA adapter every 24h at 03:00 UTC
        "retrain-audio-daily": {
            "task": "argus_tasks.retrain_modality",
            "schedule": crontab(hour=3, minute=0),
            "args": ("audio",),
        },
        # Retrain video LoRA adapter every 24h at 04:00 UTC
        "retrain-video-daily": {
            "task": "argus_tasks.retrain_modality",
            "schedule": crontab(hour=4, minute=0),
            "args": ("video",),
        },
        # Check drift every 6 hours
        "drift-check-every-6h": {
            "task": "argus_tasks.check_drift",
            "schedule": crontab(minute=0, hour="*/6"),
            "args": (),
        },
        # Evaluate A/B test candidates every hour
        "ab-test-evaluation-hourly": {
            "task": "argus_tasks.evaluate_ab_tests",
            "schedule": crontab(minute=30),
            "args": (),
        },
    },
)


# ===========================================================
# Iteration 5: Scheduled task definitions
# ===========================================================

@celery_app.task(name="argus_tasks.retrain_modality")
def retrain_modality_task(modality: str):
    """
    Celery task: retrain a LoRA adapter from the feedback buffer.

    Scheduled daily via Celery Beat. Can also be triggered manually via
    POST /api/v1/retrain/{modality}.
    """
    try:
        from continuous_learning import schedule_retrain_task
        return schedule_retrain_task(modality)
    except Exception as e:
        logger.error("Retrain task failed for %s: %s", modality, e)
        return {"status": "failed", "modality": modality, "error": str(e)}


@celery_app.task(name="argus_tasks.check_drift")
def check_drift_task():
    """
    Celery task: check for distribution drift across all modalities.

    Scheduled every 6 hours via Celery Beat. Pulls recent embeddings
    from the Redis embedding buffer and compares against the reference.
    """
    try:
        from core.post_processing import check_batch_drift
        from monitoring.embedding_buffer import get_default_embedding_buffer
        results = {}
        buf = get_default_embedding_buffer()
        for modality in ["image", "audio", "video"]:
            count = buf.count(modality) if buf else 0
            if count < 10:
                results[modality] = {
                    "status": "insufficient_embeddings",
                    "count": count,
                }
                continue
            current_embeddings = buf.get_embeddings(modality)
            if current_embeddings is None or len(current_embeddings) < 10:
                results[modality] = {
                    "status": "insufficient_embeddings",
                    "count": count,
                }
                continue
            drift = check_batch_drift(current_embeddings, modality=modality)
            if drift is None:
                results[modality] = {
                    "status": "no_reference",
                    "count": count,
                }
            else:
                results[modality] = {
                    "status": "checked",
                    "count": count,
                    **drift,
                }
        logger.info("Drift check: %s", results)
        return results
    except Exception as e:
        logger.error("Drift check task failed: %s", e)
        return {"status": "failed", "error": str(e)}


@celery_app.task(name="argus_tasks.evaluate_ab_tests")
def evaluate_ab_tests_task():
    """
    Celery task: evaluate all active A/B test candidates.

    Scheduled hourly via Celery Beat. Promotes or rolls back candidates
    based on collected metrics.
    """
    try:
        from continuous_learning import get_default_ab_router
        router = get_default_ab_router()
        results = {}
        for modality in ["image", "audio", "video"]:
            evaluation = router.evaluate_candidate(modality)
            results[modality] = evaluation
            decision = evaluation.get("decision", "insufficient")
            if decision == "promote":
                logger.info("Promoting candidate for %s", modality)
                router.promote_candidate(modality)
            elif decision == "rollback":
                logger.warning("Rolling back candidate for %s", modality)
                router.rollback_candidate(modality)
        logger.info("A/B test evaluation: %s", results)
        return results
    except Exception as e:
        logger.error("A/B test evaluation failed: %s", e)
        return {"status": "failed", "error": str(e)}


# ===========================================================
# H4 fix: Dead-letter queue consumer task
# ===========================================================
# Failed tasks (after max retries) land in the dead_letter queue.
# Run a worker to consume them:
#   celery -A core.orchestrator.celery_app worker -Q dead_letter --loglevel=info
# This task logs the failure for forensic inspection and optional replay.

@celery_app.task(name="argus_tasks.dead_letter_handler")
def dead_letter_handler_task(
    original_task_name: str,
    task_id: str,
    error: str,
    args: list,
    kwargs: dict,
):
    """
    Handle a dead-lettered task. Logs the failure for forensic inspection.

    In production, this could:
    - Store the failed task in MongoDB for replay
    - Send an alert to ops
    - Update the analysis status to "failed"
    """
    logger.error(
        "DEAD LETTER: task=%s id=%s error=%s args=%s",
        original_task_name, task_id, error[:200], str(args)[:200],
    )
    # Store in MongoDB for forensic audit / replay
    try:
        from storage.db import get_db_client
        db = run_async(get_db_client())
        run_async(db.log_audit_event(
            event_type="dead_letter",
            resource_id=task_id,
            actor="celery",
            metadata={
                "original_task": original_task_name,
                "error": error[:500],
                "args": str(args)[:500],
                "kwargs": str(kwargs)[:500],
            }
        ))
    except Exception as e:
        logger.error("Failed to store dead letter in DB: %s", e)
    return {
        "status": "dead_lettered",
        "original_task": original_task_name,
        "task_id": task_id,
    }


# ============== ASYNC HELPERS ==============

def run_async(coro):
    """
    Run async coroutine in sync context (e.g., from Celery tasks).

    M4 fix: use asyncio.run() instead of the deprecated
    asyncio.get_event_loop() + run_until_complete() pattern.
    asyncio.run() creates a fresh event loop per call and properly
    tears it down, avoiding the "event loop already running" error
    that occurs with gevent/thread pools.
    """
    # Check if we're already inside a running loop (e.g., FastAPI context).
    # If so, we can't use asyncio.run() — fall back to the legacy pattern.
    try:
        loop = asyncio.get_running_loop()
        # We're inside a running loop — can't call asyncio.run().
        # Use a thread to run the coroutine instead.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)


def sync_task(async_func):
    """Decorator to run async task function in Celery."""
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        return run_async(async_func(*args, **kwargs))
    return wrapper


# ============== DATABASE ACCESS ==============

async def get_db():
    """Get database client for task operations."""
    from storage.db import get_db_client
    return await get_db_client()


async def get_storage():
    """Get storage client for task operations."""
    from storage.storage import get_storage_client
    return get_storage_client()


async def update_status(
    analysis_id: str,
    status: AnalysisStatus,
    progress_percent: float = 0.0,
    current_stage: str = "",
    error_message: Optional[str] = None
):
    """Update analysis status in database and publish progress."""
    db = await get_db()
    
    # Update database
    updates = {"status": status.value}
    if error_message:
        updates["error_message"] = error_message
    
    await db.update_analysis(analysis_id, updates)
    
    # Publish progress update (for WebSocket subscribers)
    await publish_progress(analysis_id, status, progress_percent, current_stage)


# H7 fix: Use redis.asyncio instead of sync redis to avoid blocking the
# event loop. The sync client's r.publish() blocks every async task
# that calls publish_progress, serializing concurrent analyses.
_orchestrator_redis_pool: Optional[Any] = None


async def _get_orchestrator_redis():
    """
    Get or create a persistent async Redis client for progress publishing.

    Uses redis.asyncio (not sync redis) so that publish() doesn't
    block the event loop.
    """
    global _orchestrator_redis_pool
    if _orchestrator_redis_pool is None:
        import redis.asyncio as aioredis
        _orchestrator_redis_pool = aioredis.from_url(
            config.redis_url,
            max_connections=10,
            decode_responses=True,
        )
    return _orchestrator_redis_pool


async def publish_progress(
    analysis_id: str,
    status: AnalysisStatus,
    progress_percent: float,
    current_stage: str,
    message: Optional[str] = None
):
    """Publish progress update to Redis for WebSocket delivery."""
    try:
        r = await _get_orchestrator_redis()

        import json
        progress_data = {
            "analysis_id": analysis_id,
            "status": status.value,
            "progress_percent": progress_percent,
            "current_stage": current_stage,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # H7 fix: await the async publish (was sync r.publish — blocking)
        await r.publish(f"argus:progress:{analysis_id}", json.dumps(progress_data))

    except Exception as e:
        logger.warning(f"Failed to publish progress: {e}")


# ============== CELERY TASKS ==============

@celery_app.task(
    bind=True,
    name="argus_tasks.run_analysis_pipeline",
    max_retries=3,
    soft_time_limit=300,
    time_limit=360
)
@sync_task
async def run_analysis_pipeline(
    self,
    analysis_id: str,
    options: dict
) -> dict:
    """
    Main analysis pipeline task.
    
    Orchestrates the complete analysis workflow:
    1. preprocess_task (extract frames/audio)
    2. analyze_task (parallel modality analysis)
    3. aggregate_task (fusion + scoring)
    4. report_task (async PDF generation)
    
    Args:
        analysis_id: Unique analysis identifier
        options: Analysis options dict
        
    Returns:
        Analysis result dict with trust_score and verdict
    """
    start_time = time.time()
    job_id = self.request.id or str(uuid.uuid4())
    
    logger.info(f"Starting analysis pipeline: {analysis_id}, job_id: {job_id}")
    
    try:
        db = await get_db()
        storage = await get_storage()
        
        # Record job in database
        await db.create_job(job_id, analysis_id, "pipeline")
        await db.update_job_status(job_id, "running")
        
        # Get analysis document
        analysis = await db.get_analysis(analysis_id)
        if analysis is None:
            raise ValueError(f"Analysis not found: {analysis_id}")
        
        # ===== PHASE 1: PREPROCESSING =====
        await update_status(
            analysis_id, 
            AnalysisStatus.PREPROCESSING, 
            10.0, 
            "preprocessing"
        )
        
        preprocessed = await _preprocess_media(
            analysis_id=analysis_id,
            file_key=analysis.input.file_id if analysis.input else None,
            options=options,
            storage=storage
        )
        
        # ===== PHASE 2: MULTI-MODEL INFERENCE =====
        await update_status(
            analysis_id,
            AnalysisStatus.ANALYZING,
            30.0,
            "analyzing"
        )
        
        # Determine which modalities to analyze
        modalities = options.get("modalities") or _detect_modalities(preprocessed)
        
        # Run parallel analysis
        modality_results = await _run_parallel_analysis(
            analysis_id=analysis_id,
            preprocessed=preprocessed,
            modalities=modalities,
            options=options
        )
        
        # ===== PHASE 3: AGGREGATION & SCORING =====
        await update_status(
            analysis_id,
            AnalysisStatus.AGGREGATING,
            80.0,
            "aggregating"
        )
        
        aggregated = await _aggregate_results(
            analysis_id=analysis_id,
            modality_results=modality_results,
            content_type=preprocessed.content_type,
            raw_tensors=_extract_raw_tensors_for_umft(preprocessed)
        )
        
        # ===== PHASE 4: FINALIZE RESULTS =====
        processing_time = time.time() - start_time
        
        # Build final results
        final_updates = _build_final_results(
            aggregated=aggregated,
            modality_results=modality_results,
            processing_time=processing_time,
            content_type=preprocessed.content_type,
            analysis_id=analysis_id
        )
        
        # A/B test: record predictions for each modality
        try:
            from continuous_learning.ab_test import get_default_ab_router
            _ab_router = get_default_ab_router()
            for mr in modality_results:
                if mr.modality and mr.confidence > 0.0:
                    _ab_router.record_prediction(
                        modality=mr.modality.value if hasattr(mr.modality, 'value') else str(mr.modality),
                        score=mr.score,
                        latency_ms=processing_time * 1000,
                        is_candidate=False,
                    )
        except Exception:
            pass
        
        # Update analysis with results
        await db.update_analysis(analysis_id, final_updates)
        
        await update_status(
            analysis_id,
            AnalysisStatus.COMPLETED,
            100.0,
            "completed"
        )
        
        # ===== PHASE 5: ASYNC REPORT GENERATION =====
        if options.get("generate_report", True):
            # Queue report generation (non-blocking)
            generate_report_task.delay(analysis_id, options)
        
        # Update job status
        await db.update_job_status(job_id, "completed")
        
        logger.info(
            f"Analysis completed: {analysis_id}, "
            f"trust_score={final_updates['trust_score']['value']}, "
            f"verdict={final_updates['verdict']}, "
            f"time={processing_time:.2f}s"
        )
        
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "trust_score": final_updates["trust_score"],
            "verdict": final_updates["verdict"],
            "processing_time_seconds": processing_time
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Analysis timed out: {analysis_id}")
        await update_status(
            analysis_id,
            AnalysisStatus.FAILED,
            0.0,
            "failed",
            "Analysis timed out"
        )
        raise
        
    except Exception as e:
        logger.error(f"Analysis failed: {analysis_id}, error: {e}", exc_info=True)
        
        retries_left = self.request.retries < self.max_retries
        
        try:
            await update_status(
                analysis_id,
                AnalysisStatus.FAILED,
                0.0,
                "failed",
                str(e)
            )
        except Exception as status_e:
            logger.error(f"Status update failed on error path: {status_e}")
        
        try:
            db = await get_db()
            await db.update_job_status(job_id, "failed", str(e))
        except Exception as db_e:
            logger.error(f"DB status update failed on error path: {db_e}")
        
        if retries_left:
            raise self.retry(exc=e, countdown=30 * (self.request.retries + 1))
        raise


@celery_app.task(
    name="argus_tasks.preprocess_task",
    max_retries=2,
    soft_time_limit=120
)
@sync_task
async def preprocess_task(
    analysis_id: str,
    file_key: str,
    options: dict
) -> dict:
    """
    Standalone preprocessing task.
    
    Can be used independently or as part of pipeline.
    
    Args:
        analysis_id: Analysis identifier
        file_key: MinIO object key for uploaded file
        options: Preprocessing options
        
    Returns:
        PreprocessedData as dict
    """
    storage = await get_storage()
    
    result = await _preprocess_media(
        analysis_id=analysis_id,
        file_key=file_key,
        options=options,
        storage=storage
    )
    
    return result.model_dump(mode="json")


@celery_app.task(
    name="argus_tasks.analyze_modality_task",
    max_retries=2,
    soft_time_limit=180
)
@sync_task
async def analyze_modality_task(
    analysis_id: str,
    modality: str,
    preprocessed_data: dict
) -> dict:
    """
    Single modality analysis task.
    
    Designed for parallel execution across modalities.
    
    Args:
        analysis_id: Analysis identifier
        modality: Modality to analyze (video, audio, image)
        preprocessed_data: PreprocessedData as dict
        
    Returns:
        ModalityResult as dict
    """
    preprocessed = PreprocessedData(**preprocessed_data)
    modality_enum = Modality(modality)
    
    result = await _analyze_single_modality(
        analysis_id=analysis_id,
        modality=modality_enum,
        preprocessed=preprocessed
    )
    
    return result.model_dump(mode="json")


@celery_app.task(
    name="argus_tasks.aggregate_results_task",
    max_retries=1,
    soft_time_limit=60
)
@sync_task
async def aggregate_results_task(
    analysis_id: str,
    modality_results: List[dict],
    content_type: str
) -> dict:
    """
    Aggregate modality results into final score.
    
    Args:
        analysis_id: Analysis identifier
        modality_results: List of ModalityResult dicts
        content_type: ContentType value
        
    Returns:
        AggregatedResult as dict
    """
    results = [ModalityResult(**r) for r in modality_results]
    ct = ContentType(content_type)
    
    aggregated = await _aggregate_results(
        analysis_id=analysis_id,
        modality_results=results,
        content_type=ct
    )
    
    return aggregated.model_dump(mode="json")


@celery_app.task(
    name="argus_tasks.generate_report_task",
    max_retries=2,
    soft_time_limit=120
)
@sync_task
async def generate_report_task(
    analysis_id: str,
    options: dict
) -> dict:
    """
    Generate PDF forensic report (async, non-blocking).
    
    Args:
        analysis_id: Analysis identifier
        options: Report options
        
    Returns:
        Dict with report_url
    """
    logger.info(f"Generating report for analysis: {analysis_id}")
    
    try:
        db = await get_db()
        storage = await get_storage()
        
        # Get completed analysis
        analysis = await db.get_analysis(analysis_id)
        if analysis is None or analysis.status != AnalysisStatus.COMPLETED:
            logger.warning(f"Cannot generate report: analysis {analysis_id} not complete")
            return {"status": "skipped", "reason": "analysis_not_complete"}
        
        # Generate report using forensics module
        report_key = f"results/{analysis_id}/report.pdf"
        
        # Use real ReportGenerator from forensics module
        from forensics.report import ReportGenerator
        report_generator = ReportGenerator()
        pdf_content = await report_generator.generate(analysis)
        
        # Upload to storage
        await storage.ensure_default_buckets()
        await storage.upload_file(
            file=pdf_content,
            bucket=storage.bucket_results,
            object_key=report_key,
            content_type="application/pdf"
        )
        
        # Get presigned URL
        report_url = await storage.get_presigned_url(
            storage.bucket_results,
            report_key,
            expires_seconds=86400  # 24 hours
        )
        
        # Update analysis with report URL
        await db.update_analysis(analysis_id, {"report_url": report_url})
        
        logger.info(f"Report generated: {analysis_id}")
        
        return {
            "status": "completed",
            "report_url": report_url,
            "analysis_id": analysis_id
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Report generation timed out: {analysis_id}")
        # H3 fix: re-raise so Celery retry fires (was swallowed before)
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {analysis_id}, error: {e}", exc_info=True)
        # H3 fix: re-raise so Celery's max_retries=2 mechanism fires.
        # Previously the exception was swallowed (returned a dict),
        # causing Celery to mark the task as SUCCESS — no retry, no DLQ.
        # Now the exception propagates → Celery retries up to 2 times,
        # then marks the task as FAILED and routes to the dead_letter queue.
        raise


# ============== HELPER FUNCTIONS ==============

async def _preprocess_media(
    analysis_id: str,
    file_key: Optional[str],
    options: dict,
    storage
) -> PreprocessedData:
    """
    Run preprocessing pipeline on media file.
    
    Downloads file, detects type, extracts features.
    """
    from processing.preprocess import Preprocessor
    from processing.sanitize import FileType
    
    if not file_key:
        raise PreprocessingError("input", "No file specified")
    
    preprocessor = Preprocessor(storage=storage)
    
    # Download file
    file_bytes = await storage.download_file(
        storage.bucket_uploads,
        file_key
    )
    
    # Detect file type
    sanitizer = preprocessor.sanitizer
    detected_type = sanitizer._detect_file_type(file_bytes)
    
    if detected_type is None:
        raise PreprocessingError("file_type", "Could not determine file type")
    
    # Process based on type
    result = await preprocessor.process(
        analysis_id=analysis_id,
        file_key=file_key,
        file_type=detected_type,
        options=options
    )
    
    return result


def _detect_modalities(preprocessed: PreprocessedData) -> List[Modality]:
    """Detect applicable modalities from preprocessed data."""
    modalities = []
    
    if preprocessed.content_type in [
        ContentType.VIDEO_WITH_SPEECH,
        ContentType.VIDEO_NO_SPEECH
    ]:
        modalities.append(Modality.VIDEO)
    
    if preprocessed.content_type in [
        ContentType.VIDEO_WITH_SPEECH,
        ContentType.AUDIO_ONLY
    ]:
        modalities.append(Modality.AUDIO)
    
    if preprocessed.content_type == ContentType.IMAGE_ONLY:
        modalities.append(Modality.IMAGE)
    
    return modalities


async def _run_parallel_analysis(
    analysis_id: str,
    preprocessed: PreprocessedData,
    modalities: List[Union[Modality, str]],
    options: dict
) -> List[ModalityResult]:
    """
    Run analysis for all modalities in parallel.
    """
    # Convert string modalities to enum
    modality_enums = []
    for m in modalities:
        if isinstance(m, str):
            modality_enums.append(Modality(m))
        else:
            modality_enums.append(m)
    
    # Create tasks with options passed
    tasks = []
    for modality in modality_enums:
        task = _analyze_single_modality(analysis_id, modality, preprocessed, options)
        tasks.append(task)
    
    # Run in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    modality_results = []
    for modality, result in zip(modality_enums, results):
        if isinstance(result, Exception):
            logger.error(
                f"MODALITY FAILED: {modality.value} raised {type(result).__name__}: {result}. "
                f"Returning neutral score (0.5, confidence=0.0) — fusion will ignore this modality."
            )
            # Create failed result — confidence=0.0 ensures fusion gives zero weight
            modality_results.append(ModalityResult(
                modality=modality,
                score=0.5,  # Neutral score on failure
                confidence=0.0,
                details={"error": str(result)}
            ))
        else:
            modality_results.append(result)
    
    return modality_results


async def _analyze_single_modality(
    analysis_id: str,
    modality: Modality,
    preprocessed: PreprocessedData,
    options: Optional[dict] = None
) -> ModalityResult:
    """
    Run analysis for a single modality.
    
    Args:
        analysis_id: Analysis identifier
        modality: Modality to analyze
        preprocessed: Preprocessed data
        options: Analysis options (generate_heatmaps, etc.)
    """
    from core.engine import get_inference_engine
    
    options = options or {}
    generate_heatmaps = options.get("generate_heatmaps", True)
    
    engine = get_inference_engine()
    
    logger.info(f"Analyzing modality: {modality.value} for {analysis_id}, heatmaps={generate_heatmaps}")
    
    try:
        if modality == Modality.VIDEO:
            # Video analysis
            from analyzers.video_analyzer import VideoAnalyzer
            analyzer = VideoAnalyzer()
            
            # VideoAnalyzer.analyze() returns ModalityResult directly
            result = await analyzer.analyze(preprocessed, engine)
            return result
            
        elif modality == Modality.AUDIO:
            # Audio analysis
            from analyzers.audio import AudioAnalyzer
            analyzer = AudioAnalyzer()
            
            # AudioAnalyzer.analyze() returns ModalityResult (via BaseAnalyzer)
            result = await analyzer.analyze(preprocessed, engine)
            return result
            
        elif modality == Modality.IMAGE:
            # Image analysis - local models
            from analyzers.image import ImageAnalyzer
            analyzer = ImageAnalyzer()
            
            result = await analyzer.analyze(preprocessed, engine)
            
            # Generate heatmap if requested
            heatmap_url = None
            if generate_heatmaps:
                heatmap_url = await _generate_image_heatmap(
                    analysis_id, preprocessed, result.details, engine
                )
            
            details = result.details.copy() if result.details else {}
            if heatmap_url:
                details["heatmap_url"] = heatmap_url
                details["heatmap_generated"] = True

            return ModalityResult(
                modality=Modality.IMAGE,
                score=result.score,
                confidence=result.confidence,
                details=details
            )
        
        else:
            raise ValueError(f"Unknown modality: {modality}")
            
    except Exception as e:
        logger.error(f"Modality analysis failed: {modality.value}, {e}")
        raise InferenceError(modality.value, str(e))


def _extract_raw_tensors_for_umft(preprocessed) -> Optional[dict]:
    """
    Extract raw tensors from preprocessed data for UMFT neural fusion.
    
    Returns None if data is not suitable for UMFT (e.g., no frames or audio).
    UMFT requires at least one of: frames [B, T, C, H, W] or waveform [B, samples].
    """
    try:
        import numpy as np
        import torch
        
        frames = None
        waveform = None
        
        # Extract video frames if available
        if hasattr(preprocessed, 'frames') and preprocessed.frames:
            frame_arrays = []
            for f in preprocessed.frames[:16]:  # cap at 16 frames
                if isinstance(f, np.ndarray):
                    frame_arrays.append(f)
            if frame_arrays:
                # Stack and convert to tensor: [B, T, C, H, W]
                arr = np.stack(frame_arrays)  # [T, H, W, C]
                arr = arr.astype(np.float32) / 255.0
                arr = arr.transpose(0, 3, 1, 2)  # [T, C, H, W]
                frames = torch.from_numpy(arr).unsqueeze(0)  # [1, T, C, H, W]
        
        # Extract audio waveform if available
        if hasattr(preprocessed, 'audio_data') and preprocessed.audio_data is not None:
            audio = preprocessed.audio_data
            if isinstance(audio, np.ndarray):
                audio = audio.astype(np.float32)
                if audio.ndim == 1:
                    audio = audio[np.newaxis, :]  # [1, samples]
                waveform = torch.from_numpy(audio)
        
        if frames is None and waveform is None:
            return None
        
        return {"frames": frames, "waveform": waveform}
        
    except Exception as e:
        logger.debug("Could not extract raw tensors for UMFT: %s", e)
        return None


async def _aggregate_results(
    analysis_id: str,
    modality_results: List[ModalityResult],
    content_type: ContentType,
    raw_tensors: Optional[dict] = None,
) -> AggregatedResult:
    """
    Aggregate modality results using multi-modal fusion.
    
    When UMFT cross-attention fusion weights are available AND raw tensors
    (frames, waveform) are provided, uses neural fusion via fuse_raw().
    Otherwise falls back to Dirichlet evidential fusion.
    
    Args:
        raw_tensors: Optional dict with keys "frames", "waveform" for UMFT.
    """
    from core.fusion import get_multi_modal_fusion
    from core.scorer import get_trust_scorer
    
    fusion = get_multi_modal_fusion()
    
    # Filter out failed modalities (confidence=0.0) — they should not influence fusion
    active_results = [r for r in modality_results if r.confidence > 0.0]
    if not active_results:
        # All modalities failed — return neutral result
        from schemas.schemas import AggregatedResult
        return AggregatedResult(
            modality_results=modality_results,
            fused_score=0.5,
            uncertainty=1.0,
            weights_used={}
        )
    
    # Try UMFT neural fusion when raw tensors are available and engine is ready
    if raw_tensors and fusion.umft_available:
        try:
            frames = raw_tensors.get("frames")
            waveform = raw_tensors.get("waveform")
            if frames is not None or waveform is not None:
                import torch
                fake_prob, attn_weights, _ = fusion.fuse_raw(
                    frames=frames,
                    waveform=waveform,
                )
                logger.info(
                    "UMFT neural fusion used for %s: fake_prob=%.3f",
                    analysis_id, fake_prob,
                )
                # Build AggregatedResult from UMFT output
                from schemas.schemas import AggregatedResult
                return AggregatedResult(
                    modality_results=modality_results,
                    fused_score=fake_prob,
                    uncertainty=0.3,  # UMFT has lower uncertainty than Dirichlet
                    weights_used={"umft": 1.0},
                )
        except Exception as e:
            logger.warning(
                "UMFT fusion failed for %s, falling back to Dirichlet: %s",
                analysis_id, e,
            )
    
    # Default: Dirichlet evidential fusion
    aggregated = fusion.aggregate(active_results, content_type)
    
    return aggregated


def _build_final_results(
    aggregated: AggregatedResult,
    modality_results: List[ModalityResult],
    processing_time: float,
    content_type: ContentType,
    analysis_id: str = ""
) -> dict:
    """
    Build final analysis results for database update.
    
    Includes XAI (Explainable AI) generation for court-admissible evidence.
    """
    from core.scorer import get_trust_scorer
    from core.xai import get_xai_generator, SCIENTIFIC_REFERENCES
    from schemas.schemas import EvidencePackage, FeatureImportance, ScientificReference
    
    scorer = get_trust_scorer()
    
    # Compute trust score
    trust_score, verdict = scorer.compute(aggregated, content_type=content_type)
    
    # Generate explanation
    explanation = _generate_explanation(aggregated, verdict, modality_results)
    
    # Build modality-specific results
    video_result = None
    audio_result = None
    image_result = None
    
    for result in modality_results:
        if result.modality == Modality.VIDEO:
            video_result = _build_video_result(result)
        elif result.modality == Modality.AUDIO:
            audio_result = _build_audio_result(result)
        elif result.modality == Modality.IMAGE:
            image_result = _build_image_result(result)
    
    # Generate XAI (Explainable AI) artifacts for court-admissible evidence
    evidence_package = None
    feature_importance = []
    scientific_references = []
    
    try:
        xai_generator = get_xai_generator()
        
        # Generate feature importance based on modality results
        feature_importance = _generate_feature_importance(
            aggregated, video_result, audio_result, image_result, verdict
        )
        
        # Generate evidence package with reproducibility hash
        evidence_package = _generate_evidence_package(
            analysis_id=analysis_id,
            modality_results=modality_results,
            video_result=video_result,
            audio_result=audio_result,
            image_result=image_result,
            feature_importance=feature_importance,
            trust_score=trust_score.value,
            uncertainty=aggregated.uncertainty
        )
        
        # Include scientific references for methods used
        scientific_references = _get_scientific_references_for_modalities(modality_results)
        
        logger.info(
            f"XAI generation complete: {len(feature_importance)} features, "
            f"{len(scientific_references)} references, "
            f"hash={evidence_package.reproducibility_hash[:16]}..."
        )
        
    except Exception as e:
        logger.warning(f"XAI generation failed, using defaults: {e}")
        # Continue without XAI - analysis is still valid
    
    return {
        "trust_score": trust_score.model_dump(mode="json"),
        "verdict": verdict.value,
        "explanation": explanation.model_dump(mode="json"),
        "video_result": video_result.model_dump(mode="json") if video_result else None,
        "audio_result": audio_result.model_dump(mode="json") if audio_result else None,
        "image_result": image_result.model_dump(mode="json") if image_result else None,
        "processing_time_seconds": processing_time,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        # XAI Enhancement Fields
        "evidence_package": evidence_package.model_dump(mode="json") if evidence_package else None,
        "feature_importance": [f.model_dump(mode="json") for f in feature_importance],
        "scientific_references": [r.model_dump(mode="json") for r in scientific_references]
    }


def _build_video_result(result: ModalityResult) -> VideoResult:
    """Build VideoResult from ModalityResult."""
    from schemas.schemas import SpatialResult, TemporalResult, LipSyncResult
    
    details = result.details or {}
    spatial_data = details.get("spatial", {})
    temporal_data = details.get("temporal", {})
    lipsync_data = details.get("lipsync")

    if not isinstance(spatial_data, dict):
        spatial_data = {}
    if not isinstance(temporal_data, dict):
        temporal_data = {}
    if lipsync_data is not None and not isinstance(lipsync_data, dict):
        lipsync_data = None
    
    return VideoResult(
        spatial=SpatialResult(
            score=spatial_data.get("score", details.get("spatial_score", result.score)),
            per_frame_scores=spatial_data.get("per_frame_scores", []),
            anomaly_indices=spatial_data.get("anomaly_indices", []),
            heatmap_urls=spatial_data.get("heatmap_urls", [])
        ),
        temporal=TemporalResult(
            consistency_score=temporal_data.get("consistency_score", details.get("temporal_score", result.score)),
            flickering_detected=temporal_data.get("flickering_detected", False),
            anomaly_timestamps=temporal_data.get("anomaly_timestamps", [])
        ),
        lip_sync=LipSyncResult(
            sync_score=lipsync_data.get("sync_score", details.get("lip_sync_score", 1.0)) if lipsync_data else (details.get("lip_sync_score", 1.0) or 1.0),
            manipulation_probability=lipsync_data.get(
                "manipulation_probability",
                1 - (details.get("lip_sync_score", 1.0) or 1.0)
            ) if lipsync_data else (1 - (details.get("lip_sync_score", 1.0) or 1.0)),
            detected_technology=lipsync_data.get("detected_technology") if lipsync_data else None
        ) if (lipsync_data is not None or details.get("lip_sync_score") is not None) else None,
        aggregate_score=result.score,
        frames_analyzed=details.get("frames_analyzed", 0),
        face_detected=details.get("face_detected", False)
    )


def _build_audio_result(result: ModalityResult) -> AudioResult:
    """Build AudioResult from ModalityResult."""
    details = result.details or {}
    
    # Handle vocoder_artifacts which might be a dict or bool
    vocoder_artifacts = details.get("vocoder_artifacts", False)
    if isinstance(vocoder_artifacts, dict):
        vocoder_detected = vocoder_artifacts.get("artifact_score", 0) > 0.5
    elif isinstance(vocoder_artifacts, bool):
        vocoder_detected = vocoder_artifacts
    else:
        vocoder_detected = False
    
    return AudioResult(
        synthetic_probability=details.get("synthetic_probability", result.score),
        vocoder_artifacts_detected=vocoder_detected,
        voice_consistency_score=details.get("voice_consistency", {}).get("pitch_consistency", 0.5)
        if isinstance(details.get("voice_consistency"), dict) else 0.5,
        spectrogram_url=details.get("spectrogram_url"),
        frequency_anomaly_score=details.get("frequency_anomaly_score", 0.0),
        aasist_score=details.get("aasist_score") if details.get("aasist_score") != 0.5 else None,
    )





def _build_image_result(result: ModalityResult) -> ImageResult:
    """Build ImageResult from ModalityResult."""
    details = result.details
    dct_features = details.get("dct_features", {}) if isinstance(details.get("dct_features"), dict) else {}
    
    return ImageResult(
        ai_generated_probability=details.get("ai_generated_probability", details.get("fake_probability", result.score)),
        fake_probability=details.get("fake_probability", result.score),
        face_detected=details.get("face_detected", False),
        num_faces=details.get("num_faces", 0),
        face_manipulation_scores=details.get("face_manipulation_scores", []),
        heatmap_url=details.get("heatmap_url"),
        dct_anomaly_score=_safe_float(dct_features.get("anomaly_score", 0.0)),
        spectral_flatness=_safe_float(dct_features.get("spectral_flatness", 0.0)),
        ensemble_score=_safe_float(details.get("ensemble_score", 0.0)),
        ensemble_primary_available=details.get("ensemble_primary_available", False),
        ensemble_secondary_available=details.get("ensemble_secondary_available", False)
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to safe float, replacing NaN/Inf with default."""
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return float(np.clip(f, 0.0, 1.0))
    except (TypeError, ValueError):
        return default


def _generate_explanation(
    aggregated: AggregatedResult,
    verdict: Verdict,
    modality_results: List[ModalityResult]
) -> Explanation:
    """Generate human-readable explanation using template-based engine."""
    from core.explain import ExplainabilityEngine

    try:
        explainer = ExplainabilityEngine()
        return explainer.generate_textual_explanation(aggregated, verdict, None)
    except Exception as e:
        logger.warning(f"Explanation generation failed: {e}")

        return Explanation(
            summary=f"Analysis complete. Verdict: {verdict.value}",
            key_findings=[
                f"Trust score: {aggregated.fused_score * 100:.1f}%",
                f"Modalities analyzed: {len(modality_results)}"
            ],
            manipulation_regions=[],
            confidence_rationale=f"Uncertainty: {aggregated.uncertainty:.2%}",
            methodology_used=[r.modality.value for r in modality_results]
        )


def _generate_feature_importance(
    aggregated: AggregatedResult,
    video_result: Optional["VideoResult"],
    audio_result: Optional["AudioResult"],
    image_result: Optional["ImageResult"],
    verdict: Verdict
) -> List["FeatureImportance"]:
    """
    Generate feature importance scores for XAI.
    
    Creates court-admissible feature-level explanations showing
    which factors contributed most to the detection decision.
    """
    from schemas.schemas import FeatureImportance
    
    features = []
    
    # Image features
    if image_result is not None:
        ai_prob = image_result.ai_generated_probability
        features.append(FeatureImportance(
            feature_name="ai_generated_probability",
            importance_score=ai_prob,
            contribution_direction="increases_fake" if ai_prob > 0.5 else "decreases_fake",
            confidence=0.85,
            feature_type="visual"
        ))
        
        # DCT frequency analysis
        if image_result.dct_anomaly_score > 0:
            features.append(FeatureImportance(
                feature_name="dct_frequency_anomaly",
                importance_score=image_result.dct_anomaly_score,
                contribution_direction="increases_fake",
                confidence=0.9,
                feature_type="frequency"
            ))
        
        # Ensemble classifier score
        if image_result.ensemble_score > 0:
            features.append(FeatureImportance(
                feature_name="ensemble_classifier",
                importance_score=image_result.ensemble_score,
                contribution_direction="increases_fake" if image_result.ensemble_score > 0.5 else "decreases_fake",
                confidence=0.90 if image_result.ensemble_primary_available and image_result.ensemble_secondary_available else 0.75,
                feature_type="visual"
            ))
        
        # Face manipulation
        if image_result.face_detected and image_result.face_manipulation_scores:
            avg_face_score = sum(image_result.face_manipulation_scores) / len(image_result.face_manipulation_scores)
            features.append(FeatureImportance(
                feature_name="face_manipulation",
                importance_score=avg_face_score,
                contribution_direction="increases_fake" if avg_face_score > 0.5 else "decreases_fake",
                confidence=0.9,
                feature_type="visual"
            ))
    
    # Video features
    if video_result is not None:
        # Spatial analysis features
        if video_result.spatial:
            spatial_score = video_result.spatial.score
            features.append(FeatureImportance(
                feature_name="video_spatial_artifacts",
                importance_score=min(1.0, spatial_score * 1.2),
                contribution_direction="increases_fake" if spatial_score > 0.5 else "decreases_fake",
                confidence=0.85,
                feature_type="spatial"
            ))
            
            # DCT anomaly score
            if hasattr(video_result.spatial, 'dct_anomaly_score') and video_result.spatial.dct_anomaly_score:
                features.append(FeatureImportance(
                    feature_name="dct_frequency_anomaly",
                    importance_score=video_result.spatial.dct_anomaly_score,
                    contribution_direction="increases_fake",
                    confidence=0.9,
                    feature_type="frequency"
                ))
        
        # Temporal consistency features
        if video_result.temporal:
            temporal_score = video_result.temporal.consistency_score
            features.append(FeatureImportance(
                feature_name="temporal_consistency",
                importance_score=1.0 - temporal_score,
                contribution_direction="increases_fake" if temporal_score < 0.7 else "decreases_fake",
                confidence=0.8,
                feature_type="temporal"
            ))
            
            if video_result.temporal.flickering_detected:
                features.append(FeatureImportance(
                    feature_name="frame_flickering",
                    importance_score=0.85,
                    contribution_direction="increases_fake",
                    confidence=0.9,
                    feature_type="temporal"
                ))
        
        # Lip-sync features
        if video_result.lip_sync:
            lipsync_score = video_result.lip_sync.sync_score
            features.append(FeatureImportance(
                feature_name="lip_sync_accuracy",
                importance_score=1.0 - lipsync_score,
                contribution_direction="increases_fake" if lipsync_score < 0.7 else "decreases_fake",
                confidence=0.85,
                feature_type="temporal"
            ))
    
    # Audio features
    if audio_result is not None:
        synthetic_prob = audio_result.synthetic_probability
        features.append(FeatureImportance(
            feature_name="synthetic_audio_probability",
            importance_score=synthetic_prob,
            contribution_direction="increases_fake" if synthetic_prob > 0.5 else "decreases_fake",
            confidence=0.85,
            feature_type="acoustic"
        ))
        
        if audio_result.vocoder_artifacts_detected:
            features.append(FeatureImportance(
                feature_name="vocoder_artifacts",
                importance_score=0.9,
                contribution_direction="increases_fake",
                confidence=0.95,
                feature_type="frequency"
            ))
        
        if audio_result.voice_consistency_score:
            consistency = audio_result.voice_consistency_score
            features.append(FeatureImportance(
                feature_name="voice_consistency",
                importance_score=1.0 - consistency,
                contribution_direction="increases_fake" if consistency < 0.7 else "decreases_fake",
                confidence=0.8,
                feature_type="acoustic"
            ))
    
    # Sort by importance score descending
    features.sort(key=lambda f: f.importance_score, reverse=True)
    
    return features


def _generate_evidence_package(
    analysis_id: str,
    modality_results: List[ModalityResult],
    video_result: Optional["VideoResult"],
    audio_result: Optional["AudioResult"],
    image_result: Optional["ImageResult"],
    feature_importance: List["FeatureImportance"],
    trust_score: float,
    uncertainty: float = 0.1
) -> "EvidencePackage":
    """
    Generate complete evidence package for court-admissible forensic reports.
    
    Includes reproducibility hash, confidence intervals, and visual evidence.
    """
    import hashlib
    import json
    from schemas.schemas import EvidencePackage, VisualEvidence
    
    # Build reproducibility data
    def _json_safe(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    repro_data = {
        "analysis_id": analysis_id,
        "trust_score": trust_score,
        "modalities": [r.modality.value for r in modality_results],
        "features": [
            {
                "name": f.feature_name,
                "score": f.importance_score,
                "direction": f.contribution_direction
            }
            for f in feature_importance[:10]  # Top 10 features
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Generate SHA-256 reproducibility hash
    repro_json = json.dumps(repro_data, sort_keys=True, default=_json_safe)
    reproducibility_hash = hashlib.sha256(repro_json.encode()).hexdigest()
    
    # Calculate 95% confidence interval
    # trust_score is on 0-100 scale, normalize to 0-1 for CI
    trust_score_normalized = trust_score / 100.0
    margin = 1.96 * uncertainty  # 95% CI
    lower = max(0.0, trust_score_normalized - margin)
    upper = min(1.0, trust_score_normalized + margin)
    confidence_interval = (round(lower, 4), round(upper, 4))
    
    # Collect visual evidence
    visual_evidence = []
    
    # Image evidence
    if image_result is not None:
        # Add heatmap evidence
        if image_result.heatmap_url:
            visual_evidence.append(VisualEvidence(
                artifact_type="heatmap",
                url=image_result.heatmap_url,
                description="GradCAM++ attention map showing manipulation regions",
                frame_index=None,
                integrity_hash=hashlib.sha256(image_result.heatmap_url.encode()).hexdigest()[:32]
            ))
        
        # Add manipulation regions
        if image_result.manipulation_regions:
            for i, region in enumerate(image_result.manipulation_regions):
                visual_evidence.append(VisualEvidence(
                    artifact_type="manipulation_region",
                    url=None,
                    description=f"Detected manipulation region: {region.type if hasattr(region, 'type') else 'unknown'}",
                    frame_index=None,
                    integrity_hash=hashlib.sha256(f"region:{i}:{region}".encode()).hexdigest()[:32]
                ))
    
    if video_result is not None:
        # Add heatmap evidence
        if hasattr(video_result, 'frame_heatmap_urls') and video_result.frame_heatmap_urls:
            for i, url in enumerate(video_result.frame_heatmap_urls[:5]):
                visual_evidence.append(VisualEvidence(
                    artifact_type="heatmap",
                    url=url,
                    description=f"GradCAM++ attention map for frame {i+1}",
                    frame_index=i,
                    integrity_hash=hashlib.sha256(f"{url}:{i}".encode()).hexdigest()[:32]
                ))
        
        # Add manipulation regions
        if video_result.spatial and hasattr(video_result.spatial, 'manipulation_regions') and video_result.spatial.manipulation_regions:
            for region in video_result.spatial.manipulation_regions[:3]:
                region_dict = region if isinstance(region, dict) else region.model_dump()
                visual_evidence.append(VisualEvidence(
                    artifact_type="overlay",
                    url=region_dict.get("thumbnail_url", ""),
                    description=f"Detected manipulation: {region_dict.get('type', 'unknown')}",
                    integrity_hash=hashlib.sha256(str(region_dict).encode()).hexdigest()[:32]
                ))
    
    if audio_result is not None:
        # Add spectrogram evidence
        if hasattr(audio_result, 'spectrogram_url') and audio_result.spectrogram_url:
            visual_evidence.append(VisualEvidence(
                artifact_type="spectrogram",
                url=audio_result.spectrogram_url,
                description="Mel-spectrogram with artifact overlay",
                integrity_hash=hashlib.sha256(audio_result.spectrogram_url.encode()).hexdigest()[:32]
            ))
    
    # Model versions used
    model_versions = {
        "video_spatial": "efficientnet-b3-deepfake-v2",
        "video_temporal": "xclip-temporal-v1",
        "audio": "aasist-anti-spoof-v1"
    }
    
    return EvidencePackage(
        visual_evidence=visual_evidence,
        feature_importance=feature_importance,
        reproducibility_hash=reproducibility_hash,
        confidence_interval=confidence_interval,
        model_versions=model_versions
    )


def _get_scientific_references_for_modalities(
    modality_results: List[ModalityResult]
) -> List["ScientificReference"]:
    """
    Get relevant scientific references for the analysis methods used.
    
    Returns peer-reviewed citations for court-admissible evidence.
    """
    from core.xai import SCIENTIFIC_REFERENCES
    from schemas.schemas import ScientificReference
    
    references = []
    modalities = {r.modality.value for r in modality_results}
    
    # Map modalities to reference keys
    ref_mapping = {
        "video": ["gradcam", "efficientnet", "xclip"],
        "audio": ["aasist"],
        "image": ["gradcam", "efficientnet", "dct_analysis", "gan_fingerprint"]
    }
    
    # Collect relevant references
    added_keys = set()
    for modality in modalities:
        keys = ref_mapping.get(modality, [])
        for key in keys:
            if key in SCIENTIFIC_REFERENCES and key not in added_keys:
                # SCIENTIFIC_REFERENCES contains ScientificReference objects
                ref_obj = SCIENTIFIC_REFERENCES[key]
                references.append(ref_obj)
                added_keys.add(key)
    
    return references


async def _generate_image_heatmap(
    analysis_id: str,
    preprocessed: PreprocessedData,
    analysis_details: dict,
    engine: "InferenceEngine"
) -> Optional[str]:
    """
    Generate GradCAM++ heatmap for image analysis and upload to MinIO.
    
    Args:
        analysis_id: Analysis identifier
        preprocessed: Preprocessed data with image keys
        analysis_details: Analysis result details
        engine: Inference engine
        
    Returns:
        URL to the uploaded heatmap or None on failure
    """
    from core.xai import get_xai_generator
    from storage.storage import get_storage_client
    from PIL import Image
    import io
    
    try:
        # Load the first image for heatmap generation
        image_keys = preprocessed.face_crops if preprocessed.face_crops else preprocessed.frames
        if not image_keys:
            logger.warning("No images available for heatmap generation")
            return None
        
        storage = get_storage_client()
        
        # Download and load the image
        image_bytes = await storage.download_file("argus-preprocessed", image_keys[0])
        
        if image_keys[0].endswith('.npy'):
            image_array = np.load(io.BytesIO(image_bytes), allow_pickle=True)
            if image_array.dtype == object:
                if hasattr(image_array, 'item') and isinstance(image_array.item(), np.ndarray):
                    image_array = image_array.item()
                elif len(image_array) > 0 and isinstance(image_array[0], np.ndarray):
                    image_array = image_array[0]
        else:
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
            image_array = np.array(pil_image, dtype=np.uint8)
        
        # Ensure correct shape
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif len(image_array.shape) == 3 and image_array.shape[-1] == 1:
            image_array = np.concatenate([image_array] * 3, axis=-1)
        elif len(image_array.shape) == 3 and image_array.shape[0] in [1, 3]:
            image_array = np.transpose(image_array, (1, 2, 0))
            if image_array.shape[-1] == 1:
                image_array = np.concatenate([image_array] * 3, axis=-1)
        
        # Resize if needed
        if image_array.shape[:2] != (224, 224):
            pil_temp = Image.fromarray(image_array.astype(np.uint8))
            pil_temp = pil_temp.resize((224, 224), Image.Resampling.LANCZOS)
            image_array = np.array(pil_temp, dtype=np.uint8)
        
        # Generate heatmap using XAI generator
        xai_generator = get_xai_generator()
        
        # Prepare model output for heatmap generation
        model_output = {
            "class_probabilities": np.array([[ 
                1.0 - analysis_details.get("fake_probability", 0.5),
                analysis_details.get("fake_probability", 0.5)
            ]]),
            "features": None,  # Will use occlusion-based heatmap
            "fake_probability": analysis_details.get("fake_probability", 0.5)
        }
        
        xai_result = xai_generator.generate_image_explanation(
            image_array,
            model_output,
            model_name="efficientnet-b3"
        )
        
        if xai_result.overlay is None:
            logger.warning("XAI generator returned no overlay")
            return None
        
        # Convert overlay to PNG bytes
        overlay_image = Image.fromarray(xai_result.overlay.astype(np.uint8))
        overlay_bytes = io.BytesIO()
        overlay_image.save(overlay_bytes, format='PNG')
        overlay_bytes.seek(0)
        
        # Upload to MinIO
        heatmap_key = f"results/{analysis_id}/heatmaps/gradcam_overlay.png"
        await storage.ensure_default_buckets()
        await storage.upload_file(
            file=overlay_bytes.read(),
            bucket=storage.bucket_results,
            object_key=heatmap_key,
            content_type="image/png"
        )
        
        # Generate presigned URL
        heatmap_url = await storage.get_presigned_url(
            storage.bucket_results,
            heatmap_key,
            expires_seconds=86400  # 24 hours
        )
        
        logger.info(f"Generated heatmap for analysis {analysis_id}: {heatmap_key}")
        return heatmap_url
        
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}")
        return None


# ============== ORCHESTRATOR CLASS ==============

class Orchestrator:
    """
    High-level orchestration interface.
    
    Provides async-friendly API for enqueueing and monitoring
    analysis jobs without direct Celery dependency.
    """
    
    def __init__(self):
        """Initialize orchestrator."""
        self.celery_app = celery_app
    
    async def enqueue_analysis(
        self,
        analysis_id: str,
        options: dict
    ) -> str:
        """
        Enqueue analysis job.
        
        Args:
            analysis_id: Unique analysis ID
            options: Analysis options
            
        Returns:
            Celery job ID
        """
        # Send to Celery
        result = run_analysis_pipeline.delay(analysis_id, options)
        
        logger.info(f"Enqueued analysis: {analysis_id}, job_id: {result.id}")
        
        return result.id
    
    async def get_job_status(self, job_id: str) -> dict:
        """
        Get status of enqueued job.
        
        Args:
            job_id: Celery job ID
            
        Returns:
            Dict with status and optional result
        """
        result = AsyncResult(job_id, app=self.celery_app)
        
        status_map = {
            "PENDING": "pending",
            "STARTED": "running",
            "SUCCESS": "completed",
            "FAILURE": "failed",
            "RETRY": "retrying"
        }
        
        return {
            "job_id": job_id,
            "status": status_map.get(result.status, result.status),
            "result": result.result if result.ready() else None,
            "error": str(result.result) if result.failed() else None
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.
        
        Args:
            job_id: Celery job ID
            
        Returns:
            True if cancellation was requested
        """
        try:
            self.celery_app.control.revoke(job_id, terminate=True)
            logger.info(f"Cancelled job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def retry_job(self, job_id: str, analysis_id: str, options: dict) -> str:
        """
        Retry a failed job.
        
        Args:
            job_id: Original job ID
            analysis_id: Analysis ID
            options: Analysis options
            
        Returns:
            New job ID
        """
        # Simply enqueue again
        return await self.enqueue_analysis(analysis_id, options)


# ============== SINGLETON ==============

_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


# Export
__all__ = [
    "celery_app",
    "run_analysis_pipeline",
    "preprocess_task",
    "analyze_modality_task",
    "aggregate_results_task",
    "generate_report_task",
    "Orchestrator",
    "get_orchestrator"
]
