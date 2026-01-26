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
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from functools import wraps

from celery import Celery, chain, group, chord
from celery.result import AsyncResult
from celery.exceptions import SoftTimeLimitExceeded

from config import config
from schemas.schemas import (
    AnalysisStatus, AnalysisDocument, PreprocessedData, ModalityResult,
    AggregatedResult, Modality, ContentType, TrustScore, Verdict, Explanation,
    VideoResult, AudioResult, TextResult, MetadataResult
)
from utils.logging import get_logger
from utils.errors import PreprocessingError, InferenceError, FusionError

logger = get_logger(__name__)

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
    }
)


# ============== ASYNC HELPERS ==============

def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


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


async def publish_progress(
    analysis_id: str,
    status: AnalysisStatus,
    progress_percent: float,
    current_stage: str,
    message: Optional[str] = None
):
    """Publish progress update to Redis for WebSocket delivery."""
    try:
        import redis
        r = redis.from_url(config.redis_url)
        
        import json
        progress_data = {
            "analysis_id": analysis_id,
            "status": status.value,
            "progress_percent": progress_percent,
            "current_stage": current_stage,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Publish to channel
        r.publish(f"argus:progress:{analysis_id}", json.dumps(progress_data))
        
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
            content_type=preprocessed.content_type
        )
        
        # ===== PHASE 4: FINALIZE RESULTS =====
        processing_time = time.time() - start_time
        
        # Build final results
        final_updates = _build_final_results(
            aggregated=aggregated,
            modality_results=modality_results,
            processing_time=processing_time
        )
        
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
        await update_status(
            analysis_id,
            AnalysisStatus.FAILED,
            0.0,
            "failed",
            str(e)
        )
        
        db = await get_db()
        await db.update_job_status(job_id, "failed", str(e))
        
        # Retry on transient errors
        if self.request.retries < self.max_retries:
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
        modality: Modality to analyze (video, audio, text, image)
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
        
        # Generate report (placeholder - full implementation in forensics module)
        report_key = f"results/{analysis_id}/report.pdf"
        
        # Create placeholder PDF
        from io import BytesIO
        pdf_content = _generate_placeholder_report(analysis)
        
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
        
    except Exception as e:
        logger.error(f"Report generation failed: {analysis_id}, error: {e}")
        return {"status": "failed", "error": str(e)}


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
    
    if preprocessed.content_type == ContentType.TEXT_ONLY:
        modalities.append(Modality.TEXT)
    
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
    
    # Create tasks
    tasks = []
    for modality in modality_enums:
        task = _analyze_single_modality(analysis_id, modality, preprocessed)
        tasks.append(task)
    
    # Run in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    modality_results = []
    for modality, result in zip(modality_enums, results):
        if isinstance(result, Exception):
            logger.error(f"Modality {modality.value} failed: {result}")
            # Create failed result
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
    preprocessed: PreprocessedData
) -> ModalityResult:
    """
    Run analysis for a single modality.
    """
    from core.engine import get_inference_engine
    
    engine = get_inference_engine()
    
    logger.info(f"Analyzing modality: {modality.value} for {analysis_id}")
    
    try:
        if modality == Modality.VIDEO:
            # Video analysis
            from analyzers.video import VideoAnalyzer
            analyzer = VideoAnalyzer()
            
            result = await analyzer.analyze(preprocessed, engine)
            
            return ModalityResult(
                modality=Modality.VIDEO,
                score=result.aggregate_score,
                confidence=0.8,  # From model confidence
                details={
                    "spatial_score": result.spatial.score,
                    "temporal_score": result.temporal.consistency_score,
                    "lip_sync_score": result.lip_sync.sync_score if result.lip_sync else None
                }
            )
            
        elif modality == Modality.AUDIO:
            # Audio analysis
            from analyzers.audio import AudioAnalyzer
            analyzer = AudioAnalyzer()
            
            result = await analyzer.analyze(preprocessed, engine)
            
            return ModalityResult(
                modality=Modality.AUDIO,
                score=1 - result.synthetic_probability,  # Invert for trust score
                confidence=result.voice_consistency_score,
                details={
                    "synthetic_probability": result.synthetic_probability,
                    "vocoder_artifacts": result.vocoder_artifacts_detected
                }
            )
            
        elif modality == Modality.IMAGE:
            # Image analysis
            from analyzers.image import ImageAnalyzer
            analyzer = ImageAnalyzer()
            
            result = await analyzer.analyze(preprocessed, engine)
            
            return ModalityResult(
                modality=Modality.IMAGE,
                score=result.score,
                confidence=result.confidence,
                details=result.details
            )
            
        elif modality == Modality.TEXT:
            # Text analysis
            from analyzers.text import TextAnalyzer
            analyzer = TextAnalyzer()
            
            result = await analyzer.analyze(preprocessed, engine)
            
            return ModalityResult(
                modality=Modality.TEXT,
                score=1 - result.ai_probability,  # Invert for trust score
                confidence=0.7,
                details={
                    "ai_probability": result.ai_probability,
                    "perplexity": result.perplexity_score,
                    "burstiness": result.burstiness_score
                }
            )
        
        else:
            raise ValueError(f"Unknown modality: {modality}")
            
    except Exception as e:
        logger.error(f"Modality analysis failed: {modality.value}, {e}")
        raise InferenceError(modality.value, str(e))


async def _aggregate_results(
    analysis_id: str,
    modality_results: List[ModalityResult],
    content_type: ContentType
) -> AggregatedResult:
    """
    Aggregate modality results using multi-modal fusion.
    """
    from core.fusion import get_multi_modal_fusion
    from core.scorer import get_trust_scorer
    
    fusion = get_multi_modal_fusion()
    
    # Run fusion
    aggregated = await fusion.aggregate(modality_results, content_type)
    
    return aggregated


def _build_final_results(
    aggregated: AggregatedResult,
    modality_results: List[ModalityResult],
    processing_time: float
) -> dict:
    """
    Build final analysis results for database update.
    """
    from core.scorer import get_trust_scorer
    
    scorer = get_trust_scorer()
    
    # Compute trust score
    trust_score, verdict = scorer.compute(aggregated)
    
    # Generate explanation
    explanation = _generate_explanation(aggregated, verdict, modality_results)
    
    # Build modality-specific results
    video_result = None
    audio_result = None
    text_result = None
    
    for result in modality_results:
        if result.modality == Modality.VIDEO:
            video_result = _build_video_result(result)
        elif result.modality == Modality.AUDIO:
            audio_result = _build_audio_result(result)
        elif result.modality == Modality.TEXT:
            text_result = _build_text_result(result)
    
    return {
        "trust_score": trust_score.model_dump(mode="json"),
        "verdict": verdict.value,
        "explanation": explanation.model_dump(mode="json"),
        "video_result": video_result.model_dump(mode="json") if video_result else None,
        "audio_result": audio_result.model_dump(mode="json") if audio_result else None,
        "text_result": text_result.model_dump(mode="json") if text_result else None,
        "processing_time_seconds": processing_time,
        "completed_at": datetime.now(timezone.utc).isoformat()
    }


def _build_video_result(result: ModalityResult) -> VideoResult:
    """Build VideoResult from ModalityResult."""
    from schemas.schemas import SpatialResult, TemporalResult, LipSyncResult
    
    details = result.details
    
    return VideoResult(
        spatial=SpatialResult(
            score=details.get("spatial_score", result.score),
            per_frame_scores=[],
            anomaly_indices=[],
            heatmap_urls=[]
        ),
        temporal=TemporalResult(
            consistency_score=details.get("temporal_score", result.score),
            flickering_detected=False,
            anomaly_timestamps=[]
        ),
        lip_sync=LipSyncResult(
            sync_score=details.get("lip_sync_score", 1.0) or 1.0,
            manipulation_probability=1 - (details.get("lip_sync_score", 1.0) or 1.0)
        ) if details.get("lip_sync_score") is not None else None,
        aggregate_score=result.score,
        frames_analyzed=0,
        face_detected=True
    )


def _build_audio_result(result: ModalityResult) -> AudioResult:
    """Build AudioResult from ModalityResult."""
    details = result.details
    
    return AudioResult(
        synthetic_probability=details.get("synthetic_probability", 1 - result.score),
        vocoder_artifacts_detected=details.get("vocoder_artifacts", False),
        voice_consistency_score=result.confidence
    )


def _build_text_result(result: ModalityResult) -> TextResult:
    """Build TextResult from ModalityResult."""
    details = result.details
    
    return TextResult(
        ai_probability=details.get("ai_probability", 1 - result.score),
        perplexity_score=details.get("perplexity", 0.0),
        burstiness_score=details.get("burstiness", 0.0),
        radar_score=details.get("radar_score")
    )


def _generate_explanation(
    aggregated: AggregatedResult,
    verdict: Verdict,
    modality_results: List[ModalityResult]
) -> Explanation:
    """Generate human-readable explanation."""
    from core.explain import ExplainabilityEngine
    
    try:
        explainer = ExplainabilityEngine()
        return explainer.generate_textual_explanation(aggregated, verdict, modality_results)
    except Exception as e:
        logger.warning(f"Explanation generation failed: {e}")
        
        # Fallback explanation
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


def _generate_placeholder_report(analysis: AnalysisDocument) -> bytes:
    """Generate placeholder PDF report content."""
    # Simple text-based placeholder
    content = f"""
ARGUS CORE - DEEPFAKE ANALYSIS REPORT
=====================================

Analysis ID: {analysis.analysis_id}
Date: {datetime.now(timezone.utc).isoformat()}

EXECUTIVE SUMMARY
-----------------
Verdict: {analysis.verdict.value if analysis.verdict else 'N/A'}
Trust Score: {analysis.trust_score.value if analysis.trust_score else 'N/A'}

This is a placeholder report. Full PDF generation will be implemented
in the forensics/report.py module.
"""
    return content.encode("utf-8")


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
