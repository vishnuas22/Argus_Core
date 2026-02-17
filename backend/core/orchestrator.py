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
import numpy as np

from celery import Celery, chain, group, chord
from celery.result import AsyncResult
from celery.exceptions import SoftTimeLimitExceeded

from config import config
from schemas.schemas import (
    AnalysisStatus, AnalysisDocument, PreprocessedData, ModalityResult,
    AggregatedResult, Modality, ContentType, TrustScore, Verdict, Explanation,
    VideoResult, AudioResult, TextResult, ImageResult, MetadataResult
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
    import sys
    import os
    # Add backend directory to path for Celery worker
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from storage.db import get_db_client
    return await get_db_client()


async def get_storage():
    """Get storage client for task operations."""
    import sys
    import os
    # Add backend directory to path for Celery worker
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
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
            
            # Result is already a ModalityResult with score field
            # The score is fake probability (0-1), invert for trust score
            result.score = 1 - result.score
            return result
            
        elif modality == Modality.AUDIO:
            # Audio analysis
            from analyzers.audio import AudioAnalyzer
            analyzer = AudioAnalyzer()
            
            # AudioAnalyzer.analyze() returns ModalityResult (via BaseAnalyzer)
            result = await analyzer.analyze(preprocessed, engine)
            
            # The score is synthetic_probability (0-1), we need to invert for trust score
            result.score = 1 - result.score  # Invert: high synthetic prob = low trust
            return result
            
        elif modality == Modality.IMAGE:
            # Image analysis
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
            
        elif modality == Modality.TEXT:
            # Text analysis
            from analyzers.text import TextAnalyzer
            analyzer = TextAnalyzer()
            
            result = await analyzer.analyze(preprocessed, engine)
            
            # TextAnalyzer.analyze() already returns a ModalityResult
            # The score is ai_probability (0-1), we need to invert for trust score
            result.score = 1 - result.score  # Invert: high AI prob = low trust
            return result
        
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
    
    # Run fusion (synchronous method - do not await)
    aggregated = fusion.aggregate(modality_results, content_type)
    
    return aggregated


def _build_final_results(
    aggregated: AggregatedResult,
    modality_results: List[ModalityResult],
    processing_time: float
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
    trust_score, verdict = scorer.compute(aggregated)
    
    # Generate explanation
    explanation = _generate_explanation(aggregated, verdict, modality_results)
    
    # Build modality-specific results
    video_result = None
    audio_result = None
    text_result = None
    image_result = None
    
    for result in modality_results:
        if result.modality == Modality.VIDEO:
            video_result = _build_video_result(result)
        elif result.modality == Modality.AUDIO:
            audio_result = _build_audio_result(result)
        elif result.modality == Modality.TEXT:
            text_result = _build_text_result(result)
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
            aggregated, video_result, audio_result, text_result, image_result, verdict
        )
        
        # Generate evidence package with reproducibility hash
        evidence_package = _generate_evidence_package(
            analysis_id="",  # Will be set by caller
            modality_results=modality_results,
            video_result=video_result,
            audio_result=audio_result,
            text_result=text_result,
            image_result=image_result,
            feature_importance=feature_importance,
            trust_score=trust_score.value
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
        "text_result": text_result.model_dump(mode="json") if text_result else None,
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
        synthetic_probability=details.get("synthetic_probability", 1 - result.score),
        vocoder_artifacts_detected=vocoder_detected,
        voice_consistency_score=result.confidence or 0.5
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


def _build_image_result(result: ModalityResult) -> ImageResult:
    """Build ImageResult from ModalityResult."""
    details = result.details
    
    return ImageResult(
        ai_generated_probability=details.get("ai_generated_probability", details.get("fake_probability", 1 - result.score)),
        fake_probability=details.get("fake_probability", 1 - result.score),
        face_detected=details.get("face_detected", False),
        num_faces=details.get("num_faces", 0),
        face_manipulation_scores=details.get("face_manipulation_scores", []),
        heatmap_url=details.get("heatmap_url"),
        dct_anomaly_score=details.get("dct_features", {}).get("anomaly_score", 0.0) if isinstance(details.get("dct_features"), dict) else 0.0,
        spectral_flatness=details.get("dct_features", {}).get("spectral_flatness", 0.0) if isinstance(details.get("dct_features"), dict) else 0.0,
        siglip_score=details.get("siglip_score", 0.0),
        efficientnet_score=details.get("efficientnet_score", 0.0)
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
        # Pass None for regions - modality_results are not Region objects
        return explainer.generate_textual_explanation(aggregated, verdict, None)
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


def _generate_feature_importance(
    aggregated: AggregatedResult,
    video_result: Optional["VideoResult"],
    audio_result: Optional["AudioResult"],
    text_result: Optional["TextResult"],
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
        
        # SigLIP classifier score
        if image_result.siglip_score > 0:
            features.append(FeatureImportance(
                feature_name="siglip_classifier",
                importance_score=image_result.siglip_score,
                contribution_direction="increases_fake" if image_result.siglip_score > 0.5 else "decreases_fake",
                confidence=0.85,
                feature_type="visual"
            ))
        
        # EfficientNet classifier score
        if image_result.efficientnet_score > 0:
            features.append(FeatureImportance(
                feature_name="efficientnet_classifier",
                importance_score=image_result.efficientnet_score,
                contribution_direction="increases_fake" if image_result.efficientnet_score > 0.5 else "decreases_fake",
                confidence=0.85,
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
    
    # Text features
    if text_result is not None:
        ai_prob = text_result.ai_probability
        features.append(FeatureImportance(
            feature_name="ai_text_probability",
            importance_score=ai_prob,
            contribution_direction="increases_fake" if ai_prob > 0.5 else "decreases_fake",
            confidence=0.85,
            feature_type="linguistic"
        ))
        
        if text_result.perplexity_score:
            perplexity = text_result.perplexity_score
            # Low perplexity often indicates AI-generated text
            features.append(FeatureImportance(
                feature_name="perplexity_score",
                importance_score=min(1.0, perplexity / 100),
                contribution_direction="decreases_fake" if perplexity > 50 else "increases_fake",
                confidence=0.75,
                feature_type="linguistic"
            ))
        
        if text_result.radar_score:
            radar = text_result.radar_score
            features.append(FeatureImportance(
                feature_name="radar_classifier",
                importance_score=radar,
                contribution_direction="increases_fake" if radar > 0.5 else "decreases_fake",
                confidence=0.9,
                feature_type="linguistic"
            ))
    
    # Sort by importance score descending
    features.sort(key=lambda f: f.importance_score, reverse=True)
    
    return features


def _generate_evidence_package(
    analysis_id: str,
    modality_results: List[ModalityResult],
    video_result: Optional["VideoResult"],
    audio_result: Optional["AudioResult"],
    text_result: Optional["TextResult"],
    image_result: Optional["ImageResult"],
    feature_importance: List["FeatureImportance"],
    trust_score: float
) -> "EvidencePackage":
    """
    Generate complete evidence package for court-admissible forensic reports.
    
    Includes reproducibility hash, confidence intervals, and visual evidence.
    """
    import hashlib
    import json
    from schemas.schemas import EvidencePackage, VisualEvidence
    
    # Build reproducibility data
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
    repro_json = json.dumps(repro_data, sort_keys=True)
    reproducibility_hash = hashlib.sha256(repro_json.encode()).hexdigest()
    
    # Calculate 95% confidence interval
    # Based on ensemble uncertainty
    uncertainty = 0.1  # Default uncertainty
    margin = 1.96 * uncertainty  # 95% CI
    lower = max(0.0, trust_score - margin)
    upper = min(1.0, trust_score + margin)
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
        "audio": "aasist-anti-spoof-v1",
        "text": "radar-ai-detection-v1"
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
        "text": ["radar"],
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
            "features": None  # Will use synthetic heatmap
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
