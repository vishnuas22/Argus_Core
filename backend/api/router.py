"""
Argus Core - API Router
=======================
HTTP endpoints for the deepfake detection platform.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - api/router.py

Role: Define all HTTP endpoints. Route requests to appropriate handlers.
Handle request validation and response serialization.

Integration:
- Imports: api/deps.py, schemas/schemas.py, core/orchestrator.py
- Inputs: AnalyzeRequest, UploadFile
- Outputs: AnalysisResponse, AnalysisStatusResponse

Why this approach: Dependency injection enables testing and modularity.
Async handlers prevent blocking during I/O operations.

Endpoints:
- POST /api/v1/analyze - Upload and analyze media
- GET /api/v1/analyze/{analysis_id} - Get analysis status/result
- GET /api/v1/analyze/{analysis_id}/detail - Get detailed results
- DELETE /api/v1/analyze/{analysis_id} - Delete analysis
- POST /api/v1/analyze/text - Analyze text for AI generation
- GET /api/v1/health - Health check
- GET /api/v1/models - List available models
"""

import uuid
from typing import Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, UploadFile, File, Form, Query, Path, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from config import config
from schemas.schemas import (
    AnalysisDocument, AnalysisStatus, AnalysisResponse, AnalysisDetailResponse,
    AnalyzeOptions, FileInput, Modality, TrustScore, Verdict, Explanation,
    ProgressUpdate, ErrorResponse
)
from storage.storage import StorageClient
from storage.db import DatabaseClient
from processing.sanitize import InputSanitizer, SanitizedFile
from core.engine import InferenceEngine
from api.deps import (
    get_db, get_storage, get_sanitizer_standard, get_sanitizer_aggressive,
    get_orchestrator, get_correlation_id, get_current_user, get_current_user_optional,
    get_analysis_deps, AnalysisDependencies, check_rate_limit, get_engine
)
from utils.logging import get_logger
from utils.errors import AnalysisNotFoundError, InvalidFileError, ValidationError

logger = get_logger(__name__)

# Create router with prefix and tags
router = APIRouter(prefix="/api/v1", tags=["analysis"])


# ============== ANALYSIS ENDPOINTS ==============

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and analyze media",
    description="""
    Upload a media file for deepfake analysis.
    
    Supported formats:
    - Images: JPEG, PNG, WebP
    - Videos: MP4, WebM, MOV, AVI
    - Audio: MP3, WAV, OGG
    - Text: Plain text (use /analyze/text endpoint instead)
    
    The analysis runs asynchronously. Use the returned analysis_id
    to poll for results via GET /analyze/{analysis_id}.
    
    Maximum file size: 500MB
    Maximum video duration: 5 minutes
    """,
    responses={
        202: {"description": "Analysis started"},
        400: {"model": ErrorResponse, "description": "Invalid file"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal error"}
    }
)
async def analyze_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Media file to analyze"),
    generate_report: bool = Form(default=True, description="Generate PDF report"),
    generate_heatmaps: bool = Form(default=True, description="Generate GradCAM heatmaps"),
    defense_level: str = Form(default="standard", description="Adversarial defense level: none, standard, aggressive"),
    modalities: Optional[str] = Form(default=None, description="Comma-separated modalities to analyze (auto-detect if empty)"),
    deps: AnalysisDependencies = Depends(get_analysis_deps),
    user: Optional[dict] = Depends(get_current_user_optional),
    _rate_limited: None = Depends(check_rate_limit)
):
    """
    Upload and analyze media for deepfake detection.
    
    Returns immediately with analysis_id. Actual analysis runs in background.
    """
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    
    logger.info(
        f"Analysis request received",
        extra={
            "analysis_id": analysis_id,
            "file_name": file.filename,
            "content_type": file.content_type,
            "correlation_id": deps.correlation_id
        }
    )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate and sanitize file
        sanitizer = InputSanitizer(defense_level=defense_level)
        sanitized = await sanitizer.validate(
            file_content=file_content,
            filename=file.filename or "unnamed",
            content_type=file.content_type
        )
        
        # Parse modalities if provided
        parsed_modalities = None
        if modalities:
            parsed_modalities = [
                Modality(m.strip().lower()) 
                for m in modalities.split(",") 
                if m.strip()
            ]
        
        # Create analysis options
        options = AnalyzeOptions(
            modalities=parsed_modalities,
            generate_report=generate_report,
            generate_heatmaps=generate_heatmaps,
            defense_level=defense_level
        )
        
        # Upload file to storage
        file_key = f"uploads/{analysis_id}/{sanitized.original_filename}"
        await deps.storage.ensure_default_buckets()
        await deps.storage.upload_file(
            file=sanitized.content,
            bucket=deps.storage.bucket_uploads,
            object_key=file_key,
            content_type=sanitized.mime_type
        )
        
        # Create file input record
        file_input = FileInput(
            file_id=file_key,
            file_type=sanitized.file_type.value,
            original_filename=sanitized.original_filename,
            file_hash=sanitized.file_hash,
            file_size=sanitized.file_size,
            duration_seconds=sanitized.duration_seconds
        )
        
        # Create analysis document
        analysis = AnalysisDocument(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            input=file_input,
            options=options
        )
        
        # Insert into database
        await deps.db.insert_analysis(analysis)
        
        # Queue analysis job
        try:
            orchestrator = await get_orchestrator()
            await orchestrator.enqueue_analysis(
                analysis_id=analysis_id,
                options=options.model_dump()
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue job, will process synchronously: {e}")
            # If orchestrator unavailable, update status for manual processing
            await deps.db.update_analysis_status(
                analysis_id, 
                AnalysisStatus.PENDING,
                error_message="Queued for processing"
            )
        
        # Log audit event
        await deps.db.log_audit_event(
            event_type="analysis_created",
            resource_id=analysis_id,
            actor=user.get("user_id", "anonymous") if user else "anonymous",
            metadata={
                "filename": sanitized.original_filename,
                "file_hash": sanitized.file_hash,
                "file_size": sanitized.file_size
            }
        )
        
        logger.info(f"Analysis created: {analysis_id}")
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            created_at=analysis.created_at
        )
        
    except InvalidFileError as e:
        logger.warning(f"Invalid file upload: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Analysis creation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "ANALYSIS_CREATION_FAILED",
                "message": str(e)
            }
        )


@router.get(
    "/analyze/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get analysis status and results",
    description="""
    Get the current status and results of an analysis.
    
    Poll this endpoint to check analysis progress.
    Once status is 'completed', results will be included.
    """,
    responses={
        200: {"description": "Analysis found"},
        404: {"model": ErrorResponse, "description": "Analysis not found"}
    }
)
async def get_analysis(
    analysis_id: str = Path(..., description="Analysis ID"),
    db: DatabaseClient = Depends(get_db)
):
    """Get analysis status and basic results."""
    analysis = await db.get_analysis(analysis_id)
    
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "ANALYSIS_NOT_FOUND",
                "message": f"Analysis not found: {analysis_id}",
                "details": {"analysis_id": analysis_id}
            }
        )
    
    return AnalysisResponse(
        analysis_id=analysis.analysis_id,
        status=analysis.status,
        trust_score=analysis.trust_score,
        verdict=analysis.verdict,
        explanation=analysis.explanation,
        report_url=analysis.report_url,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at
    )


@router.get(
    "/analyze/{analysis_id}/detail",
    response_model=AnalysisDetailResponse,
    summary="Get detailed analysis results",
    description="""
    Get full detailed results including per-modality scores.
    
    Only available after analysis is completed.
    """,
    responses={
        200: {"description": "Detailed results"},
        404: {"model": ErrorResponse, "description": "Analysis not found"},
        400: {"model": ErrorResponse, "description": "Analysis not complete"}
    }
)
async def get_analysis_detail(
    analysis_id: str = Path(..., description="Analysis ID"),
    db: DatabaseClient = Depends(get_db)
):
    """Get detailed analysis results with all modality data."""
    analysis = await db.get_analysis(analysis_id)
    
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "ANALYSIS_NOT_FOUND",
                "message": f"Analysis not found: {analysis_id}"
            }
        )
    
    if analysis.status != AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ANALYSIS_NOT_COMPLETE",
                "message": f"Analysis is still {analysis.status.value}",
                "details": {"current_status": analysis.status.value}
            }
        )
    
    return AnalysisDetailResponse(
        analysis_id=analysis.analysis_id,
        status=analysis.status,
        trust_score=analysis.trust_score,
        verdict=analysis.verdict,
        explanation=analysis.explanation,
        report_url=analysis.report_url,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
        video_result=analysis.video_result,
        audio_result=analysis.audio_result,
        image_result=getattr(analysis, 'image_result', None),
        metadata_result=analysis.metadata_result,
        processing_time_seconds=analysis.processing_time_seconds,
        # XAI Enhancement Fields
        evidence_package=getattr(analysis, 'evidence_package', None),
        feature_importance=getattr(analysis, 'feature_importance', []),
        scientific_references=getattr(analysis, 'scientific_references', [])
    )


@router.delete(
    "/analyze/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete analysis",
    description="Delete an analysis and all associated data.",
    responses={
        204: {"description": "Analysis deleted"},
        404: {"model": ErrorResponse, "description": "Analysis not found"}
    }
)
async def delete_analysis(
    analysis_id: str = Path(..., description="Analysis ID"),
    db: DatabaseClient = Depends(get_db),
    storage: StorageClient = Depends(get_storage),
    user: dict = Depends(get_current_user)
):
    """Delete analysis and associated files."""
    # Check if analysis exists
    analysis = await db.get_analysis(analysis_id)
    
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "ANALYSIS_NOT_FOUND",
                "message": f"Analysis not found: {analysis_id}"
            }
        )
    
    # Delete from database
    await db.delete_analysis(analysis_id)
    
    # Delete associated files from storage
    try:
        # Delete uploaded file
        if analysis.input:
            await storage.delete_file(
                storage.bucket_uploads,
                analysis.input.file_id
            )
        
        # Delete preprocessed data
        prefix = f"preprocessed/{analysis_id}/"
        objects = await storage.list_objects(storage.bucket_preprocessed, prefix)
        for obj in objects:
            await storage.delete_file(storage.bucket_preprocessed, obj)
        
        # Delete results (heatmaps, reports)
        prefix = f"results/{analysis_id}/"
        objects = await storage.list_objects(storage.bucket_results, prefix)
        for obj in objects:
            await storage.delete_file(storage.bucket_results, obj)
            
    except Exception as e:
        logger.warning(f"Failed to delete some files for {analysis_id}: {e}")
    
    # Log audit event
    await db.log_audit_event(
        event_type="analysis_deleted",
        resource_id=analysis_id,
        actor=user.get("user_id", "anonymous") if user else "anonymous"
    )
    
    logger.info(f"Analysis deleted: {analysis_id}")
    
    return None


@router.get(
    "/analyze",
    response_model=List[AnalysisResponse],
    summary="List analyses",
    description="List analyses with optional filtering.",
    responses={
        200: {"description": "List of analyses"}
    }
)
async def list_analyses(
    status_filter: Optional[AnalysisStatus] = Query(
        default=None, 
        alias="status",
        description="Filter by status"
    ),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Skip count for pagination"),
    db: DatabaseClient = Depends(get_db)
):
    """List analyses with optional status filter."""
    analyses = await db.list_analyses(
        status=status_filter,
        limit=limit,
        offset=offset
    )
    
    return analyses


# ============== UTILITY ENDPOINTS ==============

@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is running and healthy.",
    tags=["system"]
)
async def health_check(
    db: DatabaseClient = Depends(get_db),
    storage: StorageClient = Depends(get_storage),
):
    """Health check endpoint for load balancers and model status monitoring."""
    from api.health import run_health_check
    return await run_health_check(db, storage)


@router.get(
    "/models",
    summary="List available models",
    description="Get list of available detection models and their status.",
    tags=["system"]
)
async def list_models():
    """List available detection models."""
    from models.registry import get_model_registry
    from models.manager import get_model_manager
    
    registry = get_model_registry()
    manager = get_model_manager()
    loaded_models = manager.get_loaded_models()
    
    models = []
    
    for name in registry.list_models(category=None):
        try:
            metadata = registry.get_model_metadata(name)
            is_loaded = name in loaded_models
            file_exists = registry.model_file_exists(name)
            
            category_val = metadata.category.value if hasattr(metadata.category, 'value') else str(metadata.category)
            
            models.append({
                "name": name,
                "category": category_val,
                "vram_mb": metadata.vram_mb,
                "loaded": is_loaded,
                "file_exists": file_exists,
                "version": metadata.version,
                "description": metadata.description
            })
        except Exception as e:
            logger.warning(f"Failed to get metadata for model {name}: {e}")
    
    return {
        "models": models,
        "count": len(models),
        "loaded_count": len(loaded_models),
        "vram_usage_mb": manager.get_vram_usage(),
        "vram_limit_mb": manager.max_vram_mb
    }


@router.get(
    "/stats",
    summary="Get analysis statistics",
    description="Get aggregate statistics about analyses.",
    tags=["system"]
)
async def get_stats(
    db: DatabaseClient = Depends(get_db)
):
    """Get analysis statistics using efficient count queries."""
    stats = {
        "total": 0,
        "by_status": {},
        "by_verdict": {}
    }
    
    for status_val in AnalysisStatus:
        count = await db.count_analyses(status=status_val)
        stats["by_status"][status_val.value] = count
        stats["total"] += count
    
    return stats


# ============== REPORT ENDPOINTS ==============

@router.get(
    "/analyze/{analysis_id}/report",
    summary="Get analysis report",
    description="Get or generate PDF report for analysis.",
    responses={
        200: {"description": "Report URL"},
        404: {"model": ErrorResponse, "description": "Analysis not found"},
        400: {"model": ErrorResponse, "description": "Analysis not complete"}
    }
)
async def get_report(
    analysis_id: str = Path(..., description="Analysis ID"),
    regenerate: bool = Query(default=False, description="Force regenerate report"),
    db: DatabaseClient = Depends(get_db),
    storage: StorageClient = Depends(get_storage)
):
    """Get or generate PDF forensic report."""
    analysis = await db.get_analysis(analysis_id)
    
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "ANALYSIS_NOT_FOUND", "message": f"Analysis not found: {analysis_id}"}
        )
    
    if analysis.status != AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "ANALYSIS_NOT_COMPLETE", "message": "Analysis must be complete to generate report"}
        )
    
    # Check if report exists
    if analysis.report_url and not regenerate:
        return {"report_url": analysis.report_url}
    
    # Generate report (placeholder - actual implementation in forensics module)
    report_key = f"results/{analysis_id}/report.pdf"
    
    try:
        # Generate presigned URL for download
        report_url = await storage.get_presigned_url(
            storage.bucket_results,
            report_key,
            expires_seconds=3600
        )
        
        return {"report_url": report_url}
        
    except Exception as e:
        logger.warning(f"Report not available: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "REPORT_NOT_FOUND", "message": "Report has not been generated yet"}
        )


@router.get(
    "/analyze/{analysis_id}/heatmaps",
    summary="Get analysis heatmaps",
    description="Get GradCAM heatmap URLs for analysis.",
    responses={
        200: {"description": "Heatmap URLs"},
        404: {"model": ErrorResponse, "description": "Analysis not found"}
    }
)
async def get_heatmaps(
    analysis_id: str = Path(..., description="Analysis ID"),
    db: DatabaseClient = Depends(get_db),
    storage: StorageClient = Depends(get_storage)
):
    """Get GradCAM heatmap URLs for visualization."""
    analysis = await db.get_analysis(analysis_id)
    
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "ANALYSIS_NOT_FOUND", "message": f"Analysis not found: {analysis_id}"}
        )
    
    # List heatmap files
    prefix = f"results/{analysis_id}/heatmaps/"
    
    try:
        objects = await storage.list_objects(storage.bucket_results, prefix)
        
        heatmaps = []
        for obj in objects:
            url = await storage.get_presigned_url(
                storage.bucket_results,
                obj,
                expires_seconds=3600
            )
            heatmaps.append({
                "key": obj,
                "url": url
            })
        
        return {"heatmaps": heatmaps, "count": len(heatmaps)}
        
    except Exception as e:
        logger.warning(f"Failed to list heatmaps: {e}")
        return {"heatmaps": [], "count": 0}


@router.get(
    "/analyze/{analysis_id}/xai",
    summary="Get XAI explanations",
    description="Get Explainable AI explanations including feature importance, scientific references, and evidence package",
    responses={
        200: {"description": "XAI data for the analysis"},
        404: {"model": ErrorResponse, "description": "Analysis not found"}
    }
)
async def get_xai_explanations(
    analysis_id: str = Path(..., description="Analysis ID"),
    db: DatabaseClient = Depends(get_db),
    storage: StorageClient = Depends(get_storage)
):
    """Get XAI explanations, feature importance, and evidence package."""
    analysis = await db.get_analysis(analysis_id)

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "ANALYSIS_NOT_FOUND", "message": f"Analysis not found: {analysis_id}"}
        )

    # Build XAI response from AnalysisDocument fields
    explanation = analysis.explanation
    evidence_pkg = analysis.evidence_package

    xai_response = {
        "analysis_id": analysis_id,
        "image_xai": None,
        "video_xai": None,
        "audio_xai": None,
        "text_xai": None,
        "evidence_package": evidence_pkg.model_dump() if evidence_pkg else None,
    }

    summary_text = explanation.summary if explanation else ""
    confidence_rationale = explanation.confidence_rationale if explanation else ""
    methodology = explanation.methodology_used if explanation else []

    # Populate image XAI
    if analysis.image_result:
        img = analysis.image_result
        xai_response["image_xai"] = {
            "explanation": {
                "summary": summary_text,
                "confidence_rationale": confidence_rationale,
                "methodology_used": methodology,
            },
            "manipulation_regions": [
                mr.model_dump() for mr in (img.manipulation_regions or [])
            ],
            "heatmap_urls": [img.heatmap_url] if img.heatmap_url else [],
            "overlay_url": img.heatmap_url,
        }

    # Populate video XAI
    if analysis.video_result:
        vid = analysis.video_result
        xai_response["video_xai"] = {
            "explanation": {
                "summary": summary_text,
                "confidence_rationale": confidence_rationale,
                "methodology_used": methodology,
            },
            "manipulation_regions": [],
            "heatmap_urls": vid.frame_heatmap_urls or [],
            "temporal_heatmap_url": vid.temporal_heatmap_url,
        }

    # Populate audio XAI
    if analysis.audio_result:
        aud = analysis.audio_result
        xai_response["audio_xai"] = {
            "explanation": {
                "summary": summary_text,
                "confidence_rationale": confidence_rationale,
                "methodology_used": methodology,
            },
            "artifact_regions": [
                ar.model_dump() for ar in (aud.artifact_regions or [])
            ],
            "spectrogram_overlay_url": aud.spectrogram_url,
        }

    # Populate text XAI
    if analysis.text_result:
        txt = analysis.text_result
        xai_response["text_xai"] = {
            "explanation": {
                "summary": summary_text,
                "confidence_rationale": confidence_rationale,
                "methodology_used": methodology,
            },
            "token_attributions": [
                ta.model_dump() for ta in (txt.token_attributions or [])
            ],
        }

    return xai_response


@router.get(
    "/analyze/{analysis_id}/xai/heatmaps",
    summary="Get XAI heatmap overlays",
    description="Get heatmap overlay URLs for XAI visualization",
    responses={
        200: {"description": "XAI heatmap URLs"},
        404: {"model": ErrorResponse, "description": "Analysis not found"}
    }
)
async def get_xai_heatmaps(
    analysis_id: str = Path(..., description="Analysis ID"),
    db: DatabaseClient = Depends(get_db),
    storage: StorageClient = Depends(get_storage)
):
    """Get XAI heatmap overlay URLs from the evidence package."""
    analysis = await db.get_analysis(analysis_id)

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "ANALYSIS_NOT_FOUND", "message": f"Analysis not found: {analysis_id}"}
        )

    heatmaps = []

    # Get heatmaps from evidence package
    if analysis.evidence_package and analysis.evidence_package.visual_evidence:
        for ve in analysis.evidence_package.visual_evidence:
            if ve.artifact_type in ("heatmap", "overlay"):
                heatmaps.append({
                    "type": ve.artifact_type,
                    "url": ve.url,
                    "description": ve.description,
                })

    # Also include direct heatmap URL from image result
    if analysis.image_result and analysis.image_result.heatmap_url:
        heatmaps.append({
            "type": "gradcam_overlay",
            "url": analysis.image_result.heatmap_url,
            "description": "GradCAM++ attention overlay",
        })

    return {"heatmaps": heatmaps, "count": len(heatmaps)}


# Export router
__all__ = ["router"]
