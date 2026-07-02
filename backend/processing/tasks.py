"""
Argus Core - Celery Tasks
=========================
Asynchronous task processing for heavy operations.

Handles background processing tasks like video analysis,
model inference, and report generation.
"""

from core.orchestrator import celery_app  # Use single canonical Celery app
from utils.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(name="tasks.health_check")
def health_check():
    """
    Health check task to verify Celery worker is operational.
    
    Returns:
        Dict with status and worker info
    """
    return {
        "status": "healthy",
        "worker": "celery",
        "message": "Celery worker is operational"
    }


@celery_app.task(name="tasks.process_video_analysis")
def process_video_analysis(file_id: str, analysis_id: str):
    """
    Process video analysis asynchronously.

    Routes to the main orchestrator pipeline.
    
    Args:
        file_id: ID of the uploaded file
        analysis_id: ID of the analysis record
        
    Returns:
        Dict with analysis results
    """
    logger.info("Delegating video analysis %s to orchestrator pipeline", analysis_id)
    from core.orchestrator import run_analysis_pipeline
    return run_analysis_pipeline.delay(analysis_id)


@celery_app.task(name="tasks.generate_report")
def generate_report(analysis_id: str, report_type: str = "pdf"):
    """
    Generate analysis report asynchronously.

    Routes to the forensics report generator via the orchestrator.
    
    Args:
        analysis_id: ID of the completed analysis
        report_type: Type of report (pdf, json, etc.)
        
    Returns:
        Dict with report generation status
    """
    logger.info("Delegating report generation for %s to orchestrator", analysis_id)
    from core.orchestrator import generate_report_task
    return generate_report_task.delay(analysis_id)
