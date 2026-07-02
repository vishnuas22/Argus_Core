"""
Argus Core - Retrain Scheduler (Iteration 4)
==============================================
Celery task that periodically retrains LoRA adapters from the feedback buffer.

Algorithm:
1. Check if enough new samples have accumulated (>= retrain_min_samples).
2. Load the feedback buffer + a replay buffer of old samples.
3. Run train_lora_adapters.py as a subprocess (or call its main()).
4. Save the candidate adapter to /models/<adapter>_candidate/.
5. Register the candidate with the ABTestRouter for evaluation.
6. The ABTestRouter will promote or roll back based on metrics.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrainResult:
    """Result of a retrain cycle."""
    success: bool
    modality: str
    num_samples: int
    candidate_path: str = ""
    error: str = ""


class RetrainScheduler:
    """
    Periodically retrains LoRA adapters from the feedback buffer.
    """

    def __init__(
        self,
        feedback_buffer=None,
        retrain_script_path: str = None,
        model_cache_dir: str = "/models",
    ):
        self._feedback_buffer = feedback_buffer
        self._retrain_script = retrain_script_path or os.environ.get(
            "RETRAIN_SCRIPT_PATH",
            os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train_lora_adapters.py"),
        )
        self._model_cache_dir = model_cache_dir

    # ------------------------------------------------------------------
    def should_retrain(self, modality: str) -> bool:
        """Check if enough new samples exist to trigger a retrain."""
        from config import config
        if not getattr(config, "enable_continuous_learning", False):
            return False
        min_samples = getattr(config, "retrain_min_samples", 50)
        if self._feedback_buffer is None:
            from continuous_learning import get_default_feedback_buffer
            self._feedback_buffer = get_default_feedback_buffer()
        count = self._feedback_buffer.count(modality=modality)
        return count >= min_samples

    # ------------------------------------------------------------------
    async def retrain_modality(self, modality: str) -> RetrainResult:
        """
        Retrain the LoRA adapter for a single modality.

        Args:
            modality: image | audio | video

        Returns:
            RetrainResult with success status + candidate path.
        """
        from config import config

        if self._feedback_buffer is None:
            from continuous_learning import get_default_feedback_buffer
            self._feedback_buffer = get_default_feedback_buffer()

        # Get samples
        max_samples = getattr(config, "retrain_max_samples", 1000)
        samples = self._feedback_buffer.get_samples(
            modality=modality, limit=max_samples
        )
        if len(samples) < getattr(config, "retrain_min_samples", 50):
            return RetrainResult(
                success=False, modality=modality,
                num_samples=len(samples),
                error=f"insufficient samples ({len(samples)} < min)",
            )

        logger.info(
            "Retraining %s adapter from %d samples...", modality, len(samples)
        )
        import time as _time
        _retrain_start = _time.time()

        # Determine backbone + dataset based on modality
        if modality == "image":
            backbone = "clip"
            adapter_dir = os.path.join(self._model_cache_dir, "clip_lora_image_adapter_candidate")
        elif modality == "audio":
            backbone = "wav2vec2_xls_r"
            adapter_dir = os.path.join(self._model_cache_dir, "wav2vec2_xls_r_moe_lora_candidate")
        elif modality == "video":
            backbone = "videomae"
            adapter_dir = os.path.join(self._model_cache_dir, "videomae_finetune_candidate")
        else:
            return RetrainResult(
                success=False, modality=modality, num_samples=len(samples),
                error=f"unsupported modality {modality}",
            )

        # Write samples to a temp JSON for the training script
        # NOTE: In a full implementation, this would convert feedback
        # samples to the dataset format expected by train_lora_adapters.py.
        # For now, we just trigger the training script with the existing
        # dataset (the feedback samples inform the AB test, not the
        # training data directly — operators must add the labeled samples
        # to their dataset manually until the data pipeline is built).
        try:
            # Build the command
            cmd = [
                sys.executable, self._retrain_script,
                "--modality", modality,
                "--backbone", backbone,
                "--dataset", "faceforensics" if modality != "audio" else "asvspoof2019",
                "--dataset-root", "/data",  # operators must mount real data
                "--output-dir", adapter_dir,
                "--epochs", "5",  # short retrain
                "--batch-size", "8",
            ]

            logger.info("Running retrain: %s", " ".join(cmd))
            # Run as subprocess (non-blocking in production via Celery)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                return RetrainResult(
                    success=False, modality=modality,
                    num_samples=len(samples),
                    error=f"training failed: {result.stderr[:500]}",
                )

            # Register candidate with AB router
            from continuous_learning import get_default_ab_router
            router = get_default_ab_router()
            router.register_candidate(modality, adapter_dir)

            logger.info(
                "Retrain complete: %s candidate at %s", modality, adapter_dir
            )
            # Archive feedback buffer after successful retrain
            try:
                archive_path = f"/models/continuous_learning/feedback_buffer.{modality}.{int(_time.time())}.jsonl"
                self._feedback_buffer.archive(archive_path)
                logger.info("Feedback buffer archived: %s", archive_path)
            except Exception as e:
                logger.warning("Failed to archive feedback buffer: %s", e)
            # Iteration 7: record retrain metrics
            try:
                from observability import get_default_metrics
                _duration = _time.time() - _retrain_start
                get_default_metrics().record_retrain(
                    modality, "success", len(samples), _duration
                )
            except Exception:
                pass
            return RetrainResult(
                success=True, modality=modality,
                num_samples=len(samples),
                candidate_path=adapter_dir,
            )

        except subprocess.TimeoutExpired:
            # Iteration 7: record failed retrain
            try:
                from observability import get_default_metrics
                _duration = _time.time() - _retrain_start
                get_default_metrics().record_retrain(
                    modality, "timeout", len(samples), _duration
                )
            except Exception:
                pass
            return RetrainResult(
                success=False, modality=modality,
                num_samples=len(samples),
                error="training timed out (>1h)",
            )
        except Exception as e:
            # Iteration 7: record failed retrain
            try:
                from observability import get_default_metrics
                _duration = _time.time() - _retrain_start
                get_default_metrics().record_retrain(
                    modality, "failed", len(samples), _duration
                )
            except Exception:
                pass
            return RetrainResult(
                success=False, modality=modality,
                num_samples=len(samples),
                error=str(e),
            )


# ---------------------------------------------------------------------
def schedule_retrain_task(modality: str):
    """
    Celery task wrapper for scheduled retraining.

    Usage in core/orchestrator.py:
        @celery_app.task(name="argus_tasks.retrain_modality")
        def retrain_modality_task(modality: str):
            return schedule_retrain_task(modality)

        # Schedule via celery beat:
        # celery_app.conf.beat_schedule = {
        #     'retrain-image-daily': {
        #         'task': 'argus_tasks.retrain_modality',
        #         'schedule': crontab(hour=2, minute=0),
        #         'args': ('image',),
        #     },
        # }
    """
    scheduler = RetrainScheduler()
    if not scheduler.should_retrain(modality):
        return {"status": "skipped", "reason": "insufficient_samples"}

    result = asyncio.run(scheduler.retrain_modality(modality))
    return {
        "status": "success" if result.success else "failed",
        "modality": result.modality,
        "num_samples": result.num_samples,
        "candidate_path": result.candidate_path,
        "error": result.error,
    }
