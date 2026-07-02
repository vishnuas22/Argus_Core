"""
Argus Core - Continuous Learning Module (Iteration 4)
======================================================
Online LoRA adapter retraining as new labeled samples arrive.

Research grounding:
- Continuous learning for deepfake detection (Yan et al., "DeepfakeBench",
  NeurIPS 2023 D&B): the deepfake detection distribution is non-stationary
  — new forgery families appear weekly. Static detectors degrade; online
  retraining is required.
- Replay-based continual learning (Chaudhry et al., "On Tiny Episodic
  Memories", NeurIPS Workshop 2019): mix new samples with a small buffer
  of old samples to avoid catastrophic forgetting.
- A/B testing for safe rollout: route a fraction of traffic to the new
  adapter; if metrics improve, promote; if they regress, roll back.

Components:
1. FeedbackBuffer — appends labeled samples (modality, input_hash,
   label, predicted_score) to a JSON Lines file.
2. RetrainScheduler — Celery task that runs every N hours, checks if
   enough new samples have accumulated, and triggers a LoRA retrain.
3. ABTestRouter — routes a configurable fraction of inference requests
   to the candidate adapter for evaluation.
4. PromoteOrRollback — compares candidate vs production metrics;
   promotes or rolls back automatically.

Strict-compat: pure-additive. Existing analyzers unchanged. Operators
opt in via config.enable_continuous_learning.
"""

from continuous_learning.feedback_buffer import (
    FeedbackBuffer,
    FeedbackSample,
    get_default_feedback_buffer,
)
from continuous_learning.retrain_scheduler import (
    RetrainScheduler,
    schedule_retrain_task,
)
from continuous_learning.ab_test import (
    ABTestRouter,
    get_default_ab_router,
)

__all__ = [
    "FeedbackBuffer", "FeedbackSample", "get_default_feedback_buffer",
    "RetrainScheduler", "schedule_retrain_task",
    "ABTestRouter", "get_default_ab_router",
]
