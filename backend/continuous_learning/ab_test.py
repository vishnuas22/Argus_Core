"""
Argus Core - A/B Test Router (Iteration 4)
============================================
Routes a fraction of inference requests to a candidate adapter for
evaluation, then promotes or rolls back based on metrics.

Algorithm:
1. When a candidate adapter is registered, route X% of traffic to it.
2. Collect predictions + feedback for both production and candidate.
3. After N samples, compare metrics (accuracy, EER, calibration).
4. If candidate is better: promote (swap production <-> candidate).
5. If candidate is worse: roll back (unregister candidate).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CandidateMetrics:
    """Metrics for a candidate adapter."""
    modality: str
    candidate_path: str
    num_predictions: int = 0
    num_correct: int = 0
    num_feedback: int = 0
    # For AUC computation
    scores: List[float] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    # For latency comparison
    latencies_ms: List[float] = field(default_factory=list)


class ABTestRouter:
    """
    Routes traffic between production and candidate adapters.
    """

    _PERSIST_PATH = "/models/continuous_learning/ab_test_candidates.json"

    def __init__(self, ab_ratio: float = 0.1, min_samples_for_decision: int = 100):
        self._ab_ratio = ab_ratio
        self._min_samples = min_samples_for_decision
        self._candidates: Dict[str, CandidateMetrics] = {}
        self._lock = threading.Lock()
        self._rng = random.Random(42)
        self._load_from_disk()
        logger.info(
            "ABTestRouter initialized: ratio=%.2f, min_samples=%d",
            ab_ratio, min_samples_for_decision,
        )

    def _load_from_disk(self) -> None:
        """Load persisted candidate state from disk."""
        try:
            if os.path.exists(self._PERSIST_PATH):
                with open(self._PERSIST_PATH) as f:
                    data = json.load(f)
                for modality, info in data.items():
                    self._candidates[modality] = CandidateMetrics(
                        modality=modality,
                        candidate_path=info.get("candidate_path", ""),
                        num_predictions=info.get("num_predictions", 0),
                        num_correct=info.get("num_correct", 0),
                        num_feedback=info.get("num_feedback", 0),
                        scores=info.get("scores", []),
                        labels=info.get("labels", []),
                        latencies_ms=info.get("latencies_ms", []),
                    )
                logger.info("Loaded %d A/B test candidates from disk", len(self._candidates))
        except Exception as e:
            logger.debug("No persisted A/B test state: %s", e)

    def _save_to_disk(self) -> None:
        """Persist candidate state to disk."""
        try:
            os.makedirs(os.path.dirname(self._PERSIST_PATH), exist_ok=True)
            data = {}
            for modality, metrics in self._candidates.items():
                data[modality] = {
                    "candidate_path": metrics.candidate_path,
                    "num_predictions": metrics.num_predictions,
                    "num_correct": metrics.num_correct,
                    "num_feedback": metrics.num_feedback,
                    "scores": metrics.scores[-100:],  # keep last 100 for disk
                    "labels": metrics.labels[-100:],
                    "latencies_ms": metrics.latencies_ms[-100:],
                }
            with open(self._PERSIST_PATH, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug("Failed to persist A/B test state: %s", e)

    # ------------------------------------------------------------------
    def register_candidate(self, modality: str, candidate_path: str) -> None:
        """Register a new candidate adapter for A/B testing."""
        with self._lock:
            self._candidates[modality] = CandidateMetrics(
                modality=modality, candidate_path=candidate_path,
            )
            self._save_to_disk()
        logger.info(
            "Candidate registered: %s -> %s (routing %.0f%% of traffic)",
            modality, candidate_path, self._ab_ratio * 100,
        )

    def unregister_candidate(self, modality: str) -> None:
        """Remove a candidate (after roll back or promote)."""
        with self._lock:
            self._candidates.pop(modality, None)
            self._save_to_disk()

    # ------------------------------------------------------------------
    def should_route_to_candidate(self, modality: str) -> bool:
        """
        Decide whether the current request should use the candidate adapter.

        Args:
            modality: image | audio | video

        Returns:
            True if the request should use the candidate, False for production.
        """
        with self._lock:
            if modality not in self._candidates:
                return False
            # Check if we've collected enough samples for a decision
            metrics = self._candidates[modality]
            if metrics.num_predictions >= self._min_samples:
                # Decision time — stop routing to candidate until promoted/rolled back
                return False
        return self._rng.random() < self._ab_ratio

    # ------------------------------------------------------------------
    def record_prediction(
        self,
        modality: str,
        score: float,
        latency_ms: float,
        is_candidate: bool,
    ) -> None:
        """Record a prediction for A/B metrics."""
        if not is_candidate:
            return  # Only track candidate metrics for now
        with self._lock:
            if modality not in self._candidates:
                return
            metrics = self._candidates[modality]
            metrics.num_predictions += 1
            metrics.scores.append(score)
            metrics.latencies_ms.append(latency_ms)
        # Iteration 7: record A/B test prediction
        try:
            from observability import get_default_metrics
            get_default_metrics().ab_test_predictions.labels(
                modality=modality, is_candidate=str(is_candidate)
            ).inc()
        except Exception:
            pass

    def record_feedback(
        self,
        modality: str,
        score: float,
        label: int,
        is_candidate: bool,
    ) -> None:
        """Record feedback (ground-truth label) for a prediction."""
        if not is_candidate:
            return
        with self._lock:
            if modality not in self._candidates:
                return
            metrics = self._candidates[modality]
            metrics.num_feedback += 1
            metrics.labels.append(label)
            pred = 1 if score >= 0.5 else 0
            if pred == label:
                metrics.num_correct += 1

    # ------------------------------------------------------------------
    def evaluate_candidate(self, modality: str) -> Dict[str, Any]:
        """
        Evaluate the candidate's metrics. Returns a decision dict.

        Returns:
            Dict with:
                - "decision": "promote" | "rollback" | "insufficient"
                - "candidate_accuracy": float
                - "candidate_auc": float
                - "candidate_latency_ms": float
                - "num_samples": int
        """
        with self._lock:
            if modality not in self._candidates:
                return {"decision": "no_candidate"}
            metrics = self._candidates[modality]

        if metrics.num_feedback < self._min_samples:
            return {
                "decision": "insufficient",
                "num_samples": metrics.num_feedback,
                "min_required": self._min_samples,
            }

        # Compute candidate metrics
        accuracy = metrics.num_correct / max(metrics.num_feedback, 1)
        try:
            from sklearn.metrics import roc_auc_score
            if len(set(metrics.labels)) >= 2:
                auc = float(roc_auc_score(metrics.labels, metrics.scores[:len(metrics.labels)]))
            else:
                auc = 0.5
        except Exception:
            auc = 0.5
        avg_latency = float(sum(metrics.latencies_ms) / max(len(metrics.latencies_ms), 1))

        # Decision logic: promote if accuracy > 0.85 AND auc > 0.9
        # (These thresholds should be tuned per deployment.)
        if accuracy > 0.85 and auc > 0.9:
            decision = "promote"
        elif accuracy < 0.7 or auc < 0.75:
            decision = "rollback"
        else:
            decision = "insufficient"  # need more data

        # Iteration 7: record A/B test metrics
        try:
            from observability import get_default_metrics
            get_default_metrics().record_ab_test(modality, True, accuracy, auc)
        except Exception:
            pass

        return {
            "decision": decision,
            "candidate_accuracy": accuracy,
            "candidate_auc": auc,
            "candidate_latency_ms": avg_latency,
            "num_samples": metrics.num_feedback,
        }

    # ------------------------------------------------------------------
    def promote_candidate(self, modality: str) -> bool:
        """
        Promote the candidate to production.

        Steps:
        1. Archive the current production adapter.
        2. Move the candidate adapter into production.
        3. Reload the model in the inference engine.
        4. Unregister the candidate.
        """
        with self._lock:
            if modality not in self._candidates:
                return False
            metrics = self._candidates[modality]
            candidate_path = metrics.candidate_path

        # Map modality to production adapter path
        from config import config
        production_paths = {
            "image": os.path.join(config.model_cache_dir, "clip_lora_adapter"),
            "audio": os.path.join(config.model_cache_dir, "wav2vec2_xls_r_moe_lora"),
            "video": os.path.join(config.model_cache_dir, "videomae_finetune"),
        }
        prod_path = production_paths.get(modality)
        if not prod_path:
            logger.error("No production adapter path for modality: %s", modality)
            return False

        try:
            # 1. Archive production adapter
            archive_path = prod_path + f".bak.{int(time.time())}"
            if os.path.exists(prod_path):
                shutil.move(prod_path, archive_path)
                logger.info("Archived production adapter: %s -> %s", prod_path, archive_path)

            # 2. Move candidate to production
            shutil.move(candidate_path, prod_path)
            logger.info("Promoted candidate: %s -> %s", candidate_path, prod_path)

            # 3. Reload model in the inference engine
            try:
                from models.manager import get_model_manager
                manager = get_model_manager()
                manager.unload_model(f"{modality}_lora_adapter")
                logger.info("Unloaded old adapter, will reload on next inference")
            except Exception as e:
                logger.warning("Model reload after promote failed: %s", e)

            self.unregister_candidate(modality)
            logger.info(
                "PROMOTED candidate for %s: acc=%.4f, samples=%d",
                modality, metrics.num_correct / max(metrics.num_feedback, 1),
                metrics.num_feedback,
            )
            return True
        except Exception as e:
            logger.error("Promote failed for %s: %s", modality, e)
            return False

    def rollback_candidate(self, modality: str) -> bool:
        """
        Roll back the candidate (keep production).

        Removes the candidate directory and unregisters.
        """
        with self._lock:
            if modality not in self._candidates:
                return False
            metrics = self._candidates[modality]
            candidate_path = metrics.candidate_path

        try:
            if os.path.exists(candidate_path):
                shutil.rmtree(candidate_path, ignore_errors=True)
                logger.info("Removed candidate directory: %s", candidate_path)
        except Exception as e:
            logger.warning("Failed to remove candidate: %s", e)

        logger.warning(
            "ROLLED BACK candidate for %s: %s (acc=%.4f, samples=%d)",
            modality, metrics.candidate_path,
            metrics.num_correct / max(metrics.num_feedback, 1),
            metrics.num_feedback,
        )
        self.unregister_candidate(modality)
        return True


# ---------------------------------------------------------------------
_default_router: Optional[ABTestRouter] = None


def get_default_ab_router() -> ABTestRouter:
    global _default_router
    if _default_router is None:
        from config import config
        _default_router = ABTestRouter(
            ab_ratio=getattr(config, "retrain_ab_test_ratio", 0.1),
        )
    return _default_router
