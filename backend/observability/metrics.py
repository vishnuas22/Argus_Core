"""
Argus Core - Prometheus Metrics Implementation (Iteration 6)
=============================================================
"""

from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry,
        make_asgi_app,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = Summary = None

from utils.logging import get_logger

logger = get_logger(__name__)


class ArgusMetrics:
    """
    All Prometheus metrics for the Argus platform.
    """

    def __init__(self):
        if not _PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_client not installed; metrics disabled. "
                "Install with: pip install prometheus_client"
            )
            return

        # Inference
        self.inference_total = Counter(
            "argus_inference_total",
            "Total inferences run",
            ["modality", "verdict"],
        )
        self.inference_latency = Histogram(
            "argus_inference_latency_seconds",
            "Inference latency in seconds",
            ["modality"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )

        # Drift
        self.drift_score = Gauge(
            "argus_drift_score",
            "Combined drift score [0, 1]",
            ["modality"],
        )
        self.drift_severity = Gauge(
            "argus_drift_severity",
            "Drift severity (0=none, 1=moderate, 2=major)",
            ["modality"],
        )
        self.drift_psi = Gauge(
            "argus_drift_psi",
            "Population Stability Index",
            ["modality"],
        )
        self.drift_mmd = Gauge(
            "argus_drift_mmd",
            "Maximum Mean Discrepancy",
            ["modality"],
        )

        # Retrain
        self.retrain_total = Counter(
            "argus_retrain_total",
            "Total retrain cycles",
            ["modality", "status"],
        )
        self.retrain_samples = Gauge(
            "argus_retrain_samples",
            "Number of samples in current retrain cycle",
            ["modality"],
        )
        self.retrain_duration = Histogram(
            "argus_retrain_duration_seconds",
            "Retrain cycle duration",
            ["modality"],
            buckets=(60, 300, 900, 1800, 3600, 7200, 14400),
        )

        # A/B test
        self.ab_test_predictions = Counter(
            "argus_ab_test_predictions",
            "A/B test predictions",
            ["modality", "is_candidate"],
        )
        self.ab_test_accuracy = Gauge(
            "argus_ab_test_accuracy",
            "A/B test accuracy",
            ["modality", "is_candidate"],
        )
        self.ab_test_auc = Gauge(
            "argus_ab_test_auc",
            "A/B test AUC",
            ["modality", "is_candidate"],
        )

        # Calibration
        self.calibration_ece = Gauge(
            "argus_calibration_ece",
            "Expected Calibration Error",
            ["modality"],
        )
        self.calibration_brier = Gauge(
            "argus_calibration_brier",
            "Brier score",
            ["modality"],
        )
        self.calibration_temperature = Gauge(
            "argus_calibration_temperature",
            "Fitted temperature scaler T",
            ["modality"],
        )

        # Adversarial defense
        self.adversarial_flagged = Counter(
            "argus_adversarial_flagged_total",
            "Inputs flagged by adversarial defenses",
            ["modality", "defense"],
        )
        self.conformal_route_to_human = Counter(
            "argus_conformal_route_to_human_total",
            "Inputs routed to human review by conformal prediction",
            ["modality"],
        )

        # Feedback buffer
        self.feedback_buffer_size = Gauge(
            "argus_feedback_buffer_size",
            "Number of samples in the feedback buffer",
            ["modality"],
        )

        # Model loading
        self.model_loaded = Gauge(
            "argus_model_detector_loaded",
            "Whether a model is loaded (1=yes, 0=no)",
            ["detector_name"],
        )

        # Certified robustness
        self.certified_radius = Histogram(
            "argus_certified_robustness_radius",
            "Certified ℓ₂ radius",
            ["modality"],
            buckets=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
        )
        self.certified_total = Counter(
            "argus_certified_robustness_total",
            "Total certification attempts",
            ["modality", "status"],
        )

        # Watermarking
        self.watermark_embedded = Counter(
            "argus_watermark_embedded_total",
            "Total watermarks embedded",
            ["adapter_name"],
        )
        self.watermark_verified = Counter(
            "argus_watermark_verified_total",
            "Total watermarks verified",
            ["adapter_name", "success"],
        )

        logger.info("ArgusMetrics initialized (prometheus_client available)")

    # ------------------------------------------------------------------
    def record_inference(self, modality: str, verdict: str, latency_s: float):
        """Record a single inference."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.inference_total.labels(modality=modality, verdict=verdict).inc()
        self.inference_latency.labels(modality=modality).observe(latency_s)

    def record_drift(self, modality: str, score: float, severity: str,
                     psi: float, mmd: float):
        """Record a drift check result."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.drift_score.labels(modality=modality).set(score)
        sev_map = {"none": 0, "moderate": 1, "major": 2}
        self.drift_severity.labels(modality=modality).set(
            sev_map.get(severity, 0)
        )
        self.drift_psi.labels(modality=modality).set(psi)
        self.drift_mmd.labels(modality=modality).set(mmd)

    def record_retrain(self, modality: str, status: str, num_samples: int,
                       duration_s: float):
        """Record a retrain cycle."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.retrain_total.labels(modality=modality, status=status).inc()
        self.retrain_samples.labels(modality=modality).set(num_samples)
        self.retrain_duration.labels(modality=modality).observe(duration_s)

    def record_ab_test(self, modality: str, is_candidate: bool,
                       accuracy: float, auc: float):
        """Record A/B test metrics."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.ab_test_predictions.labels(
            modality=modality, is_candidate=str(is_candidate)
        ).inc()
        self.ab_test_accuracy.labels(
            modality=modality, is_candidate=str(is_candidate)
        ).set(accuracy)
        self.ab_test_auc.labels(
            modality=modality, is_candidate=str(is_candidate)
        ).set(auc)

    def record_calibration(self, modality: str, ece: float, brier: float,
                           temperature: float):
        """Record calibration metrics."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.calibration_ece.labels(modality=modality).set(ece)
        self.calibration_brier.labels(modality=modality).set(brier)
        self.calibration_temperature.labels(modality=modality).set(temperature)

    def record_adversarial_flag(self, modality: str, defense: str):
        """Record an adversarial defense flag."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.adversarial_flagged.labels(modality=modality, defense=defense).inc()

    def record_conformal_route(self, modality: str):
        """Record a conformal route-to-human event."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.conformal_route_to_human.labels(modality=modality).inc()

    def record_feedback_buffer(self, modality: str, size: int):
        """Record feedback buffer size."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.feedback_buffer_size.labels(modality=modality).set(size)

    def record_model_loaded(self, detector_name: str, loaded: bool):
        """Record model loading state."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.model_loaded.labels(detector_name=detector_name).set(
            1 if loaded else 0
        )

    def record_certification(self, modality: str, success: bool,
                             radius: float):
        """Record a certification attempt."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.certified_total.labels(
            modality=modality, status="success" if success else "abstained"
        ).inc()
        if success:
            self.certified_radius.labels(modality=modality).observe(radius)


# ---------------------------------------------------------------------
_default_metrics: Optional[ArgusMetrics] = None


def get_default_metrics() -> ArgusMetrics:
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = ArgusMetrics()
    return _default_metrics
