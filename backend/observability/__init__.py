"""
Argus Core - Prometheus Observability (Iteration 6)
====================================================
Prometheus metrics for drift, retrain, A/B test, inference latency,
calibration, and adversarial defense.

Metrics exposed:
- argus_inference_total{modality, verdict}
- argus_inference_latency_seconds{modality} (histogram)
- argus_drift_score{modality} (gauge)
- argus_drift_severity{modality} (gauge: 0=none, 1=moderate, 2=major)
- argus_retrain_total{modality, status} (counter)
- argus_retrain_samples{modality} (gauge)
- argus_ab_test_predictions{modality, is_candidate} (counter)
- argus_ab_test_accuracy{modality, is_candidate} (gauge)
- argus_calibration_ece{modality} (gauge)
- argus_adversarial_flagged_total{modality, defense} (counter)
- argus_conformal_route_to_human_total{modality} (counter)
- argus_feedback_buffer_size{modality} (gauge)
- argus_model_loaded{detector_name} (gauge: 1=loaded, 0=not)
- argus_certified_robustness_radius{modality} (histogram)

The /metrics endpoint is exposed by the FastAPI app via prometheus_client.
"""

from observability.metrics import (
    ArgusMetrics,
    get_default_metrics,
)

__all__ = [
    "ArgusMetrics",
    "get_default_metrics",
]
