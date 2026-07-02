"""
Argus Core - Drift Detection Module (Iteration 2)
===================================================
Production drift detection on deepfake detector embeddings.

Research grounding:
- Population Stability Index (PSI): standard production drift metric.
  PSI > 0.25 = major drift, PSI > 0.10 = moderate drift.
- KL Divergence drift detector: compares binned distributions.
- Maximum Mean Discrepancy (MMD, Sutherland et al. 2017): kernel-based
  drift test that works on raw embeddings without binning.
- Alibi Detect library (2024-2026): production-grade implementations.
- For deepfake detectors: drift in the embedding distribution can
  indicate (a) a new forgery family, (b) input distribution shift
  (e.g., new camera), or (c) adversarial probing. The detector
  cannot distinguish these without labels, but flagging drift
  triggers human review.

This module computes:
1. PSI on binned embeddings.
2. MMD on raw embeddings.
3. A combined drift score and alert.

Reference distribution is stored as a compact summary (bin edges +
counts for PSI, kernel matrix eigendecomposition for MMD).

Strict-compat: pure-additive. No changes to detector interface.
"""

from monitoring.drift_detector import (
    DriftDetector,
    DriftResult,
    PSIResult,
    MMDResult,
    get_default_drift_detector,
)
from monitoring.reference_store import (
    ReferenceStore,
    get_default_reference_store,
)
from monitoring.embedding_buffer import (
    EmbeddingBuffer,
    get_default_embedding_buffer,
)

__all__ = [
    "DriftDetector", "DriftResult", "PSIResult", "MMDResult",
    "get_default_drift_detector",
    "ReferenceStore", "get_default_reference_store",
    "EmbeddingBuffer", "get_default_embedding_buffer",
]
