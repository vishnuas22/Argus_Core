"""
Argus Core - Post-Processing Pipeline (Iteration 2)
=====================================================
Unified post-processing that applies:
1. Temperature scaling (if calibration is enabled and a scaler is fitted)
2. Conformal RAPS prediction set (if conformal is enabled and fitted)
3. Drift detection check (if drift monitoring is enabled)
4. Adversarial defense flag (if RPS/gate/RS-lite triggered)

This module is the single integration point for Iteration 2 features.
Analyzers call `apply_post_processing()` on their final score to get
a calibrated score + conformal prediction set + drift flag.

Strict-compat: pure post-hoc. No changes to analyzer signatures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PostProcessingResult:
    """Container for post-processing outputs."""
    calibrated_score: float = 0.5
    original_score: float = 0.5
    temperature: float = 1.0
    conformal_set: Optional[list] = None
    is_ambiguous: bool = False
    route_to_human: bool = False
    drift_severity: str = "none"
    drift_score: float = 0.0
    adversarial_flag: bool = False
    adversarial_reason: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


# Process-wide singletons (lazy-loaded)
_temperature_scaler = None
_conformal_raps = None
_drift_detector = None
_drift_reference = None
_post_processing_initialized = False


def _init_post_processing():
    """Lazily load calibration artifacts from disk."""
    global _post_processing_initialized, _temperature_scaler, _conformal_raps
    global _drift_detector, _drift_reference
    if _post_processing_initialized:
        return
    _post_processing_initialized = True

    # Temperature scaler
    if getattr(config, "enable_calibration", False):
        ts_path = getattr(config, "temperature_scaler_path", "")
        if ts_path and os.path.exists(ts_path):
            try:
                from calibration.temperature_scaling import TemperatureScaler
                _temperature_scaler = TemperatureScaler.load(ts_path)
                logger.info(
                    "Loaded TemperatureScaler: T=%.4f, N=%d",
                    _temperature_scaler.temperature, _temperature_scaler.num_samples,
                )
            except Exception as e:
                logger.warning("Failed to load TemperatureScaler: %s", e)

    # Conformal RAPS
    if getattr(config, "enable_calibration", False):
        cp_path = getattr(config, "conformal_raps_path", "")
        if cp_path and os.path.exists(cp_path):
            try:
                from calibration.conformal import ConformalRAPS
                _conformal_raps = ConformalRAPS.load(cp_path)
                logger.info(
                    "Loaded ConformalRAPS: q_hat=%.4f, alpha=%.2f",
                    _conformal_raps.q_hat, _conformal_raps.alpha,
                )
            except Exception as e:
                logger.warning("Failed to load ConformalRAPS: %s", e)

    # Drift detector
    if getattr(config, "enable_drift_detection", False):
        try:
            from monitoring.drift_detector import DriftDetector
            _drift_detector = DriftDetector(
                psi_moderate=getattr(config, "drift_psi_moderate", 0.10),
                psi_major=getattr(config, "drift_psi_major", 0.25),
                mmd_threshold=getattr(config, "drift_mmd_threshold", 0.05),
            )
        except Exception as e:
            logger.warning("Failed to init DriftDetector: %s", e)

        # Drift reference
        ref_path = getattr(config, "drift_reference_path", "")
        if ref_path and os.path.exists(f"{ref_path}.json"):
            try:
                from monitoring.reference_store import ReferenceStore
                _drift_reference = ReferenceStore.load(ref_path)
                logger.info(
                    "Loaded drift reference: %d samples, modality=%s",
                    _drift_reference.num_samples, _drift_reference.modality,
                )
            except Exception as e:
                logger.warning("Failed to load drift reference: %s", e)


def apply_post_processing(
    score: float,
    confidence: Optional[float] = None,
    embedding: Optional[np.ndarray] = None,
    adversarial_flag: bool = False,
    adversarial_reason: str = "",
    modality: str = "image",
    analysis_id: str = "",
) -> PostProcessingResult:
    """
    Apply the full Iteration-2 post-processing pipeline to a final score.

    Args:
        score: Raw fake probability from the ensemble [0, 1].
        confidence: Optional ensemble confidence [0, 1].
        embedding: Optional embedding for drift detection (D,).
        adversarial_flag: True if an adversarial defense flagged the input.
        adversarial_reason: Reason string if flagged.
        modality: Modality tag for embedding buffer ("image" | "audio" | "video").
        analysis_id: Analysis ID for embedding buffer provenance.

    Returns:
        PostProcessingResult with calibrated score + conformal set + drift.
    """
    _init_post_processing()

    result = PostProcessingResult(
        original_score=float(score),
        calibrated_score=float(score),
    )

    # 1. Temperature scaling
    if _temperature_scaler is not None:
        try:
            result.calibrated_score = _temperature_scaler.calibrate_binary_prob(score)
            result.temperature = _temperature_scaler.temperature
        except Exception as e:
            logger.debug("Temperature scaling failed: %s", e)

    # 2. Conformal RAPS
    if _conformal_raps is not None:
        try:
            # Binary: probs = [1 - calibrated, calibrated]
            probs = np.array([1.0 - result.calibrated_score, result.calibrated_score])
            cr = _conformal_raps.predict(probs)
            result.conformal_set = cr.prediction_set
            result.is_ambiguous = cr.is_ambiguous
            result.route_to_human = cr.route_to_human
        except Exception as e:
            logger.debug("Conformal prediction failed: %s", e)

    # 3. Drift detection (if embedding provided and reference loaded)
    if _drift_detector is not None and _drift_reference is not None and embedding is not None:
        try:
            ref_emb = _drift_reference.embeddings
            if ref_emb is not None and len(ref_emb) > 0:
                ref_mean = np.mean(ref_emb, axis=0)
                ref_std = np.std(ref_emb, axis=0) + 1e-8
                z_scores = np.abs((embedding - ref_mean) / ref_std)
                max_z = float(np.max(z_scores))
                mean_z = float(np.mean(z_scores))
                # Flag if any dimension exceeds 4σ or mean exceeds 2.5σ
                if max_z > 4.0 or mean_z > 2.5:
                    result.drift_severity = "major" if max_z > 5.0 else "moderate"
                    result.drift_score = float(np.clip(max_z / 6.0, 0, 1))
                    result.route_to_human = True
                    logger.warning(
                        "Per-sample drift detected: max_z=%.2f mean_z=%.2f severity=%s",
                        max_z, mean_z, result.drift_severity,
                    )
        except Exception as e:
            logger.debug("Drift check failed: %s", e)

    # 4. Adversarial defense flag
    result.adversarial_flag = adversarial_flag
    result.adversarial_reason = adversarial_reason
    if adversarial_flag:
        # Route flagged inputs to human review
        result.route_to_human = True

    # 5. Buffer embedding for drift detection (fire-and-forget)
    if embedding is not None and len(embedding) > 0:
        try:
            from monitoring.embedding_buffer import get_default_embedding_buffer
            buf = get_default_embedding_buffer()
            if buf is not None:
                buf.append(
                    embedding=embedding,
                    modality=modality,
                    analysis_id=analysis_id,
                    score=float(score),
                )
        except Exception:
            pass

    # Iteration 7: record conformal route-to-human
    if result.route_to_human:
        try:
            from observability import get_default_metrics
            get_default_metrics().record_conformal_route(modality)
        except Exception:
            pass

    return result


def check_batch_drift(
    current_embeddings: np.ndarray,
    modality: str = "image",
) -> Optional[Dict[str, Any]]:
    """
    Check drift on a batch of embeddings against the reference.

    Args:
        current_embeddings: (N, D) array of recent embeddings.
        modality: Modality tag (must match the reference's modality).

    Returns:
        Dict with drift results, or None if drift detection is disabled
        or no reference is loaded.
    """
    _init_post_processing()
    if _drift_detector is None or _drift_reference is None:
        return None
    if _drift_reference.modality != modality:
        return None
    if _drift_reference.embeddings is None:
        return None

    try:
        result = _drift_detector.detect(
            reference=_drift_reference.embeddings,
            current=np.asarray(current_embeddings, dtype=np.float64),
        )
        return {
            "is_drifted": result.is_drifted,
            "drift_score": result.drift_score,
            "severity": result.severity,
            "psi": result.psi.psi if result.psi else None,
            "mmd": result.mmd.mmd if result.mmd else None,
            "recommendation": result.recommendation,
        }
    except Exception as e:
        logger.warning("Batch drift check failed: %s", e)
        return None
