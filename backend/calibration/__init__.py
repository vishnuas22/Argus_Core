"""
Argus Core - Calibration Module (Iteration 2)
==============================================
Production-grade probability calibration and conformal prediction for
the deepfake detector ensemble.

Research grounding:
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
  Temperature Scaling (TS) is the recommended baseline: 1-D LBFGS
  optimization of a scalar T on held-out validation logits. Reduces
  ECE from 16.53% → 1.26% on CIFAR-100 ResNet-110.
- Kull et al., "Beta Calibration: a well-founded and easily calibrated
  improvement", AISTATS 2017. 3-param alternative to Platt scaling.
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction",
  Foundations and Trends in ML 2023. Distribution-free coverage.
- RAPS (Romano et al., "Classification with Valid Procedure under
  Ambiguity", ICLR 2021): Regularized Adaptive Prediction Sets — gives
  smaller prediction sets than naive conformal at the same coverage.
- Jin et al., "Calibration of Deepfake Detectors", Visual Intelligence
  2025: Packed-Ensembles improve deepfake detector calibration.
- Shen et al., "Mirage: Evidential Deep Learning is a Mirage",
  NeurIPS 2024: shows EDL's uncertainty estimates are unreliable
  asymptotically; recommends TS on top of EDL's projected p=alpha/S.

Three components:
1. Temperature Scaling — 1-D LBFGS, fits on held-out calibration set.
2. Calibration Audit — ECE(15), MCE, Brier, NLL, reliability diagram.
3. Conformal RAPS — distribution-free prediction sets at α=0.10.

Strict-compat: pure post-hoc on DetectionResult. No changes to detector
interface. The calibration layer is applied AFTER the ensemble combiner.
"""

from calibration.temperature_scaling import (
    TemperatureScaler,
    fit_temperature_scaler,
)
from calibration.calibration_audit import (
    CalibrationAudit,
    CalibrationMetrics,
    run_calibration_audit,
)
from calibration.conformal import (
    ConformalRAPS,
    ConformalResult,
    fit_conformal_raps,
)

__all__ = [
    # Temperature scaling
    "TemperatureScaler", "fit_temperature_scaler",
    # Audit
    "CalibrationAudit", "CalibrationMetrics", "run_calibration_audit",
    # Conformal
    "ConformalRAPS", "ConformalResult", "fit_conformal_raps",
]
