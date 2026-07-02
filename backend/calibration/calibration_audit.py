"""
Argus Core - Calibration Audit
================================
Gold-standard calibration audit per the 2024-2026 protocol:
- ECE(15) — Expected Calibration Error with 15 bins
- MCE — Maximum Calibration Error
- Brier score
- NLL — Negative Log-Likelihood
- Smooth ECE (ICLR 2024) — kernel-smoothed variant, more stable
- Reliability diagram data

Research grounding:
- Guo et al., ICML 2017 — ECE definition with 15 bins is the standard.
- Blasiok & Nakkiran, "A unifying theory of calibration metrics",
  ICLR 2024 — introduces Smooth ECE and shows it is more stable than
  binned ECE for small calibration sets.
- Recommended audit: report ECE(15) + MCE + Brier + NLL + reliability
  diagram + 1000-bootstrap CIs.

Strict-compat: pure read-only audit. No changes to detector interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationMetrics:
    """Container for all calibration metrics."""
    ece_15: float = 0.0          # Expected Calibration Error (15 bins)
    mce: float = 0.0             # Maximum Calibration Error
    brier_score: float = 0.0     # Mean squared error of probabilities
    nll: float = 0.0             # Negative log-likelihood
    smooth_ece: float = 0.0      # Smooth ECE (Blasiok & Nakkiran 2024)
    accuracy: float = 0.0
    mean_confidence: float = 0.0
    num_samples: int = 0
    # Reliability diagram data
    bin_accuracies: List[float] = field(default_factory=list)
    bin_confidences: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)
    bin_edges: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "ece_15": round(self.ece_15, 6),
            "mce": round(self.mce, 6),
            "brier_score": round(self.brier_score, 6),
            "nll": round(self.nll, 6),
            "smooth_ece": round(self.smooth_ece, 6),
            "accuracy": round(self.accuracy, 6),
            "mean_confidence": round(self.mean_confidence, 6),
            "num_samples": self.num_samples,
            "bin_accuracies": [round(x, 4) for x in self.bin_accuracies],
            "bin_confidences": [round(x, 4) for x in self.bin_confidences],
            "bin_counts": self.bin_counts,
            "bin_edges": [round(x, 4) for x in self.bin_edges],
        }


class CalibrationAudit:
    """
    Computes the full calibration audit on a set of (prob, label) pairs.
    """

    def __init__(self, num_bins: int = 15, smooth_ece_kernel: float = 0.05):
        self.num_bins = num_bins
        self.smooth_ece_kernel = smooth_ece_kernel

    # ------------------------------------------------------------------
    def audit_binary(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> CalibrationMetrics:
        """
        Audit binary classifier calibration.

        Args:
            probs: (N,) array of P(class=1) in [0, 1].
            labels: (N,) array of binary labels (0 or 1).

        Returns:
            CalibrationMetrics with all metrics + reliability diagram data.
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        N = len(labels)
        if N == 0:
            return CalibrationMetrics()

        # Predicted class = argmax([1-p, p], axis=-1)
        preds = (probs >= 0.5).astype(np.int64)
        accuracy = float(np.mean(preds == labels))
        mean_conf = float(np.mean(np.max(
            np.stack([1 - probs, probs], axis=-1), axis=-1
        )))

        # ECE / MCE with equal-width bins
        bin_edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        ece = 0.0
        mce = 0.0
        for i in range(self.num_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (probs >= lo) & (probs < hi if i < self.num_bins - 1 else probs <= hi)
            count = int(mask.sum())
            bin_counts.append(count)
            if count == 0:
                bin_accuracies.append(0.0)
                bin_confidences.append((lo + hi) / 2)
                continue
            bin_acc = float(np.mean(labels[mask] == preds[mask]))
            bin_conf = float(np.mean(probs[mask]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            gap = abs(bin_acc - bin_conf)
            ece += gap * (count / N)
            mce = max(mce, gap)

        # Brier score
        brier = float(np.mean((probs - labels) ** 2))

        # NLL
        eps = 1e-12
        nll_vals = -(
            labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)
        )
        nll = float(np.mean(nll_vals))

        # Smooth ECE (Blasiok & Nakkiran 2024) — kernel-smoothed gap
        smooth_ece = self._compute_smooth_ece(probs, labels, preds)

        return CalibrationMetrics(
            ece_15=ece,
            mce=mce,
            brier_score=brier,
            nll=nll,
            smooth_ece=smooth_ece,
            accuracy=accuracy,
            mean_confidence=mean_conf,
            num_samples=N,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            bin_edges=bin_edges.tolist(),
        )

    # ------------------------------------------------------------------
    def _compute_smooth_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> float:
        """
        Smooth ECE (Blasiok & Nakkiran, ICLR 2024).
        Uses a Gaussian kernel on the confidence axis instead of hard bins.
        """
        N = len(labels)
        if N == 0:
            return 0.0
        # Confidence = max(p, 1-p)
        confidences = np.maximum(probs, 1 - probs)
        correctness = (preds == labels).astype(np.float64)
        # Kernel-smoothed accuracy and confidence on a grid
        grid = np.linspace(0.0, 1.0, 100)
        sigma = self.smooth_ece_kernel
        smoothed_acc = np.zeros_like(grid)
        smoothed_conf = np.zeros_like(grid)
        for i, g in enumerate(grid):
            weights = np.exp(-((confidences - g) ** 2) / (2 * sigma ** 2))
            weights /= (weights.sum() + 1e-12)
            smoothed_acc[i] = float(np.sum(weights * correctness))
            smoothed_conf[i] = float(np.sum(weights * confidences))
        return float(np.mean(np.abs(smoothed_acc - smoothed_conf)))


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------

def run_calibration_audit(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> CalibrationMetrics:
    """
    Run the full calibration audit on a set of (prob, label) pairs.

    Args:
        probs: (N,) array of P(class=1) in [0, 1].
        labels: (N,) array of binary labels (0 or 1).
        num_bins: Number of bins for ECE/MCE.

    Returns:
        CalibrationMetrics with all metrics + reliability diagram data.
    """
    audit = CalibrationAudit(num_bins=num_bins)
    return audit.audit_binary(probs, labels)
