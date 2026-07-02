"""
Argus Core - Temperature Scaling
=================================
1-D LBFGS optimization of a scalar temperature T on held-out logits.

After fitting, calibrated_prob = softmax(logit / T).

Research grounding:
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
  The recommended baseline calibrator. Reduces ECE from 16.53% → 1.26%
  on CIFAR-100 ResNet-110 (Table 1 of the paper).
- Berta et al., "Structured Matrix Scaling", 2025 — generalization that
  fixes matrix scaling's overfitting problem; we use plain TS for
  simplicity and because binary classifiers rarely need matrix scaling.
- Shen et al., "Mirage: Evidential Deep Learning is a Mirage",
  NeurIPS 2024 — recommends TS on top of EDL's projected p=alpha/S
  because EDL's projected probabilities are softmax-like and benefit
  from the same temperature correction.

Algorithm:
1. Collect (logit, label) pairs on a held-out calibration set (~2k samples).
2. Define loss: NLL(softmax(logit / T), label).
3. Optimize T via L-BFGS (T > 0).
4. At inference: calibrated_prob = softmax(logit / T).

Strict-compat: pure post-hoc. No changes to detector interface.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemperatureScaler:
    """
    Holds the fitted temperature T and applies it to logits.

    Attributes:
        temperature: Scalar T > 0. Default 1.0 (no calibration).
        num_samples: Number of calibration samples used to fit T.
        loss_history: NLL loss per LBFGS iteration.
    """
    temperature: float = 1.0
    num_samples: int = 0
    loss_history: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to a batch of logits.

        Args:
            logits: (N, C) array of raw logits.

        Returns:
            (N, C) array of calibrated probabilities (softmax of logit / T).
        """
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        scaled = logits / max(self.temperature, 1e-8)
        # Numerically stable softmax
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=-1, keepdims=True)

    def calibrate_binary_prob(self, prob: float) -> float:
        """
        Apply temperature scaling to a single binary probability.

        For binary classification, softmax(logit/T) can be computed from
        the probability directly via the logit transform.

        Args:
            prob: P(class=1) in [0, 1].

        Returns:
            Calibrated P(class=1) in [0, 1].
        """
        prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
        logit = np.log(prob / (1.0 - prob))
        scaled_logit = logit / max(self.temperature, 1e-8)
        return float(1.0 / (1.0 + np.exp(-scaled_logit)))

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the temperature to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump({
                "temperature": self.temperature,
                "num_samples": self.num_samples,
                "loss_history": self.loss_history,
            }, fh, indent=2)
        logger.info("Saved TemperatureScaler to %s (T=%.4f)", path, self.temperature)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        """Load a TemperatureScaler from a JSON file."""
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls(
            temperature=data["temperature"],
            num_samples=data.get("num_samples", 0),
            loss_history=data.get("loss_history", []),
        )


# ---------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------

def fit_temperature_scaler(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> TemperatureScaler:
    """
    Fit a TemperatureScaler on held-out (logits, labels).

    Uses L-BFGS to minimize NLL(softmax(logit / T), label) over scalar T > 0.

    Args:
        logits: (N, C) array of raw logits from the detector.
        labels: (N,) array of ground-truth class indices.
        max_iter: Maximum L-BFGS iterations.
        tol: L-BFGS convergence tolerance.

    Returns:
        Fitted TemperatureScaler.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    N, C = logits.shape
    if len(labels) != N:
        raise ValueError(
            f"Mismatch: {N} logits but {len(labels)} labels"
        )

    # Try scipy first (standard LBFGS)
    try:
        from scipy.optimize import minimize

        def nll_loss(T_arr):
            T = float(np.exp(T_arr[0]))  # ensure T > 0
            scaled = logits / T
            scaled = scaled - scaled.max(axis=-1, keepdims=True)
            log_softmax = scaled - np.log(np.exp(scaled).sum(axis=-1, keepdims=True))
            # Pick the log-prob of the true class
            true_log_probs = log_softmax[np.arange(N), labels]
            return -float(np.mean(true_log_probs))

        def nll_grad(T_arr):
            T = float(np.exp(T_arr[0]))
            scaled = logits / T
            scaled = scaled - scaled.max(axis=-1, keepdims=True)
            exp_scaled = np.exp(scaled)
            probs = exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)
            # Gradient w.r.t. T: dNLL/dT = (1/N) * sum( (p_true - p_label) * logit ) / T^2
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(N), labels] = 1.0
            diff = probs - one_hot  # (N, C)
            # Chain rule: dL/d(log T) = dL/dT * dT/d(log T) = dL/dT * T
            # dL/dT = (1/N) * sum( diff * logit ) / T^2
            dL_dT = float(np.mean(np.sum(diff * logits, axis=-1)) / (T * T))
            dL_dlogT = dL_dT * T
            return np.array([dL_dlogT])

        result = minimize(
            nll_loss,
            x0=np.array([0.0]),  # log(T) = 0 → T = 1
            jac=nll_grad,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": tol},
        )
        T = float(np.exp(result.x[0]))
        loss_history = [nll_loss(np.array([np.log(T)]))]

        scaler = TemperatureScaler(
            temperature=T,
            num_samples=N,
            loss_history=loss_history,
        )
        logger.info(
            "TemperatureScaler fitted: T=%.4f, NLL=%.4f, N=%d",
            T, result.fun, N,
        )
        return scaler

    except ImportError:
        logger.warning("scipy not available; falling back to grid search for T")
        return _fit_temperature_grid_search(logits, labels)


def _fit_temperature_grid_search(
    logits: np.ndarray,
    labels: np.ndarray,
    T_grid: Optional[np.ndarray] = None,
) -> TemperatureScaler:
    """Fallback grid-search fitter when scipy is unavailable."""
    if T_grid is None:
        T_grid = np.linspace(0.5, 5.0, 46)
    N = len(labels)
    best_T = 1.0
    best_nll = float("inf")
    for T in T_grid:
        scaled = logits / T
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        log_softmax = scaled - np.log(np.exp(scaled).sum(axis=-1, keepdims=True))
        nll = -float(np.mean(log_softmax[np.arange(N), labels]))
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return TemperatureScaler(
        temperature=best_T, num_samples=N, loss_history=[best_nll],
    )
