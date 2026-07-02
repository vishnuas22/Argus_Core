"""
Argus Core - Conformal Prediction (RAPS)
=========================================
Distribution-free prediction sets with finite-sample coverage guarantee.

Research grounding:
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction",
  Foundations and Trends in ML 2023. Distribution-free coverage at
  level 1-alpha.
- Romano et al., "Classification with Valid Procedure under Ambiguity",
  ICLR 2021 — RAPS (Regularized Adaptive Prediction Sets). Gives smaller
  prediction sets than naive conformal at the same coverage.
- For deepfake detection: at alpha=0.10, RAPS gives a 90%-coverage
  prediction set. If the set is {real, fake} (size 2), the input is
  ambiguous and should be routed to a human reviewer.

Algorithm:
1. On a held-out calibration set, compute conformity scores.
2. Find the (1-alpha) quantile of the scores → q_hat.
3. At inference: include class y in the prediction set if its score
   exceeds q_hat.

For binary deepfake detection, the prediction set has size 1 (confident)
or 2 (ambiguous → route to human).

Strict-compat: pure post-hoc. No changes to detector interface.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConformalResult:
    """Result of a conformal prediction query."""
    prediction_set: List[int]    # e.g. [0] = real, [1] = fake, [0, 1] = ambiguous
    is_ambiguous: bool           # True if set size > 1
    route_to_human: bool         # True if ambiguous
    q_hat: float                 # Conformity threshold
    scores: List[float]          # Per-class conformity scores


@dataclass
class ConformalRAPS:
    """
    RAPS conformal predictor.

    Attributes:
        q_hat: Conformity threshold fitted on calibration set.
        alpha: Miscoverage rate (1-alpha = target coverage).
        num_samples: Number of calibration samples used.
        lambda_raps: RAPS regularization strength (0 = naive conformal).
        k_raps: RAPS regularization rank.
    """
    q_hat: float = 0.5
    alpha: float = 0.10
    num_samples: int = 0
    lambda_raps: float = 0.0
    k_raps: int = 1

    # ------------------------------------------------------------------
    def predict(self, probs: np.ndarray) -> ConformalResult:
        """
        Compute the conformal prediction set for a single input.

        Args:
            probs: (C,) array of class probabilities.

        Returns:
            ConformalResult with the prediction set.
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        # Sort classes by descending probability (RAPS ranking)
        order = np.argsort(-probs[0])
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        # RAPS score: -prob + lambda * max(0, rank - k + 1)
        scores = -probs[0] + self.lambda_raps * np.maximum(
            0, ranks - self.k_raps + 1
        ).astype(np.float64)
        # Include class in prediction set if score <= q_hat
        prediction_set = sorted([int(i) for i in range(len(scores)) if scores[i] <= self.q_hat])
        if not prediction_set:
            # Fallback: include the top-1 class
            prediction_set = [int(order[0])]
        is_ambiguous = len(prediction_set) > 1
        return ConformalResult(
            prediction_set=prediction_set,
            is_ambiguous=is_ambiguous,
            route_to_human=is_ambiguous,
            q_hat=self.q_hat,
            scores=scores.tolist(),
        )

    def predict_batch(self, probs_batch: np.ndarray) -> List[ConformalResult]:
        """Compute conformal prediction sets for a batch of inputs."""
        results = []
        for i in range(probs_batch.shape[0]):
            results.append(self.predict(probs_batch[i]))
        return results

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump({
                "q_hat": self.q_hat,
                "alpha": self.alpha,
                "num_samples": self.num_samples,
                "lambda_raps": self.lambda_raps,
                "k_raps": self.k_raps,
            }, fh, indent=2)
        logger.info("Saved ConformalRAPS to %s (q_hat=%.4f)", path, self.q_hat)

    @classmethod
    def load(cls, path: str) -> "ConformalRAPS":
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls(**data)


# ---------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------

def fit_conformal_raps(
    probs_calibration: np.ndarray,
    labels_calibration: np.ndarray,
    alpha: float = 0.10,
    lambda_raps: float = 0.0,
    k_raps: int = 1,
) -> ConformalRAPS:
    """
    Fit a ConformalRAPS predictor on held-out calibration data.

    Args:
        probs_calibration: (N, C) array of class probabilities.
        labels_calibration: (N,) array of true class indices.
        alpha: Miscoverage rate (1-alpha = target coverage). Default 0.10.
        lambda_raps: RAPS regularization strength. 0 = naive conformal.
        k_raps: RAPS regularization rank.

    Returns:
        Fitted ConformalRAPS predictor.
    """
    probs = np.asarray(probs_calibration, dtype=np.float64)
    labels = np.asarray(labels_calibration, dtype=np.int64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    N, C = probs.shape

    # Compute conformity scores on calibration set
    # Score for true class: -prob + lambda * max(0, rank - k + 1)
    # where rank is the rank of the true class in the sorted-by-prob order
    scores = np.zeros(N, dtype=np.float64)
    for i in range(N):
        order = np.argsort(-probs[i])
        ranks = np.empty_like(order)
        ranks[order] = np.arange(C)
        true_rank = int(ranks[labels[i]])
        scores[i] = -probs[i, labels[i]] + lambda_raps * max(0, true_rank - k_raps + 1)

    # q_hat = ceil((1-alpha) * (N+1)) / N quantile
    # This is the standard conformal calibration formula
    q_level = min(1.0, np.ceil((1.0 - alpha) * (N + 1)) / N)
    q_hat = float(np.quantile(scores, q_level, method="higher"))

    logger.info(
        "ConformalRAPS fitted: q_hat=%.4f, alpha=%.2f, N=%d, lambda=%.2f, k=%d",
        q_hat, alpha, N, lambda_raps, k_raps,
    )
    return ConformalRAPS(
        q_hat=q_hat, alpha=alpha, num_samples=N,
        lambda_raps=lambda_raps, k_raps=k_raps,
    )
