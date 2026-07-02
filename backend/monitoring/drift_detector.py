"""
Argus Core - Drift Detector (Iteration 2)
==========================================
PSI + MMD drift detection on deepfake detector embeddings.
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
class PSIResult:
    """Population Stability Index result."""
    psi: float
    is_drifted: bool
    bin_counts_ref: List[int]
    bin_counts_cur: List[int]
    bin_edges: List[float]
    severity: str  # "none" | "moderate" | "major"


@dataclass
class MMDResult:
    """Maximum Mean Discrepancy result."""
    mmd: float
    is_drifted: bool
    p_value: Optional[float] = None  # if permutation test was run
    threshold: float = 0.0


@dataclass
class DriftResult:
    """Combined drift detection result."""
    psi: Optional[PSIResult] = None
    mmd: Optional[MMDResult] = None
    is_drifted: bool = False
    drift_score: float = 0.0  # combined [0, 1]
    severity: str = "none"
    recommendation: str = ""


class DriftDetector:
    """
    Combines PSI and MMD for robust drift detection on embeddings.

    PSI is good for detecting distribution shift in binned probabilities.
    MMD is good for detecting shift in the embedding manifold.
    Using both reduces false positives.
    """

    def __init__(
        self,
        num_bins: int = 20,
        psi_moderate: float = 0.10,
        psi_major: float = 0.25,
        mmd_threshold: float = 0.05,
        mmd_permutation_samples: int = 0,  # 0 = no permutation test
        seed: Optional[int] = None,
    ):
        self.num_bins = num_bins
        self.psi_moderate = psi_moderate
        self.psi_major = psi_major
        self.mmd_threshold = mmd_threshold
        self.mmd_permutation_samples = mmd_permutation_samples
        self._rng = np.random.default_rng(seed)
        logger.info(
            "DriftDetector initialized: bins=%d, psi_moderate=%.2f, psi_major=%.2f",
            num_bins, psi_moderate, psi_major,
        )

    # ------------------------------------------------------------------
    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> PSIResult:
        """
        Compute Population Stability Index.

        Args:
            reference: (N_ref, D) reference embeddings.
            current: (N_cur, D) current embeddings.

        Returns:
            PSIResult.
        """
        reference = np.asarray(reference, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if current.ndim == 1:
            current = current.reshape(-1, 1)

        # Use the first principal component for univariate PSI
        # (or compute PSI per-dimension and average)
        psi_per_dim = []
        ref_counts_all = []
        cur_counts_all = []
        edges_all = []
        for d in range(reference.shape[1]):
            ref_d = reference[:, d]
            cur_d = current[:, d]
            # Bin edges from reference (quantile bins for robustness)
            percentiles = np.linspace(0, 100, self.num_bins + 1)
            edges = np.percentile(ref_d, percentiles)
            edges = np.unique(edges)  # avoid duplicate edges
            if len(edges) < 2:
                continue
            # Add -inf and +inf at the ends to catch outliers
            edges[0] = -np.inf
            edges[-1] = np.inf
            ref_counts = np.histogram(ref_d, bins=edges)[0]
            cur_counts = np.histogram(cur_d, bins=edges)[0]
            # Normalize to proportions
            ref_props = ref_counts / max(ref_counts.sum(), 1)
            cur_props = cur_counts / max(cur_counts.sum(), 1)
            # PSI = sum( (cur - ref) * ln(cur / ref) )
            eps = 1e-6
            ref_props = np.clip(ref_props, eps, None)
            cur_props = np.clip(cur_props, eps, None)
            psi_d = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
            psi_per_dim.append(psi_d)
            if d == 0:  # store first dim's bins for the result
                ref_counts_all = ref_counts.tolist()
                cur_counts_all = cur_counts.tolist()
                edges_all = [float(e) if np.isfinite(e) else float("inf") for e in edges]

        psi = float(np.mean(psi_per_dim)) if psi_per_dim else 0.0

        if psi > self.psi_major:
            severity = "major"
            is_drifted = True
        elif psi > self.psi_moderate:
            severity = "moderate"
            is_drifted = True
        else:
            severity = "none"
            is_drifted = False

        return PSIResult(
            psi=psi,
            is_drifted=is_drifted,
            bin_counts_ref=ref_counts_all,
            bin_counts_cur=cur_counts_all,
            bin_edges=edges_all,
            severity=severity,
        )

    # ------------------------------------------------------------------
    def compute_mmd(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        kernel: str = "rbf",
    ) -> MMDResult:
        """
        Compute Maximum Mean Discrepancy with RBF kernel.

        Args:
            reference: (N_ref, D) reference embeddings.
            current: (N_cur, D) current embeddings.
            kernel: "rbf" (only one supported here).

        Returns:
            MMDResult.
        """
        reference = np.asarray(reference, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if current.ndim == 1:
            current = current.reshape(-1, 1)

        # Heuristic bandwidth: median pairwise distance in reference
        n_ref = min(reference.shape[0], 500)  # cap for speed
        sample = reference[:n_ref]
        try:
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(sample)
            sigma = float(np.median(dists[dists > 0]))
            if sigma < 1e-6:
                sigma = 1.0
        except ImportError:
            sigma = 1.0
        gamma = 1.0 / (2.0 * sigma ** 2)

        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        K_xx = self._rbf(reference, reference, gamma)
        K_yy = self._rbf(current, current, gamma)
        K_xy = self._rbf(reference, current, gamma)
        mmd = float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())

        # Optional permutation test
        p_value = None
        if self.mmd_permutation_samples > 0:
            p_value = self._permutation_test(reference, current, gamma, mmd)

        is_drifted = mmd > self.mmd_threshold
        return MMDResult(
            mmd=mmd, is_drifted=is_drifted,
            p_value=p_value, threshold=self.mmd_threshold,
        )

    def _rbf(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """RBF kernel matrix."""
        try:
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(X, Y)
            return np.exp(-gamma * dists ** 2)
        except ImportError:
            # Fallback: manual computation
            sq_dists = (
                np.sum(X ** 2, axis=1, keepdims=True)
                + np.sum(Y ** 2, axis=1)
                - 2.0 * X @ Y.T
            )
            return np.exp(-gamma * sq_dists)

    def _permutation_test(
        self, reference: np.ndarray, current: np.ndarray,
        gamma: float, observed_mmd: float,
    ) -> float:
        """Permutation test for MMD significance."""
        combined = np.vstack([reference, current])
        n_ref = reference.shape[0]
        n_total = combined.shape[0]
        count_extreme = 0
        for _ in range(self.mmd_permutation_samples):
            perm = self._rng.permutation(n_total)
            perm_ref = combined[perm[:n_ref]]
            perm_cur = combined[perm[n_ref:]]
            K_xx = self._rbf(perm_ref, perm_ref, gamma)
            K_yy = self._rbf(perm_cur, perm_cur, gamma)
            K_xy = self._rbf(perm_ref, perm_cur, gamma)
            perm_mmd = float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())
            if perm_mmd >= observed_mmd:
                count_extreme += 1
        return float(count_extreme / self.mmd_permutation_samples)

    # ------------------------------------------------------------------
    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """
        Run both PSI and MMD, combine into a single DriftResult.

        Args:
            reference: (N_ref, D) reference embeddings.
            current: (N_cur, D) current embeddings.

        Returns:
            DriftResult with both metrics + combined drift score.
        """
        psi_result = self.compute_psi(reference, current)
        mmd_result = self.compute_mmd(reference, current)

        # Combined drift score: max of normalized PSI and normalized MMD
        psi_norm = min(psi_result.psi / self.psi_major, 1.0)
        mmd_norm = min(mmd_result.mmd / max(self.mmd_threshold * 4, 1e-6), 1.0)
        drift_score = float(max(psi_norm, mmd_norm))

        # Drifted if either detector triggers
        is_drifted = psi_result.is_drifted or mmd_result.is_drifted
        if psi_result.severity == "major" or mmd_result.mmd > self.mmd_threshold * 2:
            severity = "major"
        elif is_drifted:
            severity = "moderate"
        else:
            severity = "none"

        if severity == "major":
            recommendation = (
                "Major drift detected. Recommended actions: (1) trigger human review "
                "of recent predictions, (2) collect labeled samples from the new "
                "distribution, (3) consider retraining the LoRA adapters, (4) check "
                "for new forgery families or adversarial probing."
            )
        elif severity == "moderate":
            recommendation = (
                "Moderate drift detected. Recommended actions: (1) increase logging, "
                "(2) collect a sample of the new distribution for review, (3) monitor "
                "closely for further drift."
            )
        else:
            recommendation = "No significant drift detected."

        # Iteration 7: record drift metrics
        try:
            from observability import get_default_metrics
            _psi_val = psi_result.psi if psi_result else 0.0
            _mmd_val = mmd_result.mmd if mmd_result else 0.0
            # Use "unknown" modality since the detector doesn't know it
            # — operators should call detect() per-modality and the
            # analyzer will tag the metric with the right label.
            get_default_metrics().record_drift(
                "image", drift_score, severity, _psi_val, _mmd_val
            )
        except Exception:
            pass

        return DriftResult(
            psi=psi_result,
            mmd=mmd_result,
            is_drifted=is_drifted,
            drift_score=drift_score,
            severity=severity,
            recommendation=recommendation,
        )


# ---------------------------------------------------------------------
_default_detector: Optional[DriftDetector] = None


def get_default_drift_detector() -> DriftDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = DriftDetector()
    return _default_detector
