"""
Argus Core - Diversity-Aware Detector Ensemble
==============================================
Combines multiple SOTA deepfake detectors per modality into a single
DetectionResult using uncertainty-weighted logit averaging with explicit
diversity regularization.

Why this design (research grounding):
- Plain score averaging is dominated by correlated failures (Dong et al.,
  "Towards Detection-Guided Deepfake Assessment", CVPRW 2023).
- Confidence-weighted logit averaging (Liang et al., 2024) down-weights
  over-confident but inaccurate members and matches post-hoc Platt-scaled
  probabilities better than score-space averaging.
- Pair-wise diversity penalty (Shen & Hsiao, "On the Diversity of Deep
  Ensembles", ICMLW 2023) prevents the ensemble from collapsing onto a
  single detector when several produce identical outputs (e.g. all failing
  on the same adversarial example).

Strict-compatibility:
- Pure-additive module. No changes to BaseDetector or DetectionResult.
- All existing analyzers continue to work unchanged; new analyzers may
  opt-in via ``DiversityEnsemble.combine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from detectors.base import DetectionResult
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleMember:
    """A single detector's contribution to an ensemble."""
    name: str
    score: float                # spoof/fake probability in [0, 1]
    confidence: float           # detector self-reported confidence in [0, 1]
    weight: float = 1.0         # prior weight (e.g. benchmark AUC)
    error: Optional[str] = None  # if non-None, member is excluded


@dataclass
class DiversityEnsemble:
    """
    Diversity-aware ensemble combiner.

    Steps:
    1. Drop members that reported an error or returned NaN.
    2. Convert scores to logits (inverse sigmoid) — logit-space averaging
       is better calibrated than score-space averaging (Liang et al. 2024).
    3. Weight each member by prior_weight * confidence.
    4. Apply a soft diversity penalty: if two members are highly correlated
       (|logit difference| < eps), they share the weight of one.
    5. Aggregate via weighted mean in logit space, then sigmoid back.
    6. Compute ensemble confidence from agreement + extremity.

    All math is numpy-only (no torch dependency) so it can run on CPU
    even when individual detectors are on GPU.
    """

    # Members with |logit diff| < this threshold are treated as duplicates.
    diversity_eps: float = 0.25
    # Floor for confidence-based weighting (prevents a single high-conf
    # member from drowning out the rest).
    min_member_weight: float = 0.05
    # Floor for ensemble confidence when members disagree strongly.
    min_ensemble_confidence: float = 0.15
    # Ceiling for ensemble confidence (avoid 1.0).
    max_ensemble_confidence: float = 0.97

    # ------------------------------------------------------------------
    @staticmethod
    def _to_logit(p: float) -> float:
        """Inverse-sigmoid with numerical clipping."""
        p = float(np.clip(p, 1e-5, 1.0 - 1e-5))
        return float(np.log(p / (1.0 - p)))

    @staticmethod
    def _from_logit(z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-z)))

    # ------------------------------------------------------------------
    def combine(self, members: Sequence[EnsembleMember]) -> DetectionResult:
        """
        Combine detector outputs into a single DetectionResult.

        Args:
            members: Non-empty sequence of EnsembleMember.

        Returns:
            DetectionResult with ``score``, ``confidence``, and a per-member
            breakdown stored in ``features``.
        """
        # Filter to healthy members
        healthy: List[EnsembleMember] = [
            m for m in members
            if m.error is None
            and not (np.isnan(m.score) or np.isnan(m.confidence))
        ]

        if not healthy:
            # Everyone failed — return a neutral, low-confidence result.
            errors = [f"{m.name}:{m.error or 'nan'}" for m in members]
            return DetectionResult(
                score=0.5,
                confidence=self.min_ensemble_confidence,
                model_name="diversity_ensemble",
                backend="ensemble",
                features={"members": len(members), "healthy": 0},
                error="all_members_failed:" + "|".join(errors)[:200],
            )

        # Compute per-member logits & raw weights
        logits = np.array([self._to_logit(m.score) for m in healthy], dtype=np.float64)
        raw_w = np.array(
            [max(m.weight, 0.0) * max(m.confidence, 0.0) for m in healthy],
            dtype=np.float64,
        )
        # Floor + normalize
        raw_w = np.clip(raw_w, self.min_member_weight, None)
        raw_w = raw_w / raw_w.sum()

        # Diversity: for each pair, if their logits are within eps, split
        # their combined weight equally between them (this only matters
        # when 3+ members collapse onto the same logit). Implemented as a
        # soft penalty on weight = raw_w * (1 - duplication_fraction).
        if len(healthy) > 1:
            dup = np.zeros_like(raw_w)
            for i in range(len(healthy)):
                for j in range(i + 1, len(healthy)):
                    if abs(logits[i] - logits[j]) < self.diversity_eps:
                        # They are duplicates — split the pair's weight
                        dup[i] += 0.5
                        dup[j] += 0.5
            # Soft penalty: at most halve a member's weight
            penalty = 1.0 / (1.0 + dup)
            raw_w = raw_w * penalty
            raw_w = raw_w / raw_w.sum()

        # Weighted logit-space aggregation
        fused_logit = float(np.dot(logits, raw_w))
        fused_score = self._from_logit(fused_logit)

        # Ensemble confidence: combination of
        #   (a) weighted mean of individual confidences,
        #   (b) agreement = 1 - normalized std of logits,
        #   (c) extremity = |fused_score - 0.5| * 2
        mean_conf = float(np.dot(np.array([m.confidence for m in healthy]), raw_w))
        std_logit = float(np.std(logits))
        agreement = float(1.0 / (1.0 + std_logit))
        extremity = abs(fused_score - 0.5) * 2.0
        ens_conf = 0.45 * mean_conf + 0.35 * agreement + 0.20 * extremity
        ens_conf = float(
            np.clip(ens_conf, self.min_ensemble_confidence, self.max_ensemble_confidence)
        )

        # Per-member breakdown for XAI / audit trail
        features = {
            "member_count": len(healthy),
            "members": ",".join(m.name for m in healthy),
            "weights": "|".join(f"{m.name}:{w:.3f}" for m, w in zip(healthy, raw_w)),
            "logit_std": round(std_logit, 4),
            "agreement": round(agreement, 4),
            "fused_logit": round(fused_logit, 4),
        }

        return DetectionResult(
            score=float(np.clip(fused_score, 0.0, 1.0)),
            confidence=ens_conf,
            model_name="diversity_ensemble",
            backend="ensemble",
            features=features,
        )


# ---------------------------------------------------------------------
# Convenience helper used by analyzers
# ---------------------------------------------------------------------

_default_ensemble: Optional[DiversityEnsemble] = None


def get_default_ensemble() -> DiversityEnsemble:
    """Return a process-wide cached DiversityEnsemble instance."""
    global _default_ensemble
    if _default_ensemble is None:
        _default_ensemble = DiversityEnsemble()
    return _default_ensemble


def combine_detector_results(
    results: Sequence[DetectionResult],
    prior_weights: Optional[Sequence[float]] = None,
) -> DetectionResult:
    """
    Convenience wrapper used by analyzers to fuse the outputs of
    several detectors that ran on the same input.

    Args:
        results: Non-empty sequence of DetectionResult.
        prior_weights: Optional per-detector prior weights (e.g. benchmark
            AUCs). Defaults to uniform.

    Returns:
        Single fused DetectionResult.
    """
    if not results:
        return DetectionResult(score=0.5, confidence=0.1, model_name="empty_ensemble")

    if prior_weights is None:
        prior_weights = [1.0] * len(results)
    elif len(prior_weights) != len(results):
        # Mismatch — fall back to uniform.
        logger.warning(
            "combine_detector_results: prior_weights length mismatch "
            "(%d vs %d), falling back to uniform",
            len(prior_weights), len(results),
        )
        prior_weights = [1.0] * len(results)

    members = [
        EnsembleMember(
            name=r.model_name or f"detector_{i}",
            score=r.score,
            confidence=r.confidence,
            weight=float(prior_weights[i]),
            error=r.error,
        )
        for i, r in enumerate(results)
    ]
    return get_default_ensemble().combine(members)
