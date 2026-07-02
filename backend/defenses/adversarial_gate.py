"""
Argus Core - XAI Adversarial Gate
==================================
Flag-don't-classify defense: detects inputs whose explanation is
unstable across randomized perturbations, a known signature of
adversarial examples.

Research grounding:
- Wang et al., "XAI-Based Adversarial Detection", arXiv 2024 — uses
  GradCAM consistency under random crops/flips to flag adversarial
  inputs. The intuition: legitimate inputs have stable explanations;
  adversarial inputs rely on specific pixel patterns and produce
  unstable explanations under perturbation.
- "Flag, Don't Classify" paradigm: when the gate triggers, the input
  is routed to manual review instead of being classified. This avoids
  the cat-and-mouse game of trying to classify adversarial inputs.

Algorithm:
1. Run detector on input → baseline score s0 + baseline explanation e0.
2. Apply K=3 random perturbations (small JPEG, small crop, small flip).
3. Run detector on each → scores s1..sK + explanations e1..eK.
4. Compute:
   - Score variance: var(s0..sK)
   - Explanation IoU: mean(IoU(e0, ei)) for i=1..K
5. Flag as adversarial if score_variance > threshold OR mean_IoU < threshold.

Latency: K+1 = 4 detector forward passes per input. On T4 with batch=1,
~80-120ms overhead per image (acceptable for forensic, too slow for
real-time). Disabled by default; enable for high-security deployments.

Strict-compat: pure pre-detection filter. No changes to detector interface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GateSettings:
    """Configuration for the XAI adversarial gate."""
    enabled: bool = False  # Off by default — too slow for real-time
    num_perturbations: int = 3
    # Thresholds (empirically tuned on FF++ val; verify on your data)
    score_variance_threshold: float = 0.04  # ~0.2 std on score
    explanation_iou_threshold: float = 0.40  # mean IoU below this = adversarial
    # Perturbation strengths
    jpeg_quality_range: Tuple[int, int] = (70, 90)
    crop_ratio_range: Tuple[float, float] = (0.90, 0.95)
    flip_probability: float = 0.5
    seed: Optional[int] = None


@dataclass
class GateResult:
    """Result of the adversarial gate check."""
    is_adversarial: bool
    score_variance: float
    explanation_iou: float
    reason: str = ""
    scores: Optional[List[float]] = None


class AdversarialGate:
    """
    Detects adversarial inputs by measuring explanation stability
    under randomized perturbations.
    """

    def __init__(self, settings: Optional[GateSettings] = None):
        self.settings = settings or GateSettings()
        self._rng = random.Random(self.settings.seed)
        logger.info(
            "AdversarialGate initialized: enabled=%s, K=%d",
            self.settings.enabled, self.settings.num_perturbations,
        )

    # ------------------------------------------------------------------
    def check_image(
        self,
        image: np.ndarray,
        detect_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    ) -> GateResult:
        """
        Check if an image is adversarial by measuring explanation stability.

        Args:
            image: HxWx3 uint8 RGB image.
            detect_fn: Callable that takes an image and returns
                (score, explanation_heatmap). The heatmap should be a
                2D float32 array in [0, 1].

        Returns:
            GateResult with adversarial flag + diagnostics.
        """
        if not self.settings.enabled:
            return GateResult(
                is_adversarial=False,
                score_variance=0.0,
                explanation_iou=1.0,
                reason="gate_disabled",
            )

        try:
            # Baseline
            s0, e0 = detect_fn(image)
            scores = [s0]
            ious = []

            for _ in range(self.settings.num_perturbations):
                perturbed = self._perturb_image(image)
                si, ei = detect_fn(perturbed)
                scores.append(si)
                ious.append(self._iou(e0, ei))

            score_var = float(np.var(scores))
            mean_iou = float(np.mean(ious)) if ious else 1.0

            is_adv = (
                score_var > self.settings.score_variance_threshold
                or mean_iou < self.settings.explanation_iou_threshold
            )

            reason = ""
            if is_adv:
                if score_var > self.settings.score_variance_threshold:
                    reason += f"score_var={score_var:.4f}>{self.settings.score_variance_threshold}; "
                if mean_iou < self.settings.explanation_iou_threshold:
                    reason += f"iou={mean_iou:.4f}<{self.settings.explanation_iou_threshold}"
                # Iteration 7: record adversarial flag
                try:
                    from observability import get_default_metrics
                    get_default_metrics().record_adversarial_flag("image", "adversarial_gate")
                except Exception:
                    pass

            return GateResult(
                is_adversarial=is_adv,
                score_variance=score_var,
                explanation_iou=mean_iou,
                reason=reason,
                scores=scores,
            )
        except Exception as e:
            logger.warning("AdversarialGate check failed (%s); allowing input", e)
            return GateResult(
                is_adversarial=False,
                score_variance=0.0,
                explanation_iou=1.0,
                reason=f"check_failed:{e}",
            )

    # ------------------------------------------------------------------
    def _perturb_image(self, image: np.ndarray) -> np.ndarray:
        """Apply a random small perturbation to the image."""
        perturbed = image.copy()
        # Random JPEG
        q = self._rng.randint(*self.settings.jpeg_quality_range)
        perturbed = self._jpeg(perturbed, q)
        # Random horizontal flip
        if self._rng.random() < self.settings.flip_probability:
            perturbed = perturbed[:, ::-1, :].copy()
        # Random small crop + resize back
        ratio = self._rng.uniform(*self.settings.crop_ratio_range)
        h, w = perturbed.shape[:2]
        ch, cw = int(h * ratio), int(w * ratio)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        cropped = perturbed[y0:y0 + ch, x0:x0 + cw]
        try:
            import cv2
            perturbed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            perturbed = np.array(
                Image.fromarray(cropped).resize((w, h), Image.BILINEAR)
            )
        return perturbed

    def _jpeg(self, image: np.ndarray, quality: int) -> np.ndarray:
        import io
        from PIL import Image
        pil = Image.fromarray(image)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))

    def _iou(self, a: np.ndarray, b: np.ndarray) -> float:
        """Intersection-over-union of two binary masks (threshold at 0.5)."""
        if a is None or b is None:
            return 1.0
        if a.shape != b.shape:
            try:
                import cv2
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            except ImportError:
                return 1.0
        a_mask = a > 0.5
        b_mask = b > 0.5
        intersection = float(np.logical_and(a_mask, b_mask).sum())
        union = float(np.logical_or(a_mask, b_mask).sum())
        if union < 1:
            return 1.0
        return intersection / union


# ---------------------------------------------------------------------
_default_gate: Optional[AdversarialGate] = None


def get_default_gate() -> AdversarialGate:
    global _default_gate
    if _default_gate is None:
        _default_gate = AdversarialGate()
    return _default_gate
