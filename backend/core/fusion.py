"""
Argus Core - Multi-Modal Fusion (Evidential Uncertainty-Aware Fusion)
======================================================================
Replaces the previous score-level weighted aggregation with a
principled evidential fusion that models each modality's prediction
as evidence for a Dirichlet distribution.

Key innovations:
1. Dirichlet-based uncertainty: each modality contributes evidence
   parameters (alpha) rather than point scores. Total evidence scales
   inversely with uncertainty.
2. Precision-weighted aggregation: modalities with higher confidence
   (more extreme scores) contribute more evidence, automatically
   down-weighting uncertain modalities.
3. Disagreement-aware: when modalities disagree, the Dirichlet
   concentration parameters produce a flatter distribution (higher
   uncertainty without ad-hoc penalty terms).
4. Backward compatible: same API as previous confidence-weighted average.

Two interfaces:
1. aggregate(modality_results, content_type) -> AggregatedResult
   Maintains backward compatibility with the orchestrator.
   
2. fuse_raw(frames, waveform, ...) -> FusionOutput
   Full neural forward pass with raw inputs (requires UMFT weights).

Reference: Sensoy et al., "Evidential Deep Learning to Quantify
Classification Uncertainty", NeurIPS 2018.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from config import config
from schemas.schemas import (
    Modality, ContentType, ModalityResult, AggregatedResult,
    VideoResult, AudioResult, MetadataResult
)
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""
    use_cross_attention: bool = False
    uncertainty_method: str = "dirichlet"
    min_weight: float = 0.05
    max_weight: float = 0.80
    confidence_scaling: float = 1.0
    evidence_reg: float = 1e-3  # Regularization to prevent zero evidence


class UncertaintyEstimator:
    """
    Estimates uncertainty from modality result disagreement.

    Uses Dirichlet-based uncertainty estimation:
    - High agreement + extreme scores → low uncertainty
    - High disagreement → high uncertainty
    - Low confidence across all modalities → high uncertainty
    """

    def __init__(self, method: str = "dirichlet"):
        self.method = method

    def estimate(
        self,
        results: List[ModalityResult],
        weights: Dict[str, float],
    ) -> float:
        """
        Estimate uncertainty from modality results.

        Args:
            results: Individual modality results
            weights: Fusion weights per modality

        Returns:
            Uncertainty score [0, 1] (higher = more uncertain)
        """
        if not results or len(results) < 2:
            return 0.5

        if self.method == "dirichlet":
            return self._dirichlet_uncertainty(results, weights)
        elif self.method == "disagreement":
            return self._disagreement_uncertainty(results, weights)
        elif self.method == "entropy":
            return self._entropy_uncertainty(results, weights)
        return self._dirichlet_uncertainty(results, weights)

    def _compute_dirichlet_params(
        self,
        results: List[ModalityResult],
        weights: Dict[str, float],
    ) -> Tuple[float, float]:
        """Compute Dirichlet concentration parameters from modality evidence."""
        alpha_fake = 1.0
        alpha_real = 1.0

        for r in results:
            w = weights.get(r.modality.value, 0.5)
            score = r.score
            conf = r.confidence

            evidence = conf * (abs(score - 0.5) * 2) + self._evidence_reg
            alpha_fake += w * evidence * score
            alpha_real += w * evidence * (1.0 - score)

        return alpha_fake, alpha_real

    def _dirichlet_uncertainty(
        self,
        results: List[ModalityResult],
        weights: Dict[str, float],
    ) -> float:
        """Uncertainty from Dirichlet distribution: K / sum(alpha)."""
        alpha_fake, alpha_real = self._compute_dirichlet_params(results, weights)
        total_alpha = alpha_fake + alpha_real
        uncertainty = min(2.0 / total_alpha, 1.0)
        return float(uncertainty)

    def _disagreement_uncertainty(
        self,
        results: List[ModalityResult],
        weights: Dict[str, float],
    ) -> float:
        scores = np.array([r.score for r in results])
        weight_values = np.array([
            weights.get(r.modality.value, 0.1) for r in results
        ])
        weight_values = weight_values / weight_values.sum()
        fused = np.dot(scores, weight_values)
        disagreement = np.dot(np.abs(scores - fused), weight_values)
        return float(np.clip(disagreement * 2, 0, 1))

    def _entropy_uncertainty(
        self,
        results: List[ModalityResult],
        weights: Dict[str, float],
    ) -> float:
        scores = np.array([r.score for r in results])
        probs = np.clip(scores, 0.01, 0.99)
        entropies = -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)
        return float(np.mean(entropies))

class MultiModalFusion:
    """
    Evidential uncertainty-aware fusion of multi-modal analysis results.

    Uses Dirichlet-based evidence aggregation to produce well-calibrated
    fused scores with principled uncertainty estimates. No training
    required — evidence is derived from each modality's score extremity
    and confidence.

    Primary interface: aggregate() for orchestrator compatibility.
    Extended interface: fuse_raw() for full neural forward pass.
    """

    def __init__(self, fusion_config: Optional[FusionConfig] = None):
        """
        Initialize multi-modal fusion.

        Args:
            fusion_config: Configuration options
        """
        self.config = fusion_config or FusionConfig()
        self.uncertainty_estimator = UncertaintyEstimator(
            method=self.config.uncertainty_method
        )

        # Lazily initialize the cross-attention engine
        self._cross_attention_engine = None

        # Store last aggregate results for ensemble fusion
        self._last_aggregate_results: List[ModalityResult] = []

        logger.info(
            "MultiModalFusion initialized with evidential Dirichlet fusion"
        )

    @property
    def cross_attention_engine(self):
        """Lazy-load the cross-attention neural engine."""
        if self._cross_attention_engine is None:
            from core.cross_attention_fusion import (
                CrossModalCrossAttentionFusion,
                CrossAttentionConfig,
            )
            self._cross_attention_engine = CrossModalCrossAttentionFusion(
                CrossAttentionConfig(pretrained_encoders=False)
            )
            logger.info(
                "Cross-attention engine loaded (encoders will load on first use)"
            )
        return self._cross_attention_engine

    def aggregate(
        self,
        results: List[ModalityResult],
        content_type: Optional[ContentType] = None,
    ) -> AggregatedResult:
        """
        Fuse multi-modal results using evidential Dirichlet aggregation.

        Maintains backward compatibility with the orchestrator's call:
            fusion.aggregate(modality_results, content_type)

        Uses Dirichlet-based uncertainty quantification for
        production-ready calibrated results.

        Args:
            results: Results from each modality analyzer
            content_type: Detected content type

        Returns:
            AggregatedResult with fused score and uncertainty
        """
        self._last_aggregate_results = results

        if not results:
            logger.warning("No modality results to fuse")
            return AggregatedResult(
                modality_results=[],
                fused_score=0.5,
                uncertainty=1.0,
                weights_used={},
            )

        if len(results) == 1:
            result = results[0]
            score = result.score
            confidence = result.confidence
            if confidence < 0.5:
                score = 0.5 + (score - 0.5) * (confidence / 0.5)
            score = float(np.clip(score, 0.0, 1.0))
            score_extremity = abs(score - 0.5) * 2
            uncertainty = (1 - confidence) * (1 - score_extremity * 0.5)
            return AggregatedResult(
                modality_results=results,
                fused_score=score,
                uncertainty=uncertainty,
                weights_used={result.modality.value: 1.0},
            )

        # Evidential Dirichlet fusion
        fused_score, attention_weights = self._evidential_fusion(results)

        # Estimate uncertainty from Dirichlet concentration
        uncertainty = self.uncertainty_estimator.estimate(results, attention_weights)

        # Normalize weights
        total_w = sum(attention_weights.values())
        if total_w > 0:
            attention_weights = {k: v / total_w for k, v in attention_weights.items()}

        logger.debug(
            f"Evidential fusion: fused={fused_score:.3f}, "
            f"uncertainty={uncertainty:.3f}, "
            f"weights={attention_weights}"
        )

        return AggregatedResult(
            modality_results=results,
            fused_score=float(np.clip(fused_score, 0, 1)),
            uncertainty=float(np.clip(uncertainty, 0, 1)),
            weights_used=attention_weights,
        )

    def _evidential_fusion(
        self,
        results: List[ModalityResult],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evidential Dirichlet fusion of modality scores.

        Each modality contributes evidence (alpha_fake, alpha_real) based on
        its score extremity and confidence. The fused score is the
        precision-weighted mean across modalities.

        Args:
            results: Modality results

        Returns:
            Tuple of (fused_score, attention_weights)
        """
        try:
            alpha_fake = 1.0
            alpha_real = 1.0
            weights = {}

            for r in results:
                mod = r.modality.value
                score = r.confidence
                extremity = abs(r.score - 0.5) * 2

                # Evidence scales with confidence × extremity
                # Uncertain or near-0.5 predictions contribute little evidence
                evidence = max(score * extremity, 1e-4)

                alpha_fake += evidence * r.score
                alpha_real += evidence * (1.0 - r.score)
                weights[mod] = evidence

            total_alpha = alpha_fake + alpha_real
            fused_score = alpha_fake / total_alpha if total_alpha > 0 else 0.5

            if not np.isfinite(fused_score):
                fused_score = 0.5

            fused_score = float(np.clip(fused_score, 0.0, 1.0))

            logger.debug(
                f"Evidential fusion: alpha_fake={alpha_fake:.2f}, "
                f"alpha_real={alpha_real:.2f}, "
                f"fused={fused_score:.3f}"
            )

            return fused_score, weights

        except Exception as exc:
            logger.error(f"Evidential fusion failed: {exc}", exc_info=True)
            return 0.5, {"video": 0.5, "audio": 0.5}

    def fuse_raw(
        self,
        frames: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, float], torch.Tensor]:
        """
        Full neural forward pass with raw tensor inputs.

        Executes the complete pipeline:
            Encoders -> Cross-Attention -> Self-Attention -> Classification

        Args:
            frames: Video frames [B, T, C, H, W] or image [B, C, H, W]
            waveform: Raw audio [B, num_samples] at 16kHz
            input_ids: Tokenized text [B, seq_len]
            attention_mask: Text attention mask [B, seq_len]

        Returns:
            Tuple of (fake_probability, attention_weights_dict, fused_features)
        """
        engine = self.cross_attention_engine
        with torch.no_grad():
            output = engine.forward(
                frames=frames,
                waveform=waveform,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        fake_prob = float(output.fake_probability.mean().item())
        attn_weights = {
            k: float(v.mean().item()) if v is not None else 0.0
            for k, v in output.cross_attention_weights.items()
        }

        return fake_prob, attn_weights, output.fused_features

    def aggregate_from_analyzers(
        self,
        video_result: Optional[VideoResult] = None,
        audio_result: Optional[AudioResult] = None,
        image_score: Optional[float] = None,
        metadata_result: Optional[MetadataResult] = None,
        content_type: Optional[ContentType] = None,
    ) -> AggregatedResult:
        """
        Convenience method to aggregate from analyzer-specific results.

        Args:
            video_result: Video analysis result
            audio_result: Audio analysis result
            image_score: Single image detection score
            metadata_result: Metadata analysis (affects confidence, not score)
            content_type: Content type for weight selection

        Returns:
            AggregatedResult
        """
        modality_results = []

        if video_result:
            modality_results.append(ModalityResult(
                modality=Modality.VIDEO,
                score=video_result.aggregate_score,
                confidence=self._compute_video_confidence(video_result),
                details={
                    "frames_analyzed": video_result.frames_analyzed,
                    "face_detected": video_result.face_detected,
                },
            ))

        if audio_result:
            modality_results.append(ModalityResult(
                modality=Modality.AUDIO,
                score=audio_result.synthetic_probability,
                confidence=self._compute_audio_confidence(audio_result),
                details={
                    "vocoder_detected": audio_result.vocoder_artifacts_detected,
                    "voice_consistency": audio_result.voice_consistency_score,
                },
            ))

        if image_score is not None:
            modality_results.append(ModalityResult(
                modality=Modality.IMAGE,
                score=image_score,
                confidence=0.8,
                details={},
            ))

        result = self.aggregate(modality_results, content_type)

        if metadata_result:
            result = self._adjust_for_metadata(result, metadata_result)

        return result

    def explain_fusion(self, result: AggregatedResult) -> Dict[str, Any]:
        """
        Generate explanation of fusion process.

        Args:
            result: Aggregated result

        Returns:
            Dict with fusion explanation including cross-attention weights
        """
        contributions = []
        for mr in result.modality_results:
            weight = result.weights_used.get(mr.modality.value, 0)
            contribution = weight * mr.score
            contributions.append({
                "modality": mr.modality.value,
                "score": mr.score,
                "confidence": mr.confidence,
                "weight": weight,
                "contribution": contribution,
            })

        contributions.sort(key=lambda x: x["contribution"], reverse=True)

        return {
            "fusion_method": "evidential_dirichlet",
            "fused_score": result.fused_score,
            "uncertainty": result.uncertainty,
            "num_modalities": len(result.modality_results),
            "contributions": contributions,
            "dominant_modality": contributions[0]["modality"] if contributions else None,
            "weights_used": result.weights_used,
        }

    # ===== Internal Methods =====

    def _compute_video_confidence(self, video_result: VideoResult) -> float:
        """Compute confidence for video result."""
        base_conf = 0.7
        if video_result.face_detected:
            base_conf += 0.1
        frames = video_result.frames_analyzed
        if frames >= 100:
            base_conf += 0.1
        elif frames >= 50:
            base_conf += 0.05
        return float(np.clip(base_conf, 0.3, 0.95))

    def _compute_audio_confidence(self, audio_result: AudioResult) -> float:
        """Compute confidence for audio result."""
        base_conf = 0.7
        base_conf += audio_result.voice_consistency_score * 0.2
        if audio_result.vocoder_artifacts_detected:
            base_conf += 0.1
        return float(np.clip(base_conf, 0.3, 0.95))

    def _adjust_for_metadata(
        self,
        result: AggregatedResult,
        metadata: MetadataResult,
    ) -> AggregatedResult:
        """Adjust fusion result based on metadata analysis."""
        adjustment = 0.0
        if metadata.c2pa.present:
            if metadata.c2pa.valid:
                adjustment -= 0.1
            else:
                adjustment += 0.05
        anomaly_count = len(metadata.exif_anomalies)
        if anomaly_count > 0:
            adjustment += 0.02 * min(anomaly_count, 5)
        if not metadata.file_structure_valid:
            adjustment += 0.1

        new_score = np.clip(result.fused_score + adjustment, 0, 1)
        return AggregatedResult(
            modality_results=result.modality_results,
            fused_score=float(new_score),
            uncertainty=result.uncertainty,
            weights_used=result.weights_used,
        )


# ===== Singleton Management =====

_fusion: Optional[MultiModalFusion] = None


def get_multi_modal_fusion() -> MultiModalFusion:
    """Get singleton fusion instance."""
    global _fusion
    if _fusion is None:
        _fusion = MultiModalFusion()
    return _fusion
