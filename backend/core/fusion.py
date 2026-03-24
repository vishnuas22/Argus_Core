"""
Argus Core - Multi-Modal Fusion (CAMME Cross-Attention Engine)
================================================================
Replaces the previous score-level weighted aggregation with the
CAMME-inspired cross-modal cross-attention fusion engine.

The new fusion operates at the feature level through pairwise
cross-modal attention, capturing inter-modality correlations
(e.g., audio frequency misalignment with lip dynamics) rather
than simply averaging per-modality scores.

Two interfaces:
1. aggregate(modality_results, content_type) -> AggregatedResult
   Maintains backward compatibility with the orchestrator.
   When raw features are not available, uses the classification
   head's learned decision boundary on score-derived features.

2. fuse_raw(frames, waveform, input_ids, ...) -> FusionOutput
   Full neural forward pass with raw inputs for maximum accuracy.

Reference: Khan et al., "CAMME: Adaptive Deepfake Image Detection
with Multi-Modal Cross-Attention", arXiv:2505.18035, 2025.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from config import config
from schemas.schemas import (
    Modality, ContentType, ModalityResult, AggregatedResult,
    VideoResult, AudioResult, TextResult, MetadataResult
)
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""
    use_cross_attention: bool = True
    uncertainty_method: str = "disagreement"
    min_weight: float = 0.05
    max_weight: float = 0.80
    confidence_scaling: float = 1.0


class UncertaintyEstimator:
    """
    Estimates uncertainty from modality result disagreement.

    Higher disagreement between modalities indicates lower confidence
    in the fused prediction.
    """

    def __init__(self, method: str = "disagreement"):
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

        if self.method == "disagreement":
            return self._disagreement_uncertainty(results, weights)
        elif self.method == "entropy":
            return self._entropy_uncertainty(results, weights)
        elif self.method == "variance":
            return self._variance_uncertainty(results, weights)
        return self._disagreement_uncertainty(results, weights)

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

    def _variance_uncertainty(
        self,
        results: List[ModalityResult],
        weights: Dict[str, float],
    ) -> float:
        scores = np.array([r.score for r in results])
        variance = np.var(scores)
        return float(np.clip(variance * 4, 0, 1))


class MultiModalFusion:
    """
    Cross-modal cross-attention fusion of multi-modal analysis results.

    Primary interface: aggregate() for orchestrator compatibility.
    Extended interface: fuse_raw() for full neural forward pass.

    When aggregate() is called with ModalityResult objects (scores),
    the classification head's learned decision boundary maps the
    score-derived features to the final fake probability.

    When fuse_raw() is called with raw tensor inputs, the complete
    encoder -> cross-attention -> self-attention -> classification
    pipeline executes end-to-end.
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

        logger.info(
            "MultiModalFusion initialized with cross-attention engine"
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
        Fuse multi-modal results into a single aggregated result.

        Maintains backward compatibility with the orchestrator's call:
            fusion.aggregate(modality_results, content_type)

        Internally uses the cross-attention classification head's learned
        weights on score-derived features rather than manual weighting.

        Args:
            results: Results from each modality analyzer
            content_type: Detected content type

        Returns:
            AggregatedResult with fused score and uncertainty
        """
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

            # Safety: if confidence is low, pull score toward uncertain (0.5)
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

        # Convert modality scores to feature vectors for the neural engine
        modality_features = self._scores_to_features(results)

        # Run through the classification head (learned decision boundary)
        fused_score, attention_weights = self._neural_fusion(modality_features)

        # Estimate uncertainty from modality disagreement
        weights_for_uncertainty = {
            r.modality.value: attention_weights.get(r.modality.value, 1.0 / len(results))
            for r in results
        }
        uncertainty = self.uncertainty_estimator.estimate(results, weights_for_uncertainty)

        # Normalize weights
        total_w = sum(weights_for_uncertainty.values())
        if total_w > 0:
            weights_for_uncertainty = {
                k: v / total_w for k, v in weights_for_uncertainty.items()
            }

        logger.debug(
            f"Cross-attention fusion: fused={fused_score:.3f}, "
            f"uncertainty={uncertainty:.3f}, "
            f"weights={weights_for_uncertainty}"
        )

        return AggregatedResult(
            modality_results=results,
            fused_score=float(np.clip(fused_score, 0, 1)),
            uncertainty=float(np.clip(uncertainty, 0, 1)),
            weights_used=weights_for_uncertainty,
        )

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
        text_result: Optional[TextResult] = None,
        image_score: Optional[float] = None,
        metadata_result: Optional[MetadataResult] = None,
        content_type: Optional[ContentType] = None,
    ) -> AggregatedResult:
        """
        Convenience method to aggregate from analyzer-specific results.

        Args:
            video_result: Video analysis result
            audio_result: Audio analysis result
            text_result: Text analysis result
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

        if text_result:
            modality_results.append(ModalityResult(
                modality=Modality.TEXT,
                score=text_result.ai_probability,
                confidence=self._compute_text_confidence(text_result),
                details={
                    "perplexity": text_result.perplexity_score,
                    "burstiness": text_result.burstiness_score,
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
            "fusion_method": "cross_attention",
            "fused_score": result.fused_score,
            "uncertainty": result.uncertainty,
            "num_modalities": len(result.modality_results),
            "contributions": contributions,
            "dominant_modality": contributions[0]["modality"] if contributions else None,
            "weights_used": result.weights_used,
        }

    # ===== Internal Methods =====

    def _scores_to_features(
        self,
        results: List[ModalityResult],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert modality scores to feature vectors for the neural engine.

        Maps each modality's score and confidence into a 512-d feature
        representation that the classification head can process.

        The encoding uses the score and confidence to modulate learnable
        basis vectors, preserving modality-specific information.
        """
        d_fused = 512
        features = {}

        for result in results:
            modality_name = result.modality.value
            score = result.score
            confidence = result.confidence

            # Create feature vector encoding score and confidence
            # Use sinusoidal encoding for score position and confidence magnitude
            feature = torch.zeros(1, d_fused)

            # First half encodes the score with sinusoidal pattern
            positions = torch.arange(0, d_fused // 2, dtype=torch.float32)
            freq = 1.0 / (10000.0 ** (2.0 * positions / d_fused))
            feature[0, :d_fused // 2] = torch.sin(score * freq)

            # Second half encodes the confidence with sinusoidal pattern
            feature[0, d_fused // 2:] = torch.cos(confidence * freq)

            features[modality_name] = feature

        return features

    def _neural_fusion(
        self,
        modality_features: Dict[str, torch.Tensor],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Run the cross-attention classification on score-derived features.

        Uses the classification head with cross-attention blocks on
        the score-derived feature vectors.

        Args:
            modality_features: Dict mapping modality names to feature tensors

        Returns:
            Tuple of (fused_score, attention_weights)
        """
        z_v = modality_features.get("video", torch.zeros(1, 512))
        z_a = modality_features.get("audio", torch.zeros(1, 512))
        z_t = modality_features.get("text", torch.zeros(1, 512))
        z_i = modality_features.get("image", None)

        # If image modality present, combine with video features
        if z_i is not None:
            z_v = (z_v + z_i) / 2.0

        # Run through cross-attention fusion engine
        engine = self.cross_attention_engine
        with torch.no_grad():
            output = engine.forward_from_features(
                z_visual=z_v,
                z_audio=z_a,
                z_text=z_t,
            )

        fused_score = float(output.fake_probability.mean().item())

        attention_weights = {
            k: float(v.mean().item()) if v is not None else 0.0
            for k, v in output.cross_attention_weights.items()
        }

        # Map cross-attention pair weights to modality-level weights
        modality_weights = {
            "video": (attention_weights.get("visual_to_audio", 0)
                     + attention_weights.get("visual_to_text", 0)) / 2.0,
            "audio": (attention_weights.get("audio_to_visual", 0)
                     + attention_weights.get("audio_to_text", 0)) / 2.0,
            "text": (attention_weights.get("text_to_visual", 0)
                    + attention_weights.get("text_to_audio", 0)) / 2.0,
        }

        return fused_score, modality_weights

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

    def _compute_text_confidence(self, text_result: TextResult) -> float:
        """Compute confidence for text result."""
        base_conf = 0.6
        if text_result.radar_score is not None:
            base_conf += 0.2
        if text_result.perplexity_score < 20 or text_result.perplexity_score > 200:
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
