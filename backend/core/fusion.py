"""
Argus Core - Multi-Modal Fusion
===============================
Attention-weighted fusion of multi-modal analysis results.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - core/fusion.py

SOTA Algorithms:
- Attention-Based Fusion: Learned attention weights based on modality confidence
- Uncertainty Quantification: Ensemble disagreement for confidence calibration

Integration:
- Imports: schemas/internal.py, config.py
- Inputs: List[ModalityResult]
- Outputs: AggregatedResult

Algorithm:
1. Extract confidence scores from each modality
2. Compute attention weights: softmax(confidence * learned_bias)
3. Weighted aggregation: Σ(weight_i × score_i)
4. Uncertainty estimation via ensemble disagreement
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from config import config
from schemas.schemas import (
    Modality, ContentType, ModalityResult, AggregatedResult,
    VideoResult, AudioResult, TextResult, MetadataResult
)
from utils.logging import get_logger

logger = get_logger(__name__)


# Default modality weights from config
DEFAULT_WEIGHTS = {
    Modality.VIDEO: config.score_weight_video_spatial + config.score_weight_video_temporal,
    Modality.AUDIO: config.score_weight_audio,
    Modality.TEXT: config.score_weight_text,
    Modality.IMAGE: config.score_weight_video_spatial,  # Images use spatial weight
}

# Content-type specific weight adjustments
CONTENT_TYPE_WEIGHTS = {
    ContentType.VIDEO_WITH_SPEECH: {
        Modality.VIDEO: 0.50,
        Modality.AUDIO: 0.35,
        Modality.TEXT: 0.05,
        Modality.IMAGE: 0.10
    },
    ContentType.VIDEO_NO_SPEECH: {
        Modality.VIDEO: 0.70,
        Modality.AUDIO: 0.05,
        Modality.TEXT: 0.05,
        Modality.IMAGE: 0.20
    },
    ContentType.AUDIO_ONLY: {
        Modality.VIDEO: 0.00,
        Modality.AUDIO: 0.90,
        Modality.TEXT: 0.05,
        Modality.IMAGE: 0.05
    },
    ContentType.IMAGE_ONLY: {
        Modality.VIDEO: 0.00,
        Modality.AUDIO: 0.00,
        Modality.TEXT: 0.00,
        Modality.IMAGE: 1.00
    },
    ContentType.TEXT_ONLY: {
        Modality.VIDEO: 0.00,
        Modality.AUDIO: 0.00,
        Modality.TEXT: 1.00,
        Modality.IMAGE: 0.00
    }
}


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""
    # Attention parameters
    use_attention: bool = True
    attention_temperature: float = 1.0  # Softmax temperature
    
    # Weight bounds
    min_weight: float = 0.05  # Minimum weight for any modality
    max_weight: float = 0.80  # Maximum weight for any modality
    
    # Uncertainty parameters
    uncertainty_method: str = "disagreement"  # "disagreement", "entropy", "variance"
    uncertainty_threshold: float = 0.3  # Threshold for "uncertain" flag
    
    # Confidence calibration
    confidence_scaling: float = 1.0  # Scale factor for confidence
    use_platt_scaling: bool = False  # Apply Platt scaling to outputs


class AttentionWeightComputer:
    """
    Computes attention weights for modality fusion.
    
    Uses softmax with learned biases to adapt weights based on:
    - Modality confidence (higher confidence = higher weight)
    - Content type (different base weights per content type)
    - Historical performance (optional learned biases)
    """
    
    def __init__(
        self,
        base_weights: Optional[Dict[Modality, float]] = None,
        learned_biases: Optional[Dict[Modality, float]] = None,
        temperature: float = 1.0
    ):
        """
        Initialize attention weight computer.
        
        Args:
            base_weights: Base weights per modality
            learned_biases: Learned bias adjustments
            temperature: Softmax temperature (higher = more uniform)
        """
        self.base_weights = base_weights or DEFAULT_WEIGHTS.copy()
        self.learned_biases = learned_biases or {m: 0.0 for m in Modality}
        self.temperature = temperature
    
    def compute(
        self,
        results: List[ModalityResult],
        content_type: Optional[ContentType] = None
    ) -> Dict[Modality, float]:
        """
        Compute attention weights for given results.
        
        Args:
            results: Modality results with confidence scores
            content_type: Content type for base weight selection
            
        Returns:
            Dict mapping modalities to weights (sum to 1.0)
        """
        if not results:
            return {}
        
        # Get base weights for content type
        if content_type and content_type in CONTENT_TYPE_WEIGHTS:
            base = CONTENT_TYPE_WEIGHTS[content_type].copy()
        else:
            base = self.base_weights.copy()
        
        # Compute attention scores
        modalities = []
        scores = []
        
        for result in results:
            modality = result.modality
            confidence = result.confidence
            
            # Attention score = base_weight * confidence + bias
            base_weight = base.get(modality, 0.1)
            bias = self.learned_biases.get(modality, 0.0)
            
            attention_score = base_weight * confidence + bias
            
            modalities.append(modality)
            scores.append(attention_score)
        
        # Apply softmax with temperature
        scores = np.array(scores)
        weights = self._softmax(scores / self.temperature)
        
        return {m: float(w) for m, w in zip(modalities, weights)}
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def update_biases(
        self,
        modality: Modality,
        delta: float
    ) -> None:
        """
        Update learned bias for a modality.
        
        Used for online learning from feedback.
        
        Args:
            modality: Modality to update
            delta: Bias adjustment
        """
        self.learned_biases[modality] = self.learned_biases.get(modality, 0.0) + delta


class UncertaintyEstimator:
    """
    Estimates uncertainty in fusion results.
    
    Uses ensemble disagreement and statistical measures to quantify
    how confident we should be in the final prediction.
    """
    
    def __init__(self, method: str = "disagreement"):
        """
        Initialize uncertainty estimator.
        
        Args:
            method: Uncertainty method ("disagreement", "entropy", "variance")
        """
        self.method = method
    
    def estimate(
        self,
        results: List[ModalityResult],
        weights: Dict[Modality, float]
    ) -> float:
        """
        Estimate uncertainty from modality results.
        
        Args:
            results: Individual modality results
            weights: Fusion weights
            
        Returns:
            Uncertainty score [0, 1] (higher = more uncertain)
        """
        if not results or len(results) < 2:
            return 0.5  # Default uncertainty for single modality
        
        if self.method == "disagreement":
            return self._disagreement_uncertainty(results, weights)
        elif self.method == "entropy":
            return self._entropy_uncertainty(results, weights)
        elif self.method == "variance":
            return self._variance_uncertainty(results, weights)
        else:
            return self._disagreement_uncertainty(results, weights)
    
    def _disagreement_uncertainty(
        self,
        results: List[ModalityResult],
        weights: Dict[Modality, float]
    ) -> float:
        """
        Compute uncertainty as weighted disagreement.
        
        Disagreement = weighted sum of |score_i - fused_score|
        """
        scores = np.array([r.score for r in results])
        weight_values = np.array([weights.get(r.modality, 0.1) for r in results])
        
        # Normalize weights
        weight_values = weight_values / weight_values.sum()
        
        # Fused score
        fused = np.dot(scores, weight_values)
        
        # Weighted disagreement
        disagreement = np.dot(np.abs(scores - fused), weight_values)
        
        # Normalize to [0, 1] (max disagreement is 0.5 when split 50/50)
        return float(np.clip(disagreement * 2, 0, 1))
    
    def _entropy_uncertainty(
        self,
        results: List[ModalityResult],
        weights: Dict[Modality, float]
    ) -> float:
        """
        Compute uncertainty as entropy of score distribution.
        """
        scores = np.array([r.score for r in results])
        
        # Treat scores as probabilities
        probs = np.clip(scores, 0.01, 0.99)  # Avoid log(0)
        
        # Binary entropy for each score
        entropies = -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)
        
        # Average entropy, normalized
        return float(np.mean(entropies))
    
    def _variance_uncertainty(
        self,
        results: List[ModalityResult],
        weights: Dict[Modality, float]
    ) -> float:
        """
        Compute uncertainty as variance of scores.
        """
        scores = np.array([r.score for r in results])
        
        # Variance, normalized to [0, 1] (max variance is 0.25 for binary)
        variance = np.var(scores)
        
        return float(np.clip(variance * 4, 0, 1))


class MultiModalFusion:
    """
    Attention-weighted fusion of multi-modal analysis results.
    
    Combines outputs from all analyzers (video, audio, text, image)
    into a single fused score with uncertainty quantification.
    
    The fusion process:
    1. Normalize scores from each modality
    2. Compute attention weights based on confidence and content type
    3. Apply weighted aggregation: fused = Σ(weight_i × score_i)
    4. Estimate uncertainty from modality disagreement
    5. Return calibrated result with confidence bounds
    
    Usage:
        fusion = MultiModalFusion()
        result = fusion.aggregate(modality_results, content_type)
    """
    
    def __init__(
        self,
        fusion_config: Optional[FusionConfig] = None
    ):
        """
        Initialize multi-modal fusion.
        
        Args:
            fusion_config: Fusion configuration options
        """
        self.config = fusion_config or FusionConfig()
        
        self.weight_computer = AttentionWeightComputer(
            temperature=self.config.attention_temperature
        )
        
        self.uncertainty_estimator = UncertaintyEstimator(
            method=self.config.uncertainty_method
        )
        
        logger.info(
            f"MultiModalFusion initialized: attention={self.config.use_attention}, "
            f"uncertainty={self.config.uncertainty_method}"
        )
    
    def aggregate(
        self,
        results: List[ModalityResult],
        content_type: Optional[ContentType] = None
    ) -> AggregatedResult:
        """
        Fuse multi-modal results into single aggregated result.
        
        Args:
            results: Results from each modality analyzer
            content_type: Detected content type (affects weight distribution)
            
        Returns:
            AggregatedResult with fused score and uncertainty
        """
        if not results:
            logger.warning("No modality results to fuse")
            return AggregatedResult(
                modality_results=[],
                fused_score=0.5,
                uncertainty=1.0,
                weights_used={}
            )
        
        # Handle single modality case
        if len(results) == 1:
            result = results[0]
            # For single modality, uncertainty should be based on score extremity
            # not just 1 - confidence (which double-penalizes in TrustScorer)
            # Low uncertainty for confident predictions (away from 0.5)
            score_extremity = abs(result.score - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
            # Uncertainty is inverse of both confidence and extremity
            uncertainty = (1 - result.confidence) * (1 - score_extremity * 0.5)
            return AggregatedResult(
                modality_results=results,
                fused_score=result.score,
                uncertainty=uncertainty,
                weights_used={result.modality.value: 1.0}
            )
        
        # Compute attention weights
        if self.config.use_attention:
            weights = self.weight_computer.compute(results, content_type)
        else:
            # Use uniform weights
            weights = {r.modality: 1.0 / len(results) for r in results}
        
        # Clamp weights to bounds
        total_weight = sum(weights.values())
        weights = {
            m: np.clip(w / total_weight, self.config.min_weight, self.config.max_weight)
            for m, w in weights.items()
        }
        
        # Renormalize after clamping
        total_weight = sum(weights.values())
        weights = {m: w / total_weight for m, w in weights.items()}
        
        # Compute fused score
        fused_score = 0.0
        for result in results:
            weight = weights.get(result.modality, 0.0)
            fused_score += weight * result.score
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator.estimate(results, weights)
        
        # Apply confidence scaling
        if self.config.confidence_scaling != 1.0:
            fused_score = self._apply_scaling(fused_score)
        
        # Convert weights dict to string keys for schema
        weights_str = {m.value: w for m, w in weights.items()}
        
        logger.debug(
            f"Fusion complete: fused={fused_score:.3f}, uncertainty={uncertainty:.3f}, "
            f"weights={weights_str}"
        )
        
        return AggregatedResult(
            modality_results=results,
            fused_score=float(np.clip(fused_score, 0, 1)),
            uncertainty=float(np.clip(uncertainty, 0, 1)),
            weights_used=weights_str
        )
    
    def aggregate_from_analyzers(
        self,
        video_result: Optional[VideoResult] = None,
        audio_result: Optional[AudioResult] = None,
        text_result: Optional[TextResult] = None,
        image_score: Optional[float] = None,
        metadata_result: Optional[MetadataResult] = None,
        content_type: Optional[ContentType] = None
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
                    "lip_sync_detected": (
                        video_result.lip_sync is not None and
                        video_result.lip_sync.manipulation_probability > 0.5
                    ),
                    "temporal_inconsistency": video_result.temporal.flickering_detected
                }
            ))
        
        if audio_result:
            modality_results.append(ModalityResult(
                modality=Modality.AUDIO,
                score=audio_result.synthetic_probability,
                confidence=self._compute_audio_confidence(audio_result),
                details={
                    "vocoder_detected": audio_result.vocoder_artifacts_detected,
                    "voice_consistency": audio_result.voice_consistency_score
                }
            ))
        
        if text_result:
            modality_results.append(ModalityResult(
                modality=Modality.TEXT,
                score=text_result.ai_probability,
                confidence=self._compute_text_confidence(text_result),
                details={
                    "perplexity": text_result.perplexity_score,
                    "burstiness": text_result.burstiness_score
                }
            ))
        
        if image_score is not None:
            modality_results.append(ModalityResult(
                modality=Modality.IMAGE,
                score=image_score,
                confidence=0.8,  # Default confidence for image
                details={}
            ))
        
        # Aggregate
        result = self.aggregate(modality_results, content_type)
        
        # Adjust for metadata if present
        if metadata_result:
            result = self._adjust_for_metadata(result, metadata_result)
        
        return result
    
    def _compute_video_confidence(self, video_result: VideoResult) -> float:
        """Compute confidence for video result."""
        # Higher confidence if face detected and more frames analyzed
        base_conf = 0.7
        
        if video_result.face_detected:
            base_conf += 0.1
        
        frames = video_result.frames_analyzed
        if frames >= 100:
            base_conf += 0.1
        elif frames >= 50:
            base_conf += 0.05
        
        # Reduce confidence if many anomalies detected
        if len(video_result.spatial.anomaly_indices) > 0.3 * frames:
            base_conf -= 0.1
        
        return float(np.clip(base_conf, 0.3, 0.95))
    
    def _compute_audio_confidence(self, audio_result: AudioResult) -> float:
        """Compute confidence for audio result."""
        # Higher confidence if voice consistency is good
        base_conf = 0.7
        
        base_conf += audio_result.voice_consistency_score * 0.2
        
        if audio_result.vocoder_artifacts_detected:
            base_conf += 0.1  # More confident when artifacts found
        
        return float(np.clip(base_conf, 0.3, 0.95))
    
    def _compute_text_confidence(self, text_result: TextResult) -> float:
        """Compute confidence for text result."""
        # RADAR score if available increases confidence
        base_conf = 0.6
        
        if text_result.radar_score is not None:
            base_conf += 0.2
        
        # Very low or very high perplexity increases confidence
        if text_result.perplexity_score < 20 or text_result.perplexity_score > 200:
            base_conf += 0.1
        
        return float(np.clip(base_conf, 0.3, 0.95))
    
    def _adjust_for_metadata(
        self,
        result: AggregatedResult,
        metadata: MetadataResult
    ) -> AggregatedResult:
        """
        Adjust fusion result based on metadata analysis.
        
        Metadata doesn't contribute to score directly but affects confidence.
        """
        adjustment = 0.0
        
        # C2PA presence increases authenticity confidence
        if metadata.c2pa.present:
            if metadata.c2pa.valid:
                adjustment -= 0.1  # Reduce fake probability
            else:
                adjustment += 0.05  # Invalid C2PA is suspicious
        
        # EXIF anomalies increase suspicion
        anomaly_count = len(metadata.exif_anomalies)
        if anomaly_count > 0:
            adjustment += 0.02 * min(anomaly_count, 5)
        
        # Invalid file structure is very suspicious
        if not metadata.file_structure_valid:
            adjustment += 0.1
        
        # Apply adjustment to fused score
        new_score = np.clip(result.fused_score + adjustment, 0, 1)
        
        return AggregatedResult(
            modality_results=result.modality_results,
            fused_score=float(new_score),
            uncertainty=result.uncertainty,
            weights_used=result.weights_used
        )
    
    def _apply_scaling(self, score: float) -> float:
        """Apply confidence scaling to score."""
        # Simple scaling around 0.5
        centered = score - 0.5
        scaled = centered * self.config.confidence_scaling
        return float(np.clip(scaled + 0.5, 0, 1))
    
    def get_fusion_weights(
        self,
        content_type: ContentType
    ) -> Dict[Modality, float]:
        """
        Get fusion weights for a content type.
        
        Useful for transparency and debugging.
        
        Args:
            content_type: Content type
            
        Returns:
            Base weights before attention adjustment
        """
        if content_type in CONTENT_TYPE_WEIGHTS:
            return CONTENT_TYPE_WEIGHTS[content_type].copy()
        return DEFAULT_WEIGHTS.copy()
    
    def explain_fusion(
        self,
        result: AggregatedResult
    ) -> Dict[str, Any]:
        """
        Generate explanation of fusion process.
        
        Args:
            result: Aggregated result
            
        Returns:
            Dict with fusion explanation
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
                "contribution": contribution
            })
        
        # Sort by contribution
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        
        return {
            "fused_score": result.fused_score,
            "uncertainty": result.uncertainty,
            "num_modalities": len(result.modality_results),
            "contributions": contributions,
            "dominant_modality": contributions[0]["modality"] if contributions else None,
            "weights_used": result.weights_used
        }


# Singleton instance
_fusion: Optional[MultiModalFusion] = None


def get_multi_modal_fusion() -> MultiModalFusion:
    """Get singleton fusion instance."""
    global _fusion
    if _fusion is None:
        _fusion = MultiModalFusion()
    return _fusion
