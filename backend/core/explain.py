"""
Argus Core - Explainability Engine
==================================
Generate human-interpretable explanations for deepfake detection results.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - core/explain.py

SOTA Algorithms:
- Visual: GradCAM++ (improved gradient weighting for better localization)
- Textual: Template-based generation with dynamic slot filling

Integration:
- Imports: schemas/internal.py, models/manager.py
- Inputs: model_activations, ModalityResult
- Outputs: Explanation (heatmaps + text)

Key Features:
- GradCAM++ heatmap generation for visual explanations
- Manipulation region localization from heatmaps
- Template-based textual explanation generation
- No external LLM dependencies required
"""

import asyncio
import io
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import cv2
from enum import Enum

from config import config
from schemas.schemas import (
    Modality, Verdict, ContentType, AggregatedResult, ModalityResult,
    Explanation, ManipulationRegion, VideoResult, AudioResult
)
from utils.logging import get_logger
from utils.errors import InferenceError

logger = get_logger(__name__)


class ManipulationType(str, Enum):
    """Types of detected manipulation."""
    FACE_SWAP = "face_swap"
    LIP_SYNC = "lip_sync"
    FACE_REENACTMENT = "face_reenactment"
    AUDIO_CLONE = "audio_clone"
    AI_GENERATED_IMAGE = "ai_generated_image"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    METADATA_TAMPERING = "metadata_tampering"
    UNKNOWN = "unknown"


@dataclass
class Region:
    """Detected region of interest in analysis."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    region_type: str = "face"
    frame_index: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "region_type": self.region_type,
            "frame_index": self.frame_index
        }
    
    @property
    def area(self) -> int:
        """Get region area in pixels."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get region center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class HeatmapResult:
    """Result of heatmap generation."""
    heatmap: np.ndarray
    overlay: Optional[np.ndarray] = None
    regions: List[Region] = field(default_factory=list)
    model_name: str = ""
    frame_index: Optional[int] = None


# Explanation templates for different scenarios
EXPLANATION_TEMPLATES = {
    "high_confidence_fake": {
        "summary": "Analysis indicates this content is likely manipulated with {confidence}% confidence.",
        "findings": [
            "{modality} analysis detected {manipulation_type} artifacts",
            "Trust Score: {score}/100 ({verdict})",
            "Primary indicators: {indicators}"
        ]
    },
    "high_confidence_real": {
        "summary": "Analysis indicates this content appears authentic with {confidence}% confidence.",
        "findings": [
            "No significant manipulation indicators detected",
            "Trust Score: {score}/100 ({verdict})",
            "Content passed {num_checks} authenticity checks"
        ]
    },
    "uncertain": {
        "summary": "Analysis results are inconclusive. Human review recommended.",
        "findings": [
            "Mixed signals detected across modalities",
            "Trust Score: {score}/100 ({verdict})",
            "Uncertainty factors: {uncertainty_factors}"
        ]
    },
    "spatial_artifacts": "Spatial analysis detected {artifact_type} in {region} region",
    "temporal_artifacts": "Temporal analysis found {artifact_type} at {timestamp}",
    "audio_artifacts": "Audio analysis detected {artifact_type} with {confidence}% confidence",
    "lipsync_artifacts": "Lip-sync analysis detected potential {tech_type} manipulation"
}


class ExplainabilityEngine:
    """
    Generate human-interpretable explanations for analysis results.
    
    Provides:
    - GradCAM++ visual explanations showing model attention
    - Textual summaries explaining findings
    - Manipulation region localization
    - Evidence compilation for reports
    """
    
    def __init__(
        self,
        heatmap_colormap: str = "jet",
        heatmap_alpha: float = 0.5,
        localization_threshold: float = 0.5
    ):
        """
        Initialize explainability engine.
        
        Args:
            heatmap_colormap: OpenCV colormap for heatmap visualization
            heatmap_alpha: Heatmap overlay transparency (0-1)
            localization_threshold: Threshold for region localization
        """
        self.heatmap_colormap = getattr(cv2, f"COLORMAP_{heatmap_colormap.upper()}", cv2.COLORMAP_JET)
        self.heatmap_alpha = heatmap_alpha
        self.localization_threshold = localization_threshold
        
        logger.info(
            f"ExplainabilityEngine initialized: colormap={heatmap_colormap}, "
            f"alpha={heatmap_alpha}, threshold={localization_threshold}"
        )
    
    def generate_gradcam(
        self,
        activations: np.ndarray,
        gradients: np.ndarray,
        original_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate GradCAM++ heatmap from model activations and gradients.
        
        GradCAM++ improves on GradCAM with better gradient weighting
        for more precise localization of important regions.
        
        Args:
            activations: Feature map activations [C, H, W]
            gradients: Gradients w.r.t. activations [C, H, W]
            original_size: Target size for upsampling (W, H)
            
        Returns:
            Normalized heatmap array (H, W) in [0, 1]
        """
        # GradCAM++ weighting
        # α_k^c = sum(relu(grad^2 * A)) / (sum(grad^2) + ε)
        grad_squared = np.power(gradients, 2)
        grad_cubed = np.power(gradients, 3)
        
        # Compute alpha weights
        sum_activations = np.sum(activations, axis=(1, 2), keepdims=True)
        alpha = grad_squared / (2 * grad_squared + sum_activations * grad_cubed + 1e-8)
        alpha = np.where(gradients != 0, alpha, 0)
        
        # Weighted combination
        weights = np.sum(alpha * np.maximum(gradients, 0), axis=(1, 2))
        
        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU to keep positive activations
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Upscale to original size if specified
        if original_size is not None:
            cam = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)
        
        return cam
    
    def generate_gradcam_simple(
        self,
        feature_maps: np.ndarray,
        class_weights: np.ndarray,
        original_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Simplified GradCAM using class weights.
        
        For use when gradients aren't available directly.
        
        Args:
            feature_maps: Model feature maps [C, H, W]
            class_weights: Weights for target class [C]
            original_size: Target size (W, H)
            
        Returns:
            Heatmap array (H, W) in [0, 1]
        """
        # Weighted sum of feature maps
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(class_weights):
            if i < feature_maps.shape[0]:
                cam += w * feature_maps[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        if original_size is not None:
            cam = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)
        
        return cam
    
    def generate_heatmap_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray
    ) -> np.ndarray:
        """
        Create heatmap overlay on original image.
        
        Args:
            image: Original image [H, W, 3] (RGB)
            heatmap: Heatmap [H, W] in [0, 1]
            
        Returns:
            Overlaid image [H, W, 3] (RGB)
        """
        # Ensure heatmap matches image size
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, self.heatmap_colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original
        overlay = cv2.addWeighted(
            image.astype(np.float32),
            1 - self.heatmap_alpha,
            colored_heatmap.astype(np.float32),
            self.heatmap_alpha,
            0
        ).astype(np.uint8)
        
        return overlay
    
    def localize_manipulation(
        self,
        heatmap: np.ndarray,
        threshold: Optional[float] = None,
        min_area: int = 100
    ) -> List[Region]:
        """
        Extract manipulation regions from heatmap.
        
        Uses contour detection on thresholded heatmap to find
        discrete regions of interest.
        
        Args:
            heatmap: Heatmap [H, W] in [0, 1]
            threshold: Activation threshold (None = use default)
            min_area: Minimum region area in pixels
            
        Returns:
            List of detected regions
        """
        threshold = threshold or self.localization_threshold
        
        # Threshold heatmap
        binary = (heatmap >= threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Compute confidence as mean activation in region
                region_heatmap = heatmap[y:y+h, x:x+w]
                confidence = float(np.mean(region_heatmap))
                
                regions.append(Region(
                    x=x, y=y, width=w, height=h,
                    confidence=confidence,
                    region_type="manipulation"
                ))
        
        # Sort by confidence descending
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        return regions
    
    def generate_textual_explanation(
        self,
        aggregated: AggregatedResult,
        verdict: Verdict,
        regions: Optional[List[Region]] = None
    ) -> Explanation:
        """
        Generate natural language explanation of analysis results.
        
        Uses template-based generation with dynamic slot filling.
        No external LLM dependencies required.
        
        Args:
            aggregated: Aggregated multi-modal results
            verdict: Final verdict
            regions: Detected manipulation regions
            
        Returns:
            Explanation with summary and findings
        """
        regions = regions or []
        
        # Determine explanation template based on verdict
        score = aggregated.fused_score * 100
        confidence = (1 - aggregated.uncertainty) * 100
        
        if verdict in [Verdict.FAKE, Verdict.LIKELY_FAKE]:
            template = EXPLANATION_TEMPLATES["high_confidence_fake"]
            manipulation_type = self._detect_manipulation_type(aggregated)
        elif verdict in [Verdict.AUTHENTIC, Verdict.LIKELY_AUTHENTIC]:
            template = EXPLANATION_TEMPLATES["high_confidence_real"]
            manipulation_type = None
        else:
            template = EXPLANATION_TEMPLATES["uncertain"]
            manipulation_type = None
        
        # Build key findings
        key_findings = []
        methodology = []
        
        for result in aggregated.modality_results:
            modality_finding = self._generate_modality_finding(result)
            if modality_finding:
                key_findings.append(modality_finding)
            methodology.append(f"{result.modality.value} analysis")
        
        # Generate summary
        summary = template["summary"].format(
            confidence=round(confidence, 1),
            score=round(score, 1),
            verdict=verdict.value.replace("_", " ").title(),
            manipulation_type=manipulation_type.value if manipulation_type else "unknown"
        )
        
        # Convert regions to ManipulationRegion objects
        manipulation_regions = [
            ManipulationRegion(
                region_type=r.region_type,
                location=f"({r.x}, {r.y}) - ({r.x + r.width}, {r.y + r.height})",
                confidence=r.confidence,
                frame_indices=[r.frame_index] if r.frame_index is not None else None
            )
            for r in regions
        ]
        
        # Confidence rationale
        confidence_rationale = self._generate_confidence_rationale(
            aggregated, verdict
        )
        
        return Explanation(
            summary=summary,
            key_findings=key_findings,
            manipulation_regions=manipulation_regions,
            confidence_rationale=confidence_rationale,
            methodology_used=methodology
        )
    
    def _detect_manipulation_type(
        self,
        aggregated: AggregatedResult
    ) -> ManipulationType:
        """Detect primary manipulation type from results."""
        # Analyze modality results to determine manipulation type
        for result in aggregated.modality_results:
            if result.modality == Modality.VIDEO:
                details = result.details
                if details.get("lip_sync_detected"):
                    return ManipulationType.LIP_SYNC
                if details.get("temporal_inconsistency"):
                    return ManipulationType.TEMPORAL_INCONSISTENCY
                return ManipulationType.FACE_SWAP
            
            elif result.modality == Modality.AUDIO:
                if result.score > 0.7:
                    return ManipulationType.AUDIO_CLONE
            
            elif result.modality == Modality.IMAGE:
                return ManipulationType.AI_GENERATED_IMAGE
            
        return ManipulationType.UNKNOWN
    
    def _generate_modality_finding(
        self,
        result: ModalityResult
    ) -> Optional[str]:
        """Generate finding string for a modality result."""
        score_pct = round(result.score * 100, 1)
        confidence_pct = round(result.confidence * 100, 1)
        
        if result.modality == Modality.VIDEO:
            if result.score > 0.5:
                return f"Video: {score_pct}% manipulation probability detected ({confidence_pct}% confidence)"
            return f"Video: No significant manipulation ({confidence_pct}% confidence)"
        
        elif result.modality == Modality.AUDIO:
            if result.score > 0.5:
                return f"Audio: {score_pct}% synthetic voice probability ({confidence_pct}% confidence)"
            return f"Audio: Voice appears authentic ({confidence_pct}% confidence)"
        
        elif result.modality == Modality.IMAGE:
            if result.score > 0.5:
                return f"Image: {score_pct}% deepfake probability ({confidence_pct}% confidence)"
            return f"Image: No deepfake artifacts detected ({confidence_pct}% confidence)"
        
        return None
    
    def _generate_confidence_rationale(
        self,
        aggregated: AggregatedResult,
        verdict: Verdict
    ) -> str:
        """Generate rationale for confidence level."""
        uncertainty = aggregated.uncertainty
        num_modalities = len(aggregated.modality_results)
        
        if uncertainty < 0.2:
            agreement = "strong"
        elif uncertainty < 0.4:
            agreement = "moderate"
        else:
            agreement = "weak"
        
        if verdict == Verdict.UNCERTAIN:
            return (
                f"Confidence is limited due to {agreement} agreement across "
                f"{num_modalities} modalities. Human review is recommended."
            )
        
        weights_desc = ", ".join([
            f"{m}: {w:.0%}"
            for m, w in aggregated.weights_used.items()
        ])
        
        return (
            f"Verdict based on {agreement} agreement across {num_modalities} modalities. "
            f"Weights applied: {weights_desc}. "
            f"Uncertainty score: {uncertainty:.1%}"
        )
    
    def generate_video_explanation(
        self,
        video_result: VideoResult,
        frame_heatmaps: Optional[List[HeatmapResult]] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation for video analysis.
        
        Args:
            video_result: Video analysis results
            frame_heatmaps: GradCAM results for key frames
            
        Returns:
            Detailed video explanation dict
        """
        findings = []
        
        # Spatial analysis findings
        spatial = video_result.spatial
        if spatial.anomaly_indices:
            findings.append({
                "type": "spatial_artifact",
                "description": f"Detected artifacts in {len(spatial.anomaly_indices)} frames",
                "frames": spatial.anomaly_indices,
                "confidence": spatial.score
            })
        
        # Temporal analysis findings
        temporal = video_result.temporal
        if temporal.flickering_detected:
            findings.append({
                "type": "temporal_artifact",
                "description": "Flickering artifacts detected between frames",
                "timestamps": temporal.anomaly_timestamps,
                "confidence": 1 - temporal.consistency_score
            })
        
        # Lip-sync findings
        if video_result.lip_sync:
            lip_sync = video_result.lip_sync
            if lip_sync.manipulation_probability > 0.5:
                tech = lip_sync.detected_technology or "unknown"
                findings.append({
                    "type": "lipsync_artifact",
                    "description": f"Potential {tech} lip-sync manipulation detected",
                    "confidence": lip_sync.manipulation_probability
                })
        
        # Aggregate heatmap regions
        all_regions = []
        if frame_heatmaps:
            for hm in frame_heatmaps:
                all_regions.extend(hm.regions)
        
        return {
            "aggregate_score": video_result.aggregate_score,
            "frames_analyzed": video_result.frames_analyzed,
            "face_detected": video_result.face_detected,
            "findings": findings,
            "key_regions": [r.to_dict() for r in all_regions[:10]]  # Top 10
        }
    
    def generate_audio_explanation(
        self,
        audio_result: AudioResult
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation for audio analysis.
        
        Args:
            audio_result: Audio analysis results
            
        Returns:
            Detailed audio explanation dict
        """
        findings = []
        
        if audio_result.vocoder_artifacts_detected:
            findings.append({
                "type": "vocoder_artifact",
                "description": "Detected vocoder artifacts consistent with synthetic speech",
                "confidence": audio_result.synthetic_probability
            })
        
        if audio_result.voice_consistency_score < 0.7:
            findings.append({
                "type": "voice_inconsistency",
                "description": "Voice characteristics vary unnaturally across segments",
                "confidence": 1 - audio_result.voice_consistency_score
            })
        
        return {
            "synthetic_probability": audio_result.synthetic_probability,
            "voice_consistency": audio_result.voice_consistency_score,
            "vocoder_detected": audio_result.vocoder_artifacts_detected,
            "findings": findings
        }
    
    async def generate_full_explanation(
        self,
        aggregated: AggregatedResult,
        verdict: Verdict,
        video_result: Optional[VideoResult] = None,
        audio_result: Optional[AudioResult] = None,
        frame_heatmaps: Optional[List[HeatmapResult]] = None
    ) -> Explanation:
        """
        Generate comprehensive explanation combining all modalities.
        
        Args:
            aggregated: Multi-modal fusion results
            verdict: Final verdict
            video_result: Optional video analysis
            audio_result: Optional audio analysis
            frame_heatmaps: Optional heatmaps from video
            
        Returns:
            Complete Explanation object
        """
        # Collect all regions from heatmaps
        all_regions = []
        if frame_heatmaps:
            for hm in frame_heatmaps:
                all_regions.extend(hm.regions)
        
        # Generate base explanation
        explanation = self.generate_textual_explanation(
            aggregated, verdict, all_regions
        )
        
        # Enhance with modality-specific findings
        enhanced_findings = list(explanation.key_findings)
        
        if video_result and video_result.aggregate_score > 0.5:
            video_exp = self.generate_video_explanation(video_result, frame_heatmaps)
            for finding in video_exp.get("findings", []):
                enhanced_findings.append(finding["description"])
        
        if audio_result and audio_result.synthetic_probability > 0.5:
            audio_exp = self.generate_audio_explanation(audio_result)
            for finding in audio_exp.get("findings", []):
                enhanced_findings.append(finding["description"])
        
        # Update explanation with enhanced findings
        return Explanation(
            summary=explanation.summary,
            key_findings=enhanced_findings[:10],  # Limit to top 10
            manipulation_regions=explanation.manipulation_regions,
            confidence_rationale=explanation.confidence_rationale,
            methodology_used=explanation.methodology_used
        )
    
    def heatmap_to_image(
        self,
        heatmap: np.ndarray,
        size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Convert heatmap array to PIL Image.
        
        Args:
            heatmap: Heatmap [H, W] in [0, 1]
            size: Target size (W, H)
            
        Returns:
            PIL Image object
        """
        # Convert to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(heatmap_uint8, self.heatmap_colormap)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(colored)
        
        if size:
            img = img.resize(size, Image.LANCZOS)
        
        return img
    
    def overlay_to_image(
        self,
        overlay: np.ndarray
    ) -> Image.Image:
        """
        Convert overlay array to PIL Image.
        
        Args:
            overlay: RGB image array [H, W, 3]
            
        Returns:
            PIL Image object
        """
        return Image.fromarray(overlay.astype(np.uint8))


# Singleton instance
_explainer: Optional[ExplainabilityEngine] = None


def get_explainability_engine() -> ExplainabilityEngine:
    """Get singleton explainability engine instance."""
    global _explainer
    if _explainer is None:
        _explainer = ExplainabilityEngine()
    return _explainer
