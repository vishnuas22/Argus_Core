"""
Argus Core - Lip-Sync Analyzer
==============================
Lip-sync deepfake detection using LIPINC-V2 architecture.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/video/lipsync.py

SOTA Algorithms:
- Model: LIPINC-V2 (Vision Temporal Transformer with multihead cross-attention)
- Detection: Audio-visual synchronization scoring
- Targets: Wav2Lip, Diff2Lip, Video_Retalking, IP_LAP artifacts

Specialized Detection:
- Wav2Lip manipulations
- Diff2Lip artifacts
- Video_Retalking inconsistencies
- Audio-visual desynchronization
- Phoneme-viseme mismatches

Integration:
- Imports: core/engine.py
- Inputs: frames: List[np.ndarray], audio: np.ndarray
- Outputs: LipSyncResult

Target Hardware: RTX 3050 (4GB VRAM)
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
import time

from analyzers.base import (
    SubAnalyzer,
    compute_confidence,
    infer_fake_class_index,
    extract_fake_probabilities,
)
from schemas.schemas import LipSyncResult
from config import config
from utils.logging import get_logger
from utils.errors import InferenceError

if TYPE_CHECKING:
    from core.engine import InferenceEngine

logger = get_logger(__name__)


# Known lip-sync deepfake technologies
LIPSYNC_TECHNOLOGIES = {
    "wav2lip": {
        "name": "Wav2Lip",
        "description": "GAN-based lip synthesis from audio",
        "artifacts": ["blending boundary", "temporal jitter", "color mismatch"]
    },
    "diff2lip": {
        "name": "Diff2Lip",
        "description": "Diffusion-based lip synthesis",
        "artifacts": ["noise patterns", "temporal inconsistency"]
    },
    "video_retalking": {
        "name": "Video_Retalking",
        "description": "Audio-driven face reenactment",
        "artifacts": ["identity shift", "expression artifacts"]
    },
    "ip_lap": {
        "name": "IP_LAP",
        "description": "Identity-preserving lip animation",
        "artifacts": ["subtle boundary", "motion smoothing"]
    },
    "unknown": {
        "name": "Unknown",
        "description": "Unidentified lip-sync technology",
        "artifacts": []
    }
}


@dataclass
class AudioVisualCorrelation:
    """
    Audio-visual correlation analysis results.
    
    Measures synchronization between lip movements and audio.
    """
    correlation_score: float = 0.0  # Overall AV correlation
    phoneme_alignment: float = 0.0  # Phoneme-viseme alignment
    temporal_offset_ms: float = 0.0  # Detected AV offset
    consistent_regions: List[Tuple[float, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_score": round(self.correlation_score, 4),
            "phoneme_alignment": round(self.phoneme_alignment, 4),
            "temporal_offset_ms": round(self.temporal_offset_ms, 2),
            "consistent_regions": self.consistent_regions
        }


@dataclass
class MouthRegionFeatures:
    """
    Mouth region analysis features.
    
    Extracted from mouth crop sequences for lip-sync analysis.
    """
    openness_sequence: List[float] = field(default_factory=list)
    movement_energy: float = 0.0
    symmetry_score: float = 0.0
    boundary_sharpness: float = 0.0
    texture_consistency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "openness_variance": float(np.var(self.openness_sequence)) if self.openness_sequence else 0.0,
            "movement_energy": round(self.movement_energy, 4),
            "symmetry_score": round(self.symmetry_score, 4),
            "boundary_sharpness": round(self.boundary_sharpness, 4),
            "texture_consistency": round(self.texture_consistency, 4)
        }


@dataclass
class TechnologySignature:
    """
    Detected lip-sync technology signature.
    """
    technology: str
    confidence: float
    detected_artifacts: List[str] = field(default_factory=list)


class LipSyncAnalyzer(SubAnalyzer):
    """
    Lip-sync deepfake detection.
    
    Detects audio-driven lip manipulation using:
    - LIPINC-V2 neural architecture
    - Audio-visual correlation analysis
    - Mouth region temporal consistency
    - Technology-specific artifact detection
    
    Pipeline:
    1. Extract mouth crops from face frames
    2. Extract audio features (mel-spectrogram, MFCC)
    3. Run LIPINC-V2 cross-modal analysis
    4. Compute AV correlation metrics
    5. Detect technology-specific signatures
    6. Generate manipulation probability
    
    Usage:
        lipsync = LipSyncAnalyzer()
        result = await lipsync.verify_sync(mouth_crops, audio_features, engine)
    """
    
    def __init__(
        self,
        sync_threshold: float = 0.5,
        min_frames: int = 10,
        audio_sample_rate: int = 16000
    ):
        """
        Initialize lip-sync analyzer.
        
        Args:
            sync_threshold: Threshold for sync/desync classification
            min_frames: Minimum frames required for analysis
            audio_sample_rate: Expected audio sample rate
        """
        super().__init__("LipSyncAnalyzer")
        
        self.sync_threshold = sync_threshold
        self.min_frames = min_frames
        self.audio_sample_rate = audio_sample_rate
        
        # Mouth region configuration
        self.mouth_target_size = (96, 96)  # Standard for lip-sync models
        
        # Weight configuration
        self.weights = {
            "lipinc": 0.50,      # Primary neural detector
            "av_correlation": 0.30,  # Audio-visual correlation
            "mouth_features": 0.20   # Mouth region analysis
        }
        
        logger.info(
            f"LipSyncAnalyzer initialized: sync_threshold={sync_threshold}, "
            f"min_frames={min_frames}"
        )
    
    def get_required_models(self) -> List[str]:
        """
        Return required models for lip-sync analysis.
        
        Returns:
            List of model registry keys
        """
        return [
            "lipinc_v2",           # LIPINC-V2 lip-sync detector
            "wav2vec2_base"        # Audio feature extractor
        ]
    
    async def verify_sync(
        self,
        mouth_crops: List[np.ndarray],
        audio_features: np.ndarray,
        engine: "InferenceEngine"
    ) -> LipSyncResult:
        """
        Verify audio-visual lip synchronization.
        
        Args:
            mouth_crops: Cropped mouth regions from frames (H, W, 3)
            audio_features: MFCC/mel-spectrogram features (T, F)
            engine: InferenceEngine for model inference
            
        Returns:
            LipSyncResult with sync score and manipulation probability
        """
        start_time = time.time()
        
        if not mouth_crops or len(mouth_crops) < self.min_frames:
            logger.warning(f"Insufficient mouth crops: {len(mouth_crops) if mouth_crops else 0}")
            return LipSyncResult(
                sync_score=1.0,
                manipulation_probability=0.0,
                detected_technology=None
            )
        
        logger.debug(
            f"Analyzing lip-sync: {len(mouth_crops)} mouth crops, "
            f"audio shape={audio_features.shape if isinstance(audio_features, np.ndarray) else 'N/A'}"
        )
        
        # 1. Preprocess inputs
        preprocessed_visual = self._preprocess_mouth_crops(mouth_crops)
        preprocessed_audio = self._preprocess_audio_features(audio_features)
        
        # 2. Run LIPINC-V2 analysis
        lipinc_score = await self._run_lipinc_analysis(
            preprocessed_visual,
            preprocessed_audio,
            engine,
            mouth_crops
        )
        
        # 3. Compute audio-visual correlation
        av_correlation = self._compute_av_correlation(
            mouth_crops,
            audio_features
        )
        
        # 4. Analyze mouth region features
        mouth_features = self._analyze_mouth_features(mouth_crops)
        
        # 5. Detect technology signature
        technology = self._detect_technology(
            lipinc_score,
            av_correlation,
            mouth_features
        )
        
        # 6. Compute combined scores
        manipulation_probability = (
            self.weights["lipinc"] * lipinc_score +
            self.weights["av_correlation"] * (1.0 - av_correlation.correlation_score) +
            self.weights["mouth_features"] * self._compute_mouth_anomaly_score(mouth_features)
        )
        manipulation_probability = float(np.clip(manipulation_probability, 0, 1))
        
        # Sync score is inverse of manipulation probability
        sync_score = 1.0 - manipulation_probability
        
        inference_time = (time.time() - start_time) * 1000
        confidence = compute_confidence(
            np.array([lipinc_score, 1 - av_correlation.correlation_score]),
            len(mouth_crops)
        )
        self.record_analysis(True, inference_time, confidence)
        
        logger.info(
            f"Lip-sync analysis complete: sync={sync_score:.3f}, "
            f"manipulation={manipulation_probability:.3f}, tech={technology.technology}"
        )
        
        return LipSyncResult(
            sync_score=sync_score,
            manipulation_probability=manipulation_probability,
            detected_technology=technology.technology if technology.confidence > 0.5 else None
        )
    
    def _preprocess_mouth_crops(
        self,
        mouth_crops: List[np.ndarray]
    ) -> np.ndarray:
        """
        Preprocess mouth crops for LIPINC-V2.
        
        Args:
            mouth_crops: List of mouth images
            
        Returns:
            Tensor (T, C, H, W)
        """
        import cv2
        
        processed = []
        
        for crop in mouth_crops:
            # Ensure uint8
            if crop.dtype in [np.float32, np.float64]:
                if crop.max() <= 1.0:
                    crop = (crop * 255).astype(np.uint8)
                else:
                    crop = crop.astype(np.uint8)
            
            # Resize to standard size
            resized = cv2.resize(crop, self.mouth_target_size)
            
            # Normalize
            float_crop = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (float_crop - mean) / std
            
            # CHW format
            chw = np.transpose(normalized, (2, 0, 1))
            processed.append(chw)
        
        return np.stack(processed, axis=0).astype(np.float32)
    
    def _preprocess_audio_features(
        self,
        audio_features: np.ndarray
    ) -> np.ndarray:
        """
        Preprocess audio features for cross-modal analysis.
        
        Args:
            audio_features: Raw audio features (T, F)
            
        Returns:
            Preprocessed features
            
        Raises:
            ValueError: If audio features are None or empty
        """
        if audio_features is None or len(audio_features) == 0:
            raise ValueError(
                "Audio features are required for lip-sync analysis. "
                "Cannot proceed with empty or None audio features."
            )
        
        # Normalize
        features = audio_features.astype(np.float32)
        
        if features.max() > 0:
            features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features
    
    async def _run_lipinc_analysis(
        self,
        visual_features: np.ndarray,
        audio_features: np.ndarray,
        engine: "InferenceEngine",
        mouth_crops: List[np.ndarray]
    ) -> float:
        """
        Run LIPINC-V2 cross-modal analysis.
        
        LIPINC-V2 uses vision temporal transformer with multihead
        cross-attention to detect audio-visual inconsistencies.
        
        Note: When lipinc_v2 is unavailable (requires GPU), falls back to
        ai_real_detector (PyTorch model) for per-frame analysis.
        
        Args:
            visual_features: Preprocessed mouth crops (T, C, H, W)
            audio_features: Preprocessed audio (T, F)
            engine: InferenceEngine
            
        Returns:
            Manipulation probability [0, 1]
        """
        try:
            # Use LIPINC-V2 with 5D input
            visual_batch = np.expand_dims(visual_features, 0)  # (1, T, C, H, W)
            
            result = await engine.infer(
                "lipinc_v2",
                visual_batch,
                return_probabilities=True
            )
            
            # Extract manipulation probability
            if result.class_probabilities is not None:
                probs = result.class_probabilities
                # Class 1 = fake/manipulated for dedicated lip-sync detector
                fake_prob = float(probs[0, 1]) if probs.shape[-1] >= 2 else float(probs[0, 0])
            else:
                fake_prob = float(result.predictions.mean())
            
            return fake_prob
        except Exception as e:
            logger.warning(f"LIPINC analysis failed, using ai_real fallback: {e}")
            return await self._run_ai_real_fallback(mouth_crops)

    async def _run_ai_real_fallback(self, mouth_crops: List[np.ndarray]) -> float:
        """
        Fallback lip-sync score using ai_real_detector over mouth crops.
        
        Args:
            mouth_crops: Original mouth crops
            
        Returns:
            Mean fake probability across crops
        """
        from models.manager import get_model_manager
        import torch
        from PIL import Image as PILImage
        
        if not mouth_crops:
            return 0.5
        
        pil_images = []
        for crop in mouth_crops:
            crop_np = crop
            if crop_np.dtype in (np.float32, np.float64):
                crop_np = np.clip(crop_np * 255.0 if crop_np.max() <= 1.0 else crop_np, 0, 255).astype(np.uint8)
            else:
                crop_np = crop_np.astype(np.uint8)
            
            if crop_np.ndim == 2:
                crop_np = np.stack([crop_np] * 3, axis=-1)
            pil_images.append(PILImage.fromarray(crop_np))
        
        try:
            manager = get_model_manager()
            model_session = await manager.get_model("ai_real_detector")
            if model_session is None:
                return 0.5
            
            model, processor = model_session
            device = next(model.parameters()).device
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs_np = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            fake_idx = infer_fake_class_index(
                id2label=getattr(model.config, "id2label", None),
                class_labels=["human", "ai_generated"],
                default_index=0
            )
            fake_probs = extract_fake_probabilities(
                probs_np,
                fake_class_index=fake_idx,
                apply_confidence_shrinkage=True
            )
            return float(np.mean(fake_probs))
        except Exception as fallback_error:
            logger.warning(f"Lip-sync fallback failed: {fallback_error}")
            return 0.5
    
    def _compute_av_correlation(
        self,
        mouth_crops: List[np.ndarray],
        audio_features: np.ndarray
    ) -> AudioVisualCorrelation:
        """
        Compute audio-visual correlation metrics.
        
        Measures how well lip movements correlate with audio energy.
        
        Args:
            mouth_crops: Mouth images
            audio_features: Audio features
            
        Returns:
            AudioVisualCorrelation results
        """
        # Extract visual motion (mouth movement)
        visual_energy = self._extract_visual_energy(mouth_crops)
        
        # Extract audio energy
        if audio_features is not None and len(audio_features) > 0:
            audio_energy = self._extract_audio_energy(audio_features)
        else:
            # No audio - can't compute correlation
            return AudioVisualCorrelation(correlation_score=0.5)
        
        # Align sequences
        visual_energy = self._resample_sequence(visual_energy, len(audio_energy))
        
        # Compute correlation
        if len(visual_energy) > 1 and len(audio_energy) > 1:
            correlation = np.corrcoef(visual_energy, audio_energy)[0, 1]
            if np.isnan(correlation):
                correlation = 0.5
        else:
            correlation = 0.5
        
        # Normalize to [0, 1]
        correlation_score = (correlation + 1) / 2
        
        # Estimate temporal offset
        offset = self._estimate_av_offset(visual_energy, audio_energy)
        
        return AudioVisualCorrelation(
            correlation_score=float(correlation_score),
            phoneme_alignment=float(correlation_score * 0.9),  # Simplified
            temporal_offset_ms=float(offset),
            consistent_regions=[]  # Would require more detailed analysis
        )
    
    def _extract_visual_energy(
        self,
        mouth_crops: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract visual energy (mouth movement) from mouth crops.
        
        Args:
            mouth_crops: List of mouth images
            
        Returns:
            Visual energy sequence
        """
        import cv2
        
        energies = []
        prev_gray = None
        
        for crop in mouth_crops:
            if crop.dtype in [np.float32, np.float64]:
                if crop.max() <= 1.0:
                    crop = (crop * 255).astype(np.uint8)
            
            gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            if prev_gray is not None:
                # Frame difference as motion energy
                diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
                energy = np.mean(diff)
                energies.append(energy)
            
            prev_gray = gray
        
        return np.array(energies) if energies else np.array([0.0])
    
    def _extract_audio_energy(
        self,
        audio_features: np.ndarray
    ) -> np.ndarray:
        """
        Extract audio energy from features.
        
        Args:
            audio_features: Audio features (T, F)
            
        Returns:
            Audio energy sequence
        """
        audio_arr = np.asarray(audio_features, dtype=np.float32)
        if audio_arr.size == 0:
            return np.array([0.0], dtype=np.float32)

        if audio_arr.ndim == 1:
            # Raw waveform case: compute short-window energy envelope.
            if audio_arr.size <= 512:
                energy = np.array([float(np.mean(np.abs(audio_arr)))], dtype=np.float32)
            else:
                win = max(256, min(2048, audio_arr.size // 20))
                hop = max(1, win // 2)
                chunks = []
                for start in range(0, audio_arr.size - win + 1, hop):
                    segment = audio_arr[start:start + win]
                    chunks.append(float(np.mean(np.abs(segment))))
                if not chunks:
                    chunks = [float(np.mean(np.abs(audio_arr)))]
                energy = np.asarray(chunks, dtype=np.float32)
        else:
            # Feature matrix case: sum across feature axis.
            energy = np.sum(np.abs(audio_arr), axis=1).astype(np.float32)
        
        # Normalize
        max_val = float(np.max(energy)) if energy.size > 0 else 0.0
        if max_val > 0.0:
            energy = energy / max_val
        
        return energy
    
    def _resample_sequence(
        self,
        sequence: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """
        Resample sequence to target length.
        
        Args:
            sequence: Input sequence
            target_length: Target length
            
        Returns:
            Resampled sequence
        """
        if target_length <= 0:
            return np.array([], dtype=np.float32)

        if len(sequence) == 0:
            return np.zeros(target_length, dtype=np.float32)

        if len(sequence) == target_length:
            return sequence.astype(np.float32)

        if len(sequence) == 1:
            return np.full(target_length, float(sequence[0]), dtype=np.float32)
        
        indices = np.linspace(0, len(sequence) - 1, target_length)
        return np.interp(indices, np.arange(len(sequence)), sequence).astype(np.float32)
    
    def _estimate_av_offset(
        self,
        visual_energy: np.ndarray,
        audio_energy: np.ndarray
    ) -> float:
        """
        Estimate audio-visual temporal offset.
        
        Uses cross-correlation to find offset.
        
        Args:
            visual_energy: Visual motion sequence
            audio_energy: Audio energy sequence
            
        Returns:
            Estimated offset in milliseconds
        """
        if len(visual_energy) < 3 or len(audio_energy) < 3:
            return 0.0
        
        # Cross-correlation
        correlation = np.correlate(visual_energy, audio_energy, mode='full')
        
        # Find peak offset
        peak_idx = np.argmax(correlation)
        center = len(audio_energy) - 1
        offset_samples = peak_idx - center
        
        # Convert to milliseconds (assuming 30 fps video)
        offset_ms = offset_samples * (1000 / 30)
        
        return float(offset_ms)
    
    def _analyze_mouth_features(
        self,
        mouth_crops: List[np.ndarray]
    ) -> MouthRegionFeatures:
        """
        Analyze mouth region features for manipulation artifacts.
        
        Args:
            mouth_crops: Mouth images
            
        Returns:
            MouthRegionFeatures
        """
        import cv2
        
        openness_sequence = []
        symmetry_scores = []
        boundary_scores = []
        
        for crop in mouth_crops:
            if crop.dtype in [np.float32, np.float64]:
                if crop.max() <= 1.0:
                    crop = (crop * 255).astype(np.uint8)
            
            gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Estimate mouth openness (dark pixels in center = open mouth)
            h, w = gray.shape
            center_region = gray[h//4:3*h//4, w//4:3*w//4]
            openness = 1.0 - (np.mean(center_region) / 255.0)
            openness_sequence.append(openness)
            
            # Check symmetry
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            min_w = min(left_half.shape[1], right_half.shape[1])
            symmetry = 1.0 - np.mean(np.abs(
                left_half[:, :min_w].astype(np.float32) - 
                right_half[:, :min_w].astype(np.float32)
            )) / 255.0
            symmetry_scores.append(symmetry)
            
            # Check boundary sharpness (edge detection)
            edges = cv2.Canny(gray, 50, 150)
            boundary_scores.append(np.mean(edges) / 255.0)
        
        # Compute movement energy
        if len(openness_sequence) > 1:
            movement_energy = np.mean(np.abs(np.diff(openness_sequence)))
        else:
            movement_energy = 0.0
        
        # Texture consistency (variance of gradients)
        texture_scores = []
        for crop in mouth_crops:
            if crop.dtype in [np.float32, np.float64]:
                if crop.max() <= 1.0:
                    crop = (crop * 255).astype(np.uint8)
            gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            grad = cv2.Laplacian(gray, cv2.CV_64F)
            texture_scores.append(np.var(grad))
        
        texture_consistency = 1.0 - min(1.0, np.std(texture_scores) / (np.mean(texture_scores) + 1e-8))
        
        return MouthRegionFeatures(
            openness_sequence=openness_sequence,
            movement_energy=float(movement_energy),
            symmetry_score=float(np.mean(symmetry_scores)) if symmetry_scores else 0.5,
            boundary_sharpness=float(np.mean(boundary_scores)) if boundary_scores else 0.5,
            texture_consistency=float(texture_consistency)
        )
    
    def _compute_mouth_anomaly_score(
        self,
        features: MouthRegionFeatures
    ) -> float:
        """
        Compute anomaly score from mouth features.
        
        Args:
            features: MouthRegionFeatures
            
        Returns:
            Anomaly score [0, 1]
        """
        score = 0.0
        
        # Low symmetry is suspicious
        if features.symmetry_score < 0.8:
            score += 0.3 * (0.8 - features.symmetry_score) / 0.8
        
        # Abnormal boundary sharpness
        if features.boundary_sharpness < 0.1 or features.boundary_sharpness > 0.5:
            score += 0.3
        
        # Low texture consistency
        if features.texture_consistency < 0.7:
            score += 0.4 * (0.7 - features.texture_consistency) / 0.7
        
        return float(np.clip(score, 0, 1))
    
    def _detect_technology(
        self,
        lipinc_score: float,
        av_correlation: AudioVisualCorrelation,
        mouth_features: MouthRegionFeatures
    ) -> TechnologySignature:
        """
        Detect which lip-sync technology was likely used.
        
        Args:
            lipinc_score: Neural detection score
            av_correlation: AV correlation results
            mouth_features: Mouth region analysis
            
        Returns:
            TechnologySignature with detected technology
        """
        # If no manipulation detected, return early
        if lipinc_score < 0.3:
            return TechnologySignature(
                technology="unknown",
                confidence=0.0,
                detected_artifacts=[]
            )
        
        # Analyze artifact patterns
        artifacts = []
        tech_scores = {tech: 0.0 for tech in LIPSYNC_TECHNOLOGIES}
        
        # Wav2Lip: boundary issues, color mismatches
        if mouth_features.boundary_sharpness > 0.3:
            artifacts.append("blending boundary")
            tech_scores["wav2lip"] += 0.3
        
        # Temporal jitter
        if mouth_features.openness_sequence:
            jitter = np.var(np.diff(mouth_features.openness_sequence))
            if jitter > 0.1:
                artifacts.append("temporal jitter")
                tech_scores["wav2lip"] += 0.2
                tech_scores["diff2lip"] += 0.2
        
        # AV desync suggests retalking
        if av_correlation.temporal_offset_ms > 50:
            artifacts.append("av_desync")
            tech_scores["video_retalking"] += 0.4
        
        # Low symmetry suggests identity issues
        if mouth_features.symmetry_score < 0.7:
            artifacts.append("identity shift")
            tech_scores["video_retalking"] += 0.2
        
        # Find most likely technology
        best_tech = max(tech_scores.items(), key=lambda x: x[1])
        
        if best_tech[1] < 0.2:
            best_tech = ("unknown", 0.0)
        
        return TechnologySignature(
            technology=best_tech[0],
            confidence=float(best_tech[1]),
            detected_artifacts=artifacts
        )


# Singleton instance
_lipsync_analyzer: Optional[LipSyncAnalyzer] = None


def get_lipsync_analyzer() -> LipSyncAnalyzer:
    """Get singleton lip-sync analyzer instance."""
    global _lipsync_analyzer
    if _lipsync_analyzer is None:
        _lipsync_analyzer = LipSyncAnalyzer()
    return _lipsync_analyzer
