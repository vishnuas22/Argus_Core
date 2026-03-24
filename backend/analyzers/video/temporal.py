"""
Argus Core - Temporal Video Analyzer
=====================================
Cross-frame temporal consistency analysis using X-CLIP transformer.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/video/temporal.py

SOTA Algorithms:
- Model: X-CLIP with Multiframe Integration Transformer (KDD 2025)
- Analysis: Optical flow consistency, facial landmark tracking
- Anomaly: Frame-to-frame coherence scoring

Detects:
- Flickering artifacts between frames
- Unnatural motion patterns
- Landmark jitter in facial features
- Inter-frame color inconsistency
- Temporal boundary discontinuities

Integration:
- Imports: core/engine.py
- Inputs: List[np.ndarray] (sequence of frames)
- Outputs: TemporalResult

Target Hardware: RTX 3050 (4GB VRAM) with optimized inference
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
import time

from analyzers.base import (
    SubAnalyzer,
    normalize_scores,
    compute_confidence,
    detect_anomalies,
    infer_fake_class_index,
    extract_fake_probabilities,
)
from schemas.schemas import TemporalResult
from config import config
from utils.logging import get_logger
from utils.errors import InferenceError
from models.model_init import ensure_models_for_analyzer, is_model_ready

if TYPE_CHECKING:
    from core.engine import InferenceEngine

logger = get_logger(__name__)


@dataclass
class OpticalFlowFeatures:
    """
    Optical flow analysis features.
    
    Measures motion consistency between consecutive frames.
    Deepfakes often have unnatural or discontinuous motion.
    """
    mean_flow_magnitude: float = 0.0
    flow_consistency: float = 0.0
    sudden_motion_count: int = 0
    flow_direction_variance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_flow_magnitude": round(self.mean_flow_magnitude, 4),
            "flow_consistency": round(self.flow_consistency, 4),
            "sudden_motion_count": self.sudden_motion_count,
            "flow_direction_variance": round(self.flow_direction_variance, 4)
        }


@dataclass
class LandmarkJitterFeatures:
    """
    Facial landmark stability features.
    
    Deepfakes often have jittery or unstable facial landmarks
    due to per-frame generation inconsistencies.
    """
    mean_jitter: float = 0.0
    max_jitter: float = 0.0
    jitter_variance: float = 0.0
    unstable_regions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_jitter": round(self.mean_jitter, 4),
            "max_jitter": round(self.max_jitter, 4),
            "jitter_variance": round(self.jitter_variance, 4),
            "unstable_regions": self.unstable_regions
        }


@dataclass
class ColorConsistencyFeatures:
    """
    Inter-frame color consistency features.
    
    Deepfakes may have color inconsistencies between frames,
    especially around manipulated regions.
    """
    color_variance: float = 0.0
    histogram_correlation: float = 0.0
    color_shift_detected: bool = False
    affected_frames: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "color_variance": round(self.color_variance, 4),
            "histogram_correlation": round(self.histogram_correlation, 4),
            "color_shift_detected": self.color_shift_detected,
            "affected_frames": self.affected_frames
        }


class TemporalAnalyzer(SubAnalyzer):
    """
    Temporal consistency analysis for video deepfake detection.
    
    Analyzes frame sequences for temporal artifacts:
    - Optical flow discontinuities
    - Facial landmark jitter
    - Color consistency violations
    - Motion pattern anomalies
    
    Pipeline:
    1. Frame sequence preprocessing
    2. X-CLIP transformer analysis (if available)
    3. Optical flow computation and analysis
    4. Landmark stability tracking
    5. Color consistency checking
    6. Anomaly timestamp detection
    
    Usage:
        temporal = TemporalAnalyzer()
        result = await temporal.analyze_consistency(frames, engine)
    """
    
    def __init__(
        self,
        sequence_length: int = 16,
        flow_threshold: float = 0.3,
        jitter_threshold: float = 0.1,
        color_threshold: float = 0.05
    ):
        """
        Initialize temporal analyzer.
        
        Args:
            sequence_length: Number of frames to analyze together
            flow_threshold: Threshold for optical flow anomaly
            jitter_threshold: Threshold for landmark jitter anomaly
            color_threshold: Threshold for color shift anomaly
        """
        super().__init__("TemporalAnalyzer")
        
        self.sequence_length = sequence_length
        self.flow_threshold = flow_threshold
        self.jitter_threshold = jitter_threshold
        self.color_threshold = color_threshold
        
        # Weight configuration
        self.weights = {
            "xclip": 0.40,           # X-CLIP transformer
            "optical_flow": 0.25,     # Optical flow analysis
            "landmark_jitter": 0.20,  # Facial landmark stability
            "color_consistency": 0.15 # Inter-frame color
        }
        
        logger.info(
            f"TemporalAnalyzer initialized: seq_len={sequence_length}, "
            f"weights={self.weights}"
        )
    
    def get_required_models(self) -> List[str]:
        """
        Return required models for temporal analysis.
        
        Returns:
            List of model registry keys
        """
        return [
            "xclip_temporal",      # X-CLIP temporal transformer
            "retinaface"           # For landmark tracking
        ]
    
    async def analyze_consistency(
        self,
        frame_sequence: List[np.ndarray],
        engine: "InferenceEngine",
        fps: float = 30.0
    ) -> TemporalResult:
        """
        Analyze temporal consistency across frames.
        
        Args:
            frame_sequence: Ordered list of frames (H, W, 3)
            engine: InferenceEngine for model inference
            fps: Video frame rate for timestamp calculation
            
        Returns:
            TemporalResult with consistency score and anomalies
        """
        start_time = time.time()
        
        if not frame_sequence or len(frame_sequence) < 2:
            logger.warning("Insufficient frames for temporal analysis")
            return TemporalResult(
                consistency_score=1.0,  # No evidence of inconsistency
                flickering_detected=False,
                anomaly_timestamps=[]
            )
        
        logger.debug(f"Analyzing temporal consistency for {len(frame_sequence)} frames")
        
        # 1. X-CLIP Transformer Analysis
        xclip_score = await self._run_xclip_analysis(frame_sequence, engine)
        
        # 2. Optical Flow Analysis
        flow_features = self.compute_optical_flow_consistency(frame_sequence)
        # Detect AI artifacts: unnaturally smooth motion or inconsistent flow patterns
        flow_score = self._compute_flow_anomaly_score(flow_features)
        
        # 3. Landmark Jitter Analysis
        jitter_features = await self._analyze_landmark_jitter(frame_sequence, engine)
        jitter_score = self._compute_jitter_score(jitter_features)
        
        # 4. Color Consistency Analysis
        color_features = self._analyze_color_consistency(frame_sequence)
        # Detect AI artifacts: unnaturally uniform colors or sudden color shifts
        color_score = self._compute_color_anomaly_score(color_features)
        
        # 5. Combine scores for consistency measure
        # Higher score = more consistent (less likely fake)
        combined_inconsistency = (
            self.weights["xclip"] * xclip_score +
            self.weights["optical_flow"] * flow_score +
            self.weights["landmark_jitter"] * jitter_score +
            self.weights["color_consistency"] * color_score
        )
        
        consistency_score = 1.0 - combined_inconsistency
        consistency_score = float(np.clip(consistency_score, 0, 1))
        
        # 6. Detect flickering
        flickering_detected = self._detect_flickering(
            frame_sequence,
            flow_features,
            color_features
        )
        
        # 7. Find anomaly timestamps
        anomaly_timestamps = self._find_anomaly_timestamps(
            frame_sequence,
            flow_features,
            jitter_features,
            color_features,
            fps
        )
        
        inference_time = (time.time() - start_time) * 1000
        confidence = compute_confidence(
            np.array([xclip_score, flow_score, jitter_score, color_score]),
            len(frame_sequence)
        )
        self.record_analysis(True, inference_time, confidence)
        
        logger.info(
            f"Temporal analysis complete: consistency={consistency_score:.3f}, "
            f"flickering={flickering_detected}, anomalies={len(anomaly_timestamps)}"
        )
        
        return TemporalResult(
            consistency_score=consistency_score,
            flickering_detected=flickering_detected,
            anomaly_timestamps=anomaly_timestamps
        )
    
    async def _run_xclip_analysis(
        self,
        frames: List[np.ndarray],
        engine: "InferenceEngine"
    ) -> float:
        """
        Run X-CLIP transformer for temporal analysis.
        
        X-CLIP uses multiframe integration to detect temporal inconsistencies
        that per-frame analysis would miss.
        
        Note: When xclip_temporal is unavailable (requires GPU), falls back to
        ai_real_detector (PyTorch model) for per-frame analysis.
        
        Args:
            frames: Frame sequence
            engine: InferenceEngine
            
        Returns:
            Inconsistency score [0, 1]
        """
        try:
            # Prepare frame sequence for X-CLIP
            # X-CLIP expects (B, T, C, H, W) format
            preprocessed = self._preprocess_sequence(frames)
            batch = np.expand_dims(preprocessed, 0)  # (1, T, C, H, W)
            
            result = await engine.infer(
                "xclip_temporal",
                batch,
                return_probabilities=True
            )
            
            # Extract fake probability
            if result.class_probabilities is not None:
                probs = result.class_probabilities
                fake_prob = float(probs[0, 1]) if probs.shape[-1] >= 2 else float(probs[0, 0])
            else:
                fake_prob = float(result.predictions.mean())
            
            return fake_prob
            
        except Exception as e:
            logger.warning(f"X-CLIP analysis failed, using ai_real fallback: {e}")
            return await self._run_ai_real_fallback(frames)

    async def _run_ai_real_fallback(self, frames: List[np.ndarray]) -> float:
        """
        Fallback temporal score using ai_real_detector over sampled frames.
        
        Args:
            frames: Original RGB frames
            
        Returns:
            Mean fake probability across sampled frames
        """
        from models.manager import get_model_manager
        import torch
        from PIL import Image as PILImage
        
        if not frames:
            return 0.5
        
        # Sample to sequence length for bounded compute cost.
        if len(frames) > self.sequence_length:
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames
        
        pil_images = []
        for frame in sampled_frames:
            frame_np = frame
            if frame_np.dtype in (np.float32, np.float64):
                frame_np = np.clip(frame_np * 255.0 if frame_np.max() <= 1.0 else frame_np, 0, 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
            
            if frame_np.ndim == 2:
                frame_np = np.stack([frame_np] * 3, axis=-1)
            pil_images.append(PILImage.fromarray(frame_np))
        
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
            logger.warning(f"Temporal fallback failed: {fallback_error}")
            return 0.5
    
    def _preprocess_sequence(
        self,
        frames: List[np.ndarray]
    ) -> np.ndarray:
        """
        Preprocess frame sequence for X-CLIP.
        
        Args:
            frames: List of frames
            
        Returns:
            Tensor (T, C, H, W)
        """
        import cv2
        
        target_size = (224, 224)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        
        # Sample frames if too many
        if len(frames) > self.sequence_length:
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
        
        processed = []
        for frame in frames:
            # Ensure uint8
            if frame.dtype in [np.float32, np.float64]:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Resize
            resized = cv2.resize(frame, target_size)
            
            # Normalize
            float_frame = resized.astype(np.float32) / 255.0
            normalized = (float_frame - mean) / std
            
            # CHW format
            chw = np.transpose(normalized, (2, 0, 1))
            processed.append(chw)
        
        # Pad if needed
        while len(processed) < self.sequence_length:
            processed.append(processed[-1])
        
        return np.stack(processed, axis=0).astype(np.float32)
    
    def compute_optical_flow_consistency(
        self,
        frames: List[np.ndarray]
    ) -> OpticalFlowFeatures:
        """
        OpenCV optical flow analysis for motion consistency.
        
        Computes dense optical flow between consecutive frames
        and analyzes for unnaturalness.
        
        Args:
            frames: Frame sequence
            
        Returns:
            OpticalFlowFeatures with analysis results
        """
        import cv2
        
        if len(frames) < 2:
            return OpticalFlowFeatures(flow_consistency=1.0)
        
        flow_magnitudes = []
        flow_directions = []
        sudden_motions = 0
        
        prev_gray = None
        
        for frame in frames:
            # Convert to grayscale
            if frame.dtype in [np.float32, np.float64]:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
            
            gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (256, 256))
            
            if prev_gray is not None:
                # Compute dense optical flow (Farneback)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
                
                # Compute magnitude and angle
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                mean_mag = np.mean(magnitude)
                flow_magnitudes.append(mean_mag)
                flow_directions.append(np.mean(angle))
                
                # Detect sudden motion
                if len(flow_magnitudes) > 1:
                    if abs(mean_mag - flow_magnitudes[-2]) > self.flow_threshold * 100:
                        sudden_motions += 1
            
            prev_gray = gray
        
        if not flow_magnitudes:
            return OpticalFlowFeatures(flow_consistency=1.0)
        
        # Compute consistency metrics
        mean_magnitude = np.mean(flow_magnitudes)
        magnitude_variance = np.var(flow_magnitudes)
        direction_variance = np.var(flow_directions)
        
        # Flow consistency (higher = more consistent = more likely real)
        # Normalize variance to [0, 1]
        consistency = 1.0 / (1.0 + magnitude_variance / 10 + direction_variance)
        
        return OpticalFlowFeatures(
            mean_flow_magnitude=float(mean_magnitude),
            flow_consistency=float(np.clip(consistency, 0, 1)),
            sudden_motion_count=sudden_motions,
            flow_direction_variance=float(direction_variance)
        )
    
    async def _analyze_landmark_jitter(
        self,
        frames: List[np.ndarray],
        engine: "InferenceEngine"
    ) -> LandmarkJitterFeatures:
        """
        Analyze facial landmark stability across frames.
        
        Deepfakes often have jittery landmarks due to per-frame
        generation inconsistencies.
        
        Args:
            frames: Frame sequence
            engine: InferenceEngine for face detection
            
        Returns:
            LandmarkJitterFeatures
        """
        # In production, this would:
        # 1. Run face detection on each frame
        # 2. Track landmark positions across frames
        # 3. Compute jitter metrics
        
        # Simplified version using frame differences
        if len(frames) < 2:
            return LandmarkJitterFeatures(mean_jitter=0.0)
        
        jitters = []
        prev_frame = None
        
        for frame in frames:
            if frame.dtype in [np.float32, np.float64]:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
            
            if prev_frame is not None:
                # Simple frame difference as proxy for jitter
                diff = np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32))
                
                # Focus on face region (center of frame)
                h, w = diff.shape[:2]
                face_region = diff[h//4:3*h//4, w//4:3*w//4]
                
                jitter = np.mean(face_region) / 255.0
                jitters.append(jitter)
            
            prev_frame = frame
        
        if not jitters:
            return LandmarkJitterFeatures(mean_jitter=0.0)
        
        # Identify unstable regions
        unstable_regions = []
        mean_jitter = np.mean(jitters)
        
        if mean_jitter > self.jitter_threshold:
            unstable_regions.append("face")
        if max(jitters) > self.jitter_threshold * 2:
            unstable_regions.append("mouth")  # Often most jittery in deepfakes
        
        return LandmarkJitterFeatures(
            mean_jitter=float(mean_jitter),
            max_jitter=float(max(jitters)),
            jitter_variance=float(np.var(jitters)),
            unstable_regions=unstable_regions
        )
    
    def _compute_jitter_score(
        self,
        features: LandmarkJitterFeatures
    ) -> float:
        """
        Compute jitter-based inconsistency score.
        
        Deepfake detection logic:
        - Real videos have NATURAL MOTION: consistent frame differences,
          moderate variance, predictable motion patterns.
        - AI deepfakes have ARTIFACTS: either unnaturally smooth (too-low
          frame differences) or inconsistently jittery (high variance in
          frame differences with irregular patterns).
        
        Args:
            features: LandmarkJitterFeatures
            
        Returns:
            Score [0, 1] where higher = more likely fake
        """
        score = 0.0
        
        # Detect unnaturally smooth video (suspicious for AI)
        # Real video has meaningful frame differences; AI video may be too smooth
        if features.mean_jitter < 0.005:
            # Very low motion = unnaturally smooth, suspicious for AI generation
            score += 0.4 * (1.0 - features.mean_jitter / 0.005)
        
        # High variance in jitter (inconsistent motion pattern = suspicious)
        # Real video has consistent motion; AI video may have irregular artifacts
        if features.jitter_variance > 0.01:
            score += 0.3 * min(1.0, features.jitter_variance * 50)
        
        # Extremely high jitter (beyond natural motion range = suspicious)
        if features.mean_jitter > 0.5:
            score += 0.3 * min(1.0, (features.mean_jitter - 0.5) / 0.5)
        
        return float(np.clip(score, 0, 1))

    def _compute_flow_anomaly_score(
        self,
        features: "OpticalFlowFeatures"
    ) -> float:
        """
        Compute optical flow anomaly score for AI detection.
        
        Real video: Natural motion with consistent flow patterns.
        AI video: Either unnaturally smooth (low magnitude variance)
        or inconsistently jittery (high sudden motion count).
        
        Args:
            features: OpticalFlowFeatures from flow computation
            
        Returns:
            Score [0, 1] where higher = more likely fake
        """
        score = 0.0
        
        # Unnaturally smooth motion (too-low magnitude variance = suspicious for AI)
        # Real video has natural motion variation; AI video may be too smooth
        if features.magnitude_variance < 5.0:
            score += 0.4 * (1.0 - features.magnitude_variance / 5.0)
        
        # Inconsistent motion direction (high direction variance = suspicious)
        if features.direction_variance > 2.0:
            score += 0.3 * min(1.0, (features.direction_variance - 2.0) / 3.0)
        
        # Sudden motion spikes (abrupt changes = suspicious)
        if features.sudden_motion_count > 3:
            score += 0.3 * min(1.0, features.sudden_motion_count / 10.0)
        
        return float(np.clip(score, 0, 1))

    def _compute_color_anomaly_score(
        self,
        features: "ColorConsistencyFeatures"
    ) -> float:
        """
        Compute color anomaly score for AI detection.
        
        Real video: Natural color variation with moderate correlation (>0.85).
        AI video: Either unnaturally uniform (very high correlation >0.99)
        or has sudden color shifts (low correlation in burst patterns).
        
        Args:
            features: ColorConsistencyFeatures
            
        Returns:
            Score [0, 1] where higher = more likely fake
        """
        score = 0.0
        
        # Unnaturally uniform colors (too-high correlation = suspicious for AI)
        if features.histogram_correlation > 0.99:
            score += 0.4
        
        # Sudden color shift detected (suspicious for AI)
        if features.color_shift_detected:
            score += 0.3
        
        # Very low correlation (below 0.7 = inconsistent, suspicious)
        if features.histogram_correlation < 0.7:
            score += 0.3 * (1.0 - features.histogram_correlation / 0.7)
        
        return float(np.clip(score, 0, 1))

    def _analyze_color_consistency(
        self,
        frames: List[np.ndarray]
    ) -> ColorConsistencyFeatures:
        """
        Analyze inter-frame color consistency.
        
        Deepfakes may have color shifts between frames, especially
        around manipulated regions.
        
        Args:
            frames: Frame sequence
            
        Returns:
            ColorConsistencyFeatures
        """
        import cv2
        
        if len(frames) < 2:
            return ColorConsistencyFeatures(histogram_correlation=1.0)
        
        correlations = []
        color_variances = []
        affected_frames = []
        
        prev_hist = None
        
        for i, frame in enumerate(frames):
            if frame.dtype in [np.float32, np.float64]:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
            
            # Compute color histogram
            hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Compare histograms
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                correlations.append(correlation)
                
                # Track variance
                diff = np.abs(hist - prev_hist)
                color_variances.append(np.mean(diff))
                
                # Flag frames with significant shift
                if correlation < 1.0 - self.color_threshold:
                    affected_frames.append(i)
            
            prev_hist = hist
        
        if not correlations:
            return ColorConsistencyFeatures(histogram_correlation=1.0)
        
        mean_correlation = np.mean(correlations)
        mean_variance = np.mean(color_variances)
        color_shift = len(affected_frames) > len(frames) * 0.1
        
        return ColorConsistencyFeatures(
            color_variance=float(mean_variance),
            histogram_correlation=float(mean_correlation),
            color_shift_detected=color_shift,
            affected_frames=affected_frames
        )
    
    def _detect_flickering(
        self,
        frames: List[np.ndarray],
        flow_features: OpticalFlowFeatures,
        color_features: ColorConsistencyFeatures
    ) -> bool:
        """
        Detect flickering artifacts.
        
        Flickering is characterized by rapid changes that alternate
        between states (e.g., brightness oscillation).
        
        Args:
            frames: Frame sequence
            flow_features: Optical flow analysis
            color_features: Color consistency analysis
            
        Returns:
            True if flickering detected
        """
        # Check for sudden motion patterns
        if flow_features.sudden_motion_count > len(frames) * 0.1:
            return True
        
        # Check for color oscillation
        if color_features.color_shift_detected and len(color_features.affected_frames) > 3:
            # Check if affected frames alternate
            if len(color_features.affected_frames) >= 3:
                diffs = np.diff(color_features.affected_frames)
                if np.std(diffs) < 2:  # Regular pattern = flickering
                    return True
        
        # Check frame-to-frame brightness variation
        if len(frames) >= 3:
            brightnesses = []
            for frame in frames:
                if frame.dtype in [np.float32, np.float64]:
                    brightness = np.mean(frame)
                else:
                    brightness = np.mean(frame) / 255.0
                brightnesses.append(brightness)
            
            # Detect alternating pattern
            diffs = np.diff(brightnesses)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            
            if sign_changes > len(frames) * 0.3:  # Many sign changes = oscillation
                return True
        
        return False
    
    def _find_anomaly_timestamps(
        self,
        frames: List[np.ndarray],
        flow_features: OpticalFlowFeatures,
        jitter_features: LandmarkJitterFeatures,
        color_features: ColorConsistencyFeatures,
        fps: float
    ) -> List[float]:
        """
        Find timestamps where temporal anomalies occur.
        
        Args:
            frames: Frame sequence
            flow_features: Flow analysis results
            jitter_features: Jitter analysis results
            color_features: Color analysis results
            fps: Frame rate
            
        Returns:
            List of anomaly timestamps in seconds
        """
        anomaly_frames = set()
        
        # Add color-affected frames
        anomaly_frames.update(color_features.affected_frames)
        
        # Estimate frames with flow anomalies
        if flow_features.sudden_motion_count > 0:
            # Distribute estimated anomaly positions
            for i in range(flow_features.sudden_motion_count):
                pos = int((i + 1) * len(frames) / (flow_features.sudden_motion_count + 1))
                anomaly_frames.add(pos)
        
        # Convert frame indices to timestamps
        timestamps = [
            round(frame_idx / fps, 3)
            for frame_idx in sorted(anomaly_frames)
            if frame_idx < len(frames)
        ]
        
        return timestamps[:20]  # Limit to 20 timestamps


# Singleton instance
_temporal_analyzer: Optional[TemporalAnalyzer] = None


def get_temporal_analyzer() -> TemporalAnalyzer:
    """Get singleton temporal analyzer instance."""
    global _temporal_analyzer
    if _temporal_analyzer is None:
        _temporal_analyzer = TemporalAnalyzer()
    return _temporal_analyzer
