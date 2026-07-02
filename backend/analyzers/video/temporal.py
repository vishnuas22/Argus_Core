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
import cv2

from analyzers.base import (
    SubAnalyzer,
    compute_confidence,
    detect_anomalies,
)
from schemas.schemas import TemporalResult
from config import config
from utils.logging import get_logger
from utils.errors import InferenceError
from models.model_init import ensure_models_for_analyzer, is_model_ready
import threading

logger = get_logger(__name__)

# Iteration 1: SOTA video temporal detectors (lazy import)
# Iteration 4: added TimeSformerVideoDetector for further diversity
try:
    from detectors import (
        VideoMAEDetector,
        AltFreeVideoDetector,
        TimeSformerVideoDetector,
        combine_detector_results,
    )
    _SOTA_TEMPORAL_AVAILABLE = True
except ImportError as _e:
    _SOTA_TEMPORAL_AVAILABLE = False
    logger.warning("SOTA temporal detectors unavailable: %s", _e)

# PyTorch Model Cache
_videomae_model = None
_videomae_model_lock = threading.Lock()

if TYPE_CHECKING:
    from core.engine import InferenceEngine


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
    magnitude_variance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_flow_magnitude": round(self.mean_flow_magnitude, 4),
            "flow_consistency": round(self.flow_consistency, 4),
            "sudden_motion_count": self.sudden_motion_count,
            "flow_direction_variance": round(self.flow_direction_variance, 4),
            "magnitude_variance": round(self.magnitude_variance, 4)
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
            "videomae": 0.40,         # VideoMAE transformer
            "optical_flow": 0.25,     # Optical flow analysis
            "landmark_jitter": 0.20,  # Facial landmark stability
            "color_consistency": 0.15 # Inter-frame color
        }

        # ===========================================================
        # Iteration 1: SOTA video temporal detector ensemble
        # Iteration 4: added TimeSformer as 3rd detector (cc-by-nc-4.0)
        # ===========================================================
        # VideoMAE-base + AltFree (CVPR 2024) + TimeSformer (ICML 2021) —
        # fused via DiversityEnsemble. Lazy-initialized; auto-downweighted
        # when adapter weights are missing.
        # TimeSformer is gated by config.enable_timesformer (default True
        # but OFF if non-commercial restriction is unacceptable).
        self._sota_temporal_detectors = None
        self._sota_temporal_prior_weights = [0.89, 0.86, 0.84]  # VideoMAE, AltFree, TimeSformer

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
            "videomae_temporal",   # VideoMAE temporal transformer
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
        
        # 1. VideoMAE Transformer Analysis
        videomae_score = await self._run_videomae_analysis(frame_sequence, engine)
        
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
            self.weights["videomae"] * videomae_score +
            self.weights["optical_flow"] * flow_score +
            self.weights["landmark_jitter"] * jitter_score +
            self.weights["color_consistency"] * color_score
        )

        # ===========================================================
        # Iteration 1 — SOTA video temporal ensemble integration
        # (additive, strict-compat)
        # ===========================================================
        # Run VideoMAE (fine-tuned) + AltFree on the same frame sequence
        # and fuse their outputs via DiversityEnsemble. The resulting
        # fake probability is blended with the legacy inconsistency
        # score: 60% SOTA ensemble, 40% legacy heuristics.
        if (
            getattr(config, "enable_sota_detectors", False)
            and _SOTA_TEMPORAL_AVAILABLE
            and len(frame_sequence) >= 2
        ):
            try:
                sota_fake_prob = await self._run_sota_temporal_ensemble(
                    frame_sequence
                )
                if sota_fake_prob is not None:
                    # Legacy inconsistency is in [0,1] (1 = max fake).
                    # Blend in fake-probability space, then convert back.
                    legacy_fake_prob = float(combined_inconsistency)
                    blended_fake_prob = 0.6 * sota_fake_prob + 0.4 * legacy_fake_prob
                    combined_inconsistency = float(
                        np.clip(blended_fake_prob, 0.0, 1.0)
                    )
                    logger.info(
                        "SOTA temporal ensemble integrated: sota=%.4f, blended=%.4f",
                        sota_fake_prob, combined_inconsistency,
                    )
            except Exception as e:
                logger.warning("SOTA temporal ensemble failed (non-fatal): %s", e)

        consistency_score = 1.0 - combined_inconsistency
        consistency_score = float(np.clip(consistency_score, 0, 1))

        # ===========================================================
        # Iteration 3 — Post-processing (calibration + conformal)
        # (additive, strict-compat)
        # ===========================================================
        try:
            from core.post_processing import apply_post_processing
            # Convert consistency_score to fake_prob for calibration
            # consistency_score: 1 = real, 0 = fake → fake_prob = 1 - consistency
            fake_prob_for_cal = 1.0 - consistency_score
            _synth = np.array([
                fake_prob_for_cal,
                flow_features.flow_consistency,
                color_features.color_shift_detected if color_features else 0.0,
                float(len(color_features.affected_frames)) if color_features else 0.0,
                float(flow_features.sudden_motion_count),
            ], dtype=np.float64)
            pp = apply_post_processing(
                score=fake_prob_for_cal,
                confidence=abs(consistency_score - 0.5) * 2,  # extremity as confidence
                embedding=_synth,
                modality="video",
                analysis_id="",
            )
            if pp.calibrated_score != pp.original_score:
                logger.info(
                    "Temporal temperature scaling: %.4f -> %.4f (T=%.4f)",
                    pp.original_score, pp.calibrated_score, pp.temperature,
                )
                # Convert back to consistency_score
                consistency_score = 1.0 - pp.calibrated_score
        except Exception as e:
            logger.debug("Temporal post-processing failed (non-fatal): %s", e)
        
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
            np.array([videomae_score, flow_score, jitter_score, color_score]),
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

    # ------------------------------------------------------------------
    # Iteration 1: SOTA video temporal ensemble helper
    # ------------------------------------------------------------------
    async def _run_sota_temporal_ensemble(
        self,
        frame_sequence: List[np.ndarray],
    ) -> Optional[float]:
        """
        Run VideoMAE (fine-tuned) + AltFree on the frame sequence and
        fuse their outputs via DiversityEnsemble.

        Args:
            frame_sequence: List of HxWx3 RGB frames (any resolution).

        Returns:
            Fused fake probability in [0, 1], or None on failure.
        """
        if not frame_sequence:
            return None

        if self._sota_temporal_detectors is None:
            detectors = [
                VideoMAEDetector(),
                AltFreeVideoDetector(),
            ]
            # Iteration 4: add TimeSformer if enabled (cc-by-nc-4.0 license)
            if getattr(config, "enable_timesformer", True):
                try:
                    detectors.append(TimeSformerVideoDetector())
                except Exception as e:
                    logger.warning("TimeSformer init failed (non-fatal): %s", e)
            self._sota_temporal_detectors = detectors

        detector_results = []
        for det in self._sota_temporal_detectors:
            try:
                r = await det.detect(frame_sequence)
                detector_results.append(r)
            except Exception as e:
                logger.warning("SOTA temporal detector %s failed: %s", det.name, e)

        if not detector_results:
            return None

        fused = combine_detector_results(
            detector_results,
            prior_weights=self._sota_temporal_prior_weights[:len(detector_results)],
        )
        if fused.error and "all_members_failed" in fused.error:
            return None
        return float(fused.score)

    async def _run_videomae_analysis(
        self,
        frames: List[np.ndarray],
        engine: "InferenceEngine"
    ) -> float:
        """
        Run VideoMAE transformer for temporal analysis.
        Uses a PyTorch implementation for VideoMAE.
        
        Args:
            frames: Frame sequence
            engine: InferenceEngine
            
        Returns:
            Inconsistency score [0, 1]
        """
        global _videomae_model
        try:
            import torch
            from torchvision import transforms
            from analyzers.video.videomae_pytorch import VideoMAEDeepfakeDetector
            
            with _videomae_model_lock:
                if _videomae_model is None:
                    _videomae_model = VideoMAEDeepfakeDetector()
                    _device = config.device
                    if _device != "cpu":
                        _videomae_model = _videomae_model.to(_device)
                        logger.info("VideoMAE moved to %s", _device)
                    _videomae_model.eval()
            
            preprocessed = self._preprocess_sequence(frames)
            # preprocessed is (T, C, H, W)
            # VideoMAE expects (B, T, C, H, W)
            input_tensor = torch.from_numpy(preprocessed).unsqueeze(0)
            
            _device = config.device
            if _device != "cpu":
                input_tensor = input_tensor.to(_device)
                
            with torch.no_grad():
                logits = _videomae_model(input_tensor)
                probs = torch.softmax(logits, dim=-1)
                fake_prob = float(probs[0, 1].cpu().item())
            
            return fake_prob
            
        except ImportError:
            logger.warning("PyTorch/Transformers not available for VideoMAE, returning neutral score")
            return 0.5
        except Exception as e:
            logger.warning(f"VideoMAE analysis failed: {e}")
            return 0.5
    
    def _preprocess_sequence(
        self,
        frames: List[np.ndarray]
    ) -> np.ndarray:
        """
        Preprocess frame sequence for VideoMAE.
        
        Args:
            frames: List of frames
            
        Returns:
            Tensor (T, C, H, W)
        """
        import cv2
        
        target_size = (224, 224)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Sample frames if too many (VideoMAE typically uses 16 frames)
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
            flow_direction_variance=float(direction_variance),
            magnitude_variance=float(magnitude_variance)
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
        
        Uses a three-tier detection strategy:
        1. RetinaFace ONNX via InferenceEngine (most accurate)
        2. OpenCV DNN face detector (good accuracy, no extra deps)
        3. Frame-difference proxy (fallback only)
        
        Args:
            frames: Frame sequence
            engine: InferenceEngine for face detection
            
        Returns:
            LandmarkJitterFeatures
        """
        if len(frames) < 2:
            return LandmarkJitterFeatures(mean_jitter=0.0)
        
        # Try to extract real facial landmarks
        landmarks_sequence = await self._extract_face_landmarks(frames, engine)
        
        if landmarks_sequence is not None and len(landmarks_sequence) >= 2:
            # Use real landmark tracking
            return self._compute_landmark_jitter_from_tracking(landmarks_sequence)
        
        # Fallback: frame-difference proxy (legacy behavior)
        return self._compute_landmark_jitter_from_framediff(frames)
    
    async def _extract_face_landmarks(
        self,
        frames: List[np.ndarray],
        engine: "InferenceEngine"
    ) -> Optional[List[Dict[str, Tuple[float, float]]]]:
        """
        Extract facial landmarks from each frame using available detectors.
        
        Returns list of landmark dicts with keys: left_eye, right_eye, nose,
        mouth_left, mouth_right. Returns None if no detector is available.
        """
        # Tier 1: Try RetinaFace ONNX via InferenceEngine
        landmarks = await self._extract_landmarks_retinaface(frames, engine)
        if landmarks is not None:
            return landmarks
        
        # Tier 2: Try OpenCV DNN face detector (Caffe model)
        landmarks = self._extract_landmarks_opencv_dnn(frames)
        if landmarks is not None:
            return landmarks
        
        return None
    
    async def _extract_landmarks_retinaface(
        self,
        frames: List[np.ndarray],
        engine: "InferenceEngine"
    ) -> Optional[List[Dict[str, Tuple[float, float]]]]:
        """
        Extract landmarks using RetinaFace ONNX model via InferenceEngine.
        
        RetinaFace outputs: [batch, num_anchors, 15]
        where 15 = 4 (bbox) + 1 (confidence) + 10 (5 landmarks × 2 coords)
        """
        try:
            if not is_model_ready("retinaface"):
                return None
            
            landmarks_sequence = []
            
            for frame in frames:
                # Preprocess: resize to 640x640, normalize to [0,1], NCHW
                if frame.dtype in [np.float32, np.float64] and frame.max() <= 1.0:
                    frame_uint8 = (frame * 255).astype(np.uint8)
                else:
                    frame_uint8 = frame.astype(np.uint8)
                
                h_orig, w_orig = frame_uint8.shape[:2]
                resized = cv2.resize(frame_uint8, (640, 640))
                blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, :] / 255.0
                
                # Run inference
                result = await engine.infer("retinaface", blob, return_probabilities=False)
                predictions = result.predictions  # [1, num_anchors, 15]
                
                # Decode detections
                frame_landmarks = self._decode_retinaface_output(
                    predictions[0], w_orig, h_orig, conf_threshold=0.5
                )
                
                if frame_landmarks is not None:
                    landmarks_sequence.append(frame_landmarks)
                else:
                    landmarks_sequence.append(None)
            
            # Check if we got landmarks for at least 50% of frames
            valid_count = sum(1 for l in landmarks_sequence if l is not None)
            if valid_count >= len(frames) * 0.5:
                return landmarks_sequence
            return None
            
        except Exception as e:
            logger.debug(f"RetinaFace landmark extraction failed: {e}")
            return None
    
    def _decode_retinaface_output(
        self,
        predictions: np.ndarray,
        w_orig: int,
        h_orig: int,
        conf_threshold: float = 0.5
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Decode RetinaFace raw output into landmark coordinates.
        
        Args:
            predictions: [num_anchors, 15] array
            w_orig, h_orig: Original image dimensions
            conf_threshold: Minimum confidence threshold
            
        Returns:
            Dict of landmark name -> (x, y) in original image coords, or None
        """
        # Extract confidence scores (index 4)
        scores = predictions[:, 4]
        
        # Filter by confidence
        mask = scores > conf_threshold
        if not np.any(mask):
            return None
        
        filtered = predictions[mask]
        scores_filtered = scores[mask]
        
        # Get best detection
        best_idx = np.argmax(scores_filtered)
        best = filtered[best_idx]
        
        # Extract 5 landmarks (indices 5-14): left_eye, right_eye, nose, mouth_left, mouth_right
        # Each landmark is (x, y) in 640x640 space
        landmark_names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        landmarks = {}
        
        scale_x = w_orig / 640.0
        scale_y = h_orig / 640.0
        
        for i, name in enumerate(landmark_names):
            x = best[5 + i * 2] * scale_x
            y = best[6 + i * 2] * scale_y
            landmarks[name] = (float(x), float(y))
        
        return landmarks
    
    def _extract_landmarks_opencv_dnn(
        self,
        frames: List[np.ndarray]
    ) -> Optional[List[Dict[str, Tuple[float, float]]]]:
        """
        Fallback: Extract face bounding boxes using OpenCV's built-in
        face detector and derive approximate landmark positions.
        
        Less accurate than RetinaFace but requires zero extra dependencies.
        """
        try:
            # Use OpenCV's Haar cascade (always available)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if face_cascade.empty():
                return None
            
            landmarks_sequence = []
            
            for frame in frames:
                if frame.dtype in [np.float32, np.float64] and frame.max() <= 1.0:
                    gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Use largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    
                    # Derive approximate landmark positions from bounding box
                    # These are rough estimates based on average face proportions
                    landmarks = {
                        "left_eye": (x + w * 0.35, y + h * 0.35),
                        "right_eye": (x + w * 0.65, y + h * 0.35),
                        "nose": (x + w * 0.50, y + h * 0.55),
                        "mouth_left": (x + w * 0.30, y + h * 0.75),
                        "mouth_right": (x + w * 0.70, y + h * 0.75),
                    }
                    landmarks_sequence.append(landmarks)
                else:
                    landmarks_sequence.append(None)
            
            valid_count = sum(1 for l in landmarks_sequence if l is not None)
            if valid_count >= len(frames) * 0.5:
                return landmarks_sequence
            return None
            
        except Exception as e:
            logger.debug(f"OpenCV face detection failed: {e}")
            return None
    
    def _compute_landmark_jitter_from_tracking(
        self,
        landmarks_sequence: List[Optional[Dict[str, Tuple[float, float]]]]
    ) -> LandmarkJitterFeatures:
        """
        Compute jitter metrics from tracked landmark positions.
        
        Analyzes frame-to-frame displacement of each landmark,
        focusing on eye, nose, and mouth stability.
        """
        # Filter to frames with valid landmarks
        valid_frames = [(i, lm) for i, lm in enumerate(landmarks_sequence) if lm is not None]
        
        if len(valid_frames) < 2:
            return LandmarkJitterFeatures(mean_jitter=0.0)
        
        # Compute per-landmark displacement between consecutive valid frames
        landmark_names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        all_displacements = {name: [] for name in landmark_names}
        
        for idx in range(1, len(valid_frames)):
            prev_lm = valid_frames[idx - 1][1]
            curr_lm = valid_frames[idx][1]
            
            for name in landmark_names:
                if name in prev_lm and name in curr_lm:
                    px, py = prev_lm[name]
                    cx, cy = curr_lm[name]
                    displacement = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    all_displacements[name].append(displacement)
        
        # Compute jitter metrics
        all_values = []
        for name in landmark_names:
            if all_displacements[name]:
                all_values.extend(all_displacements[name])
        
        if not all_values:
            return LandmarkJitterFeatures(mean_jitter=0.0)
        
        mean_jitter = float(np.mean(all_values))
        max_jitter = float(np.max(all_values))
        jitter_variance = float(np.var(all_values))
        
        # Identify unstable regions
        unstable_regions = []
        mouth_displacements = all_displacements["mouth_left"] + all_displacements["mouth_right"]
        eye_displacements = all_displacements["left_eye"] + all_displacements["right_eye"]
        
        if mouth_displacements and np.mean(mouth_displacements) > mean_jitter * 1.5:
            unstable_regions.append("mouth")
        if eye_displacements and np.mean(eye_displacements) > mean_jitter * 1.5:
            unstable_regions.append("eyes")
        if jitter_variance > mean_jitter * 2:
            unstable_regions.append("face")
        
        return LandmarkJitterFeatures(
            mean_jitter=mean_jitter,
            max_jitter=max_jitter,
            jitter_variance=jitter_variance,
            unstable_regions=unstable_regions
        )
    
    def _compute_landmark_jitter_from_framediff(
        self,
        frames: List[np.ndarray]
    ) -> LandmarkJitterFeatures:
        """
        Fallback jitter estimation using frame differences.
        
        Used when no face detector is available.
        """
        jitters = []
        prev_frame = None
        
        for frame in frames:
            if frame.dtype in [np.float32, np.float64]:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
            
            if prev_frame is not None:
                diff = np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32))
                h, w = diff.shape[:2]
                face_region = diff[h//4:3*h//4, w//4:3*w//4]
                jitter = np.mean(face_region) / 255.0
                jitters.append(jitter)
            
            prev_frame = frame
        
        if not jitters:
            return LandmarkJitterFeatures(mean_jitter=0.0)
        
        unstable_regions = []
        mean_jitter = np.mean(jitters)
        
        if mean_jitter > self.jitter_threshold:
            unstable_regions.append("face")
        if max(jitters) > self.jitter_threshold * 2:
            unstable_regions.append("mouth")
        
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
        if features.flow_direction_variance > 2.0:
            score += 0.3 * min(1.0, (features.flow_direction_variance - 2.0) / 3.0)
        
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
