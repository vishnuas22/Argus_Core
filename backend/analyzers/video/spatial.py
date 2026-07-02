"""
Argus Core - Spatial Video Analyzer
====================================
Per-frame spatial artifact detection using EfficientNet-B3 with CLIP guidance.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/video/spatial.py

SOTA Algorithms:
- Model: EfficientNet-B3 (from DeepfakeBench) fine-tuned on FaceForensics++
- Enhancement: CLIP visual encoder for generalization to unseen forgery types
- Inference: ONNX INT8 quantized for RTX 3050

Features Detected:
- Blending boundaries between real and fake regions
- Texture inconsistencies from generation artifacts
- Frequency domain artifacts (DCT analysis for GAN fingerprints)

Integration:
- Imports: core/engine.py, core/explain.py
- Inputs: List[np.ndarray] (face crops)
- Outputs: SpatialResult (per-frame scores, heatmaps)

Target Hardware: RTX 3050 (4GB VRAM) with INT8 quantization
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
import time

from analyzers.base import (
    SubAnalyzer,
    compute_confidence,
    detect_anomalies,
)
from schemas.schemas import SpatialResult
from config import config
from utils.logging import get_logger
from utils.errors import InferenceError

logger = get_logger(__name__)

# Iteration 1: SOTA per-frame image detector (lazy import)
try:
    from detectors import CLIPLoRAImageDetector
    _SOTA_SPATIAL_AVAILABLE = True
except ImportError as _e:
    _SOTA_SPATIAL_AVAILABLE = False
    logger.warning("SOTA CLIP spatial detector unavailable: %s", _e)

if TYPE_CHECKING:
    from core.engine import InferenceEngine
    from core.explain import ExplainabilityEngine


@dataclass
class FrequencyFeatures:
    """
    Frequency domain features for GAN fingerprint detection.
    
    DCT analysis reveals characteristic patterns from:
    - Upsampling artifacts (checkerboard patterns)
    - GAN-specific frequency signatures
    - Missing high-frequency detail
    """
    dct_score: float = 0.0
    spectral_flatness: float = 0.0
    high_freq_energy: float = 0.0
    grid_artifact_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dct_score": round(self.dct_score, 4),
            "spectral_flatness": round(self.spectral_flatness, 4),
            "high_freq_energy": round(self.high_freq_energy, 4),
            "grid_artifact_detected": self.grid_artifact_detected
        }


@dataclass
class PerFrameResult:
    """
    Analysis result for a single frame.
    
    Contains detection score, confidence, and optional heatmap.
    """
    frame_index: int
    score: float  # Fake probability [0, 1]
    confidence: float
    is_anomaly: bool = False
    heatmap: Optional[np.ndarray] = None
    frequency_features: Optional[FrequencyFeatures] = None


class SpatialAnalyzer(SubAnalyzer):
    """
    Per-frame spatial artifact detection.
    
    Analyzes individual video frames for spatial manipulation artifacts:
    - Blending boundaries at face edges
    - Texture inconsistencies in synthesized regions
    - Compression artifacts around manipulated areas
    - Frequency-domain GAN signatures
    
    Pipeline:
    1. Batch preprocessing (resize, normalize)
    2. EfficientNet-B3 classification
    3. CLIP embedding for generalization (optional)
    4. Frequency domain analysis
    5. GradCAM heatmap generation for anomalies
    6. Score aggregation with anomaly detection
    
    Usage:
        spatial = SpatialAnalyzer()
        result = await spatial.analyze_frames(face_crops, engine)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 8,
        anomaly_threshold: float = 2.0,
        generate_heatmaps: bool = True,
        max_heatmaps: int = 5
    ):
        """
        Initialize spatial analyzer.
        
        Args:
            target_size: Model input size (W, H)
            batch_size: Batch size for inference
            anomaly_threshold: Z-score threshold for anomaly detection
            generate_heatmaps: Whether to generate GradCAM heatmaps
            max_heatmaps: Maximum heatmaps to generate (for storage efficiency)
        """
        super().__init__("SpatialAnalyzer")
        
        self.target_size = target_size
        self.batch_size = batch_size
        self.anomaly_threshold = anomaly_threshold
        self.generate_heatmaps = generate_heatmaps
        self.max_heatmaps = max_heatmaps
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Model-specific config
        self.efficientnet_weight = 0.7
        self.clip_weight = 0.2
        self.frequency_weight = 0.1

        # ===========================================================
        # Iteration 1: SOTA per-frame CLIP+LoRA detector
        # ===========================================================
        # Lazy-initialized. When config.enable_sota_detectors is True,
        # the analyzer runs the CLIP+LoRA detector on each face crop
        # and blends the per-frame average with the legacy aggregate.
        self._sota_clip_detector = None

        logger.info(
            f"SpatialAnalyzer initialized: target_size={target_size}, "
            f"batch_size={batch_size}, heatmaps={generate_heatmaps}"
        )
    
    def get_required_models(self) -> List[str]:
        """
        Return required models for spatial analysis.
        
        Returns:
            List of model registry keys
        """
        return [
            "deepfake_detector_v3",  # Primary deepfake image detection
            "clip_vit_b16"          # CLIP for generalization
        ]
    
    async def analyze_frames(
        self,
        face_crops: List[np.ndarray],
        engine: "InferenceEngine",
        explainer: Optional["ExplainabilityEngine"] = None
    ) -> SpatialResult:
        """
        Analyze face crops for spatial artifacts.
        
        Args:
            face_crops: List of (H, W, 3) face images (RGB, uint8 or float)
            engine: InferenceEngine for model inference
            explainer: Optional ExplainabilityEngine for heatmaps
            
        Returns:
            SpatialResult with per-frame scores and aggregate
        """
        start_time = time.time()
        
        if not face_crops:
            logger.warning("No face crops provided for spatial analysis")
            return SpatialResult(
                score=0.5,
                per_frame_scores=[],
                anomaly_indices=[],
                heatmap_urls=[]
            )
        
        logger.debug(f"Analyzing {len(face_crops)} face crops")
        
        # 1. Preprocess all frames
        preprocessed = self._preprocess_batch(face_crops)
        
        # 2. Run EfficientNet detection
        efficientnet_scores = await self._run_efficientnet(preprocessed, engine)
        
        # 3. Run CLIP analysis (optional, for generalization)
        clip_scores = await self._run_clip_analysis(preprocessed, engine)
        
        # 4. Run frequency analysis
        frequency_scores = self._run_frequency_analysis(face_crops)
        
        # 5. Combine scores per frame
        per_frame_scores = self._combine_scores(
            efficientnet_scores,
            clip_scores,
            frequency_scores
        )
        
        # 6. Detect anomalies (suspicious frames)
        anomaly_indices = detect_anomalies(
            np.array(per_frame_scores),
            threshold=self.anomaly_threshold
        )
        
        # Also flag high-score frames as anomalies
        for i, score in enumerate(per_frame_scores):
            if score > 0.7 and i not in anomaly_indices:
                anomaly_indices.append(i)
        anomaly_indices = sorted(set(anomaly_indices))
        
        # 7. Generate heatmaps for anomaly frames (if enabled)
        heatmap_urls = []
        if self.generate_heatmaps and explainer and anomaly_indices:
            heatmap_urls = await self._generate_heatmaps(
                face_crops,
                anomaly_indices,
                engine,
                explainer
            )
        
        # 8. Compute aggregate score
        aggregate_score = self._compute_aggregate_score(per_frame_scores)

        # ===========================================================
        # Iteration 1 — SOTA CLIP+LoRA per-frame integration
        # (additive, strict-compat)
        # ===========================================================
        # Run the CLIP+LoRA image detector on a sample of frames and
        # blend the per-frame average with the legacy aggregate. The
        # SOTA detector has higher benchmark AUC, so it gets the
        # majority weight; the legacy aggregate still contributes
        # EfficientNet + frequency signals.
        if (
            getattr(config, "enable_sota_detectors", False)
            and _SOTA_SPATIAL_AVAILABLE
            and face_crops
        ):
            try:
                sota_score = await self._run_sota_clip_pass(face_crops)
                if sota_score is not None:
                    aggregate_score = float(
                        0.55 * sota_score + 0.45 * aggregate_score
                    )
                    logger.info(
                        "SOTA CLIP spatial integrated: sota=%.4f, blended=%.4f",
                        sota_score, aggregate_score,
                    )
            except Exception as e:
                logger.warning("SOTA CLIP spatial pass failed (non-fatal): %s", e)

        # ===========================================================
        # Iteration 2 — RPS sanitization (applied inside _run_sota_clip_pass)
        # Iteration 3 — Post-processing (calibration + conformal + drift)
        # ===========================================================
        try:
            from core.post_processing import apply_post_processing
            pp = apply_post_processing(
                score=aggregate_score,
                confidence=compute_confidence(np.array(per_frame_scores), len(face_crops)),
            )
            if pp.calibrated_score != pp.original_score:
                logger.info(
                    "Spatial temperature scaling: %.4f -> %.4f (T=%.4f)",
                    pp.original_score, pp.calibrated_score, pp.temperature,
                )
                aggregate_score = pp.calibrated_score
        except Exception as e:
            logger.debug("Spatial post-processing failed (non-fatal): %s", e)

        inference_time = (time.time() - start_time) * 1000
        confidence = compute_confidence(np.array(per_frame_scores), len(face_crops))
        self.record_analysis(True, inference_time, confidence)

        logger.info(
            f"Spatial analysis complete: {len(face_crops)} frames, "
            f"score={aggregate_score:.3f}, anomalies={len(anomaly_indices)}"
        )

        # ===========================================================
        # Iteration 7: Prometheus metrics recording
        # ===========================================================
        try:
            from observability import get_default_metrics
            _verdict = "fake" if aggregate_score >= 0.5 else "real"
            _metrics = get_default_metrics()
            _metrics.record_inference("video", _verdict, inference_time / 1000.0)
            if getattr(config, "enable_adversarial_defenses", False) and \
               getattr(config, "enable_rps", False):
                _metrics.record_adversarial_flag("video", "rps")
        except Exception as _e:
            logger.debug("Spatial metrics recording failed: %s", _e)

        return SpatialResult(
            score=aggregate_score,
            per_frame_scores=per_frame_scores,
            anomaly_indices=anomaly_indices,
            heatmap_urls=heatmap_urls
        )

    # ------------------------------------------------------------------
    # Iteration 1: SOTA CLIP+LoRA per-frame helper
    # ------------------------------------------------------------------
    async def _run_sota_clip_pass(
        self,
        face_crops: List[np.ndarray],
    ) -> Optional[float]:
        """
        Run the CLIP+LoRA image detector on a sample of face crops
        and return the average fake probability.

        Iteration 2: applies RPS sanitization to each frame before
        detection.

        Args:
            face_crops: List of HxWx3 RGB face images.

        Returns:
            Mean fake probability in [0, 1], or None on failure.
        """
        if not face_crops:
            return None

        if self._sota_clip_detector is None:
            self._sota_clip_detector = CLIPLoRAImageDetector()

        # Iteration 2: lazy-init RPS sanitizer
        rps = None
        if getattr(config, "enable_adversarial_defenses", False) and getattr(config, "enable_rps", False):
            try:
                from defenses import get_default_rps
                rps = get_default_rps()
            except Exception as e:
                logger.debug("RPS unavailable: %s", e)

        # Sample at most 8 frames for latency control
        n = min(8, len(face_crops))
        if n == len(face_crops):
            sampled = face_crops
        else:
            idx = np.linspace(0, len(face_crops) - 1, n).astype(int)
            sampled = [face_crops[i] for i in idx]

        async def _process_crop(crop: np.ndarray) -> Optional[float]:
            try:
                if rps is not None:
                    crop = rps.sanitize_image(crop)
                r = await self._sota_clip_detector.detect(crop)
                if r.error is None:
                    return r.score
            except Exception as e:
                logger.debug("SOTA CLIP per-frame failed: %s", e)
            return None

        results = await asyncio.gather(*[_process_crop(c) for c in sampled])
        scores = [s for s in results if s is not None]
        if not scores:
            return None
        return float(np.mean(scores))
    
    def _preprocess_batch(
        self,
        images: List[np.ndarray]
    ) -> np.ndarray:
        """
        Preprocess images for model input.
        
        Args:
            images: List of (H, W, 3) images
            
        Returns:
            Batched tensor (N, 3, H, W) float32
        """
        import cv2
        
        processed = []
        
        for img in images:
            # Ensure uint8
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Resize to target size
            resized = cv2.resize(img, self.target_size)
            
            # Convert to float [0, 1]
            float_img = resized.astype(np.float32) / 255.0
            
            # ImageNet normalization
            normalized = (float_img - self.mean) / self.std
            
            # Convert to CHW
            chw = np.transpose(normalized, (2, 0, 1))
            
            processed.append(chw)
        
        return np.stack(processed, axis=0).astype(np.float32)
    
    async def _run_efficientnet(
        self,
        batch: np.ndarray,
        engine: "InferenceEngine"
    ) -> List[float]:
        """
        Run deepfake detection model for spatial analysis.
        
        Uses deepfake_detector_v3 ONNX model via the inference engine.
        
        Args:
            batch: Preprocessed batch (N, 3, H, W)
            engine: InferenceEngine
            
        Returns:
            List of fake probability scores
        """
        try:
            result = await engine.infer(
                "deepfake_detector_v3",
                batch,
                return_probabilities=True
            )
            if result.class_probabilities is not None:
                probs = result.class_probabilities
                scores = [float(p[1]) if probs.shape[-1] >= 2 else float(p[0]) for p in probs]
            else:
                scores = [float(s) for s in result.predictions]
        except Exception as e:
            logger.warning(f"Primary model inference failed: {e}")
            scores = [0.5] * len(batch)
        
        return scores
    
    async def _run_clip_analysis(
        self,
        batch: np.ndarray,
        engine: "InferenceEngine"
    ) -> List[float]:
        """
        Run CLIP embedding analysis for novel deepfake detection.
        
        CLIP provides better generalization to unseen manipulation types.
        
        Args:
            batch: Preprocessed batch
            engine: InferenceEngine
            
        Returns:
            List of anomaly scores from CLIP
        """
        try:
            # Get CLIP embeddings
            result = await engine.infer(
                "clip_vit_b16",
                batch,
                # clip_vit_b16 ONNX in this stack is static-batch; force chunk size 1.
                batch_size=1,
                return_probabilities=False
            )
            
            embeddings = np.asarray(result.predictions, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = np.expand_dims(embeddings, 0)
            elif embeddings.ndim > 2:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
            
            # Compute anomaly scores based on embedding statistics
            # Real faces cluster differently than fake faces in CLIP space
            scores = []
            
            if len(embeddings) > 1:
                # Compute mean embedding
                mean_embedding = np.mean(embeddings, axis=0)
                
                for emb in embeddings:
                    # Distance from mean (outliers may be fake)
                    distance = np.linalg.norm(emb - mean_embedding)
                    # Normalize to [0, 1] using sigmoid
                    score = 1 / (1 + np.exp(-distance / 10))
                    scores.append(float(score))
            else:
                scores = [0.5] * len(batch)
            
            return scores
            
        except Exception as e:
            logger.debug(f"CLIP analysis skipped: {e}")
            return [0.5] * len(batch)
    
    def _run_frequency_analysis(
        self,
        images: List[np.ndarray]
    ) -> List[float]:
        """
        Analyze images in frequency domain for GAN artifacts.
        
        Args:
            images: Original images (before preprocessing)
            
        Returns:
            List of frequency-based anomaly scores
        """
        import cv2
        
        scores = []
        
        for img in images:
            try:
                features = self.detect_frequency_artifacts(img)
                scores.append(features.dct_score)
            except Exception as e:
                logger.debug(f"Frequency analysis failed for frame: {e}")
                scores.append(0.5)
        
        return scores
    
    def detect_frequency_artifacts(
        self,
        image: np.ndarray
    ) -> FrequencyFeatures:
        """
        DCT analysis for GAN fingerprints.
        
        Args:
            image: Input image (H, W, 3) or (H, W)
            
        Returns:
            FrequencyFeatures with analysis results
        """
        import cv2
        
        # Convert to grayscale
        if len(image.shape) == 3:
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Resize for consistent analysis
        gray = cv2.resize(gray, (256, 256))
        gray_float = gray.astype(np.float32)
        
        # Apply DCT
        dct = cv2.dct(gray_float)
        
        h, w = dct.shape
        
        # Analyze frequency distribution
        low_freq = dct[:h//4, :w//4]
        high_freq = dct[h//2:, w//2:]
        
        energy_low = np.sum(np.abs(low_freq) ** 2)
        energy_high = np.sum(np.abs(high_freq) ** 2)
        total_energy = np.sum(np.abs(dct) ** 2)
        
        if total_energy > 0:
            high_freq_ratio = energy_high / total_energy
        else:
            high_freq_ratio = 0.0
        
        # Spectral flatness (Wiener entropy)
        dct_abs = np.abs(dct.flatten()) + 1e-10
        geometric_mean = np.exp(np.mean(np.log(dct_abs)))
        arithmetic_mean = np.mean(dct_abs)
        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        # Detect grid artifacts (common in GANs)
        grid_artifact = self._detect_grid_artifacts(dct)
        
        # Compute anomaly score
        # Low high-freq energy + high spectral flatness = suspicious
        dct_score = 0.0
        
        if high_freq_ratio < 0.05:
            dct_score += 0.4
        elif high_freq_ratio < 0.1:
            dct_score += 0.2
        
        if spectral_flatness > 0.7:
            dct_score += 0.3
        elif spectral_flatness > 0.5:
            dct_score += 0.15
        
        if grid_artifact:
            dct_score += 0.3
        
        return FrequencyFeatures(
            dct_score=float(np.clip(dct_score, 0, 1)),
            spectral_flatness=float(spectral_flatness),
            high_freq_energy=float(high_freq_ratio),
            grid_artifact_detected=grid_artifact
        )
    
    def _detect_grid_artifacts(self, dct: np.ndarray) -> bool:
        """
        Detect grid/checkerboard artifacts in DCT.
        
        These are characteristic of upsampling operations in GANs.
        
        Args:
            dct: DCT coefficients
            
        Returns:
            True if grid artifacts detected
        """
        h, w = dct.shape
        
        # Check for periodic patterns at specific frequencies
        # Grid artifacts appear as spikes at regular intervals
        
        # Sample at positions that would show grid patterns
        check_positions = [
            (h//4, w//4),
            (h//2, w//2),
            (3*h//4, 3*w//4)
        ]
        
        magnitudes = [np.abs(dct[y, x]) for y, x in check_positions]
        avg_magnitude = np.mean(np.abs(dct))
        
        # If specific positions have much higher energy than average
        if avg_magnitude > 0:
            for mag in magnitudes:
                if mag > avg_magnitude * 10:  # 10x average is suspicious
                    return True
        
        return False
    
    def _combine_scores(
        self,
        efficientnet_scores: List[float],
        clip_scores: List[float],
        frequency_scores: List[float]
    ) -> List[float]:
        """
        Combine scores from multiple detectors.
        
        Args:
            efficientnet_scores: Primary detector scores
            clip_scores: CLIP embedding scores
            frequency_scores: DCT analysis scores
            
        Returns:
            Combined per-frame scores
        """
        combined = []
        
        n_frames = len(efficientnet_scores)
        
        for i in range(n_frames):
            eff_score = efficientnet_scores[i] if i < len(efficientnet_scores) else 0.5
            clip_score = clip_scores[i] if i < len(clip_scores) else 0.5
            freq_score = frequency_scores[i] if i < len(frequency_scores) else 0.5
            
            # Weighted combination
            combined_score = (
                self.efficientnet_weight * eff_score +
                self.clip_weight * clip_score +
                self.frequency_weight * freq_score
            )
            
            combined.append(float(np.clip(combined_score, 0, 1)))
        
        return combined
    
    def _compute_aggregate_score(
        self,
        per_frame_scores: List[float]
    ) -> float:
        """
        Compute aggregate score from per-frame scores.
        
        Uses weighted combination favoring high-scoring frames.
        
        Args:
            per_frame_scores: List of per-frame scores
            
        Returns:
            Aggregate score [0, 1]
        """
        if not per_frame_scores:
            return 0.5
        
        scores = np.array(per_frame_scores)
        
        # Use multiple aggregation methods and combine
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        top_k_mean = np.mean(np.sort(scores)[-max(1, len(scores)//5):])  # Top 20%
        
        # Weighted combination (favor suspicious frames)
        aggregate = 0.4 * mean_score + 0.3 * max_score + 0.3 * top_k_mean
        
        return float(np.clip(aggregate, 0, 1))
    
    async def _generate_heatmaps(
        self,
        images: List[np.ndarray],
        anomaly_indices: List[int],
        engine: "InferenceEngine",
        explainer: "ExplainabilityEngine"
    ) -> List[str]:
        """
        Generate GradCAM heatmaps for anomaly frames.
        
        Args:
            images: Original images
            anomaly_indices: Indices of anomaly frames
            engine: InferenceEngine
            explainer: ExplainabilityEngine
            
        Returns:
            List of heatmap URLs/keys
            
        Note:
            Heatmap generation requires the XAI module to be properly configured.
            Returns empty list if explainer is not available.
        """
        heatmap_urls = []
        
        # Limit number of heatmaps
        indices_to_process = anomaly_indices[:self.max_heatmaps]
        
        if not explainer:
            logger.warning("ExplainabilityEngine not available - skipping heatmap generation")
            return heatmap_urls
        
        for idx in indices_to_process:
            if idx >= len(images):
                continue
            
            try:
                # Generate actual heatmap using explainer
                heatmap = await explainer.generate_gradcam(
                    model_name="deepfake_detector",
                    input_image=images[idx],
                    target_class=1  # Fake class
                )
                
                if heatmap is not None:
                    # Upload to storage and get URL
                    # For now, store the heatmap data reference
                    heatmap_url = f"heatmap_frame_{idx}_{id(heatmap)}"
                    heatmap_urls.append(heatmap_url)
                
            except Exception as e:
                logger.error(f"Heatmap generation failed for frame {idx}: {e}")
        
        return heatmap_urls


# Singleton instance
_spatial_analyzer: Optional[SpatialAnalyzer] = None


def get_spatial_analyzer() -> SpatialAnalyzer:
    """Get singleton spatial analyzer instance."""
    global _spatial_analyzer
    if _spatial_analyzer is None:
        _spatial_analyzer = SpatialAnalyzer()
    return _spatial_analyzer
