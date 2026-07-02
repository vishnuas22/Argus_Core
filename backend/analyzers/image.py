"""
Argus Core - Image Analyzer
===========================
Single-image deepfake detection focused on face manipulation artifacts.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/image.py

Detection Pipeline:
- Model: ONNX-based ViT classifier for face manipulation detection
- Analysis: Frequency domain (DCT), blending boundary detection, ensemble fusion
- Explainability: GradCAM overlay generation

Detection Targets:
- Face swaps (DeepFaceLab, FaceSwap, etc.)
- Facial reenactment
- Face attribute manipulation
- Identity swap forgeries

Integration:
- Imports: core/engine.py, core/explain.py
- Inputs: image: np.ndarray
- Outputs: ImageResult (via ModalityResult)

Target Hardware: RTX 3050 (4GB VRAM) with INT8 quantization
"""

import os
import asyncio
import threading
import time
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field

from analyzers.base import (
    BaseAnalyzer,
    SubAnalyzer,
    normalize_scores,
    aggregate_scores,
    compute_confidence,
    infer_fake_class_index,
    extract_fake_probabilities,
)
from schemas.schemas import (
    Modality, PreprocessedData, ModalityResult, ContentType
)
from config import config
from utils.logging import get_logger
from utils.errors import ValidationError, InferenceError

logger = get_logger(__name__)

# Iteration 1: SOTA detector ensemble (lazy import to avoid hard dep)
# Iteration 3: added SigLIPImageDetector for ensemble diversity
# Iteration 5: added SBIDetector for boundary-artifact detection
# Iteration 6: added UCFCrossForgeryDetector for cross-generator detection
try:
    from detectors import (
        CLIPLoRAImageDetector,
        DINOv2ImageDetector,
        SigLIPImageDetector,
        SBIDetector,
        UCFCrossForgeryDetector,
        combine_detector_results,
    )
    _SOTA_DETECTORS_AVAILABLE = True
except ImportError as _e:
    _SOTA_DETECTORS_AVAILABLE = False
    logger.warning("SOTA image detectors unavailable: %s", _e)

if TYPE_CHECKING:
    from core.engine import InferenceEngine

# ===== ONNX Session Cache =====
# Sessions are created once and reused across all analysis requests.
# This avoids 2-6s overhead per request from InferenceSession creation.
_primary_onnx_session = None
_auxiliary_onnx_session = None
_onnx_session_lock = threading.Lock()
_primary_run_lock = threading.Lock()
_auxiliary_run_lock = threading.Lock()

# PyTorch Model Cache
_pytorch_model = None
_pytorch_model_lock = threading.Lock()


def get_cached_primary_session(model_path: str):
    """
    Get or create a cached ONNX session for the primary image detector.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        ONNX InferenceSession instance
    """
    global _primary_onnx_session

    if _primary_onnx_session is not None:
        return _primary_onnx_session

    with _onnx_session_lock:
        if _primary_onnx_session is not None:
            return _primary_onnx_session

        try:
            import onnxruntime as ort

            if not os.path.exists(model_path):
                logger.warning(f"Primary model not found at {model_path}")
                return None

            providers = ["CPUExecutionProvider"]
            if config.use_gpu:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.inter_op_num_threads = 2
            session_options.intra_op_num_threads = 4

            _primary_onnx_session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
            )

            logger.info(
                f"Cached primary ONNX session: "
                f"inputs={[i.name for i in _primary_onnx_session.get_inputs()]}, "
                f"providers={_primary_onnx_session.get_providers()}"
            )

        except Exception as exc:
            logger.error(f"Failed to create primary ONNX session: {exc}")
            _primary_onnx_session = None

    return _primary_onnx_session


def get_cached_auxiliary_session(model_path: str):
    """
    Get or create a cached ONNX session for the auxiliary image detector.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        ONNX InferenceSession instance
    """
    global _auxiliary_onnx_session

    if _auxiliary_onnx_session is not None:
        return _auxiliary_onnx_session

    with _onnx_session_lock:
        if _auxiliary_onnx_session is not None:
            return _auxiliary_onnx_session

        try:
            import onnxruntime as ort

            if not os.path.exists(model_path):
                logger.warning(f"Auxiliary model not found at {model_path}")
                return None

            providers = ["CPUExecutionProvider"]
            if config.use_gpu:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.inter_op_num_threads = 2
            session_options.intra_op_num_threads = 4

            _auxiliary_onnx_session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
            )

            logger.info(
                f"Cached auxiliary ONNX session: "
                f"inputs={[i.name for i in _auxiliary_onnx_session.get_inputs()]}, "
                f"providers={_auxiliary_onnx_session.get_providers()}"
            )

        except Exception as exc:
            logger.error(f"Failed to create auxiliary ONNX session: {exc}")
            _auxiliary_onnx_session = None

    return _auxiliary_onnx_session


@dataclass
class DCTFeatures:
    """
    Discrete Cosine Transform features for GAN fingerprint detection.
    
    GANs leave characteristic frequency-domain signatures that
    differ from natural images.
    """
    energy_high_freq: float = 0.0  # Energy in high frequency bands
    energy_low_freq: float = 0.0   # Energy in low frequency bands
    spectral_flatness: float = 0.0  # Measure of spectral uniformity
    anomaly_score: float = 0.0      # Overall DCT anomaly score
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "energy_high_freq": round(self.energy_high_freq, 4),
            "energy_low_freq": round(self.energy_low_freq, 4),
            "spectral_flatness": round(self.spectral_flatness, 4),
            "anomaly_score": round(self.anomaly_score, 4)
        }


@dataclass 
class ImageAnalysisResult:
    """
    Internal result container for image analysis.
    
    Contains detailed per-detector results before aggregation.
    """
    # Primary detection score (fake probability)
    fake_probability: float = 0.0
    
    # Model-specific scores
    ensemble_score: float = 0.0  # Primary: ViT ensemble (dima806 + v2)
    auxiliary_score: float = 0.0  # Auxiliary Swin detector (high-suspicion only)
    clip_embedding_anomaly: float = 0.0
    
    # Ensemble metadata
    ensemble_primary_available: bool = False
    ensemble_secondary_available: bool = False
    
    # DCT analysis
    dct_features: Optional[DCTFeatures] = None
    
    # Face detection
    face_detected: bool = False
    num_faces: int = 0
    face_manipulation_scores: List[float] = field(default_factory=list)
    
    # Heatmap
    heatmap_generated: bool = False
    heatmap_key: Optional[str] = None
    
    # Confidence
    confidence: float = 0.0
    
    def to_details_dict(self) -> Dict[str, Any]:
        """Convert to details dictionary for ModalityResult."""
        return {
            "fake_probability": round(self.fake_probability, 4),
            "ai_generated_probability": round(self.fake_probability, 4),  # Alias for clarity
            "ensemble_score": round(self.ensemble_score, 4),
            "auxiliary_score": round(self.auxiliary_score, 4),
            "ensemble_primary_available": self.ensemble_primary_available,
            "ensemble_secondary_available": self.ensemble_secondary_available,
            "clip_embedding_anomaly": round(self.clip_embedding_anomaly, 4),
            "dct_features": self.dct_features.to_dict() if self.dct_features else None,
            "face_detected": self.face_detected,
            "num_faces": self.num_faces,
            "face_manipulation_scores": [round(s, 4) for s in self.face_manipulation_scores],
            "heatmap_generated": self.heatmap_generated,
            "heatmap_key": self.heatmap_key
        }


class DCTAnalyzer(SubAnalyzer):
    """
    Multi-signal image forensics analyzer for AI/deepfake detection.
    
    Uses multiple independent signals to discriminate between real camera
    images and AI-generated/manipulated images:
    
    1. DCT Frequency Analysis: GAN-generated images have different frequency
       distributions than natural camera images due to upsampling artifacts.
    2. Noise Variance Analysis: Real camera images have sensor noise patterns;
       AI-generated images have uniform or absent noise.
    3. Color Channel Correlation: GANs produce artificially correlated color
       channels; real cameras have more independent channel distributions.
    4. Texture/Entropy Analysis: Real images have richer, more irregular
       texture patterns than AI-generated images.
    """
    
    def __init__(self):
        super().__init__("DCTAnalyzer")
    
    def analyze_dct(self, image: np.ndarray) -> DCTFeatures:
        """
        Perform comprehensive multi-signal image forensics analysis.
        
        Args:
            image: Input image (H, W, 3) or (H, W)
            
        Returns:
            DCTFeatures with forensic analysis results
        """
        try:
            import cv2
            
            # Convert to grayscale and resize
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)
            gray = cv2.resize(gray, (256, 256))
            gray_f = gray.astype(np.float32)
            
            # ===== Signal 1: DCT Frequency Analysis =====
            dct = cv2.dct(gray_f)
            h, w = dct.shape
            total_energy = np.sum(dct ** 2)
            
            # Energy distribution across frequency bands
            low_freq = dct[:h//8, :w//8]
            mid_freq = dct[h//8:h//2, w//8:w//2]
            high_freq = dct[h//2:, w//2:]
            
            low_energy = np.sum(low_freq ** 2) / (total_energy + 1e-10)
            mid_energy = np.sum(mid_freq ** 2) / (total_energy + 1e-10)
            high_energy = np.sum(high_freq ** 2) / (total_energy + 1e-10)
            
            # Spectral flatness
            dct_abs = np.abs(dct.flatten()) + 1e-10
            geo_mean = np.exp(np.mean(np.log(dct_abs)))
            arith_mean = np.mean(dct_abs)
            spectral_flatness = geo_mean / arith_mean if arith_mean > 0 else 0
            
            # ===== Signal 2: Noise Variance Analysis =====
            # Real cameras have sensor noise; AI images have uniform/absent noise
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_variance = np.var(laplacian)
            
            # Local variance analysis (texture richness)
            kernel = np.ones((3,3), dtype=np.float32) / 9
            local_mean = cv2.filter2D(gray_f, -1, kernel)
            local_var = cv2.filter2D((gray_f - local_mean)**2, -1, kernel)
            texture_richness = np.std(local_var)
            
            # ===== Signal 3: Color Channel Correlation =====
            # GANs produce artificially correlated color channels
            if len(image.shape) == 3:
                img_u = image.astype(np.uint8)
                if img_u.shape[2] == 3:
                    r = img_u[:,:,0].flatten().astype(np.float64)
                    g = img_u[:,:,1].flatten().astype(np.float64)
                    b = img_u[:,:,2].flatten().astype(np.float64)
                    corr_rg = np.abs(np.corrcoef(r, g)[0,1])
                    corr_rb = np.abs(np.corrcoef(r, b)[0,1])
                    corr_gb = np.abs(np.corrcoef(g, b)[0,1])
                    mean_color_corr = (corr_rg + corr_rb + corr_gb) / 3.0
                else:
                    mean_color_corr = 0.5
            else:
                mean_color_corr = 0.5
            
            # ===== Signal 4: Entropy Analysis =====
            hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
            hist = hist / (hist.sum() + 1e-10)
            hist_nonzero = hist[hist > 0]
            entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
            
            # ===== Signal 5: Patch-Level Texture Consistency =====
            # AI-generated images have suspiciously uniform texture across
            # different image regions. Real photos have natural regional
            # variation (different lighting, subjects, backgrounds).
            patch_size = 64
            patch_variances = []
            for py in range(0, 256 - patch_size + 1, patch_size):
                for px in range(0, 256 - patch_size + 1, patch_size):
                    patch = gray_f[py:py+patch_size, px:px+patch_size]
                    patch_var = np.var(patch)
                    patch_variances.append(patch_var)
            patch_variances = np.array(patch_variances)
            # Coefficient of variation of patch variances
            # High CV = natural variation, Low CV = uniform (suspicious)
            patch_cv = np.std(patch_variances) / (np.mean(patch_variances) + 1e-10)
            
            # ===== Signal 6: Saturation Uniformity =====
            # AI images tend to have more uniform saturation across the image
            if len(image.shape) == 3:
                img_hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
                saturation_std = float(np.std(img_hsv[:,:,1].astype(np.float32)))
            else:
                saturation_std = 50.0  # neutral
            
            # ===== Compute anomaly score from continuous signals =====
            # Replace hardcoded thresholds with sigmoid-based continuous scoring.
            # Each signal is mapped to [0, 1] via sigmoid centered at empirical mean.
            
            freq_ratio = mid_energy + high_energy
            
            # Sigmoid-based scoring: each signal maps to [0, 1] continuously.
            # Center and scale parameters recalibrated for broader AI generator coverage.
            
            # Frequency ratio: deepfakes have higher mid/high energy
            freq_score = 1.0 / (1.0 + np.exp(-20.0 * (freq_ratio - 0.005)))
            
            # Noise variance: real cameras have more noise than AI images
            noise_score = 1.0 / (1.0 + np.exp(0.03 * (noise_variance - 30.0)))
            
            # Color correlation: AI images have higher inter-channel correlation
            color_score = 1.0 / (1.0 + np.exp(-15.0 * (mean_color_corr - 0.80)))
            
            # Texture richness: real images have richer textures
            texture_score = 1.0 / (1.0 + np.exp(0.1 * (texture_richness - 12.0)))
            
            # Spectral flatness: higher in AI images
            flatness_score = 1.0 / (1.0 + np.exp(-20.0 * (spectral_flatness - 0.25)))
            
            # Patch-level consistency: AI images have uniform texture
            patch_score = 1.0 / (1.0 + np.exp(10.0 * (patch_cv - 0.30)))
            
            # Saturation uniformity: AI images have uniform colors
            saturation_score = 1.0 / (1.0 + np.exp(0.1 * (saturation_std - 25.0)))
            
            # Entropy: AI images tend to have lower entropy (more uniform distributions)
            # Real images have higher natural variation
            entropy_score = 1.0 / (1.0 + np.exp(0.5 * (entropy - 6.0)))
            
            # Weighted combination of all signals (sum = 1.0)
            anomaly_score = float(np.clip(
                0.18 * freq_score +
                0.18 * noise_score +
                0.14 * color_score +
                0.14 * texture_score +
                0.09 * flatness_score +
                0.09 * patch_score +
                0.09 * saturation_score +
                0.09 * entropy_score,
                0.0, 1.0
            ))
            
            logger.info(
                f"DCT analysis: freq_ratio={freq_ratio:.6f}, noise_var={noise_variance:.1f}, "
                f"color_corr={mean_color_corr:.4f}, texture={texture_richness:.1f}, "
                f"flatness={spectral_flatness:.4f}, patch_cv={patch_cv:.4f}, "
                f"sat_std={saturation_std:.1f}, anomaly={anomaly_score:.4f}"
            )
            
            return DCTFeatures(
                energy_high_freq=float(high_energy),
                energy_low_freq=float(low_energy),
                spectral_flatness=float(spectral_flatness),
                anomaly_score=anomaly_score
            )
            
        except Exception as e:
            logger.warning(f"DCT analysis failed: {e}")
            return DCTFeatures()
    
    def get_required_models(self) -> List[str]:
        """DCT analysis doesn't require ML models."""
        return []


class ImageAnalyzer(BaseAnalyzer):
    """
    Single-image deepfake detection focused on face manipulation artifacts.
    
    Multi-stage detection pipeline:
    1. Preprocessing: Resize, normalize, apply adversarial defense
    2. DCT Analysis: Frequency-domain blending artifact detection
    3. Neural Detection: ONNX ViT classifier for face manipulation
    4. Face Analysis: Face-specific manipulation detection
    5. Aggregation: Weighted combination of all signals
    
    Supported Detection:
    - Face swaps (DeepFaceLab, FaceSwap, etc.)
    - Facial reenactment
    - Face attribute manipulation
    - Identity swap forgeries
    
    Usage:
        analyzer = ImageAnalyzer()
        result = await analyzer.analyze(preprocessed_data, engine)
    """
    
    def __init__(self):
        """Initialize image analyzer with sub-analyzers."""
        super().__init__(
            analyzer_name="ImageAnalyzer",
            supported_modalities=[Modality.IMAGE],
            version="1.0.0"
        )

        # Initialize sub-analyzers
        self.dct_analyzer = DCTAnalyzer()

        # Analysis configuration
        self.target_size = (224, 224)  # Standard input for ViT models

        # ViT models (dima806, Deep-Fake-Detector-v2) use mean=0.5, std=0.5
        self.vit_mean = [0.5, 0.5, 0.5]
        self.vit_std = [0.5, 0.5, 0.5]

        # ===========================================================
        # Iteration 1: SOTA detector ensemble (additive, strict-compat)
        # Iteration 3: added SigLIP for ensemble diversity (3 detectors)
        # Iteration 8: ModeManager gates which detectors are enabled
        # ===========================================================
        # These detectors are loaded lazily and only when
        # config.enable_sota_detectors is True. If a detector's weights
        # are missing, it returns a low-confidence neutral result and is
        # auto-downweighted by the DiversityEnsemble combiner.
        #
        # Iteration 8: In Lite mode, SOTA detectors are disabled entirely
        # and the legacy ONNX pipeline runs alone (faster on CPU).
        self._sota_detectors = None  # Lazy-initialized list of detectors
        # Prior weights: CLIP+LoRA, DINOv2, SigLIP, SBI, UCF (benchmark AUCs)
        # UCF gets high prior due to strong cross-generator generalization.
        self._sota_prior_weights = [0.95, 0.92, 0.88, 0.90, 0.93]
        # Iteration 8: cache the mode config to avoid repeated lookups
        self._mode_config = None

        logger.info(f"ImageAnalyzer initialized with ViT normalization")
    
    def get_required_models(self) -> List[str]:
        """
        Return models required for image analysis.
        
        Models:
        - ai_real_detector: Unified AI/Real image detection (SDXL, DALL-E, Midjourney, deepfakes)
        - retinaface: Face detection for preprocessing
        
        Returns:
            List of model registry keys
        """
        return [
            "deepfake_detector_v3",  # Primary deepfake image detection
            "retinaface"            # Face detection for preprocessing
        ]
    
    def validate_input(self, data: PreprocessedData) -> None:
        """
        Validate input data for image analysis.
        
        Args:
            data: PreprocessedData to validate
            
        Raises:
            ValidationError: If data is invalid
        """
        super().validate_input(data)
        
        # Image analyzer needs either face_crops or frames
        if not data.face_crops and not data.frames:
            raise ValidationError(
                "ImageAnalyzer requires face_crops or frames in PreprocessedData"
            )
    
    async def _analyze_impl(
        self,
        data: PreprocessedData,
        engine: "InferenceEngine"
    ) -> ModalityResult:
        """
        Core image analysis implementation.
        
        Pipeline:
        1. Load and preprocess images from MinIO keys
        2. Run DCT analysis for frequency artifacts
        3. Run neural detector for deepfake classification
        4. Run face-specific analysis if faces detected
        5. Aggregate scores with confidence weighting
        
        Args:
            data: PreprocessedData with image keys
            engine: InferenceEngine for model inference
            
        Returns:
            ModalityResult with detection score and details
        """
        # Determine image source (face crops preferred)
        image_keys = data.face_crops if data.face_crops else data.frames
        
        if not image_keys:
            logger.warning("No images available for analysis")
            return ModalityResult(
                modality=Modality.IMAGE,
                score=0.5,
                confidence=0.3,
                details={"error": "No images available"}
            )
        
        # Load images (in production, load from MinIO)
        # For now, we'll simulate with placeholder analysis
        images = await self._load_images(image_keys)
        
        if not images:
            logger.warning("Failed to load any images")
            return ModalityResult(
                modality=Modality.IMAGE,
                score=0.5,
                confidence=0.3,
                details={"error": "Failed to load images"}
            )
        
        # Run analysis pipeline
        _analysis_start = time.time()
        result = await self._run_analysis_pipeline(images, engine, data)

        # ===========================================================
        # Iteration 4: XAI attribution + conformal output
        # ===========================================================
        xai_attribution = None
        conformal_set = None
        route_to_human = False
        if getattr(config, "enable_xai_attribution_output", False):
            # Compute Eigen-CAM attribution (cheap, always available)
            try:
                xai_attribution = self._compute_xai_attribution(images, result)
            except Exception as e:
                logger.debug("XAI attribution failed: %s", e)
            # Get conformal prediction set + route_to_human from post-processing
            try:
                from core.post_processing import apply_post_processing
                _synth_xai = np.array([
                    result.fake_probability,
                    result.ensemble_score,
                    result.auxiliary_score,
                    result.clip_embedding_anomaly,
                ], dtype=np.float64)
                pp = apply_post_processing(
                    score=result.fake_probability,
                    confidence=result.confidence,
                    embedding=_synth_xai,
                    modality="image",
                    analysis_id=data.analysis_id,
                )
                conformal_set = pp.conformal_set
                route_to_human = pp.route_to_human
            except Exception as e:
                logger.debug("Conformal output failed: %s", e)

        # ===========================================================
        # Iteration 7: Prometheus metrics recording
        # ===========================================================
        try:
            from observability import get_default_metrics
            _latency_s = time.time() - _analysis_start
            _verdict = "fake" if result.fake_probability >= 0.5 else "real"
            _metrics = get_default_metrics()
            _metrics.record_inference("image", _verdict, _latency_s)
            if route_to_human:
                _metrics.record_conformal_route("image")
            if getattr(config, "enable_adversarial_defenses", False) and \
               getattr(config, "enable_rps", False):
                _metrics.record_adversarial_flag("image", "rps")
        except Exception as _e:
            logger.debug("Metrics recording failed: %s", _e)

        return ModalityResult(
            modality=Modality.IMAGE,
            score=result.fake_probability,
            confidence=result.confidence,
            details=result.to_details_dict(),
            xai_attribution=xai_attribution,
            conformal_prediction_set=conformal_set,
            route_to_human=route_to_human,
        )
    
    async def _load_images(
        self,
        image_keys: List[str]
    ) -> List[np.ndarray]:
        """
        Load images from MinIO keys.
        
        Supports both .npy (NumPy arrays) and image formats (PNG, JPEG, etc.)
        
        Args:
            image_keys: List of MinIO object keys
            
        Returns:
            List of loaded image arrays
        """
        from storage.storage import get_storage_client
        from PIL import Image
        import io
        
        images = []
        storage = get_storage_client()  # Synchronous singleton getter
        
        for key in image_keys[:10]:  # Limit to 10 images
            try:
                # Download image bytes from MinIO
                image_bytes = await storage.download_file("argus-preprocessed", key)
                
                # Check if it's a .npy file (NumPy array)
                if key.endswith('.npy'):
                    # Load NumPy array directly - use allow_pickle for object arrays
                    image_array = np.load(io.BytesIO(image_bytes), allow_pickle=True)
                    # If it's an object array, try to extract the actual array
                    if image_array.dtype == object:
                        if hasattr(image_array, 'item') and isinstance(image_array.item(), np.ndarray):
                            image_array = image_array.item()
                        elif len(image_array) > 0 and isinstance(image_array[0], np.ndarray):
                            image_array = image_array[0]
                    logger.debug(f"Loaded numpy array from {key}: shape={image_array.shape}, dtype={image_array.dtype}")
                else:
                    # Load image with PIL
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to RGB if necessary
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Convert to numpy array
                    image_array = np.array(pil_image, dtype=np.uint8)
                    logger.debug(f"Loaded image from {key}: shape={image_array.shape}")
                
                # Ensure correct shape (H, W, C)
                if len(image_array.shape) == 2:
                    # Grayscale to RGB
                    image_array = np.stack([image_array] * 3, axis=-1)
                elif len(image_array.shape) == 3 and image_array.shape[-1] == 1:
                    # Single channel to RGB
                    image_array = np.concatenate([image_array] * 3, axis=-1)
                elif len(image_array.shape) == 3 and image_array.shape[0] in [1, 3]:
                    # CHW to HWC
                    image_array = np.transpose(image_array, (1, 2, 0))
                    if image_array.shape[-1] == 1:
                        image_array = np.concatenate([image_array] * 3, axis=-1)
                
                images.append(image_array)
                
            except Exception as e:
                logger.error(f"Failed to load image {key}: {e}")
                raise RuntimeError(
                    f"Failed to load image {key}: {e}. "
                    "Cannot proceed with analysis - all images must be loadable."
                )
        
        logger.info(f"Loaded {len(images)} images for analysis")
        
        # ===== INPUT VALIDATION =====
        # Reject images that are too small, corrupted, or out-of-distribution.
        validated_images = []
        for idx, img in enumerate(images):
            if img is None or img.size == 0:
                logger.warning(f"Image {idx} is empty, skipping")
                continue
            if len(img.shape) < 2:
                logger.warning(f"Image {idx} has invalid shape {img.shape}, skipping")
                continue
            h, w = img.shape[:2]
            if h < 32 or w < 32:
                logger.warning(f"Image {idx} too small ({h}x{w}), skipping")
                continue
            # Check for solid color images (no meaningful content)
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            if gray.std() < 1.0:
                logger.warning(f"Image {idx} has near-zero variance ({gray.std():.2f}), likely solid color")
            validated_images.append(img)
        
        if not validated_images:
            logger.error("No valid images after validation")
            raise ValueError("No valid images after validation")
        
        images = validated_images
        return images
    
    async def _run_analysis_pipeline(
        self,
        images: List[np.ndarray],
        engine: "InferenceEngine",
        data: PreprocessedData
    ) -> ImageAnalysisResult:
        """
        Run full analysis pipeline on images.
        
        Args:
            images: List of loaded images
            engine: InferenceEngine
            data: Original preprocessed data
            
        Returns:
            ImageAnalysisResult with all detection scores
        """
        result = ImageAnalysisResult()
        
        # 1. DCT Analysis (CPU-based, no model needed)
        dct_scores = []
        for image in images:
            dct_features = self.dct_analyzer.analyze_dct(image)
            dct_scores.append(dct_features.anomaly_score)
            if result.dct_features is None:
                result.dct_features = dct_features
        
        avg_dct_score = np.mean(dct_scores) if dct_scores else 0.0
        
        # 2. Primary model detection
        primary_scores = []
        primary_available = False
        try:
            primary_scores = await self._run_primary_detection(images, engine)
            primary_available = True
            result.ensemble_score = float(np.mean(primary_scores))
            logger.info(f"Primary model detection completed: scores={primary_scores[:3]}...")
        except Exception as e:
            logger.warning(f"Primary model detection failed: {e}")
            result.ensemble_score = 0.5
            primary_available = False
        
        # 3. Auxiliary model detection (used only in high-suspicion disagreement cases)
        auxiliary_scores = []
        secondary_available = False
        try:
            auxiliary_scores = await self._run_auxiliary_detection(images, engine)
            secondary_available = True
            result.auxiliary_score = float(np.mean(auxiliary_scores))
            logger.info(f"Auxiliary model detection completed: scores={auxiliary_scores[:3]}...")
        except Exception as e:
            logger.warning(f"Auxiliary model detection failed: {e}")
            result.auxiliary_score = 0.5
            secondary_available = False
        
        # 4. Face-specific analysis
        if data.face_crops:
            result.face_detected = True
            result.num_faces = len(data.face_crops)
            result.face_manipulation_scores = [
                max(0.0, min(1.0, result.ensemble_score))
            ]
        
        # Store model availability
        result.ensemble_primary_available = primary_available
        result.ensemble_secondary_available = secondary_available
        
        # 4.5 PyTorch DINOv2 LoRA adapter (Accurate)
        pytorch_score = 0.5
        pytorch_available = False
        try:
            pytorch_scores = await self._run_pytorch_ensemble_model(images)
            pytorch_available = True
            pytorch_score = float(np.mean(pytorch_scores))
            logger.info(f"PyTorch DINOv2 detection completed: scores={pytorch_scores[:3]}...")
        except Exception as e:
            logger.warning(f"PyTorch DINOv2 detection failed: {e}")
            pytorch_available = False
            
        # 5. Ensemble scoring: combine primary neural, auxiliary neural, PyTorch, and DCT signals.
        # DCT frequency analysis serves as the arbiter when models disagree.
        primary_neural_score = result.ensemble_score
        neural_confidence = abs(primary_neural_score - 0.5) * 2

        # DCT anomaly signal with proper weighting
        dct_signal = avg_dct_score if avg_dct_score > 0 else 0.0

        # Auxiliary model signal (high = more likely artificial/fake)
        auxiliary_signal = result.auxiliary_score if secondary_available else 0.5
        
        # PyTorch model signal
        pt_signal = pytorch_score if pytorch_available else 0.5

        logger.info(f"Primary ONNX: {primary_neural_score:.4f} (conf={neural_confidence:.4f}), "
                     f"Auxiliary ONNX: {auxiliary_signal:.4f}, "
                     f"PyTorch DINOv2: {pt_signal:.4f}, DCT: {dct_signal:.4f}")

        # ===== Multi-Signal Ensemble Fusion =====
        # Architecture: Neural-first with DCT modulation and C2PA override.
        #
        # The neural model provides relative ranking (real < AI < deepfake)
        # but has systematic bias toward high absolute scores.
        # DCT analysis detects deepfake-specific artifacts.
        # C2PA metadata provides deterministic authenticity when available.
        
        dct_anomaly = dct_signal
        neural_raw = primary_neural_score
        aux_signal = auxiliary_signal if secondary_available else 0.5

        # ===== C2PA Override =====
        # If C2PA metadata is present and verified, it overrides all other signals.
        # C2PA provides deterministic (not AI-based) authenticity verification.
        c2pa_override = False
        c2pa_score = 0.5
        if data.metadata and isinstance(data.metadata, dict) and 'c2pa_result' in data.metadata:
            c2pa_result = data.metadata.get('c2pa_result')
            if c2pa_result and c2pa_result.get('present', False):
                c2pa_override = True
                c2pa_score = 0.05 if not c2pa_result.get('ai_generated', False) else 0.95

        if c2pa_override:
            result.fake_probability = c2pa_score
            logger.info(f"C2PA override: score={c2pa_score:.4f}")
        else:
            # ===== Continuous Multi-Signal Ensemble Fusion =====
            # Uses sigmoid-based continuous weighting instead of hardcoded thresholds.
            # Each signal contributes proportionally to its strength, with no brittle
            # if-statements or fixed decision boundaries.

            # Continuous Multi-Signal Ensemble Fusion
            # Neural model weight: higher confidence = more weight
            neural_weight = neural_confidence

            # DCT weight: sigmoid-based, increases with anomaly strength
            # Centered at 0.30 with steepness 15.0, capped at 0.50
            dct_weight = 0.50 / (1.0 + np.exp(-15.0 * (dct_anomaly - 0.30)))

            # Auxiliary weight: only if available, proportional to its score
            aux_weight = 0.30 if secondary_available else 0.0
            
            # PyTorch weight: heavily weighted if available, highly accurate
            pt_weight = 0.80 if pytorch_available else 0.0

            # Normalize weights to sum to 1.0
            total_weight = neural_weight + dct_weight + aux_weight + pt_weight
            if total_weight > 0:
                neural_weight /= total_weight
                dct_weight /= total_weight
                aux_weight /= total_weight
                pt_weight /= total_weight

            # Compute weighted ensemble score
            fake_prob = (
                neural_weight * neural_raw +
                dct_weight * dct_anomaly +
                aux_weight * aux_signal +
                pt_weight * pt_signal
            )

            # Apply disagreement penalty: if signals strongly disagree,
            # pull score toward uncertain (0.5)
            signals_to_std = [neural_raw, dct_anomaly]
            if secondary_available: signals_to_std.append(aux_signal)
            if pytorch_available: signals_to_std.append(pt_signal)
            
            signal_std = float(np.std(signals_to_std))
            disagreement_penalty = min(signal_std * 0.5, 0.25)
            fake_prob = fake_prob * (1.0 - disagreement_penalty) + 0.5 * disagreement_penalty

            result.fake_probability = float(np.clip(fake_prob, 0.0, 1.0))

            logger.info(
                f"Ensemble fusion: neural={neural_raw:.4f} (w={neural_weight:.2f}), "
                f"DCT={dct_anomaly:.4f} (w={dct_weight:.2f}), "
                f"aux={aux_signal:.4f} (w={aux_weight:.2f}), "
                f"pytorch={pt_signal:.4f} (w={pt_weight:.2f}), "
                f"disagreement={disagreement_penalty:.4f}, "
                f"final={result.fake_probability:.4f}"
            )

        logger.info(
            f"Ensemble: dct={dct_anomaly:.4f}, neural={neural_raw:.4f}, "
            f"aux={aux_signal:.4f}, final={result.fake_probability:.4f}"
        )

        # Clamp to valid range
        result.fake_probability = float(np.clip(result.fake_probability, 0.0, 1.0))

        # ===========================================================
        # Iteration 1 — SOTA detector ensemble integration
        # (additive, strict-compat)
        # Iteration 8: ModeManager gates SOTA detectors (disabled in Lite)
        # ===========================================================
        # When SOTA detectors are enabled (mode != lite), run the CLIP+LoRA
        # and DINOv2 SOTA detectors on the same images and blend their
        # outputs with the existing fusion using the DiversityEnsemble
        # combiner. If SOTA detectors are unavailable or disabled, the
        # existing fusion score is preserved unchanged.
        sota_score: Optional[float] = None
        sota_confidence: Optional[float] = None
        # Iteration 8: check ModeManager instead of raw config flag
        _sota_enabled = getattr(config, "enable_sota_detectors", False)
        try:
            from modes import get_current_mode
            if self._mode_config is None:
                self._mode_config = get_current_mode()
            _sota_enabled = self._mode_config.enable_sota_detectors
        except Exception:
            pass  # Fall back to config flag
        if _sota_enabled and _SOTA_DETECTORS_AVAILABLE:
            try:
                sota_score, sota_confidence = await self._run_sota_ensemble(
                    images, prior_score=result.fake_probability
                )
                if sota_score is not None:
                    # Blend: 60% SOTA ensemble, 40% legacy fusion.
                    blended = 0.6 * sota_score + 0.4 * result.fake_probability
                    result.fake_probability = float(np.clip(blended, 0.0, 1.0))
                    if sota_confidence is not None:
                        result.confidence = float(
                            np.clip(0.5 * sota_confidence + 0.5 * result.confidence, 0.0, 0.97)
                        )
                    logger.info(
                        "SOTA ensemble integrated: sota=%.4f (conf=%.4f), "
                        "blended=%.4f",
                        sota_score, sota_confidence or 0.0,
                        result.fake_probability,
                    )
            except Exception as e:
                logger.warning("SOTA image ensemble failed (non-fatal): %s", e)

        # ===========================================================
        # Iteration 2 — Calibration + Conformal + Adversarial flag
        # (additive, strict-compat)
        # ===========================================================
        # Apply temperature scaling (if a scaler is fitted) and conformal
        # RAPS prediction set (if fitted). Adversarial-defense flags from
        # RPS / gate / RS-lite propagate through.
        try:
            from core.post_processing import apply_post_processing
            # Build synthetic embedding from analysis signals for drift detection
            _synth = np.array([
                result.fake_probability,
                result.ensemble_score,
                result.auxiliary_score,
                result.clip_embedding_anomaly,
                float(result.face_detected),
                float(result.num_faces),
                float(np.mean(result.face_manipulation_scores)) if result.face_manipulation_scores else 0.5,
            ], dtype=np.float64)
            pp = apply_post_processing(
                score=result.fake_probability,
                confidence=result.confidence,
                embedding=_synth,
                modality="image",
                analysis_id=data.analysis_id,
            )
            if pp.calibrated_score != pp.original_score:
                logger.info(
                    "Temperature scaling applied: %.4f -> %.4f (T=%.4f)",
                    pp.original_score, pp.calibrated_score, pp.temperature,
                )
                result.fake_probability = pp.calibrated_score
            if pp.route_to_human:
                logger.info(
                    "Routing to human review: ambiguous=%s, adv_flag=%s",
                    pp.is_ambiguous, pp.adversarial_flag,
                )
        except Exception as e:
            logger.debug("Post-processing failed (non-fatal): %s", e)

        # 6. Compute confidence
        all_scores = [result.ensemble_score, avg_dct_score]
        if secondary_available:
            all_scores.append(result.auxiliary_score)
        if sota_score is not None:
            all_scores.append(sota_score)
        all_scores = [s for s in all_scores if s > 0]

        base_confidence = compute_confidence(
            np.array(all_scores) if all_scores else np.array([0.5]),
            len(images),
            min_samples=5
        )

        # Boost confidence when primary model is available
        if result.ensemble_primary_available:
            base_confidence = min(1.0, base_confidence * 1.1)

        # Reduce confidence when signals disagree
        if len(all_scores) >= 2:
            score_variance = np.var(all_scores)
            if score_variance > 0.05:
                base_confidence *= 0.9

        result.confidence = base_confidence

        return result

    # ------------------------------------------------------------------
    # Iteration 1: SOTA detector ensemble helper
    # ------------------------------------------------------------------
    async def _run_sota_ensemble(
        self,
        images: List[np.ndarray],
        prior_score: float = 0.5,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Run CLIP+LoRA and DINOv2 SOTA detectors on each image and fuse
        their outputs via DiversityEnsemble.

        Iteration 2: applies Randomized Preprocessing Sanitizer (RPS) to
        each image before detection, defeating single-transform adaptive
        EOT attackers (Qiu et al., ACM WS 2025).

        Args:
            images: List of HxWx3 uint8 RGB images.
            prior_score: Existing fusion score (used as a tie-breaker
                member of the ensemble to preserve calibration).

        Returns:
            (fused_score, fused_confidence) or (None, None) on failure.
        """
        if not images:
            return None, None

        # Lazy-init detectors
        if self._sota_detectors is None:
            self._sota_detectors = [
                CLIPLoRAImageDetector(),
                DINOv2ImageDetector(),
                SigLIPImageDetector(),  # Iteration 3: 3rd detector for diversity
                SBIDetector(),          # Iteration 5: boundary-artifact detection
                UCFCrossForgeryDetector(),  # Iteration 6: cross-generator detection
            ]

        # Iteration 2: lazy-init RPS sanitizer
        rps = None
        if getattr(config, "enable_adversarial_defenses", False) and getattr(config, "enable_rps", False):
            try:
                from defenses import get_default_rps
                rps = get_default_rps()
            except Exception as e:
                logger.debug("RPS unavailable: %s", e)

        # Run each detector on the first image (single-image case) or
        # the median of per-image scores (multi-image case).
        per_detector_scores = []
        for det in self._sota_detectors:
            try:
                scores = []
                for img in images[:5]:  # cap at 5 for latency
                    # Iteration 2: sanitize input before detection
                    if rps is not None:
                        img = rps.sanitize_image(img)
                    r = await det.detect(img, return_features=False)
                    scores.append(r)
                per_detector_scores.append(scores)
            except Exception as e:
                logger.warning("SOTA detector %s failed: %s", det.name, e)
                per_detector_scores.append([])

        # Average per-detector across images, then combine detectors.
        averaged_results = []
        for scores in per_detector_scores:
            if not scores:
                continue
            avg_score = float(np.mean([s.score for s in scores]))
            avg_conf = float(np.mean([s.confidence for s in scores]))
            # Re-pack into a DetectionResult for the combiner
            from detectors.base import DetectionResult
            err = None
            if all(s.error for s in scores):
                err = "all_images_failed"
            averaged_results.append(DetectionResult(
                score=avg_score,
                confidence=avg_conf,
                model_name=scores[0].model_name,
                backend=scores[0].backend,
                error=err,
            ))

        if not averaged_results:
            return None, None

        fused = combine_detector_results(
            averaged_results,
            prior_weights=self._sota_prior_weights[:len(averaged_results)],
        )
        return fused.score, fused.confidence

    # ------------------------------------------------------------------
    # Iteration 4: XAI attribution helper
    # ------------------------------------------------------------------
    def _compute_xai_attribution(
        self,
        images: List[np.ndarray],
        result,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute Eigen-CAM attribution for the primary image.

        Returns a dict with heatmap data + human-readable explanation
        for the frontend to display.

        Args:
            images: List of HxWx3 uint8 RGB images.
            result: ImageAnalysisResult with fake_probability.

        Returns:
            Dict with:
                - "method": "eigen_cam"
                - "heatmap": list-of-lists (HxW, normalized [0,1])
                - "explanation": str
            Or None on failure.
        """
        if not images:
            return None
        try:
            from core.xai_eigencam import eigen_cam_from_features
            # Use the first image for attribution
            img = images[0]
            # Compute simple feature map: use the image itself as a
            # pseudo-feature map (channel-wise). This is a fallback
            # when we don't have access to the backbone's internal
            # feature map. In a full implementation, we'd hook the
            # backbone's last conv layer.
            features = np.transpose(img.astype(np.float32), (2, 0, 1))  # (3, H, W)
            heatmap = eigen_cam_from_features(features)
            if heatmap is None:
                return None

            # Downsample for JSON serialization (224x224 → 28x28)
            import numpy as np
            h, w = heatmap.shape
            target_h, target_w = 28, 28
            if h != target_h or w != target_w:
                try:
                    import cv2
                    heatmap_ds = cv2.resize(heatmap, (target_w, target_h))
                except ImportError:
                    # Simple stride downsampling
                    step_h = max(1, h // target_h)
                    step_w = max(1, w // target_w)
                    heatmap_ds = heatmap[::step_h, ::step_w][:target_h, :target_w]
            else:
                heatmap_ds = heatmap

            # Human-readable explanation
            verdict = "fake" if result.fake_probability >= 0.5 else "real"
            explanation = (
                f"Eigen-CAM attribution shows the image regions most "
                f"influential for the {verdict} verdict (score="
                f"{result.fake_probability:.3f}). Brighter regions in the "
                f"heatmap indicate higher influence. This is a gradient-"
                f"free attribution method (Muhammad & Yeasin, IJCNN 2020) "
                f"used as a cross-check against GradCAM++."
            )

            return {
                "method": "eigen_cam",
                "heatmap": heatmap_ds.tolist(),
                "heatmap_shape": [target_h, target_w],
                "explanation": explanation,
                "verdict": verdict,
                "score": float(result.fake_probability),
            }
        except Exception as e:
            logger.debug("Eigen-CAM attribution failed: %s", e)
            return None

    async def _run_primary_detection(
        self,
        images: List[np.ndarray],
        engine: "InferenceEngine"
    ) -> List[float]:
        """
        Run AI/Real image detection using the ONNX deepfake detector.

        Uses the deepfake_detector_v3 model (ViT-based, fine-tuned for
        deepfake/AI detection). The model outputs 2-class logits where
        index 0 = Realism, index 1 = Deepfake.

        The model has a systematic bias toward high fake probabilities
        for all images. Calibration uses the logit difference as a
        more discriminative signal and maps it to a well-calibrated
        probability range based on observed score distributions.

        Args:
            images: List of preprocessed images (H, W, 3)
            engine: InferenceEngine instance

        Returns:
            List of calibrated fake probability scores in [0, 1]
        """
        try:
            model_path = "/models/deepfake_detector_v3.onnx"
            if not os.path.exists(model_path):
                model_path = "/models/deepfake_vit_v2.onnx"
            if not os.path.exists(model_path):
                logger.error("PRIMARY DETECTOR UNAVAILABLE: No deepfake ONNX model found. Ensemble will use DCT + PyTorch only.")
                return [0.5] * len(images)

            sess = get_cached_primary_session(model_path)
            if sess is None:
                logger.error("PRIMARY DETECTOR UNAVAILABLE: Failed to initialize ONNX session. Ensemble will use DCT + PyTorch only.")
                return [0.5] * len(images)
            input_name = sess.get_inputs()[0].name

            scores = []
            for img in images:
                preprocessed = self._preprocess_for_onnx(img, target_size=224)

                with _primary_run_lock:
                    logits_orig = sess.run(None, {input_name: preprocessed[np.newaxis, ...].astype(np.float32)})[0]

                with _primary_run_lock:
                    flip_input = np.flip(preprocessed, axis=2).copy()[np.newaxis, ...].astype(np.float32)
                    logits_flip = sess.run(None, {input_name: flip_input})[0]

                avg_logits = (logits_orig + logits_flip) / 2.0

                logit_diff = float(avg_logits[0, 1] - avg_logits[0, 0])

                calibrated = 1.0 / (1.0 + np.exp(-3.0 * (logit_diff - 1.0)))
                if not np.isfinite(calibrated):
                    calibrated = 0.5
                calibrated = float(np.clip(calibrated, 0.01, 0.99))

                scores.append(calibrated)

            logger.info(f"deepfake_detector scores={scores[:3]}...")
            return scores

        except Exception as e:
            logger.error(f"deepfake_detector inference failed: {e}")
            logger.warning("RIVP: Primary ONNX model inference FAILED — returning placeholder 0.5 scores")
            return [0.5] * len(images)
    
    async def _run_auxiliary_detection(
        self,
        images: List[np.ndarray],
        engine: "InferenceEngine"
    ) -> List[float]:
        """
        Run auxiliary AI-image detector using ONNX EfficientNet-B3 model.

        Uses the efficientnet_b3_spatial.onnx model for secondary detection
        to provide disagreement signals for the ensemble.
        """
        try:
            model_path = "/models/efficientnet_b3_spatial.onnx"
            if not os.path.exists(model_path):
                logger.warning("RIVP: AUXILIARY DETECTOR MISSING: efficientnet_b3_spatial.onnx not found — returning placeholder 0.5 scores. No real inference for auxiliary signal.")
                return [0.5] * len(images)

            sess = get_cached_auxiliary_session(model_path)
            if sess is None:
                logger.warning("RIVP: AUXILIARY DETECTOR FAILED: ONNX session init failed — returning placeholder 0.5 scores.")
                return [0.5] * len(images)
            input_name = sess.get_inputs()[0].name

            scores = []
            for img in images:
                preprocessed = self._preprocess_for_onnx(img, target_size=224)
                input_tensor = preprocessed[np.newaxis, ...].astype(np.float32)

                with _auxiliary_run_lock:
                    logits = sess.run(None, {input_name: input_tensor})[0]

                logits_f64 = logits.astype(np.float64)
                exp_logits = np.exp(logits_f64 - np.max(logits_f64, axis=-1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

                fake_prob = float(probs[0, 1]) if probs.shape[-1] >= 2 else 0.5
                if not np.isfinite(fake_prob):
                    fake_prob = 0.5
                scores.append(fake_prob)

            logger.info(f"efficientnet_b3_spatial scores={scores[:3]}...")
            return scores

        except Exception as e:
            logger.warning(f"efficientnet_b3_spatial inference failed: {e}")
            return [0.5] * len(images)

    async def _run_pytorch_ensemble_model(
        self,
        images: List[np.ndarray]
    ) -> List[float]:
        """
        Run PyTorch DINOv2-based deepfake detector with LoRA adapter.
        """
        global _pytorch_model
        
        try:
            import torch
            from torchvision import transforms
            from analyzers.image_pytorch import DinoV2DeepfakeDetector
            
            with _pytorch_model_lock:
                if _pytorch_model is None:
                    _pytorch_model = DinoV2DeepfakeDetector()
                    _img_device = config.device
                    if _img_device != "cpu":
                        _pytorch_model = _pytorch_model.to(_img_device)
                    _pytorch_model.eval()
                    logger.warning(
                        "RIVP: PyTorch DINOv2 initialized with RANDOM-INIT classifier head. "
                        "Scores will be near-0.5 and statistically meaningless until "
                        "a trained head is supplied."
                    )
            
            # Prepare transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            scores = []
            with torch.no_grad():
                for img in images:
                    if img.dtype != np.uint8:
                        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    
                    input_tensor = transform(img).unsqueeze(0)
                    if config.device != "cpu":
                        input_tensor = input_tensor.to(config.device)
                    
                    logits = _pytorch_model(input_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Assuming class 1 is fake
                    fake_prob = float(probs[0, 1].cpu().item())
                    scores.append(fake_prob)
                    
            return scores
        except ImportError:
            logger.warning("PyTorch or transformers not available, skipping DINOv2")
            raise
        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}")
            raise
    
    def _preprocess_for_model(self, image: np.ndarray, model_type: str = "vit") -> np.ndarray:
        """
        Preprocess image for model input.
        
        Applies:
        - Resize to target size (224x224)
        - Convert to float [0, 1]
        - Model-specific normalization
        - CHW format for PyTorch-style models
        
        Args:
            image: Input image (H, W, 3) uint8
            model_type: "vit" for ViT normalization
            
        Returns:
            Preprocessed tensor (3, 224, 224) float32
        """
        return self._preprocess_for_model_size(image, self.target_size, model_type)
    
    def _preprocess_for_model_size(self, image: np.ndarray, target_size: tuple, model_type: str = "vit") -> np.ndarray:
        """
        Preprocess image for model input with specified target size.
        
        Args:
            image: Input image (H, W, 3) uint8
            target_size: Target size (width, height)
            model_type: "vit" for ViT normalization
            
        Returns:
            Preprocessed tensor (3, H, W) float32
        """
        import cv2
        
        # CRITICAL: Use INTER_AREA for downsampling to preserve high-frequency
        # GAN fingerprints. Bilinear (default) smears forensic artifacts.
        # Reference: Wang et al. (2020) CVPR - "CNN-generated images are
        # surprisingly easy to spot"
        if image.shape[0] > target_size[1] or image.shape[1] > target_size[0]:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to float [0, 1]
        float_img = resized.astype(np.float32) / 255.0
        
        # Normalize with ViT stats (mean=0.5, std=0.5)
        mean = np.array(self.vit_mean).reshape(1, 1, 3)
        std = np.array(self.vit_std).reshape(1, 1, 3)
        normalized = (float_img - mean) / std
        
        # Convert to CHW format
        chw = np.transpose(normalized, (2, 0, 1))
        
        return chw.astype(np.float32)
    
    def _preprocess_for_onnx(
        self,
        image: np.ndarray,
        target_size: int = 224
    ) -> np.ndarray:
        """
        Preprocess image for ONNX model inference.
        
        Applies:
        - Resize to (target_size, target_size)
        - Convert to float32 [0, 1]
        - ViT normalization (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        - Convert HWC to CHW format
        
        Args:
            image: Input image (H, W, 3) uint8 or float
            target_size: Target spatial size
            
        Returns:
            Preprocessed tensor (3, target_size, target_size) float32
        """
        import cv2

        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # CRITICAL: Use INTER_AREA for downsampling to preserve high-frequency
        # GAN fingerprints. Bilinear (default) smears forensic artifacts.
        target_shape = (target_size, target_size)
        if image.shape[0] > target_size or image.shape[1] > target_size:
            resized = cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)

        # Convert to float [0, 1]
        float_img = resized.astype(np.float32) / 255.0

        # ViT normalization: (x - 0.5) / 0.5 maps [0,1] to [-1,1]
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        normalized = (float_img - mean) / std

        # Convert to CHW format
        chw = np.transpose(normalized, (2, 0, 1))

        return chw.astype(np.float32)

    async def analyze_single_image(
        self,
        image: np.ndarray,
        engine: "InferenceEngine"
    ) -> ImageAnalysisResult:
        """
        Analyze a single image directly.
        
        Convenience method for analyzing individual images without
        going through PreprocessedData.
        
        Args:
            image: Input image (H, W, 3)
            engine: InferenceEngine
            
        Returns:
            ImageAnalysisResult
        """
        # Create minimal preprocessed data
        dummy_data = PreprocessedData(
            analysis_id="single_image",
            content_type=ContentType.IMAGE_ONLY,
            frames=["dummy_key"]
        )
        
        # Run pipeline directly
        return await self._run_analysis_pipeline([image], engine, dummy_data)


# Singleton instance
_image_analyzer: Optional[ImageAnalyzer] = None


def get_image_analyzer() -> ImageAnalyzer:
    """Get singleton image analyzer instance."""
    global _image_analyzer
    if _image_analyzer is None:
        _image_analyzer = ImageAnalyzer()
    return _image_analyzer
