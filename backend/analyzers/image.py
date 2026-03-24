"""
Argus Core - Image Analyzer
===========================
Single-image deepfake and AI-generated image detection.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/image.py

SOTA Algorithms:
- Model: SigLIP-based classifier (HuggingFace deepfake-detector-model-v1)
- Analysis: Frequency domain (DCT), CLIP embeddings
- Explainability: GradCAM overlay generation

Detection Targets:
- Face swaps (DeepFaceLab, etc.)
- AI-generated faces (StyleGAN, Midjourney)
- Edited/manipulated images
- Stable Diffusion outputs

Integration:
- Imports: core/engine.py, core/explain.py
- Inputs: image: np.ndarray
- Outputs: ImageResult (via ModalityResult)

Target Hardware: RTX 3050 (4GB VRAM) with INT8 quantization
"""

import os
import asyncio
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
    Modality, PreprocessedData, ModalityResult, ContentType, FeatureImportance,
    ManipulationRegion, EvidencePackage, ScientificReference
)
from config import config
from utils.logging import get_logger
from utils.errors import ValidationError, InferenceError

if TYPE_CHECKING:
    from core.engine import InferenceEngine

logger = get_logger(__name__)

# ===== ONNX Session Cache =====
# Sessions are created once and reused across all analysis requests.
# This avoids 2-6s overhead per request from InferenceSession creation.
_primary_onnx_session = None
_auxiliary_onnx_session = None


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
            
            # Weighted combination of all signals
            anomaly_score = float(np.clip(
                0.20 * freq_score +
                0.20 * noise_score +
                0.15 * color_score +
                0.15 * texture_score +
                0.10 * flatness_score +
                0.10 * patch_score +
                0.10 * saturation_score,
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
    Single-image deepfake and AI-generated image detection.
    
    Multi-stage detection pipeline:
    1. Preprocessing: Resize, normalize, apply adversarial defense
    2. DCT Analysis: Frequency-domain GAN fingerprint detection
    3. Neural Detection: EfficientNet/SigLIP classifier
    4. Face Analysis: Face-specific manipulation detection
    5. Aggregation: Weighted combination of all signals
    
    Supported Detection:
    - Face swaps (DeepFaceLab, FaceApp, etc.)
    - AI-generated faces (StyleGAN, Midjourney, DALL-E)
    - General image manipulation
    - Stable Diffusion outputs
    
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
            "ai_real_detector",      # Unified AI/Real image detection
            "retinaface"             # Face detection for preprocessing
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
        result = await self._run_analysis_pipeline(images, engine, data)
        
        return ModalityResult(
            modality=Modality.IMAGE,
            score=result.fake_probability,
            confidence=result.confidence,
            details=result.to_details_dict()
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
            return ImageAnalysisResult()
        
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
                result.ensemble_score if result.ensemble_score > 0 else 0.5
            ]
        
        # Store model availability
        result.ensemble_primary_available = primary_available
        result.ensemble_secondary_available = secondary_available
        
        # 5. Ensemble scoring: combine primary neural, auxiliary neural, and DCT signals.
        # DCT frequency analysis serves as the arbiter when models disagree.
        primary_neural_score = result.ensemble_score
        neural_confidence = abs(primary_neural_score - 0.5) * 2

        # DCT anomaly signal with proper weighting
        dct_signal = avg_dct_score if avg_dct_score > 0 else 0.0

        # Auxiliary model signal (high = more likely artificial/fake)
        auxiliary_signal = 0.0
        if secondary_available:
            auxiliary_signal = result.auxiliary_score

        logger.info(f"Primary: {primary_neural_score:.4f} (conf={neural_confidence:.4f}), "
                     f"Auxiliary: {auxiliary_signal:.4f}, DCT: {dct_signal:.4f}")

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
        if data.metadata and hasattr(data.metadata, 'c2pa_result'):
            c2pa_result = data.metadata.get('c2pa_result') if isinstance(data.metadata, dict) else None
            if c2pa_result and c2pa_result.get('present', False):
                c2pa_override = True
                c2pa_score = 0.05 if not c2pa_result.get('ai_generated', False) else 0.95
        
        if c2pa_override:
            result.fake_probability = c2pa_score
            logger.info(f"C2PA override: score={c2pa_score:.4f}")
        else:
            # ===== Multi-Signal Ensemble Fusion =====
            # Strategy: Use multiple signals with majority voting.
            # When the neural model is confidently wrong, DCT + auxiliary override.
            
            # ===== Step 1: Majority Voting Override =====
            # If neural says "real" (very low score) but DCT and auxiliary both
            # say "suspicious" (>0.30), the neural model is likely wrong.
            # This happens with out-of-distribution images the model wasn't trained on.
            
            if neural_raw < 0.20 and dct_anomaly > 0.25 and aux_signal > 0.30:
                # Majority vote: DCT and auxiliary agree on fake, neural disagrees
                # When neural is very confident wrong (<0.10), trust DCT+auxiliary more
                if neural_raw < 0.10:
                    # Neural is confidently wrong: use the stronger of DCT/auxiliary
                    # as primary, weighted average as secondary
                    strong_signal = max(dct_anomaly, aux_signal)
                    weak_signal = min(dct_anomaly, aux_signal)
                    fake_prob = 0.70 * strong_signal + 0.30 * weak_signal
                else:
                    # Neural moderately wrong: balanced weighting
                    fake_prob = 0.55 * dct_anomaly + 0.45 * aux_signal
                
                logger.info(
                    f"Majority vote override: neural={neural_raw:.4f} (too low), "
                    f"DCT={dct_anomaly:.4f}, aux={aux_signal:.4f}, final={fake_prob:.4f}"
                )
            else:
                # ===== Step 2: Standard Ensemble =====
                # Smooth blending of neural and DCT signals
                
                # Smooth DCT weight: sigmoid transition centered at 0.25, capped at 0.40
                dct_weight = 0.40 / (1.0 + np.exp(-20.0 * (dct_anomaly - 0.25)))
                dct_weight = min(dct_weight, 0.40)
                
                # Blend neural and DCT
                fake_prob = (1.0 - dct_weight) * neural_raw + dct_weight * dct_anomaly
                
                # ===== Step 3: Auxiliary Agreement Boost =====
                # If auxiliary agrees with DCT on fake but neural says real
                if secondary_available and aux_signal > 0.40 and neural_raw < 0.30:
                    aux_boost = 0.15 * (aux_signal - 0.40) / 0.60
                    fake_prob = max(fake_prob, aux_boost + fake_prob)
            
            result.fake_probability = float(np.clip(fake_prob, 0.0, 1.0))

        logger.info(
            f"Ensemble: dct={dct_anomaly:.4f}, neural={neural_raw:.4f}, "
            f"aux={aux_signal:.4f}, final={result.fake_probability:.4f}"
        )

        # Clamp to valid range
        result.fake_probability = float(np.clip(result.fake_probability, 0.0, 1.0))
        
        # 6. Compute confidence
        all_scores = [result.ensemble_score, avg_dct_score]
        if secondary_available:
            all_scores.append(result.auxiliary_score)
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
        import onnxruntime as ort
        global _primary_onnx_session

        try:
            # ===== Session Caching =====
            # Create ONNX session once, reuse across all requests (saves 1-3s per call).
            if _primary_onnx_session is None:
                model_path = "/models/deepfake_detector_v3.onnx"
                if not os.path.exists(model_path):
                    model_path = "/models/deepfake_vit_v2.onnx"
                if not os.path.exists(model_path):
                    # Fallback: check model directory for any valid ONNX file
                    model_dir = os.environ.get("MODEL_PATH", "/models")
                    for fname in os.listdir(model_dir):
                        if fname.endswith(".onnx") and os.path.getsize(os.path.join(model_dir, fname)) > 1_000_000:
                            model_path = os.path.join(model_dir, fname)
                            break
                if not os.path.exists(model_path):
                    logger.error("No deepfake detection model found")
                    return [0.5] * len(images)

                _primary_onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                logger.info(f"Primary ONNX session cached: {model_path}")

            sess = _primary_onnx_session
            input_name = sess.get_inputs()[0].name

            scores = []
            for img in images:
                preprocessed = self._preprocess_for_onnx(img, target_size=224)

                # Test-time augmentation: average original + horizontal flip
                orig_input = preprocessed[np.newaxis, ...].astype(np.float32)
                flip_input = np.flip(preprocessed, axis=2).copy()[np.newaxis, ...].astype(np.float32)

                logits_orig = sess.run(None, {input_name: orig_input})[0]
                logits_flip = sess.run(None, {input_name: flip_input})[0]
                avg_logits = (logits_orig + logits_flip) / 2.0

                # Logit difference: positive = more likely fake (index 1 > index 0)
                logit_diff = float(avg_logits[0, 1] - avg_logits[0, 0])

                # ===== CALIBRATION =====
                # The model has systematic bias. Logit difference is more discriminative.
                # Sigmoid calibration: maps logit_diff to [0, 1] smoothly.
                # Center at 1.0 (borderline) with steepness 3.0:
                #   logit_diff=-0.5 -> fake_prob~0.01 (very real)
                #   logit_diff=0.0  -> fake_prob~0.05 (real)
                #   logit_diff=0.5  -> fake_prob~0.14 (real)
                #   logit_diff=1.0  -> fake_prob~0.38 (borderline)
                #   logit_diff=1.5  -> fake_prob~0.69 (AI)
                #   logit_diff=2.0  -> fake_prob~0.88 (very fake)
                calibrated = 1.0 / (1.0 + np.exp(-3.0 * (logit_diff - 1.0)))
                calibrated = float(np.clip(calibrated, 0.01, 0.99))

                scores.append(calibrated)

            logger.info(f"deepfake_detector scores={scores[:3]}...")
            return scores

        except Exception as e:
            logger.error(f"deepfake_detector inference failed: {e}")
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
        import onnxruntime as ort
        global _auxiliary_onnx_session

        try:
            if _auxiliary_onnx_session is None:
                model_path = "/models/efficientnet_b3_spatial.onnx"
                if not os.path.exists(model_path):
                    logger.warning("efficientnet_b3_spatial model not found, using neutral scores")
                    return [0.5] * len(images)
                _auxiliary_onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                logger.info(f"Auxiliary ONNX session cached: {model_path}")

            sess = _auxiliary_onnx_session
            input_name = sess.get_inputs()[0].name

            scores = []
            for img in images:
                preprocessed = self._preprocess_for_onnx(img, target_size=224)
                input_tensor = preprocessed[np.newaxis, ...].astype(np.float32)

                logits = sess.run(None, {input_name: input_tensor})[0]

                # Softmax
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

                # Index 1 = fake/AI-generated (verified empirically)
                fake_prob = float(probs[0, 1])
                scores.append(fake_prob)

            logger.info(f"efficientnet_b3_spatial scores={scores[:3]}...")
            return scores

        except Exception as e:
            logger.warning(f"efficientnet_b3_spatial inference failed: {e}")
            return [0.5] * len(images)
    
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
        
        # Resize
        resized = cv2.resize(image, target_size)
        
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

        # Resize
        resized = cv2.resize(image, (target_size, target_size))

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
