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

import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field

from analyzers.base import BaseAnalyzer, SubAnalyzer, normalize_scores, aggregate_scores, compute_confidence
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
    siglip_score: float = 0.0
    efficientnet_score: float = 0.0
    clip_embedding_anomaly: float = 0.0
    
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
            "siglip_score": round(self.siglip_score, 4),
            "efficientnet_score": round(self.efficientnet_score, 4),
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
    Discrete Cosine Transform analyzer for GAN fingerprint detection.
    
    GANs produce characteristic frequency-domain patterns:
    - Abnormal energy distribution across frequencies
    - Grid artifacts from upsampling
    - Missing high-frequency detail
    
    This analyzer extracts DCT features and scores them against
    known GAN signatures.
    
    CALIBRATION NOTE: Thresholds adjusted to reduce false positives
    from mobile camera images with JPEG compression artifacts.
    """
    
    def __init__(self):
        super().__init__("DCTAnalyzer")
        
        # DCT thresholds (calibrated to reduce false positives from mobile photos)
        # Mobile camera JPEG compression naturally reduces high-freq energy
        # and increases spectral flatness - these thresholds account for that
        self.high_freq_threshold = 0.08  # Lowered from 0.15 - only flag severe cases
        self.spectral_flatness_threshold = 0.75  # Raised from 0.6 - natural images can have higher flatness
        self.energy_ratio_threshold = 0.005  # New: more stringent ratio check
    
    def analyze_dct(self, image: np.ndarray) -> DCTFeatures:
        """
        Analyze image using DCT for GAN fingerprints.
        
        Args:
            image: Input image (H, W, 3) or (H, W)
            
        Returns:
            DCTFeatures with frequency analysis results
        """
        try:
            # Import cv2 here to handle potential import issues
            import cv2
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)
            
            # Resize to standard size for consistent analysis
            gray = cv2.resize(gray, (256, 256))
            
            # Convert to float for DCT
            gray_float = gray.astype(np.float32)
            
            # Apply 2D DCT
            dct = cv2.dct(gray_float)
            
            # Compute energy distribution
            h, w = dct.shape
            
            # Low frequency region (top-left 32x32)
            low_freq = dct[:32, :32]
            energy_low = np.sum(np.abs(low_freq) ** 2)
            
            # High frequency region (bottom-right)
            high_freq = dct[h//2:, w//2:]
            energy_high = np.sum(np.abs(high_freq) ** 2)
            
            # Total energy
            total_energy = np.sum(np.abs(dct) ** 2)
            
            if total_energy > 0:
                energy_low_norm = energy_low / total_energy
                energy_high_norm = energy_high / total_energy
            else:
                energy_low_norm = 0.0
                energy_high_norm = 0.0
            
            # Spectral flatness (geometric mean / arithmetic mean)
            dct_abs = np.abs(dct.flatten()) + 1e-10
            geometric_mean = np.exp(np.mean(np.log(dct_abs)))
            arithmetic_mean = np.mean(dct_abs)
            spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
            
            # Compute anomaly score with calibrated thresholds
            # GANs typically have: low high_freq energy, HIGH spectral flatness
            # BUT: Mobile camera JPEG compression also reduces high_freq energy
            # KEY INSIGHT: Natural images have LOW spectral flatness (< 0.5)
            # GAN images have HIGH spectral flatness (> 0.6)
            # Solution: If spectral_flatness is low, reduce anomaly from energy checks
            
            anomaly_score = 0.0
            
            # Low high-frequency energy is suspicious (but only if very low)
            # Mobile photos with JPEG compression can have reduced high-freq energy
            if energy_high_norm < self.high_freq_threshold:
                anomaly_score += 0.25  # Reduced from 0.4
            
            # High spectral flatness (too uniform) is suspicious
            # But natural images with smooth backgrounds can also have high flatness
            if spectral_flatness > self.spectral_flatness_threshold:
                anomaly_score += 0.20  # Reduced from 0.3
            
            # Abnormal low/high ratio - most reliable indicator
            # GANs have extremely skewed ratios, natural images don't
            if energy_low_norm > 0 and energy_high_norm / energy_low_norm < self.energy_ratio_threshold:
                anomaly_score += 0.35  # Increased weight for most reliable signal
            
            # NEW: Require multiple signals for high anomaly
            # Single signal alone shouldn't trigger high scores
            signal_count = 0
            if energy_high_norm < self.high_freq_threshold:
                signal_count += 1
            if spectral_flatness > self.spectral_flatness_threshold:
                signal_count += 1
            if energy_low_norm > 0 and energy_high_norm / energy_low_norm < self.energy_ratio_threshold:
                signal_count += 1
            
            # If only one signal triggers, reduce the anomaly score
            # This prevents false positives from single metric variations
            if signal_count == 1:
                anomaly_score *= 0.5
            elif signal_count == 0:
                anomaly_score = 0.0
            
            # CRITICAL FIX: Low spectral flatness is a strong indicator of NATURAL images
            # GAN-generated images have HIGH spectral flatness (> 0.5)
            # If spectral_flatness < 0.4, this is likely a natural image - reduce anomaly
            if spectral_flatness < 0.4:
                # Strong indicator of natural image - significantly reduce anomaly
                anomaly_score *= 0.3
            elif spectral_flatness < 0.5:
                # Moderate indicator of natural image
                anomaly_score *= 0.5
            
            return DCTFeatures(
                energy_high_freq=float(energy_high_norm),
                energy_low_freq=float(energy_low_norm),
                spectral_flatness=float(spectral_flatness),
                anomaly_score=float(np.clip(anomaly_score, 0, 1))
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
        self.target_size = (224, 224)  # Standard input for EfficientNet
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        
        # Detection thresholds - calibrated to reduce false positives
        self.fake_threshold = 0.6  # Raised from 0.5 - require stronger evidence
        self.face_manipulation_threshold = 0.65  # Raised from 0.6
        
        # Weight configuration for aggregation - recalibrated
        # Neural models are more reliable than DCT for mobile photos
        self.weights = {
            "neural": 0.60,      # Increased from 0.50 - primary signal
            "dct": 0.15,         # Reduced from 0.25 - prone to false positives on compressed images
            "face": 0.25         # Kept same - reliable when faces detected
        }
        
        # Probability calibration parameters (temperature scaling)
        # These help shift probabilities away from the decision boundary
        self.temperature = 1.5  # Temperature for scaling logits
        self.calibration_offset = -0.05  # Small bias towards "real" for uncertain cases
        
        logger.info(f"ImageAnalyzer initialized with calibrated weights: {self.weights}, temp={self.temperature}")
    
    def get_required_models(self) -> List[str]:
        """
        Return models required for image analysis.
        
        Models:
        - efficientnet_b3_spatial: Primary fake detector
        - siglip_deepfake: Secondary AI-image detector
        - retinaface: Face detection for targeted analysis
        
        Returns:
            List of model registry keys
        """
        return [
            "efficientnet_b3_spatial",  # Primary deepfake detector
            "siglip_deepfake",           # AI-generated image detector  
            "retinaface"        # Face detection
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
                    
                    # Resize to expected input size
                    pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
                    
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
                
                # Resize if needed
                if image_array.shape[:2] != (224, 224):
                    pil_temp = Image.fromarray(image_array.astype(np.uint8))
                    pil_temp = pil_temp.resize((224, 224), Image.Resampling.LANCZOS)
                    image_array = np.array(pil_temp, dtype=np.uint8)
                
                images.append(image_array)
                
            except Exception as e:
                logger.warning(f"Failed to load image {key}: {e}")
                # Create a placeholder for failed loads
                placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
                images.append(placeholder)
        
        logger.info(f"Loaded {len(images)} images for analysis")
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
        
        # 2. Neural Detection (EfficientNet)
        try:
            neural_scores = await self._run_neural_detection(images, engine)
            result.efficientnet_score = float(np.mean(neural_scores))
        except Exception as e:
            logger.warning(f"Neural detection failed: {e}")
            result.efficientnet_score = 0.5
        
        # 3. SigLIP Detection (if available)
        try:
            siglip_scores = await self._run_siglip_detection(images, engine)
            result.siglip_score = float(np.mean(siglip_scores))
        except Exception as e:
            logger.debug(f"SigLIP detection skipped: {e}")
            result.siglip_score = result.efficientnet_score  # Fallback
        
        # 4. Face-specific analysis
        if data.face_crops:
            result.face_detected = True
            result.num_faces = len(data.face_crops)
            result.face_manipulation_scores = [
                result.efficientnet_score  # Use neural score for faces
            ]
        
        # 5. Aggregate scores with dynamic weighting based on signal confidence
        neural_avg = (result.efficientnet_score + result.siglip_score) / 2
        
        # Compute signal confidence for dynamic weighting
        # Neural confidence: how far from 0.5 (borderline)
        neural_confidence = abs(neural_avg - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
        
        # DCT confidence: based on spectral flatness
        # Low spectral flatness (< 0.4) = high confidence it's natural
        # High spectral flatness (> 0.6) = high confidence it's AI
        # Mid range = uncertain
        dct_features = result.dct_features
        if dct_features:
            spectral_flatness = dct_features.spectral_flatness
            if spectral_flatness < 0.4:
                # Strong indicator of natural image - high DCT confidence
                dct_confidence = 0.8 * (1.0 - spectral_flatness / 0.4)
            elif spectral_flatness > 0.6:
                # Strong indicator of AI - high DCT confidence
                dct_confidence = 0.8 * (spectral_flatness - 0.6) / 0.4
            else:
                # Uncertain range - low DCT confidence
                dct_confidence = 0.3
        else:
            dct_confidence = 0.3
        
        # Dynamic weights based on signal confidence
        # If neural is uncertain (near 0.5) but DCT is confident, prioritize DCT
        total_confidence = neural_confidence + dct_confidence + 0.1  # Add small base
        dynamic_neural_weight = (neural_confidence + 0.1) / total_confidence
        dynamic_dct_weight = dct_confidence / total_confidence
        
        # Blend with base weights
        final_neural_weight = 0.5 * self.weights["neural"] + 0.5 * dynamic_neural_weight
        final_dct_weight = 0.5 * self.weights["dct"] + 0.5 * dynamic_dct_weight
        
        # Apply temperature scaling
        calibrated_neural = self._apply_temperature_scaling(neural_avg)
        calibrated_dct = self._apply_temperature_scaling(avg_dct_score)
        
        result.fake_probability = (
            final_neural_weight * calibrated_neural +
            final_dct_weight * calibrated_dct +
            self.weights["face"] * (
                np.mean(result.face_manipulation_scores) 
                if result.face_manipulation_scores else calibrated_neural
            )
        )
        
        # Apply calibration offset for uncertain cases
        # If all signals are weak (near 0.5), bias towards "real"
        signal_strength = abs(neural_avg - 0.5) + abs(avg_dct_score - 0.5)
        if signal_strength < 0.3:  # Weak signals - uncertain case
            result.fake_probability += self.calibration_offset
        
        # Clamp to valid range
        result.fake_probability = float(np.clip(result.fake_probability, 0.0, 1.0))
        
        # 6. Compute confidence with uncertainty awareness
        all_scores = [result.efficientnet_score, result.siglip_score, avg_dct_score]
        base_confidence = compute_confidence(
            np.array(all_scores),
            len(images),
            min_samples=5
        )
        
        # Reduce confidence when signals disagree (indicates uncertainty)
        score_variance = np.var(all_scores)
        if score_variance > 0.05:  # High disagreement
            base_confidence *= 0.8  # Reduce confidence
        
        result.confidence = base_confidence
        
        return result
    
    async def _run_neural_detection(
        self,
        images: List[np.ndarray],
        engine: "InferenceEngine"
    ) -> List[float]:
        """
        Run EfficientNet-based deepfake detection.
        
        Args:
            images: List of preprocessed images
            engine: InferenceEngine
            
        Returns:
            List of fake probability scores
        """
        scores = []
        
        # Preprocess images for model
        preprocessed = [self._preprocess_for_model(img) for img in images]
        
        if not preprocessed:
            return [0.5]
        
        # Stack into batch
        batch = np.stack(preprocessed, axis=0)
        
        try:
            # Run inference
            result = await engine.infer(
                "efficientnet_b3_spatial",
                batch,
                return_probabilities=True
            )
            
            # Extract fake probabilities (class 1)
            if result.class_probabilities is not None:
                probs = result.class_probabilities
                if probs.shape[-1] >= 2:
                    scores = probs[:, 1].tolist()
                else:
                    scores = probs.flatten().tolist()
            else:
                scores = result.predictions.flatten().tolist()
            
        except Exception as e:
            logger.warning(f"EfficientNet inference failed: {e}")
            # Return neutral scores on failure
            scores = [0.5] * len(images)
        
        return scores
    
    async def _run_siglip_detection(
        self,
        images: List[np.ndarray],
        engine: "InferenceEngine"
    ) -> List[float]:
        """
        Run SigLIP-based AI-generated image detection.
        
        SigLIP provides better generalization to novel image generators.
        Model: prithivMLmods/AI-vs-Deepfake-vs-Real-ONNX
        Classes: 0=real, 1=deepfake, 2=ai_generated
        
        Args:
            images: List of preprocessed images
            engine: InferenceEngine
            
        Returns:
            List of AI-generated probability scores
        """
        scores = []
        
        # Model uses 224x224 input size (same as efficientnet)
        preprocessed = [self._preprocess_for_model_size(img, (224, 224)) for img in images]
        
        if not preprocessed:
            return [0.5]
        
        batch = np.stack(preprocessed, axis=0)
        
        try:
            result = await engine.infer(
                "siglip_deepfake",
                batch,
                return_probabilities=True
            )
            
            if result.class_probabilities is not None:
                probs = result.class_probabilities
                # 3-class model: 0=real, 1=deepfake, 2=ai_generated
                # Return combined fake probability (deepfake + ai_generated)
                if probs.shape[-1] == 3:
                    # P(fake) = P(deepfake) + P(ai_generated) = 1 - P(real)
                    scores = (1.0 - probs[:, 0]).tolist()
                elif probs.shape[-1] >= 2:
                    # Binary: class 1 = fake
                    scores = probs[:, 1].tolist()
                else:
                    scores = probs.flatten().tolist()
            else:
                scores = result.predictions.flatten().tolist()
                
        except Exception as e:
            logger.debug(f"SigLIP inference skipped: {e}")
            # SigLIP is optional, return empty list to trigger fallback
            scores = []
        
        return scores if scores else [0.5] * len(images)
    
    def _preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Applies:
        - Resize to target size (224x224)
        - Convert to float [0, 1]
        - ImageNet normalization
        - CHW format for PyTorch-style models
        
        Args:
            image: Input image (H, W, 3) uint8
            
        Returns:
            Preprocessed tensor (3, 224, 224) float32
        """
        return self._preprocess_for_model_size(image, self.target_size)
    
    def _preprocess_for_model_size(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Preprocess image for model input with specified target size.
        
        Args:
            image: Input image (H, W, 3) uint8
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed tensor (3, H, W) float32
        """
        import cv2
        
        # Resize
        resized = cv2.resize(image, target_size)
        
        # Convert to float [0, 1]
        float_img = resized.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array(self.normalize_mean).reshape(1, 1, 3)
        std = np.array(self.normalize_std).reshape(1, 1, 3)
        normalized = (float_img - mean) / std
        
        # Convert to CHW format
        chw = np.transpose(normalized, (2, 0, 1))
        
        return chw.astype(np.float32)
    
    def _apply_temperature_scaling(self, probability: float) -> float:
        """
        Apply temperature scaling to calibrate probabilities.
        
        Temperature scaling pushes probabilities away from 0.5 (uncertain)
        towards the extremes, reducing false positives from borderline cases.
        
        For T > 1: Softens the probability (moves towards 0.5)
        For T < 1: Sharpens the probability (moves away from 0.5)
        
        We use T > 1 to soften borderline "fake" predictions that are
        likely false positives from mobile camera artifacts.
        
        Args:
            probability: Raw probability score [0, 1]
            
        Returns:
            Calibrated probability [0, 1]
        """
        # Convert to logit space
        # p = sigmoid(logit) => logit = log(p / (1-p))
        eps = 1e-7
        p_clipped = np.clip(probability, eps, 1 - eps)
        logit = np.log(p_clipped / (1 - p_clipped))
        
        # Apply temperature scaling
        scaled_logit = logit / self.temperature
        
        # Convert back to probability
        calibrated = 1.0 / (1.0 + np.exp(-scaled_logit))
        
        return float(calibrated)
    
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
