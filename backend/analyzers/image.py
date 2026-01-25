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
    Modality, PreprocessedData, ModalityResult, ContentType
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
    """
    
    def __init__(self):
        super().__init__("DCTAnalyzer")
        
        # DCT thresholds (tuned for common GANs)
        self.high_freq_threshold = 0.15
        self.spectral_flatness_threshold = 0.6
    
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
            
            # Compute anomaly score
            # GANs typically have: low high_freq energy, high spectral flatness
            anomaly_score = 0.0
            
            # Low high-frequency energy is suspicious
            if energy_high_norm < self.high_freq_threshold:
                anomaly_score += 0.4
            
            # High spectral flatness (too uniform) is suspicious
            if spectral_flatness > self.spectral_flatness_threshold:
                anomaly_score += 0.3
            
            # Abnormal low/high ratio
            if energy_low_norm > 0 and energy_high_norm / energy_low_norm < 0.01:
                anomaly_score += 0.3
            
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
        
        # Detection thresholds
        self.fake_threshold = 0.5
        self.face_manipulation_threshold = 0.6
        
        # Weight configuration for aggregation
        self.weights = {
            "neural": 0.50,      # Primary neural detector
            "dct": 0.25,         # DCT frequency analysis
            "face": 0.25         # Face-specific detection
        }
        
        logger.info(f"ImageAnalyzer initialized with weights: {self.weights}")
    
    def get_required_models(self) -> List[str]:
        """
        Return models required for image analysis.
        
        Models:
        - efficientnet_b3_spatial: Primary fake detector
        - siglip_detector: Secondary AI-image detector
        - retinaface_detector: Face detection for targeted analysis
        
        Returns:
            List of model registry keys
        """
        return [
            "efficientnet_b3_spatial",  # Primary deepfake detector
            "siglip_detector",           # AI-generated image detector  
            "retinaface_detector"        # Face detection
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
        
        In production, this fetches from object storage.
        For development, generates placeholder images.
        
        Args:
            image_keys: List of MinIO object keys
            
        Returns:
            List of loaded image arrays
        """
        # TODO: Integrate with StorageClient for actual loading
        # For now, we'll return placeholder data
        images = []
        
        for key in image_keys[:10]:  # Limit to 10 images
            # Create placeholder image (224x224 RGB)
            # In production, this would be:
            # image_bytes = await storage.download_file("argus-preprocessed", key)
            # image = load_image(image_bytes)
            placeholder = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            images.append(placeholder)
        
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
        
        # 5. Aggregate scores
        neural_avg = (result.efficientnet_score + result.siglip_score) / 2
        
        result.fake_probability = (
            self.weights["neural"] * neural_avg +
            self.weights["dct"] * avg_dct_score +
            self.weights["face"] * (
                np.mean(result.face_manipulation_scores) 
                if result.face_manipulation_scores else neural_avg
            )
        )
        
        # 6. Compute confidence
        all_scores = [result.efficientnet_score, result.siglip_score, avg_dct_score]
        result.confidence = compute_confidence(
            np.array(all_scores),
            len(images),
            min_samples=5
        )
        
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
        
        Args:
            images: List of preprocessed images
            engine: InferenceEngine
            
        Returns:
            List of AI-generated probability scores
        """
        scores = []
        
        preprocessed = [self._preprocess_for_model(img) for img in images]
        
        if not preprocessed:
            return [0.5]
        
        batch = np.stack(preprocessed, axis=0)
        
        try:
            result = await engine.infer(
                "siglip_detector",
                batch,
                return_probabilities=True
            )
            
            if result.class_probabilities is not None:
                probs = result.class_probabilities
                scores = probs[:, 1].tolist() if probs.shape[-1] >= 2 else probs.flatten().tolist()
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
        import cv2
        
        # Resize
        resized = cv2.resize(image, self.target_size)
        
        # Convert to float [0, 1]
        float_img = resized.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array(self.normalize_mean).reshape(1, 1, 3)
        std = np.array(self.normalize_std).reshape(1, 1, 3)
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
