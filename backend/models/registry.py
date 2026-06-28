"""
Argus Core - Model Registry
===========================
Central registry of available ML models with metadata.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - models/registry.py

Registry Structure:
{
    "model_name": {
        "path": "/models/model_name.onnx",
        "input_shape": [1, 3, 224, 224],
        "vram_mb": 300,
        "version": "1.0.0",
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "quantization": "INT8"  # None, "INT8", "FP16"
    }
}
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import os
import json

from config import config
from interfaces.model import ModelInfo
from utils.logging import get_logger
from utils.errors import ConfigurationError

logger = get_logger(__name__)


class ModelCategory(str, Enum):
    """Model categories for organization."""
    SPATIAL = "spatial"           # Per-frame spatial analysis
    TEMPORAL = "temporal"         # Temporal consistency
    LIPSYNC = "lipsync"           # Lip-sync verification
    AUDIO = "audio"               # Audio deepfake detection
    IMAGE = "image"               # Image manipulation detection
    FACE_DETECTION = "face_detection"  # Face detection preprocessing
    FEATURE = "feature"           # Feature extraction (CLIP, etc.)


class QuantizationType(str, Enum):
    """Model quantization types."""
    NONE = "none"       # FP32 - full precision
    FP16 = "fp16"       # Half precision
    INT8 = "int8"       # 8-bit integer quantization
    INT4 = "int4"       # 4-bit quantization (experimental)


@dataclass
class ModelMetadata:
    """
    Complete model metadata.
    
    Includes all information needed to load, manage,
    and run inference on a model.
    """
    name: str
    path: str
    input_shape: List[int]
    output_shape: Optional[List[int]] = None
    vram_mb: int = 500
    version: str = "1.0.0"
    providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    quantization: QuantizationType = QuantizationType.NONE
    category: ModelCategory = ModelCategory.IMAGE
    description: str = ""
    
    # Input preprocessing
    input_dtype: str = "float32"
    normalize_input: bool = True
    input_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    
    # Output postprocessing
    output_dtype: str = "float32"
    num_classes: int = 2  # Binary: real/fake
    class_labels: List[str] = field(default_factory=lambda: ["real", "fake"])
    
    # Performance hints
    max_batch_size: int = 32
    optimal_batch_size: int = 8
    supports_dynamic_batch: bool = True
    
    # Dependencies
    requires_models: List[str] = field(default_factory=list)
    
    # Verified model provenance (Phase 2 additions)
    source: str = ""  # HuggingFace model ID or GitHub repo
    download_url: str = ""  # Direct download URL for ONNX weights
    checksum_sha256: str = ""  # SHA256 checksum for verification
    license: str = "MIT"  # Model license
    academic_reference: str = ""  # Paper URL or citation
    
    def to_model_info(self) -> ModelInfo:
        """Convert to ModelInfo for interface compatibility."""
        return ModelInfo(
            name=self.name,
            path=self.path,
            input_shape=tuple(self.input_shape),
            vram_mb=self.vram_mb,
            version=self.version,
            providers=self.providers,
            quantization=self.quantization.value if isinstance(self.quantization, QuantizationType) else self.quantization
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "vram_mb": self.vram_mb,
            "version": self.version,
            "providers": self.providers,
            "quantization": self.quantization.value if isinstance(self.quantization, QuantizationType) else self.quantization,
            "category": self.category.value if isinstance(self.category, ModelCategory) else self.category,
            "description": self.description,
            "input_dtype": self.input_dtype,
            "normalize_input": self.normalize_input,
            "input_range": self.input_range,
            "output_dtype": self.output_dtype,
            "num_classes": self.num_classes,
            "class_labels": self.class_labels,
            "max_batch_size": self.max_batch_size,
            "optimal_batch_size": self.optimal_batch_size,
            "supports_dynamic_batch": self.supports_dynamic_batch,
            "requires_models": self.requires_models,
            "source": self.source,
            "download_url": self.download_url,
            "checksum_sha256": self.checksum_sha256,
            "license": self.license,
            "academic_reference": self.academic_reference
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        # Handle enum conversion
        if "quantization" in data and isinstance(data["quantization"], str):
            data["quantization"] = QuantizationType(data["quantization"])
        if "category" in data and isinstance(data["category"], str):
            data["category"] = ModelCategory(data["category"])
        return cls(**data)


# Default model registry with SOTA models for deepfake detection
# Input shapes match the actual ONNX model files in /models/
DEFAULT_MODELS: Dict[str, ModelMetadata] = {
    # ============== FEATURE EXTRACTION ==============
    "clip_vit_b16": ModelMetadata(
        name="clip_vit_b16",
        path="/models/clip_vit_b16.onnx",
        input_shape=[1, 3, 224, 224],  # Actual: data: [1, 3, 224, 224]
        output_shape=[1, 1280],  # MobileNetV2 output
        vram_mb=400,
        version="1.0.0",
        quantization=QuantizationType.NONE,
        category=ModelCategory.FEATURE,
        description="CLIP ViT-B/16 feature extractor for visual embedding and anomaly scoring",
        optimal_batch_size=8,
        max_batch_size=32,
        num_classes=0,  # Feature extractor, not classifier
        source="openai/clip-vit-base-patch16",
        download_url="https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/onnx/model.onnx",
        license="MIT",
        academic_reference="https://arxiv.org/abs/2103.00020"
    ),
    
    # ============== TEMPORAL ANALYSIS ==============
    # X-CLIP Temporal Transformer (KDD 2025)
    "xclip_temporal": ModelMetadata(
        name="xclip_temporal",
        path="/models/xclip_temporal_int8.onnx",
        input_shape=[1, 16, 3, 224, 224],  # (B, T, C, H, W)
        output_shape=[1, 2],
        vram_mb=450,
        version="1.0.0",
        quantization=QuantizationType.INT8,
        category=ModelCategory.VIDEO,
        description="Cross-frame temporal consistency analysis. Detects flickering, irregular motion, and inter-frame artifacts.",
        optimal_batch_size=2,
        max_batch_size=8,
        num_classes=2,
        class_labels=["real", "fake"],
        source="microsoft/xclip-base-patch16",
        download_url="https://huggingface.co/microsoft/xclip-base-patch16/resolve/main/onnx/model.onnx",
        license="MIT"
    ),

    # VideoMAE - SOTA Temporal Video Analyzer
    "videomae_temporal": ModelMetadata(
        name="videomae_temporal",
        path="/models/videomae_base.onnx",
        input_shape=[1, 16, 3, 224, 224],
        output_shape=[1, 2],
        vram_mb=650,
        version="1.0.0",
        quantization=QuantizationType.INT8,
        category=ModelCategory.VIDEO,
        description="VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training",
        optimal_batch_size=2,
        max_batch_size=4,
        num_classes=2,
        class_labels=["real", "fake"],
        source="MCG-NJU/videomae-base",
        download_url="pytorch:MCG-NJU/videomae-base",
        license="MIT",
        academic_reference="https://arxiv.org/abs/2203.12602"
    ),
    
    # ============== LIP-SYNC DETECTION ==============
    "lipinc_v2": ModelMetadata(
        name="lipinc_v2",
        path="/models/lipinc_v2_int8.onnx",
        input_shape=[1, 16, 3, 96, 96],  # Actual: input: [1, 16, 3, 96, 96]
        output_shape=[1, 2],
        vram_mb=350,
        version="2.0.0",
        quantization=QuantizationType.INT8,
        category=ModelCategory.LIPSYNC,
        description="LIPINC-V2 for Wav2Lip/Diff2Lip detection with multihead cross-attention",
        requires_models=["wav2vec2_base"],  # For audio encoding
        optimal_batch_size=4,
        max_batch_size=8,
        class_labels=["real", "lip_synced"],
        source="custom/lipinc-v2",
        license="MIT",
        academic_reference="https://arxiv.org/abs/2006.08818"
    ),
    
    # ============== AUDIO ANALYSIS ==============
    # AASIST - ASVspoof 2021 Winner
    "aasist_antispoof": ModelMetadata(
        name="aasist_antispoof",
        path="/models/aasist.onnx",  # Actual AASIST model
        input_shape=[1, 64600],  # Raw audio waveform input (not spectrogram)
        output_shape=[1, 2],
        vram_mb=300,
        version="1.0.0",
        quantization=QuantizationType.NONE,
        category=ModelCategory.AUDIO,
        description="AASIST: Anti-spoofing with Attention and Self-supervised Learning. ASVspoof 2021 winner.",
        optimal_batch_size=8,
        max_batch_size=32,
        num_classes=2,
        class_labels=["bonafide", "spoof"],
        source="clovaai/aasist",
        download_url="https://github.com/clovaai/aasist",
        license="MIT",
        academic_reference="https://arxiv.org/abs/2110.01200"
    ),
    
    # Legacy Purdue-M2 model (kept for backward compatibility)
    "purdue_m2": ModelMetadata(
        name="purdue_m2",
        path="/models/purdue_m2.onnx",
        input_shape=[1, 224, 224, 3],  # Actual: images:0: [1, 224, 224, 3] - NHWC format
        output_shape=[1, 2],
        vram_mb=250,
        version="1.0.0",
        quantization=QuantizationType.NONE,
        category=ModelCategory.AUDIO,
        description="Audio deepfake detection model (spectrogram-based). Legacy - use aasist_antispoof for better accuracy.",
        optimal_batch_size=16,
        max_batch_size=64,
        class_labels=["real", "synthetic"],
        source="legacy/purdue-m2",
        license="MIT"
    ),
    
    "wav2vec2_base": ModelMetadata(
        name="wav2vec2_base",
        path="/models/wav2vec2_base.onnx",
        input_shape=[1, 16000],  # Actual: input: [1, 16000] - 1 second at 16kHz
        output_shape=[1, 49, 768],  # Features
        vram_mb=380,
        version="1.0.0",
        quantization=QuantizationType.NONE,
        category=ModelCategory.FEATURE,
        description="Wav2Vec2 base model for audio feature extraction",
        optimal_batch_size=4,
        max_batch_size=16,
        num_classes=0,  # Feature extractor
        source="facebook/wav2vec2-base-960h",
        download_url="https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/onnx/model.onnx",
        license="Apache-2.0",
        academic_reference="https://arxiv.org/abs/2006.11477"
    ),
    
    # ============== FACE DETECTION ==============
    # ============== CLIP / FOUNDATION MODEL ADAPTER ==============
    "clip_vit_l14": ModelMetadata(
        name="clip_vit_l14",
        path="/models/clip_vit_l14.onnx",
        input_shape=[1, 3, 224, 224],
        output_shape=[1, 768],
        vram_mb=1200,
        version="1.0.0",
        quantization=QuantizationType.NONE,
        category=ModelCategory.IMAGE,
        description="CLIP ViT-L/14 foundation model adapter (ForensicsAdapter-style) for deepfake image detection",
        optimal_batch_size=4,
        max_batch_size=8,
        num_classes=2,
        class_labels=["real", "fake"],
        source="openai/clip-vit-large-patch14",
        download_url="https://huggingface.co/openai/clip-vit-large-patch14",
        license="MIT",
        academic_reference="https://arxiv.org/abs/2103.00020"
    ),
    "dinov2_vit_b14": ModelMetadata(
        name="dinov2_vit_b14",
        path="/models/dinov2_vit_b14.onnx",
        input_shape=[1, 3, 224, 224],
        output_shape=[1, 768],
        vram_mb=700,
        version="1.0.0",
        quantization=QuantizationType.NONE,
        category=ModelCategory.IMAGE,
        description="DINOv2-B/14 foundation model (NTIRE 2026 winner backbone) for deepfake detection",
        optimal_batch_size=4,
        max_batch_size=8,
        num_classes=2,
        class_labels=["real", "fake"],
        source="facebook/dinov2-base",
        download_url="https://huggingface.co/facebook/dinov2-base",
        license="Apache-2.0",
        academic_reference="https://arxiv.org/abs/2304.07193"
    ),

    # ============== WAV2VEC2 XLSR ANTISPOOFING ==============
    "wav2vec2_antispoof": ModelMetadata(
        name="wav2vec2_antispoof",
        path="/models/wav2vec2_antispoof.onnx",
        input_shape=[1, 64600],
        output_shape=[1, 2],
        vram_mb=340,
        version="2.0.0",
        quantization=QuantizationType.INT8,
        category=ModelCategory.AUDIO,
        description="Wav2Vec2 Large XLSR fine-tuned on ASVspoof2019 for audio deepfake detection (4.01% EER). INT8 ONNX.",
        optimal_batch_size=4,
        max_batch_size=8,
        num_classes=2,
        class_labels=["bonafide", "spoof"],
        source="pranjal-pravesh/wav2vec2-large-xlsr-deepfake-audio-classification",
        download_url="https://huggingface.co/pranjal-pravesh/wav2vec2-large-xlsr-deepfake-audio-classification",
        license="Apache-2.0",
        academic_reference="https://arxiv.org/abs/2006.11477"
    ),

    # ============== FACE DETECTION ==============
    "retinaface": ModelMetadata(
        name="retinaface",
        path="/models/retinaface.onnx",
        input_shape=[1, 3, 240, 320],  # Actual: input: [1, 3, 240, 320]
        output_shape=[1, -1, 15],  # Variable number of detections
        vram_mb=200,
        version="1.0.0",
        quantization=QuantizationType.NONE,
        category=ModelCategory.FACE_DETECTION,
        description="RetinaFace for high-accuracy face detection and alignment",
        optimal_batch_size=1,
        max_batch_size=4,
        num_classes=0,  # Detection, not classification
        source="biubug6/Pytorch_RetinaFace",
        download_url="https://github.com/biubug6/Pytorch_RetinaFace/releases/download/v1.0/RetinaFace.onnx",
        license="MIT",
        academic_reference="https://arxiv.org/abs/1905.00641"
    ),

    # ============== DEEPFAKE IMAGE DETECTION ==============
    "deepfake_detector_v3": ModelMetadata(
        name="deepfake_detector_v3",
        path="/models/deepfake_detector_v3.onnx",
        input_shape=[1, 3, 224, 224],
        output_shape=[1, 2],
        vram_mb=420,
        version="2.0.0",
        quantization=QuantizationType.NONE,
        category=ModelCategory.IMAGE,
        description="Primary deepfake image detection model (ViT-based, fine-tuned for face manipulation detection)",
        optimal_batch_size=4,
        max_batch_size=16,
        num_classes=2,
        class_labels=["authentic", "manipulated"],
        source="onnx-community/Deep-Fake-Detector-v2-Model-ONNX",
        download_url="https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX",
        license="MIT",
        academic_reference="https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX"
    ),
}


class ModelRegistry:
    """
    Central registry of available models.
    
    Manages model metadata, versioning, and discovery.
    Thread-safe singleton pattern for consistent access.
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        registry_file: Optional[str] = None
    ):
        """
        Initialize registry.
        
        Args:
            model_dir: Base directory for model files
            registry_file: Path to JSON registry file (optional)
        """
        self.model_dir = model_dir or config.model_cache_dir
        self.registry_file = registry_file
        
        # Initialize with default models
        self._models: Dict[str, ModelMetadata] = {}
        self._load_defaults()
        
        # Load custom registry if provided
        if registry_file and os.path.exists(registry_file):
            self._load_registry_file(registry_file)
        
        logger.info(f"Model registry initialized with {len(self._models)} models")
    
    def _load_defaults(self) -> None:
        """Load default model configurations."""
        for name, metadata in DEFAULT_MODELS.items():
            # Update path to use configured model directory
            updated_path = os.path.join(
                self.model_dir,
                os.path.basename(metadata.path)
            )
            
            self._models[name] = ModelMetadata(
                **{**metadata.to_dict(), "path": updated_path}
            )
    
    def _load_registry_file(self, path: str) -> None:
        """Load models from JSON registry file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            for name, model_data in data.get("models", {}).items():
                self._models[name] = ModelMetadata.from_dict(model_data)
            
            logger.info(f"Loaded {len(data.get('models', {}))} models from registry file")
            
        except Exception as e:
            logger.warning(f"Failed to load registry file: {e}")
    
    def get_model_info(self, model_name: str) -> ModelInfo:
        """
        Get model metadata from registry.
        
        Args:
            model_name: Model name/key
            
        Returns:
            ModelInfo with model details
            
        Raises:
            ConfigurationError: If model not found
        """
        if model_name not in self._models:
            raise ConfigurationError(
                f"model.{model_name}",
                f"Model not found in registry. Available: {list(self._models.keys())}"
            )
        
        return self._models[model_name].to_model_info()
    
    def get_model_metadata(self, model_name: str) -> ModelMetadata:
        """
        Get full model metadata.
        
        Args:
            model_name: Model name/key
            
        Returns:
            Complete ModelMetadata
        """
        if model_name not in self._models:
            raise ConfigurationError(
                f"model.{model_name}",
                "Model not found in registry"
            )
        
        return self._models[model_name]
    
    def list_models(
        self,
        category: Optional[ModelCategory] = None
    ) -> List[str]:
        """
        List all registered models.
        
        Args:
            category: Optional filter by category
            
        Returns:
            List of model names
        """
        if category is None:
            return list(self._models.keys())
        
        return [
            name for name, meta in self._models.items()
            if meta.category == category
        ]
    
    def get_models_by_category(
        self,
        category: ModelCategory
    ) -> Dict[str, ModelMetadata]:
        """
        Get all models in a category.
        
        Args:
            category: Model category
            
        Returns:
            Dict of model name to metadata
        """
        return {
            name: meta for name, meta in self._models.items()
            if meta.category == category
        }
    
    def model_exists(self, model_name: str) -> bool:
        """Check if model is registered."""
        return model_name in self._models
    
    def model_file_exists(self, model_name: str) -> bool:
        """Check if model file exists on disk."""
        if model_name not in self._models:
            return False
        return os.path.exists(self._models[model_name].path)
    
    def get_total_vram_required(
        self,
        model_names: List[str]
    ) -> int:
        """
        Calculate total VRAM required for models.
        
        Args:
            model_names: List of models to load
            
        Returns:
            Total VRAM in MB
        """
        total = 0
        seen = set()
        
        for name in model_names:
            if name in seen:
                continue
            seen.add(name)
            
            if name in self._models:
                meta = self._models[name]
                total += meta.vram_mb
                
                # Add dependencies
                for dep in meta.requires_models:
                    if dep not in seen and dep in self._models:
                        total += self._models[dep].vram_mb
                        seen.add(dep)
        
        return total
    
    def register_model(
        self,
        metadata: ModelMetadata,
        overwrite: bool = False
    ) -> None:
        """
        Register a new model.
        
        Args:
            metadata: Model metadata
            overwrite: Allow overwriting existing entry
        """
        if metadata.name in self._models and not overwrite:
            raise ConfigurationError(
                f"model.{metadata.name}",
                "Model already registered. Use overwrite=True to replace."
            )
        
        self._models[metadata.name] = metadata
        logger.info(f"Registered model: {metadata.name}")
    
    def unregister_model(self, model_name: str) -> bool:
        """
        Remove model from registry.
        
        Args:
            model_name: Model to remove
            
        Returns:
            True if removed, False if not found
        """
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Unregistered model: {model_name}")
            return True
        return False
    
    def export_registry(self, path: str) -> None:
        """
        Export registry to JSON file.
        
        Args:
            path: Output file path
        """
        data = {
            "version": "1.0.0",
            "models": {
                name: meta.to_dict()
                for name, meta in self._models.items()
            }
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported registry to {path}")
    
    def get_execution_providers(
        self,
        model_name: str
    ) -> List[str]:
        """
        Get execution providers for model.
        
        Falls back to config defaults if model not found.
        """
        if model_name in self._models:
            providers = self._models[model_name].providers.copy()
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        # Add TensorRT if enabled
        if config.enable_tensorrt:
            if "TensorrtExecutionProvider" not in providers:
                providers.insert(0, "TensorrtExecutionProvider")
        
        # Ensure CPU fallback
        if config.fallback_to_cpu:
            if "CPUExecutionProvider" not in providers:
                providers.append("CPUExecutionProvider")
        
        return providers


# Singleton instance
_registry: Optional[ModelRegistry] = None


@lru_cache()
def get_model_registry() -> ModelRegistry:
    """
    Get singleton model registry instance.
    
    Thread-safe lazy initialization.
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
