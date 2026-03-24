"""
Argus Core - Model Manager
=========================
Intelligent model loading for constrained VRAM environments.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - models/manager.py

Features:
- LRU eviction when VRAM pressure detected
- Lazy loading (models loaded on first use)
- Warmup mode (preload critical models)
- Real-time VRAM monitoring via nvidia-smi
- Hardware-aware model loading (CUDA/MPS/CPU)
- Automatic model downloading from HuggingFace

Target Hardware: RTX 3050 (4GB VRAM) with 3.5GB budget
"""

import asyncio
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from functools import lru_cache
import os

from config import config
from models.registry import ModelRegistry, ModelMetadata, get_model_registry
from interfaces.model import IModel, ModelInfo
from utils.errors import ModelLoadError, InferenceError, ConfigurationError
from utils.logging import get_logger
from utils.metrics import record_model_load, record_model_unload, update_vram_usage
from utils.hardware import (
    get_hardware_info, 
    HardwareInfo, 
    AcceleratorType,
    get_recommended_settings
)

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    """
    Container for a loaded model with usage tracking.
    Supports both ONNX Runtime sessions and PyTorch models.
    """
    name: str
    session: Any  # ONNX Runtime InferenceSession or PyTorch model
    metadata: ModelMetadata
    vram_mb: int
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    is_pytorch: bool = False  # Flag to indicate PyTorch model
    processor: Any = None  # Tokenizer/Processor for PyTorch models
    
    def touch(self) -> None:
        """Update last used timestamp and increment use count."""
        self.last_used = time.time()
        self.use_count += 1


class ModelManager:
    """
    Intelligent model loading for constrained VRAM.
    
    Manages model lifecycle with LRU eviction to stay within
    VRAM budget (3.5GB for RTX 3050).
    
    Thread-safe for concurrent access.
    """
    
    def __init__(
        self,
        max_vram_mb: Optional[int] = None,
        model_cache_dir: Optional[str] = None,
        registry: Optional[ModelRegistry] = None
    ):
        """
        Initialize model manager.
        
        Args:
            max_vram_mb: VRAM budget in MB (default from config)
            model_cache_dir: Directory for model files
            registry: Model registry instance
        """
        # Detect hardware capabilities
        self.hardware = get_hardware_info()
        
        # Set VRAM budget based on hardware
        if max_vram_mb is None:
            hw_settings = get_recommended_settings(self.hardware)
            max_vram_mb = hw_settings.get("vram_budget_mb", config.gpu_memory_limit_mb)
        
        self.max_vram_mb = max_vram_mb
        self.model_cache_dir = model_cache_dir or config.model_cache_dir
        self.registry = registry or get_model_registry()
        
        # LRU cache of loaded models (OrderedDict maintains insertion order)
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        
        # Thread safety
        self._lock = Lock()
        
        # VRAM tracking
        self._current_vram_mb = 0
        
        # ONNX Runtime options based on hardware
        self._use_gpu = self.hardware.accelerator != AcceleratorType.CPU
        self._fallback_to_cpu = config.fallback_to_cpu
        self._providers = self.hardware.available_providers
        
        logger.info(
            f"ModelManager initialized: accelerator={self.hardware.accelerator.value}, "
            f"device={self.hardware.device_name}, max_vram={self.max_vram_mb}MB, "
            f"providers={self._providers}"
        )
    
    async def get_model(
        self,
        model_name: str,
        preload_dependencies: bool = True
    ) -> Any:
        """
        Get model session, loading if necessary.
        
        Evicts LRU models if VRAM insufficient.
        
        Args:
            model_name: Model name from registry
            preload_dependencies: Load required models first
            
        Returns:
            ONNX Runtime InferenceSession
            
        Raises:
            ModelLoadError: If model cannot be loaded
        """
        with self._lock:
            # Check if already loaded
            if model_name in self._loaded:
                self._loaded[model_name].touch()
                # Move to end (most recently used)
                self._loaded.move_to_end(model_name)
                logger.debug(f"Model cache hit: {model_name}")
                loaded = self._loaded[model_name]
                # For PyTorch models, return tuple (model, processor)
                if loaded.is_pytorch:
                    return (loaded.session, loaded.processor)
                return loaded.session
        
        # Get metadata
        try:
            metadata = self.registry.get_model_metadata(model_name)
        except ConfigurationError as e:
            raise ModelLoadError(model_name, str(e))
        
        # Check if model is suitable for current hardware
        model_name = self._get_compatible_model_name(model_name, metadata)
        metadata = self.registry.get_model_metadata(model_name)
        
        # Load dependencies first
        if preload_dependencies and metadata.requires_models:
            for dep_name in metadata.requires_models:
                if dep_name not in self._loaded:
                    await self.get_model(dep_name, preload_dependencies=True)
        
        # Check if we need to evict models
        await self._ensure_vram_available(metadata.vram_mb)
        
        # Load the model
        session = await self._load_model(metadata)
        
        # Check if this is a PyTorch model (returns tuple of model, processor)
        is_pytorch = metadata.download_url and metadata.download_url.startswith("pytorch:")
        
        with self._lock:
            if is_pytorch:
                model, processor = session
                self._loaded[model_name] = LoadedModel(
                    name=model_name,
                    session=model,
                    metadata=metadata,
                    vram_mb=metadata.vram_mb,
                    is_pytorch=True,
                    processor=processor
                )
            else:
                self._loaded[model_name] = LoadedModel(
                    name=model_name,
                    session=session,
                    metadata=metadata,
                    vram_mb=metadata.vram_mb
                )
            self._current_vram_mb += metadata.vram_mb
        
        logger.info(
            f"Loaded model: {model_name} ({metadata.vram_mb}MB), "
            f"total VRAM: {self._current_vram_mb}MB"
        )
        
        return session
    
    def _get_compatible_model_name(
        self, 
        model_name: str, 
        metadata: ModelMetadata
    ) -> str:
        """
        Get hardware-compatible model name.
        
        Some models require GPU - return alternative for CPU.
        
        Args:
            model_name: Original model name
            metadata: Model metadata
            
        Returns:
            Compatible model name
        """
        # Do not auto-remap cross-modality models here.
        # Analyzer-level fallback logic handles degraded paths explicitly.
        return model_name
    
    def _validate_onnx_model(self, model_path: str) -> Tuple[bool, str]:
        """
        Validate ONNX model integrity.
        
        Checks:
        1. File exists and has non-zero size
        2. Valid ONNX magic header bytes
        3. Can be loaded and validated by ONNX checker
        4. Has at least one computational node (not identity/placeholder)
        
        Args:
            model_path: Path to ONNX model file
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not os.path.exists(model_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            return False, "File is empty (0 bytes)"
        
        if file_size < 100:
            return False, f"File too small ({file_size} bytes) - likely placeholder"
        
        # Check ONNX magic header
        # ONNX files start with magic bytes: 0x08 0x01 (little-endian) or protobuf header
        try:
            with open(model_path, 'rb') as f:
                header = f.read(8)
                # ONNX protobuf files typically start with field tag (0x08 or 0x0a)
                # or can be raw protobuf
                if len(header) < 4:
                    return False, "File too small for valid ONNX header"
        except Exception as e:
            return False, f"Cannot read file header: {e}"
        
        # Try to load and validate with ONNX library
        try:
            import onnx
            model = onnx.load(model_path)
            
            # Check model has valid graph
            if not model.graph:
                return False, "Model has no graph"
            
            # Check for computational nodes (not just identity)
            computational_nodes = [
                node for node in model.graph.node 
                if node.op_type not in ['Identity', 'Constant']
            ]
            
            if len(computational_nodes) == 0:
                return False, "Model has no computational nodes (placeholder)"
            
            # Validate model structure
            onnx.checker.check_model(model)
            
            # Get model info for logging
            input_names = [i.name for i in model.graph.input]
            output_names = [o.name for o in model.graph.output]
            node_count = len(model.graph.node)
            
            return True, f"Valid ONNX model: {node_count} nodes, inputs={input_names[:3]}, outputs={output_names[:3]}"
            
        except ImportError:
            # ONNX not available, do basic file check
            return file_size > 10000, f"File exists ({file_size} bytes) - ONNX library not available for validation"
        except Exception as e:
            return False, f"ONNX validation failed: {e}"
    
    async def _download_model_if_missing(
        self,
        metadata: ModelMetadata
    ) -> str:
        """
        Download model file if not present locally.
        
        Attempts to download from HuggingFace. Raises ModelLoadError if
        model cannot be downloaded - no placeholder fallback.
        
        Args:
            metadata: Model metadata with path info
            
        Returns:
            Path to model file (downloaded or existing)
            
        Raises:
            ModelLoadError: If model file not found and download fails
        """
        model_path = metadata.path
        alt_path = os.path.join(self.model_cache_dir, os.path.basename(model_path))
        
        # Check if real model exists with valid ONNX structure
        for path in [model_path, alt_path]:
            if os.path.exists(path):
                is_valid, reason = self._validate_onnx_model(path)
                if is_valid:
                    file_size = os.path.getsize(path)
                    logger.info(f"Found valid ONNX model: {path} ({file_size / 1024 / 1024:.1f}MB) - {reason}")
                    return path
                else:
                    logger.warning(f"Invalid model file: {path} - {reason}")
        
        # Try to download real model
        logger.info(f"Model not found locally or invalid, attempting download: {metadata.name}")
        
        try:
            from models.downloader import get_model_downloader
            downloader = get_model_downloader()
            
            # Attempt download
            downloaded_path = await downloader.download_model(metadata.name)
            if downloaded_path:
                return str(downloaded_path)
                
        except Exception as e:
            logger.error(f"Model download failed for {metadata.name}: {e}")
        
        # No placeholder fallback - fail explicitly
        raise ModelLoadError(
            model_name=metadata.name,
            reason=f"Model file not found and download failed. "
                   f"Expected path: {model_path} or {alt_path}. "
                   f"Please ensure model weights are available. "
                   f"Check AUTO_START_CONFIGURATION.md for model setup instructions."
        )
    
    async def _load_model(
        self,
        metadata: ModelMetadata
    ) -> Any:
        """
        Load model from disk into ONNX Runtime session or PyTorch.
        
        Downloads model if not present, then loads into appropriate runtime.
        Supports both ONNX and PyTorch (safetensors) model formats.
        
        Args:
            metadata: Model metadata with path and providers
            
        Returns:
            ONNX Runtime InferenceSession or PyTorch model tuple (model, processor)
        """
        # Check if this is a PyTorch model (download_url starts with "pytorch:")
        is_pytorch = metadata.download_url and metadata.download_url.startswith("pytorch:")
        
        if is_pytorch:
            return await self._load_pytorch_model(metadata)
        
        # ONNX model loading
        model_path = await self._download_model_if_missing(metadata)
        
        if not os.path.exists(model_path):
            raise ModelLoadError(
                model_name=metadata.name,
                reason=f"Model file not found after download attempt: {model_path}"
            )
        
        # Check if this is a real model or placeholder stub
        file_size = os.path.getsize(model_path)
        min_model_size = 10000  # 10KB minimum for real model weights
        
        if file_size < min_model_size:
            raise ModelLoadError(
                model_name=metadata.name,
                reason=f"Model file appears to be a placeholder stub. "
                       f"File size: {file_size} bytes (expected > 10MB for real models). "
                       f"Please download real model weights."
            )
        
        try:
            import onnxruntime as ort
            
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4
            
            # Enable memory pattern optimization
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            # Get execution providers based on hardware
            providers = self._providers.copy()
            
            # Filter to available providers
            available_providers = ort.get_available_providers()
            providers = [p for p in providers if p in available_providers]
            
            if not providers:
                providers = ["CPUExecutionProvider"]
            
            logger.info(f"Loading {metadata.name} with providers: {providers}")
            
            # Create session
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            logger.info(f"Successfully loaded real ONNX model: {metadata.name}")
            
            # Record metrics with error handling to prevent model load failures
            try:
                record_model_load(metadata.name, success=True)
            except Exception as metric_err:
                logger.warning(f"Failed to record model load metric: {metric_err}")
            
            try:
                update_vram_usage(metadata.name, metadata.vram_mb * 1024 * 1024)
            except Exception as metric_err:
                logger.warning(f"Failed to update VRAM usage metric: {metric_err}")
            
            return session
            
        except ImportError as e:
            record_model_load(metadata.name, success=False)
            raise ModelLoadError(
                model_name=metadata.name,
                reason=f"ONNX Runtime not installed. Install with: pip install onnxruntime-gpu. Error: {e}"
            )
        except RuntimeError as e:
            record_model_load(metadata.name, success=False)
            raise ModelLoadError(
                model_name=metadata.name,
                reason=f"ONNX Runtime error (likely CUDA/cuDNN mismatch): {e}"
            )
        except OSError as e:
            record_model_load(metadata.name, success=False)
            raise ModelLoadError(
                model_name=metadata.name,
                reason=f"Model file corrupted or inaccessible: {e}"
            )
        except Exception as e:
            record_model_load(metadata.name, success=False)
            raise ModelLoadError(
                model_name=metadata.name,
                reason=f"Unexpected error loading model: {type(e).__name__}: {e}"
            )
    
    async def _load_pytorch_model(
        self,
        metadata: ModelMetadata
    ) -> Tuple[Any, Any]:
        """
        Load PyTorch model from HuggingFace Hub.
        
        Downloads and loads a PyTorch model (safetensors format) directly
        using the transformers library for models that don't have ONNX versions.
        
        Args:
            metadata: Model metadata with source repo
            
        Returns:
            Tuple of (model, processor) for inference
        """
        try:
            import torch
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            from huggingface_hub import snapshot_download
            
            # Extract repo from download_url (format: "pytorch:repo/name")
            repo_id = metadata.download_url.replace("pytorch:", "")
            
            logger.info(f"Loading PyTorch model from HuggingFace: {repo_id}")
            
            # Download model files
            model_dir = await self._download_pytorch_model(repo_id, metadata)
            
            # Load model and processor
            device = "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
            
            model = AutoModelForImageClassification.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=False
            )
            model.to(device)
            model.eval()
            
            processor = AutoImageProcessor.from_pretrained(
                model_dir,
                local_files_only=True
            )
            
            logger.info(f"Successfully loaded PyTorch model: {metadata.name} on {device}")
            
            # Record metrics
            try:
                record_model_load(metadata.name, success=True)
            except Exception as metric_err:
                logger.warning(f"Failed to record model load metric: {metric_err}")
            
            return (model, processor)
            
        except ImportError as e:
            record_model_load(metadata.name, success=False)
            raise ModelLoadError(
                model_name=metadata.name,
                reason=f"PyTorch/transformers not installed. Install with: pip install torch transformers. Error: {e}"
            )
        except Exception as e:
            record_model_load(metadata.name, success=False)
            raise ModelLoadError(
                model_name=metadata.name,
                reason=f"Failed to load PyTorch model: {type(e).__name__}: {e}"
            )
    
    async def _download_pytorch_model(
        self,
        repo_id: str,
        metadata: ModelMetadata
    ) -> str:
        """
        Download PyTorch model files from HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repository ID
            metadata: Model metadata
            
        Returns:
            Path to downloaded model directory
        """
        from huggingface_hub import snapshot_download
        
        model_dir = os.path.join(self.model_cache_dir, metadata.name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if already downloaded
        if os.path.exists(os.path.join(model_dir, "config.json")):
            # Check for model files
            model_files = ["model.safetensors", "pytorch_model.bin"]
            if any(os.path.exists(os.path.join(model_dir, f)) for f in model_files):
                logger.info(f"PyTorch model already cached: {model_dir}")
                return model_dir
        
        logger.info(f"Downloading PyTorch model from {repo_id}...")
        
        try:
            # Download model files
            loop = asyncio.get_event_loop()
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor() as executor:
                downloaded_path = await loop.run_in_executor(
                    executor,
                    lambda: snapshot_download(
                        repo_id=repo_id,
                        local_dir=model_dir,
                        # Only download essential files
                        allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin", 
                                       "preprocessor_config.json", "*.json"],
                    )
                )
            
            logger.info(f"Downloaded PyTorch model to: {model_dir}")
            return model_dir
            
        except Exception as e:
            raise ModelLoadError(
                model_name=metadata.name,
                reason=f"Failed to download PyTorch model from HuggingFace: {e}"
            )
    
    async def _ensure_vram_available(
        self,
        required_mb: int
    ) -> None:
        """
        Ensure sufficient VRAM is available, evicting if necessary.
        
        Args:
            required_mb: VRAM needed for new model
        """
        # Check if we need to evict
        target_available = self.max_vram_mb - self._current_vram_mb
        
        if required_mb <= target_available:
            return
        
        # Calculate how much to free
        to_free = required_mb - target_available
        
        logger.info(
            f"VRAM pressure: need {required_mb}MB, "
            f"available {target_available}MB, freeing {to_free}MB"
        )
        
        await self.evict_lru(to_free)
    
    async def evict_lru(
        self,
        required_mb: int
    ) -> int:
        """
        Evict least recently used models to free space.
        
        Args:
            required_mb: Minimum MB to free
            
        Returns:
            Actual MB freed
        """
        freed_mb = 0
        to_evict = []
        
        with self._lock:
            # Identify models to evict (oldest first)
            for name in list(self._loaded.keys()):
                if freed_mb >= required_mb:
                    break
                
                model = self._loaded[name]
                to_evict.append(name)
                freed_mb += model.vram_mb
        
        # Evict identified models
        for name in to_evict:
            await self.unload_model(name)
        
        logger.info(f"Evicted {len(to_evict)} models, freed {freed_mb}MB")
        return freed_mb
    
    async def unload_model(
        self,
        model_name: str
    ) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Model to unload
            
        Returns:
            True if model was unloaded
        """
        with self._lock:
            if model_name not in self._loaded:
                return False
            
            model = self._loaded.pop(model_name)
            self._current_vram_mb -= model.vram_mb
            
            # Clean up session
            if hasattr(model.session, '_session'):
                del model.session._session
            del model.session
            
            # Record metrics
            record_model_unload(model_name)
            update_vram_usage(model_name, 0)
        
        logger.info(f"Unloaded model: {model_name}")
        return True
    
    async def warmup(
        self,
        model_names: Optional[List[str]] = None
    ) -> None:
        """
        Preload models into memory.
        
        Args:
            model_names: Models to preload (None = critical models)
        """
        if model_names is None:
            # Default critical models based on hardware
            if self.hardware.accelerator == AcceleratorType.CPU:
                # CPU mode: load only essential models
                model_names = [
                    "ai_real_detector",  # Unified AI/Real image detection model
                    "retinaface",  # Face detection
                ]
            else:
                # GPU mode: load more models
                model_names = [
                    "ai_real_detector",  # Unified AI/Real image detection model
                    "retinaface",  # Face detection
                ]
        
        logger.info(f"Warming up models: {model_names}")
        
        for name in model_names:
            try:
                await self.get_model(name)
            except Exception as e:
                logger.warning(f"Failed to warmup {name}: {e}")
    
    def get_vram_usage(self) -> int:
        """
        Get current VRAM usage in MB.
        
        Uses nvidia-smi for actual GPU memory or tracked value.
        
        Returns:
            VRAM usage in MB
        """
        # Only try nvidia-smi for CUDA
        if self.hardware.accelerator == AcceleratorType.CUDA:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    return int(result.stdout.strip().split('\n')[0])
                    
            except Exception as e:
                logger.debug(f"nvidia-smi not available: {e}")
        
        # Return tracked value as fallback
        return self._current_vram_mb
    
    def get_available_vram(self) -> int:
        """
        Get available VRAM in MB.
        
        Returns:
            Available VRAM
        """
        return self.max_vram_mb - self.get_vram_usage()
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names."""
        with self._lock:
            return list(self._loaded.keys())
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usage statistics for loaded models.
        
        Returns:
            Dict of model name to stats
        """
        with self._lock:
            return {
                name: {
                    "vram_mb": model.vram_mb,
                    "loaded_at": model.loaded_at,
                    "last_used": model.last_used,
                    "use_count": model.use_count,
                    "age_seconds": time.time() - model.loaded_at
                }
                for name, model in self._loaded.items()
            }
    
    def can_load_model(
        self,
        model_name: str,
        with_eviction: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if model can be loaded.
        
        Args:
            model_name: Model to check
            with_eviction: Consider evicting other models
            
        Returns:
            Tuple of (can_load, reason)
        """
        if model_name in self._loaded:
            return True, "Already loaded"
        
        try:
            metadata = self.registry.get_model_metadata(model_name)
        except ConfigurationError:
            return False, "Model not in registry"
        
        required = metadata.vram_mb
        available = self.get_available_vram()
        
        if required <= available:
            return True, f"Sufficient VRAM ({available}MB available)"
        
        if with_eviction:
            # Calculate if eviction could free enough
            evictable = self._current_vram_mb
            if required <= available + evictable:
                return True, "Can load after eviction"
        
        return False, f"Insufficient VRAM ({required}MB needed, {available}MB available)"
    
    async def clear_cache(self) -> None:
        """Unload all models from memory."""
        with self._lock:
            model_names = list(self._loaded.keys())
        
        for name in model_names:
            await self.unload_model(name)
        
        self._current_vram_mb = 0
        logger.info("Model cache cleared")
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get current hardware information.
        
        Returns:
            Dict with hardware details
        """
        return {
            "accelerator": self.hardware.accelerator.value,
            "device_name": self.hardware.device_name,
            "memory_mb": self.hardware.memory_mb,
            "supports_fp16": self.hardware.supports_fp16,
            "supports_int8": self.hardware.supports_int8,
            "providers": self._providers,
            "vram_budget_mb": self.max_vram_mb,
            "vram_used_mb": self._current_vram_mb,
            "vram_available_mb": self.get_available_vram(),
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            with self._lock:
                self._loaded.clear()
                self._current_vram_mb = 0
        except Exception:
            pass


# Singleton instance
_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get singleton model manager instance.
    """
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


async def initialize_model_manager(
    warmup_models: Optional[List[str]] = None
) -> ModelManager:
    """
    Initialize model manager with optional warmup.
    
    Args:
        warmup_models: Models to preload
        
    Returns:
        Initialized ModelManager
    """
    manager = get_model_manager()
    
    if warmup_models:
        await manager.warmup(warmup_models)
    
    return manager
