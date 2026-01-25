"""
Argus Core - Model Manager
==========================
Intelligent model loading for constrained VRAM environments.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - models/manager.py

Features:
- LRU eviction when VRAM pressure detected
- Lazy loading (models loaded on first use)
- Warmup mode (preload critical models)
- Real-time VRAM monitoring via nvidia-smi

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

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    """
    Container for a loaded model with usage tracking.
    """
    name: str
    session: Any  # ONNX Runtime InferenceSession
    metadata: ModelMetadata
    vram_mb: int
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    
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
        self.max_vram_mb = max_vram_mb or config.gpu_memory_limit_mb
        self.model_cache_dir = model_cache_dir or config.model_cache_dir
        self.registry = registry or get_model_registry()
        
        # LRU cache of loaded models (OrderedDict maintains insertion order)
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        
        # Thread safety
        self._lock = Lock()
        
        # VRAM tracking
        self._current_vram_mb = 0
        
        # ONNX Runtime options
        self._use_gpu = config.use_gpu
        self._fallback_to_cpu = config.fallback_to_cpu
        
        logger.info(
            f"ModelManager initialized: max_vram={self.max_vram_mb}MB, "
            f"gpu={self._use_gpu}, fallback_cpu={self._fallback_to_cpu}"
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
                return self._loaded[model_name].session
        
        # Get metadata
        try:
            metadata = self.registry.get_model_metadata(model_name)
        except ConfigurationError as e:
            raise ModelLoadError(model_name, str(e))
        
        # Load dependencies first
        if preload_dependencies and metadata.requires_models:
            for dep_name in metadata.requires_models:
                if dep_name not in self._loaded:
                    await self.get_model(dep_name, preload_dependencies=True)
        
        # Check if we need to evict models
        await self._ensure_vram_available(metadata.vram_mb)
        
        # Load the model
        session = await self._load_model(metadata)
        
        with self._lock:
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
    
    async def _load_model(
        self,
        metadata: ModelMetadata
    ) -> Any:
        """
        Load model from disk into ONNX Runtime session.
        
        Args:
            metadata: Model metadata with path and providers
            
        Returns:
            ONNX Runtime InferenceSession
        """
        # Check if model file exists
        model_path = metadata.path
        if not os.path.exists(model_path):
            # Try alternate locations
            alt_path = os.path.join(self.model_cache_dir, os.path.basename(model_path))
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                logger.warning(
                    f"Model file not found: {model_path}. "
                    f"Creating placeholder session for development."
                )
                return self._create_placeholder_session(metadata)
        
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
            
            # Get execution providers
            providers = self.registry.get_execution_providers(metadata.name)
            
            # Filter available providers
            available_providers = ort.get_available_providers()
            providers = [p for p in providers if p in available_providers]
            
            if not providers:
                providers = ["CPUExecutionProvider"]
            
            logger.debug(f"Loading {metadata.name} with providers: {providers}")
            
            # Create session
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            return session
            
        except ImportError:
            logger.warning("ONNX Runtime not available, using placeholder")
            return self._create_placeholder_session(metadata)
            
        except Exception as e:
            raise ModelLoadError(metadata.name, str(e))
    
    def _create_placeholder_session(
        self,
        metadata: ModelMetadata
    ) -> "PlaceholderSession":
        """
        Create a placeholder session for development/testing.
        
        Returns random outputs matching expected shape.
        """
        return PlaceholderSession(metadata)
    
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
            # Default critical models for typical analysis
            model_names = [
                "efficientnet_b3_spatial",
                "retinaface"
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
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            with self._lock:
                self._loaded.clear()
                self._current_vram_mb = 0
        except Exception:
            pass


class PlaceholderSession:
    """
    Placeholder ONNX session for development/testing.
    
    Returns random outputs matching expected shape.
    """
    
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self._inputs = [
            type("Input", (), {"name": "input", "shape": metadata.input_shape})()
        ]
        self._outputs = [
            type("Output", (), {
                "name": "output",
                "shape": metadata.output_shape or [1, 2]
            })()
        ]
    
    def run(
        self,
        output_names: Optional[List[str]],
        input_feed: Dict[str, Any],
        run_options: Any = None
    ) -> List[Any]:
        """
        Run fake inference returning random outputs.
        """
        import numpy as np
        
        batch_size = 1
        for name, value in input_feed.items():
            if hasattr(value, 'shape'):
                batch_size = value.shape[0]
                break
        
        output_shape = self.metadata.output_shape or [1, 2]
        output_shape[0] = batch_size
        
        # Generate random output
        if self.metadata.num_classes > 0:
            # Classification: return softmax-like outputs
            output = np.random.dirichlet(
                np.ones(self.metadata.num_classes),
                size=batch_size
            ).astype(np.float32)
        else:
            # Feature extraction: return random features
            output = np.random.randn(*output_shape).astype(np.float32)
        
        return [output]
    
    def get_inputs(self):
        return self._inputs
    
    def get_outputs(self):
        return self._outputs


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
