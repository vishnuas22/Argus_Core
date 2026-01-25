"""
Argus Core - Model Interface
============================
Abstract base class defining the contract for ML model wrappers.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - models/manager.py
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any
import numpy as np


class IModel(ABC):
    """
    Abstract base class for ML model wrappers.
    
    Defines contract for ONNX Runtime model sessions with
    VRAM management support for RTX 3050 (4GB) constraint.
    """
    
    @abstractmethod
    async def load(self) -> None:
        """
        Load model into memory/VRAM.
        
        Should be lazy - only load when first needed.
        
        Raises:
            ModelLoadError: If model cannot be loaded
        """
        pass
    
    @abstractmethod
    async def unload(self) -> None:
        """
        Unload model from memory/VRAM.
        
        Called by ModelManager during LRU eviction.
        """
        pass
    
    @abstractmethod
    async def infer(
        self,
        inputs: np.ndarray,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            inputs: Input tensor(s) matching model's expected shape
            batch_size: Override automatic batch size calculation
            
        Returns:
            Model output tensor(s)
            
        Raises:
            InferenceError: If inference fails
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model registry name."""
        pass
    
    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """Expected input tensor shape."""
        pass
    
    @property
    @abstractmethod
    def vram_usage_mb(self) -> int:
        """Estimated VRAM usage in megabytes."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model is currently loaded in memory."""
        pass
    
    @property
    def version(self) -> str:
        """Model version string."""
        return "1.0.0"
    
    @property
    def execution_providers(self) -> List[str]:
        """
        ONNX Runtime execution providers in preference order.
        
        Default: TensorRT > CUDA > CPU
        """
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


class ModelInfo:
    """Model metadata from registry."""
    
    def __init__(
        self,
        name: str,
        path: str,
        input_shape: Tuple[int, ...],
        vram_mb: int,
        version: str = "1.0.0",
        providers: Optional[List[str]] = None,
        quantization: Optional[str] = None
    ):
        self.name = name
        self.path = path
        self.input_shape = input_shape
        self.vram_mb = vram_mb
        self.version = version
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.quantization = quantization  # None, "INT8", "FP16"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "input_shape": self.input_shape,
            "vram_mb": self.vram_mb,
            "version": self.version,
            "providers": self.providers,
            "quantization": self.quantization
        }
