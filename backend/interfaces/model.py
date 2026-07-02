"""
Argus Core - Model Info
========================
Model metadata dataclass used by the model registry and manager.

This module provides the `ModelInfo` dataclass that holds metadata
about ML models (name, path, input shape, VRAM requirements, etc.).

Note: The original `IModel` abstract base class has been removed as it
had zero concrete implementations in the codebase.
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    """
    Model metadata from registry.
    
    Holds information about a model's location, expected input shape,
    VRAM requirements, and execution provider preferences.
    
    Used by:
    - ModelRegistry (backend/models/registry.py)
    - ModelManager (backend/models/manager.py)
    - InferenceEngine (backend/core/engine.py)
    """
    name: str
    path: str
    input_shape: Tuple[int, ...]
    vram_mb: int
    version: str = "1.0.0"
    providers: List[str] = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    quantization: Optional[str] = None  # None, "INT8", "FP16"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "input_shape": self.input_shape,
            "vram_mb": self.vram_mb,
            "version": self.version,
            "providers": self.providers,
            "quantization": self.quantization,
        }
