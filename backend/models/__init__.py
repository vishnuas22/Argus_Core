"""
Argus Core - Models Package
===========================
ML model management for deepfake detection.

Layer 3: Model Infrastructure
- registry.py: Model metadata and version tracking
- manager.py: VRAM management with LRU eviction
- optimize.py: ONNX/TensorRT optimization utilities
"""

from models.registry import ModelRegistry, get_model_registry
from models.manager import ModelManager, get_model_manager

__all__ = [
    "ModelRegistry",
    "get_model_registry",
    "ModelManager",
    "get_model_manager",
]
