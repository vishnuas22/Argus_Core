# Argus Core - Interfaces Module
# Abstract base classes defining contracts

from .analyzer import IAnalyzer
from .storage import IStorage
from .model import ModelInfo

__all__ = [
    "IAnalyzer",
    "IStorage",
    "ModelInfo",
]
