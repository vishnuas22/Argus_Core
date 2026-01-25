# Argus Core - Interfaces Module
# Abstract base classes defining contracts

from .analyzer import IAnalyzer
from .storage import IStorage
from .model import IModel

__all__ = [
    "IAnalyzer",
    "IStorage",
    "IModel",
]
