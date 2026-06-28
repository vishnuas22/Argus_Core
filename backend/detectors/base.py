from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


class DetectorBackend(str, Enum):
    ONNX = "onnx"
    PYTORCH = "pytorch"
    AUTODETECT = "autodetect"


@dataclass
class DetectionResult:
    score: float = 0.5
    confidence: float = 0.5
    model_name: str = ""
    backend: str = "onnx"
    features: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class BaseDetector(ABC):
    def __init__(
        self,
        name: str,
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
    ):
        self._name = name
        self._preferred_backend = preferred_backend
        self._model = None
        self._backend_used: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def backend_used(self) -> Optional[str]:
        return self._backend_used

    @abstractmethod
    def get_required_models(self) -> List[str]:
        ...

    @abstractmethod
    async def detect(self, *args, **kwargs) -> DetectionResult:
        ...

    def _normalize_score(self, score: float) -> float:
        return float(np.clip(score, 0.0, 1.0))
