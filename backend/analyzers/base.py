"""
Argus Core - Base Analyzer
==========================
Abstract base class defining the analyzer interface for all modality analyzers.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/base.py

This file provides:
- BaseAnalyzer abstract class with standard interface
- Common utilities shared across all analyzers
- Input validation methods
- Logging and metrics integration

All modality analyzers (video, audio, image, text, metadata) inherit from this.

Integration:
- Imports: interfaces/analyzer.py, schemas/internal.py
- Inputs: N/A (abstract)
- Outputs: N/A (abstract)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import time
import numpy as np

from interfaces.analyzer import IAnalyzer
from schemas.schemas import (
    Modality, PreprocessedData, ModalityResult, ContentType
)
from utils.logging import get_logger
from utils.errors import ValidationError

if TYPE_CHECKING:
    from core.engine import InferenceEngine

logger = get_logger(__name__)


@dataclass
class AnalyzerMetrics:
    """
    Metrics tracking for analyzer performance.
    
    Tracks timing, success rates, and model usage.
    """
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    total_inference_time_ms: float = 0.0
    average_confidence: float = 0.0
    
    def record_analysis(
        self,
        success: bool,
        inference_time_ms: float,
        confidence: float
    ) -> None:
        """Record metrics for a single analysis."""
        self.total_analyses += 1
        if success:
            self.successful_analyses += 1
        else:
            self.failed_analyses += 1
        self.total_inference_time_ms += inference_time_ms
        
        # Running average for confidence
        if self.total_analyses > 0:
            self.average_confidence = (
                (self.average_confidence * (self.total_analyses - 1) + confidence)
                / self.total_analyses
            )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_analyses == 0:
            return 0.0
        return self.successful_analyses / self.total_analyses
    
    @property
    def average_inference_time_ms(self) -> float:
        """Calculate average inference time."""
        if self.total_analyses == 0:
            return 0.0
        return self.total_inference_time_ms / self.total_analyses
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "success_rate": round(self.success_rate, 3),
            "average_inference_time_ms": round(self.average_inference_time_ms, 2),
            "average_confidence": round(self.average_confidence, 3)
        }


class BaseAnalyzer(IAnalyzer):
    """
    Abstract base class for all modality analyzers.
    
    Provides common functionality:
    - Standard analyze() workflow with timing and metrics
    - Input validation framework
    - Logging integration
    - Error handling patterns
    
    Subclasses must implement:
    - _analyze_impl(): Core analysis logic
    - get_required_models(): List of model registry keys
    - supports_modality(): Modality support check
    
    Usage:
        class ImageAnalyzer(BaseAnalyzer):
            async def _analyze_impl(self, data, engine):
                # Implementation
                return ModalityResult(...)
    """
    
    def __init__(
        self,
        analyzer_name: str,
        supported_modalities: List[Modality],
        version: str = "1.0.0"
    ):
        """
        Initialize base analyzer.
        
        Args:
            analyzer_name: Human-readable name for logging
            supported_modalities: List of modalities this analyzer handles
            version: Analyzer version for tracking
        """
        self._name = analyzer_name
        self._supported_modalities = supported_modalities
        self._version = version
        self._metrics = AnalyzerMetrics()
        
        logger.info(
            f"Initialized {analyzer_name} v{version}, "
            f"supporting modalities: {[m.value for m in supported_modalities]}"
        )
    
    @property
    def name(self) -> str:
        """Return analyzer name."""
        return self._name
    
    @property
    def version(self) -> str:
        """Return analyzer version."""
        return self._version
    
    @property
    def metrics(self) -> AnalyzerMetrics:
        """Return analyzer metrics."""
        return self._metrics
    
    def supports_modality(self, modality: Modality) -> bool:
        """
        Check if this analyzer supports the given modality.
        
        Args:
            modality: Modality to check
            
        Returns:
            True if supported
        """
        return modality in self._supported_modalities
    
    async def analyze(
        self,
        data: PreprocessedData,
        engine: "InferenceEngine"
    ) -> ModalityResult:
        """
        Run analysis on preprocessed data.
        
        This is the main entry point that wraps _analyze_impl with:
        - Input validation
        - Timing and metrics
        - Error handling
        - Logging
        
        Args:
            data: PreprocessedData containing extracted features
            engine: InferenceEngine for model inference
            
        Returns:
            ModalityResult with detection score and details
            
        Raises:
            ValidationError: If input data is invalid
            InferenceError: If model inference fails
        """
        start_time = time.time()
        success = False
        confidence = 0.0
        
        try:
            # Validate input
            self.validate_input(data)
            
            logger.debug(f"{self.name}: Starting analysis for {data.analysis_id}")
            
            # Run implementation
            result = await self._analyze_impl(data, engine)
            
            success = True
            confidence = result.confidence
            
            logger.info(
                f"{self.name}: Analysis complete for {data.analysis_id}, "
                f"score={result.score:.3f}, confidence={result.confidence:.3f}"
            )
            
            return result
            
        except ValidationError:
            logger.error(f"{self.name}: Validation failed for {data.analysis_id}")
            raise
            
        except Exception as e:
            logger.error(f"{self.name}: Analysis failed for {data.analysis_id}: {e}")
            raise
            
        finally:
            inference_time_ms = (time.time() - start_time) * 1000
            self._metrics.record_analysis(success, inference_time_ms, confidence)
    
    @abstractmethod
    async def _analyze_impl(
        self,
        data: PreprocessedData,
        engine: "InferenceEngine"
    ) -> ModalityResult:
        """
        Core analysis implementation.
        
        Subclasses must implement this method with their specific
        analysis logic.
        
        Args:
            data: Validated PreprocessedData
            engine: InferenceEngine for model inference
            
        Returns:
            ModalityResult with score and details
        """
        pass
    
    @abstractmethod
    def get_required_models(self) -> List[str]:
        """
        Return list of model registry keys needed.
        
        Used by ModelManager for VRAM planning.
        
        Returns:
            List of model names from registry
        """
        pass
    
    def validate_input(self, data: PreprocessedData) -> None:
        """
        Validate input data before analysis.
        
        Default implementation checks for valid analysis_id.
        Subclasses should override to add modality-specific validation.
        
        Args:
            data: PreprocessedData to validate
            
        Raises:
            ValidationError: If data is invalid
        """
        if not data.analysis_id:
            raise ValidationError("analysis_id is required")
        
        # Check content type compatibility
        if not self._is_content_type_compatible(data.content_type):
            raise ValidationError(
                f"{self.name} does not support content type {data.content_type}"
            )
    
    def _is_content_type_compatible(self, content_type: ContentType) -> bool:
        """
        Check if content type is compatible with this analyzer.
        
        Args:
            content_type: Detected content type
            
        Returns:
            True if compatible
        """
        modality_content_map = {
            Modality.VIDEO: [
                ContentType.VIDEO_WITH_SPEECH,
                ContentType.VIDEO_NO_SPEECH
            ],
            Modality.AUDIO: [
                ContentType.VIDEO_WITH_SPEECH,
                ContentType.AUDIO_ONLY
            ],
            Modality.IMAGE: [
                ContentType.IMAGE_ONLY,
                ContentType.VIDEO_WITH_SPEECH,
                ContentType.VIDEO_NO_SPEECH
            ],
            Modality.TEXT: [
                ContentType.TEXT_ONLY
            ]
        }
        
        for modality in self._supported_modalities:
            compatible_types = modality_content_map.get(modality, [])
            if content_type in compatible_types:
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get analyzer statistics.
        
        Returns:
            Dict with analyzer name, version, and metrics
        """
        return {
            "name": self.name,
            "version": self.version,
            "supported_modalities": [m.value for m in self._supported_modalities],
            "metrics": self._metrics.to_dict()
        }
    
    def reset_metrics(self) -> None:
        """Reset analyzer metrics."""
        self._metrics = AnalyzerMetrics()
        logger.debug(f"{self.name}: Metrics reset")


class SubAnalyzer:
    """
    Base class for sub-analyzers (e.g., spatial, temporal, lipsync within video).
    
    Sub-analyzers are components that don't implement the full IAnalyzer interface
    but provide specific analysis capabilities.
    
    Usage:
        class SpatialAnalyzer(SubAnalyzer):
            async def analyze_frames(self, frames, engine):
                # Implementation
                return SpatialResult(...)
    """
    
    def __init__(self, name: str):
        """
        Initialize sub-analyzer.
        
        Args:
            name: Sub-analyzer name for logging
        """
        self._name = name
        self._metrics = AnalyzerMetrics()
        logger.debug(f"Initialized sub-analyzer: {name}")
    
    @property
    def name(self) -> str:
        """Return sub-analyzer name."""
        return self._name
    
    @property
    def metrics(self) -> AnalyzerMetrics:
        """Return sub-analyzer metrics."""
        return self._metrics
    
    def get_required_models(self) -> List[str]:
        """
        Return models required by this sub-analyzer.
        
        Override in subclasses.
        
        Returns:
            List of model registry keys
        """
        return []
    
    def record_analysis(
        self,
        success: bool,
        inference_time_ms: float,
        confidence: float
    ) -> None:
        """Record metrics for sub-analyzer."""
        self._metrics.record_analysis(success, inference_time_ms, confidence)


# Utility functions for analyzers

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: Raw scores array
        
    Returns:
        Normalized scores in [0, 1]
    """
    if len(scores) == 0:
        return scores
    
    min_val = np.min(scores)
    max_val = np.max(scores)
    
    if max_val - min_val < 1e-8:
        return np.full_like(scores, 0.5)
    
    return (scores - min_val) / (max_val - min_val)


def aggregate_scores(
    scores: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method: str = "weighted_mean"
) -> float:
    """
    Aggregate multiple scores into single value.
    
    Args:
        scores: Array of scores
        weights: Optional weights (uniform if None)
        method: Aggregation method (weighted_mean, median, max)
        
    Returns:
        Aggregated score
    """
    if len(scores) == 0:
        return 0.5
    
    if weights is None:
        weights = np.ones_like(scores) / len(scores)
    else:
        weights = weights / np.sum(weights)
    
    if method == "weighted_mean":
        return float(np.dot(scores, weights))
    elif method == "median":
        return float(np.median(scores))
    elif method == "max":
        return float(np.max(scores))
    else:
        return float(np.dot(scores, weights))


def compute_confidence(
    scores: np.ndarray,
    num_samples: int,
    min_samples: int = 10
) -> float:
    """
    Compute confidence based on score consistency and sample count.
    
    Args:
        scores: Array of individual scores
        num_samples: Number of samples analyzed
        min_samples: Minimum samples for full confidence (only affects multi-sample)
        
    Returns:
        Confidence score [0, 1]
    """
    if len(scores) == 0:
        return 0.5  # Neutral confidence when no scores
    
    # Sample count factor
    # For single images (num_samples=1), use full confidence - this is the expected case
    # For multiple samples, scale based on count
    if num_samples == 1:
        sample_factor = 1.0  # Single image is the normal case, not a low-confidence case
    else:
        sample_factor = min(1.0, num_samples / min_samples)
    
    # Consistency factor (lower variance = higher confidence)
    variance = np.var(scores)
    consistency_factor = 1.0 / (1.0 + variance * 4)  # Smooth decay
    
    # Extremity factor (scores near 0 or 1 get higher confidence)
    mean_score = np.mean(scores)
    extremity = abs(mean_score - 0.5) * 2
    extremity_factor = 0.7 + 0.3 * extremity
    
    # Combine factors
    confidence = sample_factor * consistency_factor * extremity_factor
    
    # Allow full range - don't artificially cap at 0.3 minimum
    # Low confidence should only come from actual uncertainty signals
    return float(np.clip(confidence, 0.1, 0.95))


def detect_anomalies(
    scores: np.ndarray,
    threshold: float = 2.0
) -> List[int]:
    """
    Detect anomalous scores using z-score method.
    
    Args:
        scores: Array of scores
        threshold: Z-score threshold for anomaly
        
    Returns:
        List of indices with anomalous scores
    """
    if len(scores) < 3:
        return []
    
    mean = np.mean(scores)
    std = np.std(scores)
    
    if std < 1e-8:
        return []
    
    z_scores = np.abs((scores - mean) / std)
    anomaly_indices = np.where(z_scores > threshold)[0]
    
    return anomaly_indices.tolist()
