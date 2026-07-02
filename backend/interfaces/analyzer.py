"""
Argus Core - Analyzer Interface
===============================
Abstract base class defining the contract for all modality analyzers.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/base.py
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from schemas import PreprocessedData, ModalityResult, Modality


class IAnalyzer(ABC):
    """
    Abstract base class for all analyzers.
    
    All modality analyzers (video, audio, image, metadata) must 
    implement this interface to ensure consistent behavior across
    the analysis pipeline.
    
    Contract Requirements:
    - analyze(): Main analysis method returning standardized results
    - get_required_models(): Declare models needed for VRAM planning
    - supports_modality(): Check if analyzer handles given modality
    - validate_input(): Input validation before processing
    """
    
    @abstractmethod
    async def analyze(
        self,
        data: "PreprocessedData",
        engine: object  # InferenceEngine - avoiding circular import
    ) -> "ModalityResult":
        """
        Run analysis on preprocessed data.
        
        Args:
            data: PreprocessedData containing extracted features
            engine: InferenceEngine for model inference
            
        Returns:
            ModalityResult with detection score and details
            
        Raises:
            InferenceError: If model inference fails
            ValidationError: If input data is invalid
        """
        pass
    
    @abstractmethod
    def get_required_models(self) -> List[str]:
        """
        Return list of model registry keys needed for this analyzer.
        
        Used by ModelManager for VRAM planning and preloading.
        
        Returns:
            List of model names from registry
            
        Example:
            return ["efficientnet_b3_spatial", "xclip_temporal"]
        """
        pass
    
    @abstractmethod
    def supports_modality(self, modality: "Modality") -> bool:
        """
        Check if this analyzer handles the given modality.
        
        Args:
            modality: Modality enum value to check
            
        Returns:
            True if analyzer supports this modality
        """
        pass
    
    def validate_input(self, data: "PreprocessedData") -> None:
        """
        Validate input data before analysis.
        
        Override to add modality-specific validation.
        
        Args:
            data: PreprocessedData to validate
            
        Raises:
            ValidationError: If data is invalid
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return analyzer name for logging and metrics."""
        pass
    
    @property
    def version(self) -> str:
        """Return analyzer version for tracking."""
        return "1.0.0"
