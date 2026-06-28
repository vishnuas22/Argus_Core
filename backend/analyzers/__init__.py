"""
Argus Core - Analyzers Module
==============================
Modality-specific deepfake detection analyzers.

This module provides specialized analyzers for each media type:
- ImageAnalyzer: Single image deepfake detection
- VideoAnalyzer: Video deepfake detection with spatial, temporal, lipsync
- AudioAnalyzer: Synthetic voice detection
- MetadataAnalyzer: C2PA and EXIF metadata analysis
- MetadataAnalyzer: C2PA and EXIF metadata analysis

All analyzers inherit from BaseAnalyzer and implement the IAnalyzer interface.

Usage:
    from analyzers import get_image_analyzer, get_video_analyzer
    
    image_analyzer = get_image_analyzer()
    result = await image_analyzer.analyze(preprocessed_data, engine)
"""

from analyzers.base import (
    BaseAnalyzer,
    SubAnalyzer,
    AnalyzerMetrics,
    normalize_scores,
    aggregate_scores,
    compute_confidence,
    detect_anomalies
)

from analyzers.image import ImageAnalyzer, get_image_analyzer
from analyzers.video_analyzer import VideoAnalyzer, get_video_analyzer
from analyzers.audio import AudioAnalyzer, get_audio_analyzer
from analyzers.metadata import MetadataAnalyzer, get_metadata_analyzer

__all__ = [
    # Base classes
    "BaseAnalyzer",
    "SubAnalyzer",
    "AnalyzerMetrics",
    
    # Utility functions
    "normalize_scores",
    "aggregate_scores",
    "compute_confidence",
    "detect_anomalies",
    
    # Analyzers
    "ImageAnalyzer",
    "VideoAnalyzer",
    "AudioAnalyzer",
    "MetadataAnalyzer",
    
    # Singleton getters
    "get_image_analyzer",
    "get_video_analyzer",
    "get_audio_analyzer",
    "get_metadata_analyzer",
]
