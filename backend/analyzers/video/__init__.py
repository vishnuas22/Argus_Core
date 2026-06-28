"""
Argus Core - Video Analyzer Package
===================================
Video deepfake detection with spatial, temporal, and lip-sync analysis.
"""

# Import sub-analyzers first (no circular dependency)
from analyzers.video.spatial import SpatialAnalyzer, get_spatial_analyzer
from analyzers.video.temporal import TemporalAnalyzer, get_temporal_analyzer
from analyzers.video.lipsync import LipSyncAnalyzer, get_lipsync_analyzer

# Import types
from schemas.schemas import VideoResult, SpatialResult, TemporalResult, LipSyncResult

__all__ = [
    'SpatialAnalyzer',
    'get_spatial_analyzer',
    'TemporalAnalyzer', 
    'get_temporal_analyzer',
    'LipSyncAnalyzer',
    'get_lipsync_analyzer',
    'VideoResult',
    'SpatialResult',
    'TemporalResult',
    'LipSyncResult',
]
