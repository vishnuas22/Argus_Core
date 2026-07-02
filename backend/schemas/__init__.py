# Argus Core - Schemas Module
# All Pydantic models and data structures

from .schemas import (
    # Enums
    Modality,
    AnalysisStatus,
    Verdict,
    ContentType,
    
    # Base
    BaseSchema,
    
    # Input Schemas
    FileInput,
    AnalyzeOptions,
    
    # Result Schemas
    TrustScore,
    SpatialResult,
    TemporalResult,
    LipSyncResult,
    VideoResult,
    AudioResult,
    C2PAManifest,
    MetadataResult,
    ManipulationRegion,
    Explanation,
    
    # Documents
    AnalysisDocument,
    
    # API Schemas
    AnalysisRequest,
    AnalysisResponse,
    AnalysisDetailResponse,
    
    # Internal Schemas
    PreprocessedData,
    ModalityResult,
    AggregatedResult,
)

__all__ = [
    "Modality",
    "AnalysisStatus",
    "Verdict",
    "ContentType",
    "BaseSchema",
    "FileInput",
    "AnalyzeOptions",
    "TrustScore",
    "SpatialResult",
    "TemporalResult",
    "LipSyncResult",
    "VideoResult",
    "AudioResult",
    "C2PAManifest",
    "MetadataResult",
    "ManipulationRegion",
    "Explanation",
    "AnalysisDocument",
    "AnalysisRequest",
    "AnalysisResponse",
    "AnalysisDetailResponse",
    "PreprocessedData",
    "ModalityResult",
    "AggregatedResult",
]
