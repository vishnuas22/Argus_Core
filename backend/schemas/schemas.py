"""
Argus Core - Complete Schema Definitions
========================================
All Pydantic models for the deepfake detection platform.
This is the single source of truth for all data structures.

Implements: PRIME_ARGUS_DOCUMENT.md - Appendix A: Shared Schemas
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, timezone
import uuid


# ============== ENUMS ==============

class Modality(str, Enum):
    """Supported media modalities for analysis."""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"


class AnalysisStatus(str, Enum):
    """Analysis pipeline status tracking."""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    ANALYZING = "analyzing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class Verdict(str, Enum):
    """
    Final authenticity verdict.
    
    Score Ranges (configurable):
    - 80-100: AUTHENTIC
    - 60-79: LIKELY_AUTHENTIC
    - 40-59: UNCERTAIN (flag for human review)
    - 20-39: LIKELY_FAKE
    - 0-19: FAKE
    """
    AUTHENTIC = "authentic"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    LIKELY_FAKE = "likely_fake"
    FAKE = "fake"


class ContentType(str, Enum):
    """Detected content type for analysis routing."""
    VIDEO_WITH_SPEECH = "video_with_speech"
    VIDEO_NO_SPEECH = "video_no_speech"
    AUDIO_ONLY = "audio_only"
    IMAGE_ONLY = "image_only"
    TEXT_ONLY = "text_only"


# ============== BASE MODELS ==============

class BaseSchema(BaseModel):
    """Base schema with standard configuration."""
    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
        populate_by_name=True
    )


# ============== INPUT SCHEMAS ==============

class FileInput(BaseSchema):
    """Uploaded file metadata."""
    file_id: str
    file_type: str
    original_filename: str
    file_hash: str
    file_size: int
    duration_seconds: Optional[float] = None


class AnalyzeOptions(BaseSchema):
    """Analysis configuration options."""
    modalities: Optional[List[Modality]] = None  # None = auto-detect
    generate_report: bool = True
    generate_heatmaps: bool = True
    defense_level: str = "standard"  # none, standard, aggressive


# ============== RESULT SCHEMAS ==============

class TrustScore(BaseSchema):
    """
    Calibrated Trust Score (0-100).
    
    Uses Platt scaling for well-calibrated probabilities.
    Score represents true probability (e.g., 70 = 70% confidence).
    """
    value: float = Field(..., ge=0, le=100, description="Trust score from 0-100")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence in prediction")
    calibrated: bool = Field(default=True, description="Whether Platt calibration was applied")


class SpatialResult(BaseSchema):
    """
    Per-frame spatial artifact detection results.
    
    Uses EfficientNet-B3 with CLIP guidance for:
    - Blending boundary detection
    - Texture inconsistencies
    - Frequency domain artifacts (DCT analysis)
    """
    score: float = Field(..., ge=0, le=1, description="Aggregate spatial manipulation score")
    per_frame_scores: List[float] = Field(default_factory=list, description="Per-frame detection scores")
    anomaly_indices: List[int] = Field(default_factory=list, description="Frame indices with detected anomalies")
    heatmap_urls: List[str] = Field(default_factory=list, description="GradCAM heatmap URLs for anomaly frames")


class TemporalResult(BaseSchema):
    """
    Cross-frame temporal consistency results.
    
    Uses X-CLIP transformer for:
    - Flickering detection
    - Unnatural motion patterns
    - Landmark jitter analysis
    - Inter-frame color consistency
    """
    consistency_score: float = Field(..., ge=0, le=1, description="Temporal consistency score")
    flickering_detected: bool = Field(default=False, description="Whether flickering artifacts detected")
    anomaly_timestamps: List[float] = Field(default_factory=list, description="Timestamps of detected anomalies")


class LipSyncResult(BaseSchema):
    """
    Lip-sync deepfake detection results.
    
    Uses LIPINC-V2 for detecting:
    - Wav2Lip manipulations
    - Diff2Lip artifacts
    - Video_Retalking inconsistencies
    - Audio-visual desynchronization
    """
    sync_score: float = Field(..., ge=0, le=1, description="Audio-visual sync score")
    manipulation_probability: float = Field(..., ge=0, le=1, description="Lip-sync manipulation probability")
    detected_technology: Optional[str] = Field(default=None, description="Detected lip-sync technology if identified")


class VideoResult(BaseSchema):
    """
    Aggregated video analysis results.
    
    Combines spatial, temporal, and lip-sync sub-analyzer outputs
    with weighted ensemble voting.
    """
    spatial: SpatialResult
    temporal: TemporalResult
    lip_sync: Optional[LipSyncResult] = None
    aggregate_score: float = Field(..., ge=0, le=1, description="Weighted aggregate video score")
    frames_analyzed: int = Field(default=0, description="Number of frames analyzed")
    face_detected: bool = Field(default=False, description="Whether face was detected in video")


class AudioResult(BaseSchema):
    """
    Audio deepfake detection results.
    
    Uses Purdue-M2 architecture for:
    - Vocoder artifact detection
    - Spectral inconsistency analysis
    - Voice consistency scoring
    """
    synthetic_probability: float = Field(..., ge=0, le=1, description="Probability audio is synthetic")
    vocoder_artifacts_detected: bool = Field(default=False, description="Whether vocoder artifacts detected")
    voice_consistency_score: float = Field(..., ge=0, le=1, description="Voice consistency across segments")
    spectrogram_url: Optional[str] = Field(default=None, description="URL to mel-spectrogram visualization")


class TextResult(BaseSchema):
    """
    AI-generated text detection results.
    
    Uses RADAR model with perplexity/burstiness analysis:
    - Low perplexity = likely AI (too predictable)
    - Low burstiness = likely AI (uniform variance)
    """
    ai_probability: float = Field(..., ge=0, le=1, description="Probability text is AI-generated")
    perplexity_score: float = Field(default=0.0, description="GPT-2 perplexity score")
    burstiness_score: float = Field(default=0.0, description="Sentence length variance")
    radar_score: Optional[float] = Field(default=None, description="RADAR classifier score")


class C2PAManifest(BaseSchema):
    """
    C2PA Content Credentials data.
    
    Implements C2PA v2.3 specification for content authenticity.
    """
    present: bool = Field(default=False, description="Whether C2PA manifest exists")
    valid: Optional[bool] = Field(default=None, description="Manifest validation result")
    issuer: Optional[str] = Field(default=None, description="Certificate issuer")
    issued_at: Optional[datetime] = Field(default=None, description="Manifest issuance timestamp")
    assertions: List[Dict[str, Any]] = Field(default_factory=list, description="C2PA assertions")


class MetadataResult(BaseSchema):
    """
    Media metadata analysis results.
    
    Includes C2PA, EXIF analysis, and file structure verification.
    """
    c2pa: C2PAManifest = Field(default_factory=C2PAManifest)
    exif_anomalies: List[str] = Field(default_factory=list, description="Detected EXIF anomalies")
    file_structure_valid: bool = Field(default=True, description="File structure integrity check")


class ManipulationRegion(BaseSchema):
    """Detected manipulation region for explainability."""
    region_type: str = Field(..., description="Region type: face, mouth, background, etc.")
    location: str = Field(..., description="Description or coordinates of region")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    frame_indices: Optional[List[int]] = Field(default=None, description="Affected frame indices")


class Explanation(BaseSchema):
    """
    Human-readable explanation of analysis results.
    
    Uses GradCAM++ and template-based generation.
    """
    summary: str = Field(..., description="Executive summary of findings")
    key_findings: List[str] = Field(default_factory=list, description="Bullet-point key findings")
    manipulation_regions: List[ManipulationRegion] = Field(default_factory=list, description="Detected manipulation regions")
    confidence_rationale: str = Field(default="", description="Explanation of confidence level")
    methodology_used: List[str] = Field(default_factory=list, description="Analysis methods applied")


# ============== ANALYSIS DOCUMENT ==============

class AnalysisDocument(BaseSchema):
    """
    Complete analysis record stored in MongoDB.
    
    Tracks full lifecycle from upload to completion.
    """
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Input
    input: Optional[FileInput] = None
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)
    
    # Results (populated after analysis)
    trust_score: Optional[TrustScore] = None
    verdict: Optional[Verdict] = None
    video_result: Optional[VideoResult] = None
    audio_result: Optional[AudioResult] = None
    text_result: Optional[TextResult] = None
    metadata_result: Optional[MetadataResult] = None
    explanation: Optional[Explanation] = None
    
    # Outputs
    report_url: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None


# ============== API SCHEMAS ==============

class AnalysisRequest(BaseSchema):
    """API request for new analysis."""
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


class AnalysisResponse(BaseSchema):
    """Basic API response for analysis status."""
    analysis_id: str
    status: AnalysisStatus
    trust_score: Optional[TrustScore] = None
    verdict: Optional[Verdict] = None
    explanation: Optional[Explanation] = None
    report_url: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class AnalysisDetailResponse(AnalysisResponse):
    """Detailed API response with full results."""
    video_result: Optional[VideoResult] = None
    audio_result: Optional[AudioResult] = None
    text_result: Optional[TextResult] = None
    metadata_result: Optional[MetadataResult] = None
    processing_time_seconds: Optional[float] = None


# ============== INTERNAL SCHEMAS ==============

class PreprocessedData(BaseSchema):
    """
    Preprocessed media data ready for analysis.
    
    Contains MinIO keys to extracted frames, audio, etc.
    """
    analysis_id: str
    content_type: ContentType
    frames: Optional[List[str]] = None  # MinIO keys
    face_crops: Optional[List[str]] = None  # MinIO keys
    audio_key: Optional[str] = None  # MinIO key
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModalityResult(BaseSchema):
    """Individual modality analysis result for aggregation."""
    modality: Modality
    score: float = Field(..., ge=0, le=1, description="Detection score for this modality")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    details: Dict[str, Any] = Field(default_factory=dict)


class AggregatedResult(BaseSchema):
    """
    Multi-modal fusion result.
    
    Uses attention-weighted aggregation with uncertainty quantification.
    """
    modality_results: List[ModalityResult] = Field(default_factory=list)
    fused_score: float = Field(..., ge=0, le=1, description="Weighted fused score")
    uncertainty: float = Field(..., ge=0, le=1, description="Ensemble disagreement measure")
    weights_used: Dict[str, float] = Field(default_factory=dict, description="Weights applied per modality")


# ============== WEBSOCKET SCHEMAS ==============

class ProgressUpdate(BaseSchema):
    """WebSocket progress update message."""
    analysis_id: str
    status: AnalysisStatus
    progress_percent: float = Field(..., ge=0, le=100)
    current_stage: str
    message: Optional[str] = None


class ErrorResponse(BaseSchema):
    """Standardized error response."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
