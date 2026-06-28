"""
Argus Core - Complete Schema Definitions
========================================
All Pydantic models for the deepfake detection platform.
This is the single source of truth for all data structures.

Implements: PRIME_ARGUS_DOCUMENT.md - Appendix A: Shared Schemas
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime, timezone
import uuid


# ============== ENUMS ==============

class Modality(str, Enum):
    """Supported media modalities for deepfake analysis."""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"


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


class ManipulationRegion(BaseSchema):
    """Detected manipulation region for explainability."""
    region_type: str = Field(..., description="Region type: face, mouth, background, etc.")
    location: str = Field(..., description="Description or coordinates of region")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    frame_indices: Optional[List[int]] = Field(default=None, description="Affected frame indices")


class AudioArtifactRegion(BaseSchema):
    """
    Detected artifact region in audio spectrogram.
    
    Marks regions with synthetic voice artifacts.
    """
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds")
    freq_low: float = Field(..., ge=0, description="Low frequency bound in Hz")
    freq_high: float = Field(..., ge=0, description="High frequency bound in Hz")
    artifact_type: str = Field(..., description="Type: 'vocoder', 'spectral_gap', 'harmonic_inconsistency'")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")


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
    # XAI Enhancement Fields
    dct_anomaly_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="DCT frequency domain anomaly score"
    )
    gan_fingerprint_detected: bool = Field(
        default=False,
        description="Whether GAN fingerprint was detected"
    )
    manipulation_regions: List[ManipulationRegion] = Field(
        default_factory=list,
        description="Detailed manipulation region information"
    )
    efficientnet_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Raw EfficientNet-B3 deepfake classifier score"
    )
    clip_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="CLIP-based semantic consistency score"
    )


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
    # XAI Enhancement Fields
    motion_anomaly_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Score for unnatural motion patterns"
    )
    landmark_jitter_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Facial landmark jitter analysis score"
    )
    xclip_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="X-CLIP temporal consistency model score"
    )


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
    # XAI Enhancement Fields
    lip_region_heatmap_url: Optional[str] = Field(
        default=None,
        description="URL to lip region attention heatmap"
    )
    audio_visual_offset_ms: Optional[float] = Field(
        default=None,
        description="Detected audio-visual offset in milliseconds"
    )
    confidence_interval: Optional[Dict[str, float]] = Field(
        default=None,
        description="Confidence interval for sync score"
    )


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
    # XAI Enhancement Fields
    frame_heatmap_urls: List[str] = Field(
        default_factory=list,
        description="URLs to frame-level heatmaps"
    )
    temporal_heatmap_url: Optional[str] = Field(
        default=None,
        description="URL to temporal analysis visualization"
    )
    confidence_interval: Optional[Dict[str, float]] = Field(
        default=None,
        description="Confidence interval for aggregate score"
    )


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
    # XAI Enhancement Fields
    artifact_regions: List[AudioArtifactRegion] = Field(
        default_factory=list,
        description="Detected artifact regions in spectrogram"
    )
    frequency_anomaly_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Score indicating frequency domain anomalies"
    )
    aasist_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="AASIST anti-spoofing model score"
    )


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


class ImageResult(BaseSchema):
    """
    Image deepfake detection results.
    
    Combines multiple detection signals:
    - SigLIP/EfficientNet classifier scores
    - DCT frequency analysis
    - Face manipulation detection
    """
    ai_generated_probability: float = Field(..., ge=0, le=1, description="Probability image is AI-generated")
    fake_probability: float = Field(..., ge=0, le=1, description="Probability image is manipulated/deepfake")
    face_detected: bool = Field(default=False, description="Whether face was detected")
    num_faces: int = Field(default=0, description="Number of faces detected")
    face_manipulation_scores: List[float] = Field(default_factory=list, description="Per-face manipulation scores")
    heatmap_url: Optional[str] = Field(default=None, description="URL to GradCAM heatmap")
    # DCT Features
    dct_anomaly_score: float = Field(default=0.0, ge=0, le=1, description="DCT frequency anomaly score")
    spectral_flatness: float = Field(default=0.0, ge=0, le=1, description="Spectral flatness measure")
    # Ensemble scores
    ensemble_score: float = Field(default=0.0, ge=0, le=1, description="Ensemble model score")
    ensemble_primary_available: bool = Field(default=False, description="Primary ensemble model available")
    ensemble_secondary_available: bool = Field(default=False, description="Secondary ensemble model available")
    # XAI Enhancement Fields
    manipulation_regions: List[ManipulationRegion] = Field(
        default_factory=list,
        description="Detected manipulation regions"
    )
    confidence_interval: Optional[Dict[str, float]] = Field(
        default=None,
        description="Confidence interval for AI probability"
    )


# ============== XAI (EXPLAINABLE AI) SCHEMAS ==============

class FeatureImportance(BaseSchema):
    """
    Feature-level importance scores for XAI.
    
    Indicates which features most influenced the prediction.
    Used for court-admissible evidence documentation.
    """
    feature_name: str = Field(..., description="Name or identifier of the feature")
    importance_score: float = Field(..., ge=0, le=1, description="Importance score 0-1")
    contribution_direction: str = Field(
        ..., 
        description="Direction of contribution: 'increases_fake' or 'decreases_fake'"
    )
    confidence: float = Field(..., ge=0, le=1, description="Confidence in importance score")
    feature_type: str = Field(
        default="spatial",
        description="Type: 'spatial', 'frequency', 'temporal', 'linguistic', 'acoustic'"
    )


class VisualEvidence(BaseSchema):
    """
    Visual evidence artifact for forensic reports.
    
    Each piece of visual evidence is stored with integrity hash
    for chain of custody verification.
    """
    artifact_type: str = Field(
        ..., 
        description="Type: 'heatmap', 'spectrogram', 'frequency_plot', 'overlay', 'temporal_chart'"
    )
    url: str = Field(..., description="MinIO URL or presigned URL to the artifact")
    description: str = Field(..., description="Human-readable description of the evidence")
    frame_index: Optional[int] = Field(default=None, description="Frame number for video evidence")
    timestamp_seconds: Optional[float] = Field(default=None, description="Timestamp for audio/video evidence")
    integrity_hash: str = Field(..., description="SHA-256 hash of artifact content for chain of custody")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")


class EvidencePackage(BaseSchema):
    """
    Complete evidence package for court-admissible reports.
    
    Contains all visual evidence, feature importance data, and reproducibility
    information required for legal proceedings.
    """
    visual_evidence: List[VisualEvidence] = Field(
        default_factory=list,
        description="List of visual evidence artifacts"
    )
    feature_importance: List[FeatureImportance] = Field(
        default_factory=list,
        description="Feature importance scores"
    )
    audio_artifact_regions: Optional[List[AudioArtifactRegion]] = Field(
        default=None,
        description="Audio artifact regions for audio analysis"
    )
    model_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Model name -> version mapping"
    )
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the analysis was performed"
    )
    integrity_hash: str = Field(
        default="",
        description="SHA-256 hash of entire package for chain of custody"
    )
    reproducibility_hash: str = Field(
        default="",
        description="SHA-256 hash for reproducibility verification"
    )
    reproducibility_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters, thresholds, and settings for reproducibility"
    )
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None,
        description="95% confidence interval (lower, upper) for the prediction"
    )


class ScientificReference(BaseSchema):
    """
    Scientific reference for methodology documentation.
    
    Provides peer-reviewed citations for detection methods used.
    """
    method_name: str = Field(..., description="Name of the detection method")
    citation: str = Field(..., description="Full academic citation")
    doi: Optional[str] = Field(default=None, description="DOI link if available")
    accuracy_metrics: Optional[str] = Field(default=None, description="Known accuracy metrics")


class Explanation(BaseSchema):
    """
    Human-readable explanation of analysis results.
    
    Uses GradCAM++ and template-based generation.
    Enhanced with XAI features for court-admissible evidence.
    """
    summary: str = Field(..., description="Executive summary of findings")
    key_findings: List[str] = Field(default_factory=list, description="Bullet-point key findings")
    manipulation_regions: List[ManipulationRegion] = Field(default_factory=list, description="Detected manipulation regions")
    confidence_rationale: str = Field(default="", description="Explanation of confidence level")
    methodology_used: List[str] = Field(default_factory=list, description="Analysis methods applied")
    # XAI Enhancement Fields
    feature_importance: List[FeatureImportance] = Field(
        default_factory=list,
        description="Feature-level importance scores"
    )
    evidence_package: Optional[EvidencePackage] = Field(
        default=None,
        description="Complete evidence package with visual artifacts"
    )
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None,
        description="95% confidence interval (lower, upper)"
    )
    scientific_references: List[ScientificReference] = Field(
        default_factory=list,
        description="Peer-reviewed citations for methods used"
    )
    heatmap_urls: List[str] = Field(
        default_factory=list,
        description="URLs to GradCAM heatmap visualizations"
    )
    reproducibility_hash: str = Field(
        default="",
        description="SHA-256 hash for result reproducibility verification"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for the user"
    )
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
    image_result: Optional[ImageResult] = None
    metadata_result: Optional[MetadataResult] = None
    explanation: Optional[Explanation] = None
    
    # Outputs
    report_url: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    
    # XAI Enhancement Fields (for court-admissible evidence)
    evidence_package: Optional[EvidencePackage] = None
    feature_importance: List[FeatureImportance] = Field(default_factory=list)
    scientific_references: List[ScientificReference] = Field(default_factory=list)
    
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
    # XAI Enhancement Fields
    heatmap_urls: List[str] = Field(
        default_factory=list,
        description="URLs to GradCAM heatmap visualizations"
    )
    evidence_package_url: Optional[str] = Field(
        default=None,
        description="URL to download complete evidence package"
    )
    confidence_interval: Optional[Dict[str, float]] = Field(
        default=None,
        description="95% confidence interval: {'lower': x, 'upper': y}"
    )


class AnalysisDetailResponse(AnalysisResponse):
    """Detailed API response with full results."""
    video_result: Optional[VideoResult] = None
    audio_result: Optional[AudioResult] = None
    image_result: Optional[ImageResult] = None
    metadata_result: Optional[MetadataResult] = None
    processing_time_seconds: Optional[float] = None
    # XAI Enhancement Fields
    evidence_package: Optional[EvidencePackage] = Field(
        default=None,
        description="Complete evidence package with all XAI artifacts"
    )
    feature_importance: List[FeatureImportance] = Field(
        default_factory=list,
        description="Feature-level importance scores"
    )
    scientific_references: List[ScientificReference] = Field(
        default_factory=list,
        description="Peer-reviewed citations for methods used"
    )


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
