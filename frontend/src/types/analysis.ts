/**
 * Argus Core - Analysis Type Definitions
 * ======================================
 * Core TypeScript types matching backend schemas exactly.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - types/analysis.ts
 * 
 * Role: Define all data types for analysis flow. Ensure type safety across frontend.
 * 
 * Integration:
 * - Maps to: backend/schemas/schemas.py
 * - Used by: All components, hooks, and services
 */

// ============== ENUMS ==============

/**
 * Analysis status enum matching backend AnalysisStatus
 */
export type AnalysisStatus =
  | 'pending'
  | 'preprocessing'
  | 'analyzing'
  | 'aggregating'
  | 'completed'
  | 'failed';

/**
 * Verdict classification for analysis results
 */
export type Verdict =
  | 'authentic'
  | 'likely_authentic'
  | 'uncertain'
  | 'likely_fake'
  | 'fake';

/**
 * Supported media modalities
 */
export type Modality = 'video' | 'audio' | 'image';

/**
 * Defense level for adversarial protection
 */
export type DefenseLevel = 'none' | 'standard' | 'aggressive';

// ============== CORE TYPES ==============

/**
 * Trust Score with overall score and confidence
 */
export interface TrustScore {
  overall: number;      // 0-100
  confidence: number;   // 0-1
  breakdown: ScoreBreakdown;
}

/**
 * Score breakdown by modality with weights
 */
export interface ScoreBreakdown {
  video_spatial?: number;
  video_temporal?: number;
  video_lipsync?: number;
  audio?: number;
  text?: number;
  metadata?: number;
  weights: Record<string, number>;
}

/**
 * AI-generated explanation for verdict
 */
export interface Explanation {
  summary: string;
  confidence_statement: string;
  key_findings: string[];
  recommendations?: string[];
}

// ============== FILE INPUT ==============

/**
 * Input file information
 */
export interface FileInput {
  file_id: string;
  file_type: string;
  original_filename: string;
  file_hash: string;
  file_size: number;
  duration_seconds?: number;
}

// ============== ANALYSIS OPTIONS ==============

/**
 * Options for analysis request
 */
export interface AnalysisOptions {
  generateReport: boolean;
  generateHeatmaps: boolean;
  defenseLevel: DefenseLevel;
  modalities?: Modality[];
}

// ============== VIDEO RESULTS ==============

/**
 * Bounding box for face detection
 */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

/**
 * Per-frame analysis result
 */
export interface FrameResult {
  frame_number: number;
  timestamp_ms: number;
  score: number;
  faces_detected: number;
  bounding_boxes: BoundingBox[];
  heatmap_url?: string;
}

/**
 * Spatial analysis result (per-frame)
 */
export interface SpatialResult {
  score: number;
  confidence: number;
  model_used: string;
  frames_analyzed: number;
  anomaly_frames: FrameResult[];
  average_face_confidence: number;
}

/**
 * Temporal analysis result (consistency)
 */
export interface TemporalResult {
  score: number;
  confidence: number;
  model_used: string;
  flickering_detected: boolean;
  consistency_score: number;
  motion_artifacts: number;
}

/**
 * Lip-sync analysis result
 */
export interface LipsyncResult {
  score: number;
  confidence: number;
  model_used: string;
  sync_offset_ms: number;
  phoneme_mismatches: number;
}

/**
 * Combined video analysis result
 */
export interface VideoResult {
  aggregated_score: number;
  confidence: number;
  spatial: SpatialResult;
  temporal: TemporalResult;
  lipsync?: LipsyncResult;
  frames_processed: number;
  duration_analyzed_seconds: number;
}

// ============== AUDIO RESULTS ==============

/**
 * Audio segment with anomaly
 */
export interface AudioSegment {
  start_ms: number;
  end_ms: number;
  score: number;
  anomaly_type?: string;
}

/**
 * Audio analysis result
 * Matches backend AudioResult from schemas/schemas.py
 */
export interface AudioResult {
  score: number;
  confidence: number;
  model_used: string;
  /** Probability audio is synthetic (0-1) */
  synthetic_probability: number;
  /** Whether vocoder artifacts detected */
  vocoder_artifacts_detected: boolean;
  /** Voice consistency across segments (0-1) */
  voice_consistency_score: number;
  /** URL to mel-spectrogram visualization */
  spectrogram_url?: string;
}

// ============== IMAGE RESULTS ==============

/**
 * Image analysis result for AI-generated content
 * Matches backend ImageResult from schemas/schemas.py
 */
export interface ImageResult {
  /** Probability image is AI-generated (0-1) */
  ai_generated_probability: number;
  /** Probability image is manipulated/deepfake (0-1) */
  fake_probability: number;
  /** Whether face was detected */
  face_detected: boolean;
  /** Number of faces detected */
  num_faces: number;
  /** Per-face manipulation scores */
  face_manipulation_scores: number[];
  /** URL to GradCAM heatmap */
  heatmap_url?: string;
  /** DCT frequency anomaly score */
  dct_anomaly_score: number;
  /** Spectral flatness measure */
  spectral_flatness: number;
  /** SigLIP classifier score */
  siglip_score: number;
  /** EfficientNet classifier score */
  efficientnet_score: number;
  /** Detected manipulation regions */
  manipulation_regions: ManipulationRegion[];
  /** Confidence interval for AI probability */
  confidence_interval?: [number, number];
}

// ============== METADATA RESULTS ==============

/**
 * C2PA provenance data
 */
export interface C2PAData {
  has_manifest: boolean;
  is_valid: boolean;
  claim_generator?: string;
  assertions?: string[];
  ingredients?: string[];
  signature_info?: {
    issuer: string;
    valid_from: string;
    valid_to: string;
  };
}

/**
 * EXIF metadata
 */
export interface EXIFData {
  camera_make?: string;
  camera_model?: string;
  software?: string;
  datetime_original?: string;
  gps_location?: {
    latitude: number;
    longitude: number;
  };
  anomalies: string[];
}

/**
 * Metadata analysis result
 */
export interface MetadataResult {
  score: number;
  confidence: number;
  c2pa: C2PAData;
  exif: EXIFData;
  file_integrity: {
    hash_verified: boolean;
    structure_valid: boolean;
    suspicious_markers: string[];
  };
}

// ============== API RESPONSES ==============

/**
 * Basic analysis response from POST /api/v1/analyze
 */
export interface AnalysisResponse {
  analysis_id: string;
  status: AnalysisStatus;
  trust_score?: TrustScore;
  verdict?: Verdict;
  explanation?: Explanation;
  report_url?: string;
  created_at: string;
  completed_at?: string;
}

/**
 * Detailed analysis response from GET /api/v1/analyze/{id}/detail
 */
export interface AnalysisDetailResponse extends AnalysisResponse {
  input?: FileInput;
  video_result?: VideoResult;
  audio_result?: AudioResult;
  image_result?: ImageResult;
  metadata_result?: MetadataResult;
  processing_time_seconds?: number;
  // XAI Enhancement Fields
  evidence_package?: EvidencePackage;
  feature_importance: FeatureImportance[];
  scientific_references: ScientificReference[];
}

/**
 * Error response from API
 */
export interface ErrorResponse {
  error_code: string;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * Heatmap response
 */
export interface HeatmapResponse {
  heatmaps: Array<{
    key: string;
    url: string;
  }>;
  count: number;
}

/**
 * Health check response
 */
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  components: Record<string, string>;
}

/**
 * Model info response
 */
export interface ModelInfo {
  name: string;
  category: string;
  vram_mb: number;
  loaded: boolean;
  version: string;
}

/**
 * Models list response
 */
export interface ModelsResponse {
  models: ModelInfo[];
  count: number;
}

// ============== WEBSOCKET TYPES ==============

/**
 * WebSocket message types
 */
export type WebSocketMessageType =
  | 'status'
  | 'progress'
  | 'completed'
  | 'error'
  | 'ping'
  | 'pong';

/**
 * Progress update from WebSocket
 */
export interface ProgressUpdate {
  type: WebSocketMessageType;
  analysis_id: string;
  status: AnalysisStatus;
  progress_percent: number;
  current_stage: string;
  message?: string;
  timestamp: string;
  result?: Partial<AnalysisResponse>;
}

// ============== UI TYPES ==============

/**
 * File validation error
 */
export interface ValidationError {
  field: string;
  message: string;
}

/**
 * File validation warning
 */
export interface ValidationWarning {
  field: string;
  message: string;
}

/**
 * File info from validation
 */
export interface FileInfo {
  name: string;
  size: number;
  type: string;
  extension: string;
  preview?: string;
}

/**
 * Validation result
 */
export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  fileInfo: FileInfo | null;
}

/**
 * List query parameters
 */
export interface ListParams {
  status?: AnalysisStatus;
  limit?: number;
  offset?: number;
}

// ============== XAI TYPES ==============

/**
 * Audio artifact region detected in spectrogram
 * Maps to backend AudioArtifactRegion schema
 */
export interface AudioArtifactRegion {
  start_time: number;      // Start time in seconds
  end_time: number;        // End time in seconds
  freq_low: number;        // Low frequency bound in Hz
  freq_high: number;       // High frequency bound in Hz
  artifact_type: 'vocoder' | 'spectral_gap' | 'harmonic_inconsistency' | 'high_energy_anomaly';
  confidence: number;      // Detection confidence (0-1)
}

/**
 * Token attribution for text XAI
 * Shows which tokens contribute to AI detection
 */
export interface TokenAttribution {
  token: string;           // The token/word
  attribution_score: number;  // Contribution to AI detection (-1 to 1)
  position: number;        // Token position in text
  is_ai_indicator: boolean;   // Whether this token indicates AI generation
}

/**
 * Perplexity breakdown by text segment
 */
export interface PerplexityBreakdown {
  segment: string;         // Text segment
  perplexity: number;      // Perplexity score
  is_anomalous: boolean;   // Whether segment is anomalous
}

/**
 * Feature importance for model decision
 */
export interface FeatureImportance {
  feature_name: string;    // Feature identifier
  importance_score: number; // Importance (0-1)
  contribution_direction: 'increases_fake' | 'decreases_fake';
  confidence: number;      // Confidence in importance (0-1)
  feature_type?: string;   // Type: 'spatial', 'frequency', 'temporal', 'linguistic', 'acoustic'
  description?: string;    // Human-readable description
  modality?: string;       // Modality this feature belongs to
}

/**
 * Manipulation region in image/video
 */
export interface ManipulationRegion {
  region_type: string;     // Region type: face, mouth, background, etc.
  location: string;        // Description or coordinates
  confidence: number;      // Detection confidence (0-1)
  frame_indices?: number[]; // Affected frame indices
}

/**
 * Scientific reference for methodology
 */
export interface ScientificReference {
  method_name: string;     // Method name (e.g., "GradCAM++")
  citation: string;        // Full academic citation
  doi?: string;            // DOI link
  accuracy_metrics?: string; // Known accuracy metrics
}

/**
 * Visual evidence for reports
 */
export interface VisualEvidence {
  artifact_type: 'heatmap' | 'spectrogram' | 'frequency_plot' | 'overlay' | 'temporal_chart' | 'token_highlight' | 'attention_map';
  url: string;             // MinIO presigned URL
  description: string;     // Human-readable description
  frame_index?: number;    // For video frames
  timestamp_seconds?: number; // For audio/video timestamps
  integrity_hash: string;  // SHA-256 hash for chain of custody
  created_at?: string;     // ISO timestamp
  width?: number;          // Image width in pixels
  height?: number;         // Image height in pixels
}

/**
 * Complete XAI explanation package
 */
export interface XAIExplanation {
  feature_importance: FeatureImportance[];
  visual_evidence: VisualEvidence[];
  scientific_references: ScientificReference[];
  reproducibility_hash: string;
  confidence_interval: [number, number];  // Tuple of (lower, upper)
  model_versions: Record<string, string>;
}

/**
 * Evidence package for court-admissible forensic reports
 */
export interface EvidencePackage {
  analysis_id?: string;
  visual_evidence: VisualEvidence[];
  feature_importance: FeatureImportance[];
  scientific_references?: ScientificReference[];
  reproducibility_hash?: string;
  confidence_interval: [number, number];
  model_versions: Record<string, string>;
  // Additional fields from backend
  token_attributions?: TokenAttribution[] | null;
  perplexity_breakdown?: PerplexityBreakdown[] | null;
  audio_artifact_regions?: AudioArtifactRegion[] | null;
  analysis_timestamp?: string;
  integrity_hash?: string;
  reproducibility_data?: Record<string, unknown>;
}

/**
 * Extended AudioResult with XAI fields
 */
export interface AudioResultXAI extends AudioResult {
  artifact_regions: AudioArtifactRegion[];
  frequency_anomaly_score: number;
  aasist_score: number;
  xai_explanation?: XAIExplanation;
}

/**
 * Extended SpatialResult with XAI fields
 */
export interface SpatialResultXAI extends SpatialResult {
  dct_anomaly_score: number;
  gan_fingerprint_detected: boolean;
  manipulation_regions: ManipulationRegion[];
  efficientnet_score: number;
  clip_score: number;
  xai_explanation?: XAIExplanation;
}

/**
 * Extended TemporalResult with XAI fields
 */
export interface TemporalResultXAI extends TemporalResult {
  motion_anomaly_score: number;
  landmark_jitter_score: number;
  xclip_score: number;
}

/**
 * Extended VideoResult with XAI fields
 */
export interface VideoResultXAI extends VideoResult {
  frame_heatmap_urls: string[];
  temporal_heatmap_url?: string;
  confidence_interval: [number, number];
  spatial: SpatialResultXAI;
  temporal: TemporalResultXAI;
  xai_explanation?: XAIExplanation;
}

/**
 * Extended analysis detail response with XAI data
 */
export interface AnalysisDetailResponseXAI extends AnalysisDetailResponse {
  video_result?: VideoResultXAI;
  audio_result?: AudioResultXAI;
  xai_evidence_package?: {
    feature_importance: FeatureImportance[];
    visual_evidence: VisualEvidence[];
    scientific_references: ScientificReference[];
    reproducibility_hash: string;
    confidence_interval: [number, number];
  };
}

/**
 * XAI heatmap response with metadata
 */
export interface XAIHeatmapResponse {
  heatmaps: Array<{
    key: string;
    url: string;
    frame_index?: number;
    timestamp_ms?: number;
    overlay_url?: string;  // Combined overlay URL
  }>;
  spectrogram_overlay?: {
    url: string;
    artifact_regions: AudioArtifactRegion[];
  };
  token_highlights?: Array<{
    text: string;
    highlights: Array<{
      start: number;
      end: number;
      score: number;
    }>;
  }>;
  count: number;
}

/**
 * Modality-specific XAI data structure
 * Used by XAI store and hooks
 */
export interface ModalityXAI {
  explanation: XAIExplanation | null;
  artifactRegions: AudioArtifactRegion[] | ManipulationRegion[];
  tokenAttributions?: TokenAttribution[];
  heatmapUrls: string[];
  overlayUrl: string | null;
}

// ============== EXPORTS ==============

export default {
  // Re-export for convenience
};
