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
export type Modality = 'video' | 'audio' | 'image' | 'text';

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

// ============== TEXT RESULTS ==============

/**
 * Text analysis result for AI-generated content
 * Matches backend TextResult from schemas/schemas.py
 */
export interface TextResult {
  score: number;
  confidence: number;
  model_used: string;
  /** Probability text is AI-generated (0-1) */
  ai_probability: number;
  /** GPT-2 perplexity score */
  perplexity_score: number;
  /** Sentence length variance (burstiness) */
  burstiness_score: number;
  /** RADAR classifier score */
  radar_score?: number;
  /** Word count analyzed */
  word_count?: number;
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
  text_result?: TextResult;
  metadata_result?: MetadataResult;
  processing_time_seconds?: number;
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

// ============== EXPORTS ==============

export default {
  // Re-export for convenience
};
