/**
 * Argus Core - Application Constants
 * ===================================
 * Centralized constants for the application.
 */

// ============== API CONFIGURATION ==============

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000';

// ============== FILE UPLOAD ==============

export const MAX_FILE_SIZE_MB = 500;
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

export const ACCEPTED_FILE_TYPES = {
  video: ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo'],
  audio: ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac'],
  image: ['image/jpeg', 'image/png', 'image/webp'],
} as const;

export const ACCEPTED_EXTENSIONS = {
  video: ['.mp4', '.webm', '.mov', '.avi'],
  audio: ['.mp3', '.wav', '.ogg', '.flac'],
  image: ['.jpg', '.jpeg', '.png', '.webp'],
} as const;

export const ALL_ACCEPTED_TYPES = [
  ...ACCEPTED_FILE_TYPES.video,
  ...ACCEPTED_FILE_TYPES.audio,
  ...ACCEPTED_FILE_TYPES.image,
];

export const ALL_ACCEPTED_EXTENSIONS = [
  ...ACCEPTED_EXTENSIONS.video,
  ...ACCEPTED_EXTENSIONS.audio,
  ...ACCEPTED_EXTENSIONS.image,
];

// ============== ANALYSIS ==============

export const MAX_VIDEO_DURATION_SECONDS = 300; // 5 minutes
export const MIN_TEXT_LENGTH = 50;
export const MAX_TEXT_LENGTH = 100000;

// ============== VERDICT THRESHOLDS ==============

export const VERDICT_THRESHOLDS = {
  authentic: 80,
  likely_authentic: 60,
  uncertain: 40,
  likely_fake: 20,
  fake: 0,
} as const;

// ============== SCORE WEIGHTS ==============

export const DEFAULT_SCORE_WEIGHTS = {
  video_spatial: 0.30,
  video_temporal: 0.25,
  audio: 0.20,
  metadata: 0.15,
  text: 0.10,
} as const;

// ============== PIPELINE STAGES ==============

export const PIPELINE_STAGES = [
  { id: 'pending', label: 'Queued', progress: 0 },
  { id: 'preprocessing', label: 'Preprocessing', progress: 15 },
  { id: 'analyzing', label: 'Analyzing', progress: 50 },
  { id: 'aggregating', label: 'Scoring', progress: 85 },
  { id: 'completed', label: 'Complete', progress: 100 },
] as const;

// ============== WEBSOCKET ==============

export const WS_RECONNECT_DELAY_MS = 3000;
export const WS_MAX_RECONNECT_ATTEMPTS = 5;
export const WS_PING_INTERVAL_MS = 30000;

// ============== UI ==============

export const TOAST_DURATION_MS = 5000;
export const ANIMATION_DURATION_MS = 300;
export const DEBOUNCE_DELAY_MS = 300;

// ============== PAGINATION ==============

export const DEFAULT_PAGE_SIZE = 20;
export const MAX_PAGE_SIZE = 100;

// ============== LOCAL STORAGE KEYS ==============

export const STORAGE_KEYS = {
  theme: 'argus-theme',
  authToken: 'argus-auth-token',
  recentAnalyses: 'argus-recent-analyses',
} as const;
