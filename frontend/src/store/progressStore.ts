/**
 * Argus Core - Progress Store
 * ===========================
 * Real-time progress state management for WebSocket updates using Zustand.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - store/progressStore.ts
 * 
 * Role: Manage real-time progress state from WebSocket updates. Stores progress
 * entries keyed by analysis_id to support multiple concurrent analyses.
 * 
 * Integration:
 * - Updated by: hooks/useWebSocket.ts on WebSocket message
 * - Used by: ProgressIndicator.tsx, AnalysisTimeline.tsx, analysis/[id]/page.tsx
 * - Data source: WebSocket messages from /ws/analysis/{id}
 * 
 * WebSocket Message Types Handled:
 * - status: Initial status when connecting
 * - progress: Progress updates during analysis
 * - completed: Final results when analysis completes
 * - error: Error messages
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { AnalysisStatus, TrustScore, Verdict } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Individual progress entry for an analysis
 */
export interface ProgressEntry {
  /** Current analysis status */
  status: AnalysisStatus;
  /** Progress percentage (0-100) */
  progressPercent: number;
  /** Current processing stage name */
  currentStage: string;
  /** Optional human-readable message */
  message?: string;
  /** Timestamp of last update */
  timestamp: string;
  /** Trust score (available when completed) */
  trustScore?: TrustScore;
  /** Verdict (available when completed) */
  verdict?: Verdict;
  /** Report URL (available when completed) */
  reportUrl?: string;
  /** Error code (if failed) */
  errorCode?: string;
  /** Error message (if failed) */
  errorMessage?: string;
}

/**
 * WebSocket connection state
 */
export interface ConnectionState {
  /** Whether WebSocket is connected */
  isConnected: boolean;
  /** Last connection error */
  error: string | null;
  /** Reconnection attempt count */
  reconnectAttempts: number;
  /** Last successful ping timestamp */
  lastPing: string | null;
}

/**
 * Progress store state interface
 */
interface ProgressState {
  /** Progress entries keyed by analysis_id */
  progress: Record<string, ProgressEntry>;
  /** Connection states keyed by analysis_id */
  connections: Record<string, ConnectionState>;
}

/**
 * Progress store actions interface
 */
interface ProgressActions {
  // Progress actions
  setProgress: (analysisId: string, update: Partial<ProgressEntry>) => void;
  clearProgress: (analysisId: string) => void;
  clearAllProgress: () => void;
  
  // Connection state actions
  setConnectionState: (analysisId: string, state: Partial<ConnectionState>) => void;
  clearConnectionState: (analysisId: string) => void;
  
  // Selectors
  getProgress: (analysisId: string) => ProgressEntry | undefined;
  getConnectionState: (analysisId: string) => ConnectionState | undefined;
  
  // Computed
  isAnalysisComplete: (analysisId: string) => boolean;
  isAnalysisFailed: (analysisId: string) => boolean;
  isAnalysisInProgress: (analysisId: string) => boolean;
}

// ============== INITIAL STATE ==============

const initialState: ProgressState = {
  progress: {},
  connections: {},
};

/**
 * Default progress entry for new analyses
 */
const defaultProgressEntry: ProgressEntry = {
  status: 'pending',
  progressPercent: 0,
  currentStage: 'pending',
  timestamp: new Date().toISOString(),
};

/**
 * Default connection state
 */
const defaultConnectionState: ConnectionState = {
  isConnected: false,
  error: null,
  reconnectAttempts: 0,
  lastPing: null,
};

// ============== STORE ==============

/**
 * Progress Store with Zustand
 * Manages real-time progress state from WebSocket updates
 */
export const useProgressStore = create<ProgressState & ProgressActions>()(
  devtools(
    (set, get) => ({
      // Initial state
      ...initialState,

      // ============== PROGRESS ACTIONS ==============

      /**
       * Set or update progress for an analysis
       * Merges with existing progress entry if present
       */
      setProgress: (analysisId, update) => {
        set(
          (state) => {
            const existing = state.progress[analysisId] || defaultProgressEntry;
            const newEntry: ProgressEntry = {
              ...existing,
              ...update,
              timestamp: update.timestamp || new Date().toISOString(),
            };

            return {
              progress: {
                ...state.progress,
                [analysisId]: newEntry,
              },
            };
          },
          false,
          'setProgress'
        );
      },

      /**
       * Clear progress for a specific analysis
       */
      clearProgress: (analysisId) => {
        set(
          (state) => {
            const { [analysisId]: _, ...rest } = state.progress;
            return { progress: rest };
          },
          false,
          'clearProgress'
        );
      },

      /**
       * Clear all progress entries
       */
      clearAllProgress: () => {
        set({ progress: {} }, false, 'clearAllProgress');
      },

      // ============== CONNECTION STATE ACTIONS ==============

      /**
       * Set connection state for an analysis
       */
      setConnectionState: (analysisId, state) => {
        set(
          (currentState) => {
            const existing = currentState.connections[analysisId] || defaultConnectionState;
            return {
              connections: {
                ...currentState.connections,
                [analysisId]: {
                  ...existing,
                  ...state,
                },
              },
            };
          },
          false,
          'setConnectionState'
        );
      },

      /**
       * Clear connection state for an analysis
       */
      clearConnectionState: (analysisId) => {
        set(
          (state) => {
            const { [analysisId]: _, ...rest } = state.connections;
            return { connections: rest };
          },
          false,
          'clearConnectionState'
        );
      },

      // ============== SELECTORS ==============

      /**
       * Get progress entry for an analysis
       */
      getProgress: (analysisId) => {
        return get().progress[analysisId];
      },

      /**
       * Get connection state for an analysis
       */
      getConnectionState: (analysisId) => {
        return get().connections[analysisId];
      },

      // ============== COMPUTED ==============

      /**
       * Check if analysis is complete
       */
      isAnalysisComplete: (analysisId) => {
        const entry = get().progress[analysisId];
        return entry?.status === 'completed';
      },

      /**
       * Check if analysis has failed
       */
      isAnalysisFailed: (analysisId) => {
        const entry = get().progress[analysisId];
        return entry?.status === 'failed';
      },

      /**
       * Check if analysis is in progress
       */
      isAnalysisInProgress: (analysisId) => {
        const entry = get().progress[analysisId];
        if (!entry) return false;
        return (
          entry.status === 'pending' ||
          entry.status === 'preprocessing' ||
          entry.status === 'analyzing' ||
          entry.status === 'aggregating'
        );
      },
    }),
    {
      name: 'argus-progress-store',
      enabled: process.env.NODE_ENV === 'development',
    }
  )
);

// ============== HELPER SELECTORS ==============

/**
 * Selector for progress entry with defaults
 */
export const selectProgress = (analysisId: string) => (state: ProgressState & ProgressActions) => {
  return state.progress[analysisId] || defaultProgressEntry;
};

/**
 * Selector for connection state with defaults
 */
export const selectConnection = (analysisId: string) => (state: ProgressState & ProgressActions) => {
  return state.connections[analysisId] || defaultConnectionState;
};

/**
 * Selector for progress percentage
 */
export const selectProgressPercent = (analysisId: string) => (state: ProgressState) => {
  return state.progress[analysisId]?.progressPercent ?? 0;
};

/**
 * Selector for current stage
 */
export const selectCurrentStage = (analysisId: string) => (state: ProgressState) => {
  return state.progress[analysisId]?.currentStage ?? 'pending';
};

/**
 * Selector for status
 */
export const selectStatus = (analysisId: string) => (state: ProgressState) => {
  return state.progress[analysisId]?.status ?? 'pending';
};

/**
 * Selector for checking if connected
 */
export const selectIsConnected = (analysisId: string) => (state: ProgressState & ProgressActions) => {
  return state.connections[analysisId]?.isConnected ?? false;
};

// ============== STAGE MAPPING ==============

/**
 * Map status to human-readable stage names
 */
export const STAGE_LABELS: Record<AnalysisStatus, string> = {
  pending: 'Waiting to start',
  preprocessing: 'Preprocessing file',
  analyzing: 'Running analysis',
  aggregating: 'Calculating score',
  completed: 'Analysis complete',
  failed: 'Analysis failed',
};

/**
 * Map status to progress percentage ranges
 * Used for estimating progress when not provided
 */
export const STAGE_PROGRESS_RANGES: Record<AnalysisStatus, [number, number]> = {
  pending: [0, 5],
  preprocessing: [5, 20],
  analyzing: [20, 80],
  aggregating: [80, 95],
  completed: [100, 100],
  failed: [0, 0],
};

/**
 * Get estimated progress for a status
 */
export function getEstimatedProgress(status: AnalysisStatus): number {
  const [min, max] = STAGE_PROGRESS_RANGES[status];
  return Math.round((min + max) / 2);
}

export default useProgressStore;
