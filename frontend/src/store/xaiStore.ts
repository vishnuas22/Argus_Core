/**
 * Argus Core - XAI Store
 * ======================
 * Explainable AI state management using Zustand.
 * 
 * Implements: XAI_FRONTEND_IMPLEMENTATION.md - Section 4.1 - store/xaiStore.ts
 * 
 * Role: Manage XAI explanation state including feature importance,
 * visual evidence, scientific references, and reproducibility data.
 * 
 * Integration:
 * - Used by: XAIExplanationPanel, FeatureImportanceTable, XAIEvidenceGallery
 * - Updates: On analysis completion, XAI data fetch
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type {
  XAIExplanation,
  FeatureImportance,
  VisualEvidence,
  ScientificReference,
  AudioArtifactRegion,
  ManipulationRegion,
} from '@/types/analysis';

// ============== TYPES ==============

/**
 * XAI loading status
 */
export type XAIStatus = 'idle' | 'loading' | 'success' | 'error';

/**
 * Modality type for XAI data
 */
export type ModalityType = 'image' | 'video' | 'audio';

/**
 * XAI data for a specific modality
 */
export interface ModalityXAI {
  explanation: XAIExplanation | null;
  artifactRegions: AudioArtifactRegion[] | ManipulationRegion[];
  heatmapUrls: string[];
  overlayUrl: string | null;
}

/**
 * XAI state interface
 */
interface XAIState {
  // Analysis ID this XAI belongs to
  analysisId: string | null;
  
  // Overall status
  status: XAIStatus;
  error: string | null;
  
  // Per-modality XAI data
  imageXAI: ModalityXAI | null;
  videoXAI: ModalityXAI | null;
  audioXAI: ModalityXAI | null;
  
  // Active modality for display
  activeModality: ModalityType;
  
  // Evidence package for reports
  evidencePackage: {
    featureImportance: FeatureImportance[];
    visualEvidence: VisualEvidence[];
    scientificReferences: ScientificReference[];
    reproducibilityHash: string;
    confidenceInterval: [number, number];
  } | null;
  
  // Cache timestamps
  lastFetched: number | null;
}

/**
 * XAI actions interface
 */
interface XAIActions {
  // Set analysis ID
  setAnalysisId: (id: string | null) => void;
  
  // Set XAI data for a modality
  setImageXAI: (data: ModalityXAI | null) => void;
  setVideoXAI: (data: ModalityXAI | null) => void;
  setAudioXAI: (data: ModalityXAI | null) => void;
  
  // Set evidence package
  setEvidencePackage: (pkg: XAIState['evidencePackage']) => void;
  
  // Set active modality
  setActiveModality: (modality: ModalityType) => void;
  
  // Status management
  setStatus: (status: XAIStatus) => void;
  setError: (error: string | null) => void;
  
  // Reset
  reset: () => void;
  resetModality: (modality: ModalityType) => void;
}

// ============== INITIAL STATE ==============

const initialModalityXAI: ModalityXAI = {
  explanation: null,
  artifactRegions: [],
  heatmapUrls: [],
  overlayUrl: null,
};

const initialState: XAIState = {
  analysisId: null,
  status: 'idle',
  error: null,
  imageXAI: null,
  videoXAI: null,
  audioXAI: null,
  activeModality: 'image',
  evidencePackage: null,
  lastFetched: null,
};

// ============== STORE ==============

/**
 * XAI Store with Zustand
 * Manages XAI explanation state with devtools and persistence
 */
export const useXAIStore = create<XAIState & XAIActions>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        ...initialState,

        // ============== ANALYSIS ID ==============

        /**
         * Set the analysis ID this XAI belongs to
         */
        setAnalysisId: (id) => {
          const currentId = get().analysisId;
          
          // If ID changed, reset all data
          if (id !== currentId) {
            set(
              {
                analysisId: id,
                status: 'idle',
                error: null,
                imageXAI: null,
                videoXAI: null,
                audioXAI: null,
                evidencePackage: null,
                lastFetched: null,
              },
              false,
              'setAnalysisId'
            );
          } else {
            set({ analysisId: id }, false, 'setAnalysisId');
          }
        },

        // ============== MODALITY XAI SETTERS ==============

        /**
         * Set XAI data for image modality
         */
        setImageXAI: (data) => {
          set(
            {
              imageXAI: data,
              status: 'success',
              lastFetched: Date.now(),
            },
            false,
            'setImageXAI'
          );
        },

        /**
         * Set XAI data for video modality
         */
        setVideoXAI: (data) => {
          set(
            {
              videoXAI: data,
              status: 'success',
              lastFetched: Date.now(),
            },
            false,
            'setVideoXAI'
          );
        },

        /**
         * Set XAI data for audio modality
         */
        setAudioXAI: (data) => {
          set(
            {
              audioXAI: data,
              status: 'success',
              lastFetched: Date.now(),
            },
            false,
            'setAudioXAI'
          );
        },

        // ============== EVIDENCE PACKAGE ==============

        /**
         * Set the complete evidence package for reports
         */
        setEvidencePackage: (pkg) => {
          set({ evidencePackage: pkg }, false, 'setEvidencePackage');
        },

        // ============== ACTIVE MODALITY ==============

        /**
         * Set the active modality for display
         */
        setActiveModality: (modality) => {
          set({ activeModality: modality }, false, 'setActiveModality');
        },

        // ============== STATUS MANAGEMENT ==============

        /**
         * Set loading status
         */
        setStatus: (status) => {
          set({ status }, false, 'setStatus');
        },

        /**
         * Set error message
         */
        setError: (error) => {
          set(
            {
              error,
              status: error ? 'error' : 'idle',
            },
            false,
            'setError'
          );
        },

        // ============== RESET ==============

        /**
         * Reset entire store to initial state
         */
        reset: () => {
          set(initialState, false, 'reset');
        },

        /**
         * Reset a specific modality's XAI data
         */
        resetModality: (modality) => {
          const updates: Partial<XAIState> = {};
          
          switch (modality) {
            case 'image':
              updates.imageXAI = null;
              break;
            case 'video':
              updates.videoXAI = null;
              break;
            case 'audio':
              updates.audioXAI = null;
              break;
          }
          
          set(updates, false, 'resetModality');
        },
      }),
      {
        name: 'argus-xai-store',
        // Only persist specific fields
        partialize: (state) => ({
          analysisId: state.analysisId,
          lastFetched: state.lastFetched,
        }),
      }
    ),
    {
      name: 'argus-xai-store',
      enabled: process.env.NODE_ENV === 'development',
    }
  )
);

// ============== SELECTORS ==============

/**
 * Selector for analysis ID
 */
export const selectAnalysisId = (state: XAIState & XAIActions) => ({
  analysisId: state.analysisId,
  setAnalysisId: state.setAnalysisId,
});

/**
 * Selector for loading state
 */
export const selectXAIStatus = (state: XAIState & XAIActions) => ({
  status: state.status,
  error: state.error,
  setStatus: state.setStatus,
  setError: state.setError,
});

/**
 * Selector for active modality
 */
export const selectActiveModality = (state: XAIState & XAIActions) => ({
  activeModality: state.activeModality,
  setActiveModality: state.setActiveModality,
});

/**
 * Selector for image XAI data
 */
export const selectImageXAI = (state: XAIState & XAIActions) => ({
  imageXAI: state.imageXAI,
  setImageXAI: state.setImageXAI,
});

/**
 * Selector for video XAI data
 */
export const selectVideoXAI = (state: XAIState & XAIActions) => ({
  videoXAI: state.videoXAI,
  setVideoXAI: state.setVideoXAI,
});

/**
 * Selector for audio XAI data
 */
export const selectAudioXAI = (state: XAIState & XAIActions) => ({
  audioXAI: state.audioXAI,
  setAudioXAI: state.setAudioXAI,
});

/**
 * Selector for evidence package
 */
export const selectEvidencePackage = (state: XAIState & XAIActions) => ({
  evidencePackage: state.evidencePackage,
  setEvidencePackage: state.setEvidencePackage,
});

/**
 * Get XAI data for current active modality
 */
export const selectActiveModalityXAI = (state: XAIState) => {
  switch (state.activeModality) {
    case 'image':
      return state.imageXAI;
    case 'video':
      return state.videoXAI;
    case 'audio':
      return state.audioXAI;
    default:
      return null;
  }
};

/**
 * Check if XAI data is available for any modality
 */
export const selectHasXAI = (state: XAIState) =>
  state.imageXAI !== null ||
  state.videoXAI !== null ||
  state.audioXAI !== null ||
  state.evidencePackage !== null;

/**
 * Check if XAI data needs to be fetched
 * Returns true if no data or data is stale (> 5 minutes old)
 */
export const selectNeedsFetch = (state: XAIState) => {
  if (state.status === 'loading') return false;
  if (!state.lastFetched) return true;
  
  const fiveMinutes = 5 * 60 * 1000;
  return Date.now() - state.lastFetched > fiveMinutes;
};

/**
 * Get all feature importance across modalities
 */
export const selectAllFeatureImportance = (state: XAIState): FeatureImportance[] => {
  const allFeatures: FeatureImportance[] = [];
  
  if (state.imageXAI?.explanation?.feature_importance) {
    allFeatures.push(...state.imageXAI.explanation.feature_importance);
  }
  if (state.videoXAI?.explanation?.feature_importance) {
    allFeatures.push(...state.videoXAI.explanation.feature_importance);
  }
  if (state.audioXAI?.explanation?.feature_importance) {
    allFeatures.push(...state.audioXAI.explanation.feature_importance);
  }
  
  return allFeatures;
};

/**
 * Get all visual evidence across modalities
 */
export const selectAllVisualEvidence = (state: XAIState): VisualEvidence[] => {
  const allEvidence: VisualEvidence[] = [];
  
  if (state.imageXAI?.explanation?.visual_evidence) {
    allEvidence.push(...state.imageXAI.explanation.visual_evidence);
  }
  if (state.videoXAI?.explanation?.visual_evidence) {
    allEvidence.push(...state.videoXAI.explanation.visual_evidence);
  }
  if (state.audioXAI?.explanation?.visual_evidence) {
    allEvidence.push(...state.audioXAI.explanation.visual_evidence);
  }
  
  return allEvidence;
};

/**
 * Get combined scientific references (deduplicated)
 */
export const selectScientificReferences = (state: XAIState): ScientificReference[] => {
  const referencesMap = new Map<string, ScientificReference>();
  
  const addReferences = (xai: ModalityXAI | null) => {
    if (xai?.explanation?.scientific_references) {
      for (const ref of xai.explanation.scientific_references) {
        if (!referencesMap.has(ref.method_name)) {
          referencesMap.set(ref.method_name, ref);
        }
      }
    }
  };
  
  addReferences(state.imageXAI);
  addReferences(state.videoXAI);
  addReferences(state.audioXAI);
  
  return Array.from(referencesMap.values());
};

export default useXAIStore;
