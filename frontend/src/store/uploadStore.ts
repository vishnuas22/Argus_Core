/**
 * Argus Core - Upload Store
 * =========================
 * Zustand store for upload state management.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - store/uploadStore.ts
 * 
 * Role: Manage file upload state including selected file, preview,
 * validation results, and upload progress.
 * 
 * Integration:
 * - Used by: components/upload/UploadZone, components/upload/FileCard, components/analysis/AnalysisForm
 * - Connects to: hooks/useFileValidation, services/analysisApi
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { ValidationResult, FileInfo, AnalysisOptions, DefenseLevel } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Upload status enum
 */
export type UploadStatus = 
  | 'idle'
  | 'validating'
  | 'ready'
  | 'uploading'
  | 'processing'
  | 'completed'
  | 'error';

/**
 * Upload state interface
 */
interface UploadState {
  // File state
  file: File | null;
  preview: string | null;
  fileInfo: FileInfo | null;
  
  // Validation state
  validationResult: ValidationResult | null;
  
  // Upload progress
  status: UploadStatus;
  uploadProgress: number;  // 0-100
  errorMessage: string | null;
  
  // Analysis options
  options: AnalysisOptions;
  
  // Analysis result
  analysisId: string | null;
}

/**
 * Upload actions interface
 */
interface UploadActions {
  // File actions
  setFile: (file: File, preview: string | null, fileInfo: FileInfo) => void;
  clearFile: () => void;
  
  // Validation actions
  setValidationResult: (result: ValidationResult) => void;
  
  // Upload progress actions
  setStatus: (status: UploadStatus) => void;
  setUploadProgress: (progress: number) => void;
  setError: (message: string | null) => void;
  
  // Options actions
  setOptions: (options: Partial<AnalysisOptions>) => void;
  setDefenseLevel: (level: DefenseLevel) => void;
  toggleGenerateReport: () => void;
  toggleGenerateHeatmaps: () => void;
  
  // Analysis result
  setAnalysisId: (id: string) => void;
  
  // Reset
  reset: () => void;
}

// ============== INITIAL STATE ==============

const defaultOptions: AnalysisOptions = {
  generateReport: true,
  generateHeatmaps: true,
  defenseLevel: 'standard',
  modalities: undefined,  // Auto-detect
};

const initialState: UploadState = {
  file: null,
  preview: null,
  fileInfo: null,
  validationResult: null,
  status: 'idle',
  uploadProgress: 0,
  errorMessage: null,
  options: defaultOptions,
  analysisId: null,
};

// ============== STORE ==============

/**
 * Upload Store with Zustand
 * Manages file upload and analysis submission state
 */
export const useUploadStore = create<UploadState & UploadActions>()(
  devtools(
    (set, get) => ({
      // Initial state
      ...initialState,
      
      // ============== FILE ACTIONS ==============
      
      /**
       * Set selected file with preview and metadata
       */
      setFile: (file, preview, fileInfo) => {
        // Revoke previous preview URL to prevent memory leaks
        const previousPreview = get().preview;
        if (previousPreview) {
          URL.revokeObjectURL(previousPreview);
        }
        
        set(
          {
            file,
            preview,
            fileInfo,
            status: 'validating',
            errorMessage: null,
            uploadProgress: 0,
            analysisId: null,
          },
          false,
          'setFile'
        );
      },
      
      /**
       * Clear selected file
       */
      clearFile: () => {
        // Revoke preview URL
        const preview = get().preview;
        if (preview) {
          URL.revokeObjectURL(preview);
        }
        
        set(
          {
            file: null,
            preview: null,
            fileInfo: null,
            validationResult: null,
            status: 'idle',
            uploadProgress: 0,
            errorMessage: null,
            analysisId: null,
          },
          false,
          'clearFile'
        );
      },
      
      // ============== VALIDATION ACTIONS ==============
      
      /**
       * Set validation result
       */
      setValidationResult: (result) => {
        set(
          {
            validationResult: result,
            status: result.isValid ? 'ready' : 'error',
            errorMessage: result.isValid 
              ? null 
              : result.errors[0]?.message ?? 'Validation failed',
          },
          false,
          'setValidationResult'
        );
      },
      
      // ============== UPLOAD PROGRESS ACTIONS ==============
      
      /**
       * Set upload status
       */
      setStatus: (status) => {
        set({ status }, false, 'setStatus');
      },
      
      /**
       * Set upload progress percentage
       */
      setUploadProgress: (progress) => {
        set({ uploadProgress: progress }, false, 'setUploadProgress');
      },
      
      /**
       * Set error message
       */
      setError: (message) => {
        set(
          {
            errorMessage: message,
            status: message ? 'error' : get().status,
          },
          false,
          'setError'
        );
      },
      
      // ============== OPTIONS ACTIONS ==============
      
      /**
       * Update analysis options
       */
      setOptions: (options) => {
        set(
          (state) => ({
            options: { ...state.options, ...options },
          }),
          false,
          'setOptions'
        );
      },
      
      /**
       * Set defense level
       */
      setDefenseLevel: (level) => {
        set(
          (state) => ({
            options: { ...state.options, defenseLevel: level },
          }),
          false,
          'setDefenseLevel'
        );
      },
      
      /**
       * Toggle report generation option
       */
      toggleGenerateReport: () => {
        set(
          (state) => ({
            options: { 
              ...state.options, 
              generateReport: !state.options.generateReport 
            },
          }),
          false,
          'toggleGenerateReport'
        );
      },
      
      /**
       * Toggle heatmap generation option
       */
      toggleGenerateHeatmaps: () => {
        set(
          (state) => ({
            options: { 
              ...state.options, 
              generateHeatmaps: !state.options.generateHeatmaps 
            },
          }),
          false,
          'toggleGenerateHeatmaps'
        );
      },
      
      // ============== ANALYSIS RESULT ==============
      
      /**
       * Set analysis ID after successful submission
       */
      setAnalysisId: (id) => {
        set(
          {
            analysisId: id,
            status: 'completed',
          },
          false,
          'setAnalysisId'
        );
      },
      
      // ============== RESET ==============
      
      /**
       * Reset store to initial state
       */
      reset: () => {
        // Revoke preview URL
        const preview = get().preview;
        if (preview) {
          URL.revokeObjectURL(preview);
        }
        
        set(initialState, false, 'reset');
      },
    }),
    {
      name: 'argus-upload-store',
      enabled: process.env.NODE_ENV === 'development',
    }
  )
);

// ============== SELECTORS ==============

/**
 * Selector for file state
 */
export const selectFile = (state: UploadState & UploadActions) => ({
  file: state.file,
  preview: state.preview,
  fileInfo: state.fileInfo,
  setFile: state.setFile,
  clearFile: state.clearFile,
});

/**
 * Selector for upload progress
 */
export const selectUploadProgress = (state: UploadState & UploadActions) => ({
  status: state.status,
  uploadProgress: state.uploadProgress,
  errorMessage: state.errorMessage,
  setStatus: state.setStatus,
  setUploadProgress: state.setUploadProgress,
  setError: state.setError,
});

/**
 * Selector for analysis options
 */
export const selectOptions = (state: UploadState & UploadActions) => ({
  options: state.options,
  setOptions: state.setOptions,
  setDefenseLevel: state.setDefenseLevel,
  toggleGenerateReport: state.toggleGenerateReport,
  toggleGenerateHeatmaps: state.toggleGenerateHeatmaps,
});

/**
 * Selector to check if ready to submit
 */
export const selectCanSubmit = (state: UploadState & UploadActions) => {
  return (
    state.file !== null &&
    state.validationResult?.isValid === true &&
    state.status === 'ready'
  );
};

/**
 * Selector for validation state
 */
export const selectValidation = (state: UploadState & UploadActions) => ({
  validationResult: state.validationResult,
  setValidationResult: state.setValidationResult,
});

export default useUploadStore;
