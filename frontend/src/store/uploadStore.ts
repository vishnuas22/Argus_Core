/**
 * Argus Core - Upload Store
 * =========================
 * Upload state management using Zustand.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - store/uploadStore.ts
 * 
 * Role: Manage file upload state including file selection, preview generation,
 * upload progress tracking, and validation errors.
 * 
 * Integration:
 * - Used by: UploadZone.tsx, FileCard.tsx, AnalysisForm.tsx
 * - Updates: On file selection, upload progress, completion
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { ValidationError, ValidationWarning, FileInfo } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Upload status enum
 */
export type UploadStatus = 'idle' | 'validating' | 'uploading' | 'processing' | 'complete' | 'error';

/**
 * Upload state interface
 */
interface UploadState {
  // File state
  file: File | null;
  preview: string | null;
  fileInfo: FileInfo | null;
  
  // Validation state
  isValid: boolean;
  validationErrors: ValidationError[];
  validationWarnings: ValidationWarning[];
  
  // Upload progress state
  status: UploadStatus;
  uploadProgress: number;
  error: string | null;
  
  // Analysis options (carried to form)
  analysisId: string | null;
}

/**
 * Upload actions interface
 */
interface UploadActions {
  // File actions
  setFile: (file: File, preview?: string | null, fileInfo?: FileInfo | null) => void;
  clearFile: () => void;
  
  // Validation actions
  setValidation: (
    isValid: boolean, 
    errors: ValidationError[], 
    warnings: ValidationWarning[]
  ) => void;
  
  // Upload actions
  setStatus: (status: UploadStatus) => void;
  setUploadProgress: (progress: number) => void;
  setError: (error: string | null) => void;
  
  // Analysis tracking
  setAnalysisId: (id: string | null) => void;
  
  // Reset
  reset: () => void;
}

// ============== INITIAL STATE ==============

const initialState: UploadState = {
  file: null,
  preview: null,
  fileInfo: null,
  isValid: false,
  validationErrors: [],
  validationWarnings: [],
  status: 'idle',
  uploadProgress: 0,
  error: null,
  analysisId: null,
};

// ============== STORE ==============

/**
 * Upload Store with Zustand
 * Manages file upload state with devtools support
 */
export const useUploadStore = create<UploadState & UploadActions>()(
  devtools(
    (set, get) => ({
      // Initial state
      ...initialState,

      // ============== FILE ACTIONS ==============

      /**
       * Set the selected file with optional preview and info
       */
      setFile: (file, preview = null, fileInfo = null) => {
        // Revoke previous preview URL to prevent memory leak
        const currentPreview = get().preview;
        if (currentPreview && currentPreview.startsWith('blob:')) {
          URL.revokeObjectURL(currentPreview);
        }

        set(
          {
            file,
            preview,
            fileInfo,
            status: 'idle',
            isValid: true,
            validationErrors: [],
            validationWarnings: [],
            uploadProgress: 0,
            error: null,
            analysisId: null,
          },
          false,
          'setFile'
        );
      },

      /**
       * Clear the selected file and reset state
       */
      clearFile: () => {
        // Revoke preview URL
        const currentPreview = get().preview;
        if (currentPreview && currentPreview.startsWith('blob:')) {
          URL.revokeObjectURL(currentPreview);
        }

        set(
          {
            file: null,
            preview: null,
            fileInfo: null,
            isValid: false,
            validationErrors: [],
            validationWarnings: [],
            status: 'idle',
            uploadProgress: 0,
            error: null,
            analysisId: null,
          },
          false,
          'clearFile'
        );
      },

      // ============== VALIDATION ACTIONS ==============

      /**
       * Set validation results
       */
      setValidation: (isValid, errors, warnings) => {
        set(
          {
            isValid,
            validationErrors: errors,
            validationWarnings: warnings,
            status: isValid ? 'idle' : 'error',
            error: errors.length > 0 ? errors[0].message : null,
          },
          false,
          'setValidation'
        );
      },

      // ============== UPLOAD ACTIONS ==============

      /**
       * Set current upload status
       */
      setStatus: (status) => {
        set({ status }, false, 'setStatus');
      },

      /**
       * Set upload progress (0-100)
       */
      setUploadProgress: (progress) => {
        set(
          { 
            uploadProgress: Math.min(100, Math.max(0, progress)),
            status: progress < 100 ? 'uploading' : 'processing',
          },
          false,
          'setUploadProgress'
        );
      },

      /**
       * Set error message
       */
      setError: (error) => {
        set(
          {
            error,
            status: error ? 'error' : get().status,
          },
          false,
          'setError'
        );
      },

      // ============== ANALYSIS TRACKING ==============

      /**
       * Set analysis ID after successful upload
       */
      setAnalysisId: (id) => {
        set(
          {
            analysisId: id,
            status: id ? 'complete' : get().status,
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
        const currentPreview = get().preview;
        if (currentPreview && currentPreview.startsWith('blob:')) {
          URL.revokeObjectURL(currentPreview);
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
 * Selector for validation state
 */
export const selectValidation = (state: UploadState & UploadActions) => ({
  isValid: state.isValid,
  validationErrors: state.validationErrors,
  validationWarnings: state.validationWarnings,
  setValidation: state.setValidation,
});

/**
 * Selector for upload progress
 */
export const selectUploadProgress = (state: UploadState & UploadActions) => ({
  status: state.status,
  uploadProgress: state.uploadProgress,
  error: state.error,
  setStatus: state.setStatus,
  setUploadProgress: state.setUploadProgress,
  setError: state.setError,
});

/**
 * Selector for analysis ID
 */
export const selectAnalysisId = (state: UploadState & UploadActions) => ({
  analysisId: state.analysisId,
  setAnalysisId: state.setAnalysisId,
});

/**
 * Check if upload is in progress
 */
export const selectIsUploading = (state: UploadState) =>
  state.status === 'uploading' || state.status === 'processing';

/**
 * Check if file is ready for upload
 */
export const selectIsReadyForUpload = (state: UploadState) =>
  state.file !== null && state.isValid && state.status === 'idle';

export default useUploadStore;
