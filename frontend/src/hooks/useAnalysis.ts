/**
 * Argus Core - Analysis Hook
 * ===========================
 * TanStack Query mutations for analysis CRUD operations.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - hooks/useAnalysis.ts
 * 
 * Role: Provide mutation hooks for analysis operations. Handle optimistic updates,
 * cache invalidation, and upload progress tracking.
 * 
 * Integration:
 * - Imports: @tanstack/react-query, services/analysisApi
 * - Used by: AnalysisForm.tsx, AnalysisCard.tsx
 * - Updates: Query cache on mutation success
 * - Backend: POST /api/v1/analyze, DELETE /api/v1/analyze/{id}
 */

import { useMutation, useQueryClient, UseMutationResult } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { analysisApi } from '@/services/analysisApi';
import { useUploadStore } from '@/store/uploadStore';
import type { 
  AnalysisResponse, 
  AnalysisOptions, 
  DefenseLevel 
} from '@/types/analysis';
import { getErrorMessage } from '@/services/api';

// ============== TYPES ==============

/**
 * Input for submit analysis mutation
 */
export interface SubmitAnalysisInput {
  file: File;
  options: AnalysisOptions;
  onUploadProgress?: (progress: number) => void;
}

/**
 * Input for text analysis mutation
 */
export interface SubmitTextAnalysisInput {
  text: string;
  generateReport?: boolean;
}

/**
 * Default analysis options
 */
export const DEFAULT_ANALYSIS_OPTIONS: AnalysisOptions = {
  generateReport: true,
  generateHeatmaps: true,
  defenseLevel: 'standard' as DefenseLevel,
};

/**
 * Return type for useAnalysis hook
 */
export interface UseAnalysisReturn {
  /** Mutation for submitting media analysis */
  submitAnalysis: UseMutationResult<AnalysisResponse, Error, SubmitAnalysisInput>;
  /** Mutation for submitting text analysis */
  submitTextAnalysis: UseMutationResult<AnalysisResponse, Error, SubmitTextAnalysisInput>;
  /** Mutation for deleting analysis */
  deleteAnalysis: UseMutationResult<void, Error, string>;
  /** Helper to check if any mutation is pending */
  isSubmitting: boolean;
  /** Helper to check if delete is pending */
  isDeleting: boolean;
}

// ============== QUERY KEYS ==============

/**
 * Query key factory for analysis queries
 */
export const analysisKeys = {
  all: ['analyses'] as const,
  lists: () => [...analysisKeys.all, 'list'] as const,
  list: (filters: Record<string, unknown>) => [...analysisKeys.lists(), filters] as const,
  details: () => [...analysisKeys.all, 'detail'] as const,
  detail: (id: string) => [...analysisKeys.details(), id] as const,
};

// ============== HOOK ==============

/**
 * useAnalysis hook
 * 
 * Provides mutations for analysis operations with automatic cache management.
 * 
 * @returns Object containing mutation functions and state
 * 
 * @example
 * ```tsx
 * const { submitAnalysis, isSubmitting } = useAnalysis();
 * 
 * const handleSubmit = async (file: File) => {
 *   await submitAnalysis.mutateAsync({
 *     file,
 *     options: { generateReport: true, generateHeatmaps: true, defenseLevel: 'standard' },
 *     onUploadProgress: (progress) => console.log(`Upload: ${progress}%`),
 *   });
 * };
 * ```
 */
export function useAnalysis(): UseAnalysisReturn {
  const queryClient = useQueryClient();
  const router = useRouter();
  const { setUploadProgress, setStatus, setError, setAnalysisId, reset } = useUploadStore();

  // ============== SUBMIT ANALYSIS MUTATION ==============

  const submitAnalysis = useMutation<AnalysisResponse, Error, SubmitAnalysisInput>({
    mutationFn: async ({ file, options, onUploadProgress }) => {
      // Track upload progress
      const progressHandler = (progress: number) => {
        setUploadProgress(progress);
        onUploadProgress?.(progress);
      };

      return analysisApi.submitAnalysis(file, options, progressHandler);
    },
    
    onMutate: () => {
      // Set uploading status when mutation starts
      setStatus('uploading');
      setError(null);
    },
    
    onSuccess: (data) => {
      // Update upload store with analysis ID
      setAnalysisId(data.analysis_id);
      setStatus('complete');
      
      // Invalidate analysis list cache
      queryClient.invalidateQueries({ queryKey: analysisKeys.lists() });
      
      // Navigate to analysis page
      router.push(`/analysis/${data.analysis_id}`);
    },
    
    onError: (error) => {
      // Set error state
      const message = getErrorMessage(error);
      setError(message);
      setStatus('error');
    },
  });

  // ============== SUBMIT TEXT ANALYSIS MUTATION ==============

  const submitTextAnalysis = useMutation<AnalysisResponse, Error, SubmitTextAnalysisInput>({
    mutationFn: async ({ text, generateReport = false }) => {
      return analysisApi.submitTextAnalysis(text, generateReport);
    },
    
    onMutate: () => {
      setStatus('uploading');
      setError(null);
    },
    
    onSuccess: (data) => {
      setAnalysisId(data.analysis_id);
      setStatus('complete');
      
      // Invalidate analysis list cache
      queryClient.invalidateQueries({ queryKey: analysisKeys.lists() });
      
      // Navigate to analysis page
      router.push(`/analysis/${data.analysis_id}`);
    },
    
    onError: (error) => {
      const message = getErrorMessage(error);
      setError(message);
      setStatus('error');
    },
  });

  // ============== DELETE ANALYSIS MUTATION ==============

  const deleteAnalysis = useMutation<void, Error, string>({
    mutationFn: async (analysisId: string) => {
      return analysisApi.deleteAnalysis(analysisId);
    },
    
    onSuccess: (_, analysisId) => {
      // Remove from cache
      queryClient.removeQueries({ queryKey: analysisKeys.detail(analysisId) });
      
      // Invalidate list to refresh
      queryClient.invalidateQueries({ queryKey: analysisKeys.lists() });
    },
    
    onError: (error) => {
      console.error('Failed to delete analysis:', getErrorMessage(error));
    },
  });

  // ============== COMPUTED STATE ==============

  const isSubmitting = submitAnalysis.isPending || submitTextAnalysis.isPending;
  const isDeleting = deleteAnalysis.isPending;

  return {
    submitAnalysis,
    submitTextAnalysis,
    deleteAnalysis,
    isSubmitting,
    isDeleting,
  };
}

// ============== STANDALONE FUNCTIONS ==============

/**
 * Prefetch analysis detail for faster navigation
 */
export function usePrefetchAnalysis() {
  const queryClient = useQueryClient();

  return (analysisId: string) => {
    queryClient.prefetchQuery({
      queryKey: analysisKeys.detail(analysisId),
      queryFn: () => analysisApi.getAnalysis(analysisId),
      staleTime: 30000, // 30 seconds
    });
  };
}

/**
 * Reset analysis mutations (useful for cleanup)
 */
export function useResetAnalysisMutations() {
  const { reset } = useUploadStore();
  
  return () => {
    reset();
  };
}

export default useAnalysis;
