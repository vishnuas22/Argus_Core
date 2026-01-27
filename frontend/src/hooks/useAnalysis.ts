/**
 * Argus Core - Analysis Hook
 * ==========================
 * TanStack Query mutations for analysis CRUD operations.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - hooks/useAnalysis.ts
 * 
 * Role: Submit analysis, delete analysis mutations with TanStack Query.
 * Handles loading states, error handling, and cache invalidation.
 * 
 * Integration:
 * - Imports: @tanstack/react-query, services/analysisApi
 * - Used by: components/analysis/AnalysisForm, pages
 * - Updates: Query cache on success
 */

import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { analysisApi } from '@/services/analysisApi';
import type {
  AnalysisResponse,
  AnalysisDetailResponse,
  AnalysisOptions,
  AnalysisStatus,
  ListParams,
} from '@/types/analysis';

// ============== QUERY KEYS ==============

export const analysisKeys = {
  all: ['analyses'] as const,
  lists: () => [...analysisKeys.all, 'list'] as const,
  list: (params?: ListParams) => [...analysisKeys.lists(), params] as const,
  details: () => [...analysisKeys.all, 'detail'] as const,
  detail: (id: string) => [...analysisKeys.details(), id] as const,
  status: (id: string) => [...analysisKeys.all, 'status', id] as const,
};

// ============== TYPES ==============

export interface SubmitAnalysisInput {
  file: File;
  options: AnalysisOptions;
  onUploadProgress?: (progress: number) => void;
}

export interface SubmitTextAnalysisInput {
  text: string;
  generateReport?: boolean;
}

export interface UseAnalysisReturn {
  submitAnalysis: ReturnType<typeof useMutation<AnalysisResponse, Error, SubmitAnalysisInput>>;
  submitTextAnalysis: ReturnType<typeof useMutation<AnalysisResponse, Error, SubmitTextAnalysisInput>>;
  deleteAnalysis: ReturnType<typeof useMutation<void, Error, string>>;
}

// ============== HOOK ==============

/**
 * Hook for analysis mutations
 * 
 * Provides submit and delete mutations with proper cache invalidation.
 */
export function useAnalysis(): UseAnalysisReturn {
  const queryClient = useQueryClient();
  
  /**
   * Submit media analysis mutation
   */
  const submitAnalysis = useMutation<AnalysisResponse, Error, SubmitAnalysisInput>({
    mutationFn: async ({ file, options, onUploadProgress }) => {
      return analysisApi.submitAnalysis(file, options, onUploadProgress);
    },
    onSuccess: (data) => {
      // Invalidate list queries to show new analysis
      queryClient.invalidateQueries({ queryKey: analysisKeys.lists() });
      
      // Optionally pre-populate the analysis cache
      queryClient.setQueryData(
        analysisKeys.detail(data.analysis_id),
        data
      );
    },
    onError: (error) => {
      console.error('Analysis submission failed:', error);
    },
  });
  
  /**
   * Submit text analysis mutation
   */
  const submitTextAnalysis = useMutation<AnalysisResponse, Error, SubmitTextAnalysisInput>({
    mutationFn: async ({ text, generateReport }) => {
      return analysisApi.submitTextAnalysis(text, generateReport);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: analysisKeys.lists() });
      queryClient.setQueryData(
        analysisKeys.detail(data.analysis_id),
        data
      );
    },
    onError: (error) => {
      console.error('Text analysis submission failed:', error);
    },
  });
  
  /**
   * Delete analysis mutation
   */
  const deleteAnalysis = useMutation<void, Error, string>({
    mutationFn: async (analysisId) => {
      return analysisApi.deleteAnalysis(analysisId);
    },
    onSuccess: (_, analysisId) => {
      // Remove from cache
      queryClient.removeQueries({ queryKey: analysisKeys.detail(analysisId) });
      
      // Invalidate lists
      queryClient.invalidateQueries({ queryKey: analysisKeys.lists() });
    },
    onError: (error) => {
      console.error('Analysis deletion failed:', error);
    },
  });
  
  return {
    submitAnalysis,
    submitTextAnalysis,
    deleteAnalysis,
  };
}

// ============== QUERY HOOKS ==============

/**
 * Hook to fetch analysis by ID
 * Automatically polls when status is not completed
 */
export function useAnalysisStatus(
  analysisId: string | null,
  options?: { enabled?: boolean; refetchInterval?: number | false }
) {
  return useQuery<AnalysisResponse, Error>({
    queryKey: analysisKeys.status(analysisId ?? ''),
    queryFn: () => {
      if (!analysisId) throw new Error('No analysis ID');
      return analysisApi.getAnalysis(analysisId);
    },
    enabled: !!analysisId && (options?.enabled !== false),
    refetchInterval: (query) => {
      const data = query.state.data;
      // Stop polling when completed or failed
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false;
      }
      // Poll every 2 seconds while in progress
      return options?.refetchInterval ?? 2000;
    },
    staleTime: 1000, // Consider data stale after 1 second
  });
}

/**
 * Hook to fetch detailed analysis results
 * Only fetches when analysis is completed
 */
export function useAnalysisDetail(
  analysisId: string | null,
  options?: { enabled?: boolean }
) {
  return useQuery<AnalysisDetailResponse, Error>({
    queryKey: analysisKeys.detail(analysisId ?? ''),
    queryFn: () => {
      if (!analysisId) throw new Error('No analysis ID');
      return analysisApi.getAnalysisDetail(analysisId);
    },
    enabled: !!analysisId && (options?.enabled !== false),
    staleTime: 5 * 60 * 1000, // 5 minutes - detailed results don't change
    gcTime: 30 * 60 * 1000, // Keep in cache for 30 minutes
  });
}

/**
 * Hook to fetch analysis list with pagination
 */
export function useAnalysisList(params?: ListParams) {
  return useQuery<AnalysisResponse[], Error>({
    queryKey: analysisKeys.list(params),
    queryFn: () => analysisApi.listAnalyses(params),
    staleTime: 30 * 1000, // 30 seconds
  });
}

/**
 * Hook to get report URL
 */
export function useAnalysisReport(
  analysisId: string | null,
  options?: { enabled?: boolean; regenerate?: boolean }
) {
  return useQuery<{ reportUrl: string }, Error>({
    queryKey: [...analysisKeys.detail(analysisId ?? ''), 'report'],
    queryFn: () => {
      if (!analysisId) throw new Error('No analysis ID');
      return analysisApi.getReport(analysisId, options?.regenerate);
    },
    enabled: !!analysisId && (options?.enabled !== false),
    staleTime: 60 * 60 * 1000, // 1 hour - URLs expire but not that fast
  });
}

/**
 * Hook to get heatmap URLs
 */
export function useAnalysisHeatmaps(
  analysisId: string | null,
  options?: { enabled?: boolean }
) {
  return useQuery({
    queryKey: [...analysisKeys.detail(analysisId ?? ''), 'heatmaps'],
    queryFn: () => {
      if (!analysisId) throw new Error('No analysis ID');
      return analysisApi.getHeatmaps(analysisId);
    },
    enabled: !!analysisId && (options?.enabled !== false),
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

export default useAnalysis;
