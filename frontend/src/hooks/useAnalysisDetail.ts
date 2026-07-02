/**
 * Argus Core - Analysis Detail Hook
 * ==================================
 * TanStack Query hook for fetching detailed analysis results.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - hooks/useAnalysisDetail.ts
 * 
 * Role: Provide query hook for fetching detailed analysis results. Handles loading,
 * error states, and automatic refetching when analysis completes.
 * 
 * Integration:
 * - Imports: @tanstack/react-query, services/analysisApi
 * - Used by: analysis/[id]/page.tsx, ResultsPanel.tsx, ModalityTabs.tsx
 * - Updates: Automatically refetches when WebSocket reports completion
 * - Backend: GET /api/v1/analyze/{id}, GET /api/v1/analyze/{id}/detail
 */

import { useQuery, useQueryClient, UseQueryResult } from '@tanstack/react-query';
import { analysisApi } from '@/services/analysisApi';
import { analysisKeys } from '@/hooks/useAnalysis';
import { useProgressStore } from '@/store/progressStore';
import type { 
  AnalysisResponse, 
  AnalysisDetailResponse,
  HeatmapResponse 
} from '@/types/analysis';

// ============== TYPES ==============

/**
 * Return type for useAnalysisDetail hook
 */
export interface UseAnalysisDetailReturn {
  /** Basic analysis data */
  analysis: AnalysisResponse | undefined;
  /** Detailed analysis data including modality results */
  detail: AnalysisDetailResponse | undefined;
  /** Whether basic analysis is loading */
  isLoading: boolean;
  /** Whether detail is loading */
  isDetailLoading: boolean;
  /** Error from either query */
  error: Error | null;
  /** Refetch analysis data */
  refetch: () => Promise<void>;
  /** Whether analysis is complete */
  isComplete: boolean;
  /** Whether analysis failed */
  isFailed: boolean;
  /** Whether analysis is in progress */
  isInProgress: boolean;
}

/**
 * Options for useAnalysisDetail hook
 */
export interface UseAnalysisDetailOptions {
  /** Whether to fetch detail on complete (default: true) */
  fetchDetailOnComplete?: boolean;
  /** Refetch interval in ms when in progress (default: disabled - uses WebSocket) */
  refetchInterval?: number;
  /** Whether to enable the query (default: true) */
  enabled?: boolean;
}

// ============== HOOK ==============

/**
 * useAnalysisDetail hook
 * 
 * Fetches analysis data with automatic detail fetching when complete.
 * Uses WebSocket for real-time updates during analysis.
 * 
 * @param analysisId - Analysis ID to fetch
 * @param options - Query options
 * @returns Analysis data and query state
 * 
 * @example
 * ```tsx
 * const { analysis, detail, isLoading, isComplete } = useAnalysisDetail(analysisId);
 * 
 * if (isLoading) return <Skeleton />;
 * if (isComplete && detail) {
 *   return <ResultsPanel data={detail} />;
 * }
 * return <ProgressIndicator analysisId={analysisId} />;
 * ```
 */
export function useAnalysisDetail(
  analysisId: string,
  options: UseAnalysisDetailOptions = {}
): UseAnalysisDetailReturn {
  const {
    fetchDetailOnComplete = true,
    refetchInterval,
    enabled = true,
  } = options;

  const queryClient = useQueryClient();

  // Get progress from store to determine if complete
  const progressStatus = useProgressStore((state) => 
    state.progress[analysisId]?.status
  );

  // ============== BASIC ANALYSIS QUERY ==============

  const analysisQuery = useQuery<AnalysisResponse, Error>({
    queryKey: analysisKeys.detail(analysisId),
    queryFn: () => analysisApi.getAnalysis(analysisId),
    enabled: enabled && !!analysisId,
    staleTime: 30000, // 30 seconds
    // Poll every 5s as fallback when WebSocket is unavailable
    refetchInterval: refetchInterval ?? ((query) => {
      const analysisData = query.state.data;
      if (!analysisData) return 5000;
      if (analysisData.status === 'completed' || analysisData.status === 'failed') return false;
      return 5000;
    }),
    // Retry configuration
    retry: (failureCount, error) => {
      // Don't retry 404s
      if (error.message.includes('404') || error.message.includes('not found')) {
        return false;
      }
      return failureCount < 3;
    },
  });

  // Determine completion from both query data and progress store
  const isComplete = analysisQuery.data?.status === 'completed' || progressStatus === 'completed';
  const isFailed = analysisQuery.data?.status === 'failed' || progressStatus === 'failed';

  // ============== DETAILED ANALYSIS QUERY ==============

  const detailQuery = useQuery<AnalysisDetailResponse, Error>({
    queryKey: [...analysisKeys.detail(analysisId), 'full'],
    queryFn: () => analysisApi.getAnalysisDetail(analysisId),
    enabled: enabled && !!analysisId && isComplete && fetchDetailOnComplete,
    staleTime: 60000, // 1 minute - details don't change often
    retry: 2,
  });

  // ============== COMPUTED STATE ==============

  const isInProgress = !isComplete && !isFailed && 
    (analysisQuery.data?.status === 'pending' ||
     analysisQuery.data?.status === 'preprocessing' ||
     analysisQuery.data?.status === 'analyzing' ||
     analysisQuery.data?.status === 'aggregating' ||
     progressStatus === 'pending' ||
     progressStatus === 'preprocessing' ||
     progressStatus === 'analyzing' ||
     progressStatus === 'aggregating');

  // ============== REFETCH ==============

  const refetch = async () => {
    await queryClient.invalidateQueries({ queryKey: analysisKeys.detail(analysisId) });
    if (isComplete) {
      await queryClient.invalidateQueries({ 
        queryKey: [...analysisKeys.detail(analysisId), 'full'] 
      });
    }
  };

  // ============== RETURN ==============

  return {
    analysis: analysisQuery.data,
    detail: detailQuery.data,
    isLoading: analysisQuery.isLoading,
    isDetailLoading: detailQuery.isLoading,
    error: analysisQuery.error || detailQuery.error,
    refetch,
    isComplete,
    isFailed,
    isInProgress,
  };
}

// ============== ADDITIONAL HOOKS ==============

/**
 * Hook to fetch heatmaps for an analysis
 */
export function useHeatmaps(
  analysisId: string,
  options: { enabled?: boolean } = {}
): UseQueryResult<HeatmapResponse, Error> {
  const { enabled = true } = options;

  return useQuery<HeatmapResponse, Error>({
    queryKey: [...analysisKeys.detail(analysisId), 'heatmaps'],
    queryFn: () => analysisApi.getHeatmaps(analysisId),
    enabled: enabled && !!analysisId,
    staleTime: 300000, // 5 minutes - heatmaps don't change
  });
}

/**
 * Hook to fetch report URL for an analysis
 */
export function useReport(
  analysisId: string,
  options: { enabled?: boolean; regenerate?: boolean } = {}
): UseQueryResult<{ reportUrl: string }, Error> {
  const { enabled = true, regenerate = false } = options;

  return useQuery<{ reportUrl: string }, Error>({
    queryKey: [...analysisKeys.detail(analysisId), 'report', regenerate],
    queryFn: () => analysisApi.getReport(analysisId, regenerate),
    enabled: enabled && !!analysisId,
    staleTime: regenerate ? 0 : 300000, // 5 minutes unless regenerating
  });
}

/**
 * Prefetch analysis detail for faster navigation
 */
export function usePrefetchAnalysisDetail() {
  const queryClient = useQueryClient();

  return (analysisId: string) => {
    // Prefetch basic analysis
    queryClient.prefetchQuery({
      queryKey: analysisKeys.detail(analysisId),
      queryFn: () => analysisApi.getAnalysis(analysisId),
      staleTime: 30000,
    });
  };
}

export default useAnalysisDetail;
