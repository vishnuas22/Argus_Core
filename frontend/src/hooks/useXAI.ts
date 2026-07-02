/**
 * Argus Core - XAI Hook
 * =====================
 * TanStack Query hook for fetching Explainable AI data.
 * 
 * Implements: XAI_FRONTEND_IMPLEMENTATION.md - Section 4.2 - hooks/useXAI.ts
 * 
 * Role: Provide query hooks for fetching XAI explanations, heatmaps,
 * and evidence packages. Integrates with Zustand store for state management.
 * 
 * Integration:
 * - Imports: @tanstack/react-query, services/analysisApi, store/xaiStore
 * - Used by: XAIExplanationPanel, XAIEvidenceGallery, modality panels
 * - Updates: Zustand store on successful fetch
 * - Backend: GET /api/v1/analyze/{id}/xai, GET /api/v1/analyze/{id}/heatmaps
 */

import { useQuery, useQueryClient, UseQueryResult } from '@tanstack/react-query';
import { useEffect, useCallback } from 'react';
import { analysisApi, XAIResponse } from '@/services/analysisApi';
import { useXAIStore, selectNeedsFetch, selectHasXAI } from '@/store/xaiStore';
import { analysisKeys } from '@/hooks/useAnalysis';
import type {
  XAIExplanation,
  XAIHeatmapResponse,
  FeatureImportance,
  VisualEvidence,
  ScientificReference,
  AudioArtifactRegion,
  ManipulationRegion,
  ModalityXAI,
  AnalysisDetailResponseXAI,
} from '@/types/analysis';

// ============== TYPES ==============

/**
 * Normalize confidence_interval from backend dict format to frontend array format
 */
function normalizeConfidenceInterval(ci: unknown): [number, number] | null {
  if (!ci) return null;
  if (Array.isArray(ci) && ci.length === 2) return [ci[0], ci[1]];
  if (typeof ci === 'object' && ci !== null && 'lower' in ci && 'upper' in ci) {
    return [Number((ci as Record<string, unknown>).lower), Number((ci as Record<string, unknown>).upper)];
  }
  return null;
}

/**
 * Return type for useXAI hook
 */
export interface UseXAIReturn {
  /** XAI data */
  data: XAIResponse | undefined;
  /** Whether XAI is loading */
  isLoading: boolean;
  /** Whether XAI data is being fetched in background */
  isFetching: boolean;
  /** Error from query */
  error: Error | null;
  /** Refetch XAI data */
  refetch: () => Promise<void>;
  /** Whether XAI data is available */
  hasXAI: boolean;
  /** Active modality */
  activeModality: 'image' | 'video' | 'audio';
  /** Set active modality */
  setActiveModality: (modality: 'image' | 'video' | 'audio') => void;
  /** Get XAI for specific modality */
  getModalityXAI: (modality: 'image' | 'video' | 'audio') => ModalityXAI | null;
}

/**
 * Options for useXAI hook
 */
export interface UseXAIOptions {
  /** Whether to enable the query (default: true) */
  enabled?: boolean;
  /** Whether to sync with store (default: true) */
  syncWithStore?: boolean;
  /** Stale time in ms (default: 5 minutes) */
  staleTime?: number;
}

/**
 * Return type for useFeatureImportance hook
 */
export interface UseFeatureImportanceReturn {
  features: FeatureImportance[];
  isLoading: boolean;
  error: Error | null;
  /** Features sorted by importance (descending) */
  sortedFeatures: FeatureImportance[];
  /** Features that increase fake probability */
  fakeIndicators: FeatureImportance[];
  /** Features that decrease fake probability */
  authenticIndicators: FeatureImportance[];
}

/**
 * Return type for useVisualEvidence hook
 */
export interface UseVisualEvidenceReturn {
  evidence: VisualEvidence[];
  isLoading: boolean;
  error: Error | null;
  /** Evidence grouped by type */
  evidenceByType: Record<string, VisualEvidence[]>;
  /** Heatmap evidence */
  heatmaps: VisualEvidence[];
  /** Spectrogram overlays */
  spectrograms: VisualEvidence[];
}

// ============== QUERY KEYS ==============

/**
 * Query keys for XAI data
 */
export const xaiKeys = {
  all: ['xai'] as const,
  byAnalysis: (analysisId: string) => [...xaiKeys.all, analysisId] as const,
  explanation: (analysisId: string) => [...xaiKeys.byAnalysis(analysisId), 'explanation'] as const,
  heatmaps: (analysisId: string) => [...xaiKeys.byAnalysis(analysisId), 'heatmaps'] as const,
  evidence: (analysisId: string) => [...xaiKeys.byAnalysis(analysisId), 'evidence'] as const,
};

// ============== MAIN HOOK ==============

/**
 * useXAI hook
 * 
 * Fetches XAI explanation data for an analysis. Automatically syncs
 * with the Zustand store for cross-component state access.
 * 
 * @param analysisId - Analysis ID to fetch XAI for
 * @param options - Query options
 * @returns XAI data and query state
 * 
 * @example
 * ```tsx
 * const { data, isLoading, hasXAI, activeModality, setActiveModality } = useXAI(analysisId);
 * 
 * if (isLoading) return <Skeleton />;
 * if (!hasXAI) return <div>No XAI data available</div>;
 * 
 * return (
 *   <XAIExplanationPanel 
 *     explanation={data?.image_xai?.explanation} 
 *     modality={activeModality}
 *   />
 * );
 * ```
 */
export function useXAI(
  analysisId: string,
  options: UseXAIOptions = {}
): UseXAIReturn {
  const {
    enabled = true,
    syncWithStore = true,
    staleTime = 300000, // 5 minutes
  } = options;

  const queryClient = useQueryClient();
  
  // Store access
  const setAnalysisId = useXAIStore((state) => state.setAnalysisId);
  const setImageXAI = useXAIStore((state) => state.setImageXAI);
  const setVideoXAI = useXAIStore((state) => state.setVideoXAI);
  const setAudioXAI = useXAIStore((state) => state.setAudioXAI);
  const setEvidencePackage = useXAIStore((state) => state.setEvidencePackage);
  const setStatus = useXAIStore((state) => state.setStatus);
  const setError = useXAIStore((state) => state.setError);
  const activeModality = useXAIStore((state) => state.activeModality);
  const setActiveModality = useXAIStore((state) => state.setActiveModality);
  const hasXAI = useXAIStore(selectHasXAI);

  // Sync analysis ID with store
  useEffect(() => {
    if (syncWithStore) {
      setAnalysisId(analysisId);
    }
  }, [analysisId, syncWithStore, setAnalysisId]);

  // ============== QUERY ==============

  const query = useQuery<XAIResponse, Error>({
    queryKey: xaiKeys.explanation(analysisId),
    queryFn: async () => {
      // First try to get XAI from detail endpoint
      try {
        const detail = await analysisApi.getAnalysisDetail(analysisId) as AnalysisDetailResponseXAI;
        
        // Transform detail response to XAI response
        const xaiResponse: XAIResponse = {
          analysis_id: analysisId,
        };

        // Check for image XAI data
        if (detail.image_result) {
          const imageFeatures = detail.feature_importance.filter(f => f.feature_type === 'visual' || f.modality === 'image' || !f.modality);
          if (imageFeatures.length > 0 || detail.evidence_package) {
            xaiResponse.image_xai = {
              explanation: {
                feature_importance: imageFeatures,
                visual_evidence: detail.evidence_package?.visual_evidence || [],
                scientific_references: detail.scientific_references,
                reproducibility_hash: detail.evidence_package?.reproducibility_hash || '',
                confidence_interval: normalizeConfidenceInterval(detail.image_result.confidence_interval) || normalizeConfidenceInterval(detail.evidence_package?.confidence_interval) || [0.25, 0.75],
                model_versions: detail.evidence_package?.model_versions || {},
              },
              manipulation_regions: detail.image_result.manipulation_regions || [],
              heatmap_urls: detail.image_result.heatmap_url ? [detail.image_result.heatmap_url] : [],
              overlay_url: detail.image_result.heatmap_url || null,
            };
          }
        }

        // Check for video XAI data
        if (detail.video_result) {
          const videoXaiExplanation = detail.video_result.xai_explanation;
          if (videoXaiExplanation || detail.feature_importance.length > 0) {
            xaiResponse.video_xai = {
              explanation: videoXaiExplanation || {
                feature_importance: detail.feature_importance.filter(f => f.modality === 'video' || !f.modality),
                visual_evidence: detail.evidence_package?.visual_evidence || [],
                scientific_references: detail.scientific_references,
                reproducibility_hash: detail.evidence_package?.reproducibility_hash || '',
                confidence_interval: normalizeConfidenceInterval(detail.evidence_package?.confidence_interval) || [0.25, 0.75],
                model_versions: detail.evidence_package?.model_versions || {},
              },
              manipulation_regions: detail.video_result.spatial?.manipulation_regions || [],
              heatmap_urls: detail.video_result.frame_heatmap_urls || [],
              temporal_heatmap_url: detail.video_result.temporal_heatmap_url || null,
            };
          }
        }

        // Check for audio XAI data
        if (detail.audio_result?.xai_explanation) {
          xaiResponse.audio_xai = {
            explanation: detail.audio_result.xai_explanation,
            artifact_regions: detail.audio_result.artifact_regions || [],
            spectrogram_overlay_url: null,
          };
        }

        // Add evidence package if available
        if (detail.evidence_package) {
          xaiResponse.evidence_package = detail.evidence_package;
        } else if (detail.feature_importance.length > 0) {
          // Create evidence package from feature importance if not directly available
          xaiResponse.evidence_package = {
            feature_importance: detail.feature_importance,
            visual_evidence: [],
            scientific_references: detail.scientific_references,
            reproducibility_hash: '',
            confidence_interval: [0.25, 0.75],
            model_versions: {},
          };
        }

        return xaiResponse;
      } catch (error) {
        // If detail endpoint fails, try dedicated XAI endpoint
        const response = await analysisApi.getXAI(analysisId);
        return response;
      }
    },
    enabled: enabled && !!analysisId,
    staleTime,
    retry: 2,
    refetchOnWindowFocus: false,
  });

  // ============== SYNC WITH STORE ==============

  useEffect(() => {
    if (!syncWithStore) return;

    if (query.isLoading) {
      setStatus('loading');
    } else if (query.error) {
      setError(query.error.message);
    } else if (query.data) {
      // Sync each modality's XAI data
      if (query.data.image_xai) {
        setImageXAI({
          explanation: query.data.image_xai.explanation,
          artifactRegions: query.data.image_xai.manipulation_regions,
          heatmapUrls: query.data.image_xai.heatmap_urls,
          overlayUrl: query.data.image_xai.overlay_url,
        });
      }

      if (query.data.video_xai) {
        setVideoXAI({
          explanation: query.data.video_xai.explanation,
          artifactRegions: query.data.video_xai.manipulation_regions,
          heatmapUrls: query.data.video_xai.heatmap_urls,
          overlayUrl: query.data.video_xai.temporal_heatmap_url,
        });
      }

      if (query.data.audio_xai) {
        setAudioXAI({
          explanation: query.data.audio_xai.explanation,
          artifactRegions: query.data.audio_xai.artifact_regions,
          heatmapUrls: [],
          overlayUrl: query.data.audio_xai.spectrogram_overlay_url,
        });
      }

      if (query.data.evidence_package) {
        setEvidencePackage({
          featureImportance: query.data.evidence_package.feature_importance,
          visualEvidence: query.data.evidence_package.visual_evidence,
          scientificReferences: query.data.evidence_package.scientific_references || [],
          reproducibilityHash: query.data.evidence_package.reproducibility_hash || '',
          confidenceInterval: query.data.evidence_package.confidence_interval,
        });
      }

      setStatus('success');
    }
  }, [
    query.data,
    query.isLoading,
    query.error,
    syncWithStore,
    setImageXAI,
    setVideoXAI,
    setAudioXAI,
    setEvidencePackage,
    setStatus,
    setError,
  ]);

  // ============== HELPERS ==============

  const refetch = useCallback(async () => {
    await queryClient.invalidateQueries({ queryKey: xaiKeys.explanation(analysisId) });
  }, [queryClient, analysisId]);

  const getModalityXAI = useCallback(
    (modality: 'image' | 'video' | 'audio'): ModalityXAI | null => {
      const data = query.data;
      if (!data) return null;

      switch (modality) {
        case 'image':
          return data.image_xai
            ? {
                explanation: data.image_xai.explanation,
                artifactRegions: data.image_xai.manipulation_regions,
                heatmapUrls: data.image_xai.heatmap_urls,
                overlayUrl: data.image_xai.overlay_url,
              }
            : null;
        case 'video':
          return data.video_xai
            ? {
                explanation: data.video_xai.explanation,
                artifactRegions: data.video_xai.manipulation_regions,
                heatmapUrls: data.video_xai.heatmap_urls,
                overlayUrl: data.video_xai.temporal_heatmap_url,
              }
            : null;
        case 'audio':
          return data.audio_xai
            ? {
                explanation: data.audio_xai.explanation,
                artifactRegions: data.audio_xai.artifact_regions,
                heatmapUrls: [],
                overlayUrl: data.audio_xai.spectrogram_overlay_url,
              }
            : null;
        default:
          return null;
      }
    },
    [query.data]
  );

  // ============== RETURN ==============

  return {
    data: query.data,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    error: query.error,
    refetch,
    hasXAI,
    activeModality,
    setActiveModality,
    getModalityXAI,
  };
}

// ============== ADDITIONAL HOOKS ==============

/**
 * Hook to fetch XAI heatmaps with overlay URLs
 */
export function useXAIHeatmaps(
  analysisId: string,
  options: { enabled?: boolean } = {}
): UseQueryResult<XAIHeatmapResponse, Error> {
  const { enabled = true } = options;

  return useQuery<XAIHeatmapResponse, Error>({
    queryKey: xaiKeys.heatmaps(analysisId),
    queryFn: () => analysisApi.getXAIHeatmaps(analysisId),
    enabled: enabled && !!analysisId,
    staleTime: 300000, // 5 minutes
  });
}

/**
 * Hook to get feature importance with computed helpers
 */
export function useFeatureImportance(
  analysisId: string,
  options: UseXAIOptions = {}
): UseFeatureImportanceReturn {
  const { data, isLoading, error } = useXAI(analysisId, options);

  // Aggregate features from all modalities
  const features: FeatureImportance[] = [];
  
  if (data?.image_xai?.explanation?.feature_importance) {
    features.push(...data.image_xai.explanation.feature_importance);
  }
  if (data?.video_xai?.explanation?.feature_importance) {
    features.push(...data.video_xai.explanation.feature_importance);
  }
  if (data?.audio_xai?.explanation?.feature_importance) {
    features.push(...data.audio_xai.explanation.feature_importance);
  }

  // Sort by importance score (descending)
  const sortedFeatures = [...features].sort((a, b) => b.importance_score - a.importance_score);

  // Filter by contribution direction
  const fakeIndicators = features.filter(
    (f) => f.contribution_direction === 'increases_fake'
  );
  const authenticIndicators = features.filter(
    (f) => f.contribution_direction === 'decreases_fake'
  );

  return {
    features,
    isLoading,
    error,
    sortedFeatures,
    fakeIndicators,
    authenticIndicators,
  };
}

/**
 * Hook to get visual evidence with computed helpers
 */
export function useVisualEvidence(
  analysisId: string,
  options: UseXAIOptions = {}
): UseVisualEvidenceReturn {
  const { data, isLoading, error } = useXAI(analysisId, options);

  // Aggregate evidence from all modalities
  const evidence: VisualEvidence[] = [];
  
  if (data?.image_xai?.explanation?.visual_evidence) {
    evidence.push(...data.image_xai.explanation.visual_evidence);
  }
  if (data?.video_xai?.explanation?.visual_evidence) {
    evidence.push(...data.video_xai.explanation.visual_evidence);
  }
  if (data?.audio_xai?.explanation?.visual_evidence) {
    evidence.push(...data.audio_xai.explanation.visual_evidence);
  }

  // Group by type
  const evidenceByType: Record<string, VisualEvidence[]> = {};
  for (const e of evidence) {
    if (!evidenceByType[e.artifact_type]) {
      evidenceByType[e.artifact_type] = [];
    }
    evidenceByType[e.artifact_type].push(e);
  }

  // Filter by type
  const heatmaps = evidence.filter((e) => e.artifact_type === 'heatmap');
  const spectrograms = evidence.filter((e) => e.artifact_type === 'spectrogram');

  return {
    evidence,
    isLoading,
    error,
    evidenceByType,
    heatmaps,
    spectrograms,
  };
}

/**
 * Hook to get scientific references (deduplicated)
 */
export function useScientificReferences(
  analysisId: string,
  options: UseXAIOptions = {}
): {
  references: ScientificReference[];
  isLoading: boolean;
  error: Error | null;
  referencesByMethod: Record<string, ScientificReference>;
} {
  const { data, isLoading, error } = useXAI(analysisId, options);

  // Aggregate and deduplicate references
  const referencesMap = new Map<string, ScientificReference>();
  
  const addReferences = (xai?: { explanation?: XAIExplanation }) => {
    if (xai?.explanation?.scientific_references) {
      for (const ref of xai.explanation.scientific_references) {
        if (!referencesMap.has(ref.method_name)) {
          referencesMap.set(ref.method_name, ref);
        }
      }
    }
  };

  addReferences(data?.image_xai);
  addReferences(data?.video_xai);
  addReferences(data?.audio_xai);

  const references = Array.from(referencesMap.values());
  const referencesByMethod = Object.fromEntries(referencesMap);

  return {
    references,
    isLoading,
    error,
    referencesByMethod,
  };
}

/**
 * Hook to get reproducibility information
 */
export function useReproducibility(
  analysisId: string,
  options: UseXAIOptions = {}
): {
  hash: string | null;
  confidenceInterval: [number, number] | null;
  modelVersions: Record<string, string>;
  isLoading: boolean;
  error: Error | null;
} {
  const { data, isLoading, error } = useXAI(analysisId, options);

  // Get from evidence package or first available modality
  const hash = data?.evidence_package?.reproducibility_hash ??
    data?.image_xai?.explanation?.reproducibility_hash ??
    data?.video_xai?.explanation?.reproducibility_hash ??
    data?.audio_xai?.explanation?.reproducibility_hash ??
    null;

  const confidenceInterval = data?.evidence_package?.confidence_interval ??
    data?.image_xai?.explanation?.confidence_interval ??
    data?.video_xai?.explanation?.confidence_interval ??
    data?.audio_xai?.explanation?.confidence_interval ??
    null;

  const modelVersions = data?.image_xai?.explanation?.model_versions ??
    data?.video_xai?.explanation?.model_versions ??
    data?.audio_xai?.explanation?.model_versions ??
    {};

  return {
    hash,
    confidenceInterval,
    modelVersions,
    isLoading,
    error,
  };
}

/**
 * Prefetch XAI data for faster navigation
 */
export function usePrefetchXAI() {
  const queryClient = useQueryClient();

  return (analysisId: string) => {
    queryClient.prefetchQuery({
      queryKey: xaiKeys.explanation(analysisId),
      queryFn: () => analysisApi.getXAI(analysisId),
      staleTime: 300000,
    });
  };
}

export default useXAI;
