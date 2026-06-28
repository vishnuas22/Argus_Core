/**
 * Argus Core - Analysis API Service
 * ==================================
 * API functions for analysis endpoints.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - services/analysisApi.ts
 * 
 * Role: Analysis API endpoint functions with typed responses.
 * 
 * Integration:
 * - Imports: services/api.ts, types/analysis.ts
 * - Used by: hooks/useAnalysis.ts, hooks/useAnalysisDetail.ts
 * - Backend: api/router.py endpoints
 */

import api from './api';
import type {
  AnalysisResponse,
  AnalysisDetailResponse,
  AnalysisOptions,
  HeatmapResponse,
  ListParams,
  XAIHeatmapResponse,
  XAIExplanation,
  FeatureImportance,
  VisualEvidence,
  ScientificReference,
  AudioArtifactRegion,
  TokenAttribution,
  ManipulationRegion,
  EvidencePackage,
} from '@/types/analysis';

/**
 * XAI response from backend
 */
export interface XAIResponse {
  analysis_id: string;
  image_xai?: {
    explanation: XAIExplanation;
    manipulation_regions: ManipulationRegion[];
    heatmap_urls: string[];
    overlay_url: string | null;
  };
  video_xai?: {
    explanation: XAIExplanation;
    manipulation_regions: ManipulationRegion[];
    heatmap_urls: string[];
    temporal_heatmap_url: string | null;
  };
  audio_xai?: {
    explanation: XAIExplanation;
    artifact_regions: AudioArtifactRegion[];
    spectrogram_overlay_url: string | null;
  };
  text_xai?: {
    explanation: XAIExplanation;
    token_attributions: TokenAttribution[];
  };
  evidence_package?: EvidencePackage;
}

/**
 * Analysis API service object
 */
export const analysisApi = {
  /**
   * Submit media file for analysis
   * POST /api/v1/analyze
   * 
   * @param file - Media file to analyze
   * @param options - Analysis options
   * @param onUploadProgress - Upload progress callback (0-100)
   * @returns Analysis response with analysis_id
   */
  submitAnalysis: async (
    file: File,
    options: AnalysisOptions,
    onUploadProgress?: (progress: number) => void
  ): Promise<AnalysisResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('generate_report', String(options.generateReport));
    formData.append('generate_heatmaps', String(options.generateHeatmaps));
    formData.append('defense_level', options.defenseLevel);
    
    if (options.modalities && options.modalities.length > 0) {
      formData.append('modalities', options.modalities.join(','));
    }

    const response = await api.post<AnalysisResponse>('/api/v1/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000, // 2 minute timeout for large uploads
      onUploadProgress: (event) => {
        if (event.total && onUploadProgress) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onUploadProgress(progress);
        }
      },
    });

    return transformAnalysisResponse(response.data);
  },

  /**
   * Submit text for AI-generated content analysis
   * POST /api/v1/analyze/text
   * 
   * @param text - Text content to analyze
   * @param generateReport - Generate PDF report
   * @returns Analysis response with analysis_id
   */
  submitTextAnalysis: async (
    text: string,
    generateReport: boolean = false
  ): Promise<AnalysisResponse> => {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('generate_report', String(generateReport));

    const response = await api.post<AnalysisResponse>('/api/v1/analyze/text', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    return transformAnalysisResponse(response.data);
  },

  /**
   * Get analysis status and basic results
   * GET /api/v1/analyze/{id}
   * 
   * @param id - Analysis ID
   * @returns Analysis response
   */
  getAnalysis: async (id: string): Promise<AnalysisResponse> => {
    const response = await api.get<AnalysisResponse>(`/api/v1/analyze/${id}`);
    return transformAnalysisResponse(response.data);
  },

  /**
   * Get detailed analysis results
   * GET /api/v1/analyze/{id}/detail
   * 
   * @param id - Analysis ID
   * @returns Detailed analysis response with all modality results
   */
  getAnalysisDetail: async (id: string): Promise<AnalysisDetailResponse> => {
    const response = await api.get<AnalysisDetailResponse>(`/api/v1/analyze/${id}/detail`);
    return transformDetailResponse(response.data);
  },

  /**
   * List analyses with optional filtering
   * GET /api/v1/analyze
   * 
   * @param params - Query parameters (status, limit, offset)
   * @returns Array of analysis responses
   */
  listAnalyses: async (params?: ListParams): Promise<AnalysisResponse[]> => {
    const response = await api.get<AnalysisResponse[]>('/api/v1/analyze', { params });
    return response.data.map(transformAnalysisResponse);
  },

  /**
   * Delete analysis and associated data
   * DELETE /api/v1/analyze/{id}
   * 
   * @param id - Analysis ID
   */
  deleteAnalysis: async (id: string): Promise<void> => {
    await api.delete(`/api/v1/analyze/${id}`);
  },

  /**
   * Get or generate PDF report
   * GET /api/v1/analyze/{id}/report
   * 
   * @param id - Analysis ID
   * @param regenerate - Force regenerate report
   * @returns Report URL object
   */
  getReport: async (id: string, regenerate: boolean = false): Promise<{ reportUrl: string }> => {
    const response = await api.get<{ report_url: string }>(
      `/api/v1/analyze/${id}/report`,
      { params: { regenerate } }
    );
    return { reportUrl: response.data.report_url };
  },

  /**
   * Get GradCAM heatmap URLs
   * GET /api/v1/analyze/{id}/heatmaps
   * 
   * @param id - Analysis ID
   * @returns Heatmap response with URLs
   */
  getHeatmaps: async (id: string): Promise<HeatmapResponse> => {
    const response = await api.get<HeatmapResponse>(`/api/v1/analyze/${id}/heatmaps`);
    return response.data;
  },

  /**
   * Get XAI explanation data
   * GET /api/v1/analyze/{id}/xai
   * 
   * @param id - Analysis ID
   * @returns XAI response with explanations for all modalities
   */
  getXAI: async (id: string): Promise<XAIResponse> => {
    const response = await api.get<XAIResponse>(`/api/v1/analyze/${id}/xai`);
    return response.data;
  },

  /**
   * Get XAI heatmaps with overlay URLs
   * GET /api/v1/analyze/{id}/xai/heatmaps
   * 
   * @param id - Analysis ID
   * @returns XAI heatmap response with overlay URLs
   */
  getXAIHeatmaps: async (id: string): Promise<XAIHeatmapResponse> => {
    const response = await api.get<XAIHeatmapResponse>(`/api/v1/analyze/${id}/xai/heatmaps`);
    return response.data;
  },
};

/**
 * Transform backend snake_case response to frontend camelCase
 */
function transformAnalysisResponse(data: AnalysisResponse | Record<string, unknown>): AnalysisResponse {
  return {
    analysis_id: data.analysis_id as string,
    status: data.status as AnalysisResponse['status'],
    trust_score: data.trust_score as AnalysisResponse['trust_score'],
    verdict: data.verdict as AnalysisResponse['verdict'],
    explanation: data.explanation as AnalysisResponse['explanation'],
    report_url: data.report_url as string | undefined,
    created_at: data.created_at as string,
    completed_at: data.completed_at as string | undefined,
  };
}

/**
 * Transform detailed response
 */
function transformDetailResponse(data: AnalysisDetailResponse | Record<string, unknown>): AnalysisDetailResponse {
  return {
    ...transformAnalysisResponse(data),
    input: data.input as AnalysisDetailResponse['input'],
    video_result: data.video_result as AnalysisDetailResponse['video_result'],
    audio_result: data.audio_result as AnalysisDetailResponse['audio_result'],
    image_result: data.image_result as AnalysisDetailResponse['image_result'],
    metadata_result: data.metadata_result as AnalysisDetailResponse['metadata_result'],
    processing_time_seconds: data.processing_time_seconds as number | undefined,
    // XAI Enhancement Fields
    evidence_package: data.evidence_package as AnalysisDetailResponse['evidence_package'],
    feature_importance: (data.feature_importance || []) as AnalysisDetailResponse['feature_importance'],
    scientific_references: (data.scientific_references || []) as AnalysisDetailResponse['scientific_references'],
  };
}

export default analysisApi;
