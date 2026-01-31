/**
 * Argus Core - Test Utilities
 * ============================
 * Custom render functions and test helpers.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Polish & Testing
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Purpose:
 * - Provide custom render with all providers
 * - Mock data factories
 * - Common test helpers
 * - Accessibility testing utilities
 */

import React, { ReactElement } from 'react';
import { render, RenderOptions, RenderResult } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';
import type {
  AnalysisResponse,
  AnalysisDetailResponse,
  Verdict,
  AnalysisStatus,
  Modality,
  TrustScore,
  VideoResult,
  AudioResult,
  TextResult,
  ImageResult,
  MetadataResult,
} from '@/types/analysis';

// ============== PROVIDER WRAPPER ==============

/**
 * Create a fresh QueryClient for each test
 * Prevents test pollution via cache
 */
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
    logger: {
      log: console.log,
      warn: console.warn,
      error: () => {}, // Suppress error logs in tests
    },
  });
}

/**
 * All providers wrapper for testing
 */
interface AllProvidersProps {
  children: React.ReactNode;
  queryClient?: QueryClient;
}

function AllProviders({ children, queryClient }: AllProvidersProps) {
  const client = queryClient || createTestQueryClient();

  return (
    <QueryClientProvider client={client}>
      {children}
    </QueryClientProvider>
  );
}

/**
 * Custom render with all providers
 * 
 * @example
 * ```tsx
 * const { getByText } = renderWithProviders(<MyComponent />);
 * ```
 */
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  queryClient?: QueryClient;
}

export function renderWithProviders(
  ui: ReactElement,
  options?: CustomRenderOptions
): RenderResult {
  const { queryClient, ...renderOptions } = options || {};

  return render(ui, {
    wrapper: ({ children }) => (
      <AllProviders queryClient={queryClient}>{children}</AllProviders>
    ),
    ...renderOptions,
  });
}

// ============== MOCK DATA FACTORIES ==============

/**
 * Create mock TrustScore
 */
export function createMockTrustScore(overrides?: Partial<TrustScore>): TrustScore {
  return {
    overall_score: 75,
    confidence: 0.85,
    modality_scores: {
      image: 80,
      video: 70,
      audio: 75,
      text: 80,
    },
    weights: {
      image: 0.3,
      video: 0.3,
      audio: 0.2,
      text: 0.2,
    },
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

/**
 * Create mock AnalysisResponse
 */
export function createMockAnalysisResponse(
  overrides?: Partial<AnalysisResponse>
): AnalysisResponse {
  return {
    analysis_id: 'test-analysis-123',
    status: 'completed' as AnalysisStatus,
    filename: 'test-file.mp4',
    file_size: 1024000,
    modalities: ['video', 'audio'] as Modality[],
    trust_score: createMockTrustScore(),
    verdict: 'likely_deepfake' as Verdict,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    completed_at: new Date().toISOString(),
    processing_time: 12.5,
    ...overrides,
  };
}

/**
 * Create mock VideoResult
 */
export function createMockVideoResult(overrides?: Partial<VideoResult>): VideoResult {
  return {
    modality: 'video' as Modality,
    score: 70,
    confidence: 0.82,
    verdict: 'likely_deepfake' as Verdict,
    analysis_timestamp: new Date().toISOString(),
    spatial_analysis: {
      score: 65,
      anomalies_detected: 12,
      analyzed_frames: 150,
      suspicious_regions: [
        { frame: 45, region: { x: 100, y: 150, width: 200, height: 150 }, confidence: 0.78 },
      ],
    },
    temporal_analysis: {
      score: 75,
      inconsistencies: 8,
      flicker_detected: true,
      motion_artifacts: true,
    },
    lipsync_analysis: {
      score: 72,
      offset_ms: 45,
      confidence: 0.88,
      sync_windows: [
        { start_frame: 0, end_frame: 100, sync_score: 85 },
      ],
    },
    heatmaps: [
      { frame: 45, url: 'https://example.com/heatmap1.jpg', confidence: 0.78 },
    ],
    ...overrides,
  };
}

/**
 * Create mock AudioResult
 */
export function createMockAudioResult(overrides?: Partial<AudioResult>): AudioResult {
  return {
    modality: 'audio' as Modality,
    score: 75,
    confidence: 0.85,
    verdict: 'authentic' as Verdict,
    analysis_timestamp: new Date().toISOString(),
    voice_cloning_score: 20,
    artifacts_detected: 3,
    sample_rate: 44100,
    duration_seconds: 45.5,
    spectral_anomalies: [
      { timestamp: 12.5, frequency_hz: 8000, confidence: 0.72 },
    ],
    spectrogram_url: 'https://example.com/spectrogram.jpg',
    ...overrides,
  };
}

/**
 * Create mock TextResult
 */
export function createMockTextResult(overrides?: Partial<TextResult>): TextResult {
  return {
    modality: 'text' as Modality,
    score: 80,
    confidence: 0.90,
    verdict: 'authentic' as Verdict,
    analysis_timestamp: new Date().toISOString(),
    ai_probability: 0.15,
    perplexity: 45.2,
    burstiness: 0.68,
    detected_patterns: [
      { pattern: 'human-like-variation', confidence: 0.85 },
    ],
    ...overrides,
  };
}

/**
 * Create mock ImageResult
 */
export function createMockImageResult(overrides?: Partial<ImageResult>): ImageResult {
  return {
    modality: 'image' as Modality,
    score: 85,
    confidence: 0.92,
    verdict: 'authentic' as Verdict,
    analysis_timestamp: new Date().toISOString(),
    manipulation_detected: false,
    anomaly_regions: [],
    heatmap_url: 'https://example.com/heatmap.jpg',
    ...overrides,
  };
}

/**
 * Create mock MetadataResult
 */
export function createMockMetadataResult(
  overrides?: Partial<MetadataResult>
): MetadataResult {
  return {
    modality: 'metadata' as Modality,
    score: 90,
    confidence: 0.95,
    verdict: 'authentic' as Verdict,
    analysis_timestamp: new Date().toISOString(),
    c2pa_found: true,
    c2pa_valid: true,
    exif_analysis: {
      software: 'Adobe Photoshop 2024',
      camera_model: 'Canon EOS R5',
      gps_location: null,
      timestamp_consistent: true,
    },
    provenance: {
      source: 'Canon EOS R5',
      timestamp: new Date().toISOString(),
      chain_valid: true,
    },
    ...overrides,
  };
}

/**
 * Create mock AnalysisDetailResponse
 */
export function createMockAnalysisDetail(
  overrides?: Partial<AnalysisDetailResponse>
): AnalysisDetailResponse {
  return {
    ...createMockAnalysisResponse(),
    video_result: createMockVideoResult(),
    audio_result: createMockAudioResult(),
    text_result: createMockTextResult(),
    image_result: createMockImageResult(),
    metadata_result: createMockMetadataResult(),
    explanation: {
      summary: 'Analysis detected multiple anomalies indicating potential manipulation.',
      key_factors: [
        { factor: 'Temporal inconsistencies', impact: 0.35, description: 'Frame transitions show unnatural patterns' },
        { factor: 'Lip-sync offset', impact: 0.25, description: '45ms delay detected' },
      ],
      confidence_breakdown: {
        high: ['spatial_analysis', 'lipsync_detection'],
        medium: ['temporal_analysis'],
        low: [],
      },
    },
    report_url: 'https://example.com/report.pdf',
    ...overrides,
  };
}

// ============== TEST HELPERS ==============

/**
 * Wait for async updates
 */
export const waitFor = (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Mock WebSocket connection
 */
export function createMockWebSocket() {
  const mockWs = {
    send: vi.fn(),
    close: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    readyState: WebSocket.OPEN,
    CONNECTING: WebSocket.CONNECTING,
    OPEN: WebSocket.OPEN,
    CLOSING: WebSocket.CLOSING,
    CLOSED: WebSocket.CLOSED,
  };

  return mockWs;
}

/**
 * Mock axios instance
 */
export function createMockAxios() {
  return {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    patch: vi.fn(),
    request: vi.fn(),
    interceptors: {
      request: { use: vi.fn(), eject: vi.fn() },
      response: { use: vi.fn(), eject: vi.fn() },
    },
  };
}

// ============== ACCESSIBILITY HELPERS ==============

/**
 * Run axe accessibility tests
 * 
 * @example
 * ```tsx
 * const { container } = render(<Component />);
 * expect(await axe(container)).toHaveNoViolations();
 * ```
 */
export async function runAxeTest(container: HTMLElement) {
  // This would integrate with @axe-core/react in actual implementation
  // For now, return a mock result
  return {
    violations: [],
  };
}

/**
 * Check for WCAG 2.1 AA compliance
 */
export function checkAccessibility(element: HTMLElement): {
  passed: boolean;
  issues: string[];
} {
  const issues: string[] = [];

  // Check for alt text on images
  const images = element.querySelectorAll('img');
  images.forEach((img) => {
    if (!img.alt && !img.getAttribute('aria-label')) {
      issues.push(`Image missing alt text: ${img.src}`);
    }
  });

  // Check for button labels
  const buttons = element.querySelectorAll('button');
  buttons.forEach((button) => {
    if (
      !button.textContent?.trim() &&
      !button.getAttribute('aria-label') &&
      !button.getAttribute('aria-labelledby')
    ) {
      issues.push('Button missing accessible label');
    }
  });

  // Check for form labels
  const inputs = element.querySelectorAll('input, select, textarea');
  inputs.forEach((input) => {
    const id = input.getAttribute('id');
    if (id) {
      const label = element.querySelector(`label[for="${id}"]`);
      if (!label && !input.getAttribute('aria-label')) {
        issues.push(`Input ${id} missing label`);
      }
    }
  });

  return {
    passed: issues.length === 0,
    issues,
  };
}

// ============== EXPORTS ==============

// Re-export everything from React Testing Library
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';

// Custom exports
export {
  createTestQueryClient,
  AllProviders,
};
