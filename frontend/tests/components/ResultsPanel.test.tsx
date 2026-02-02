/**
 * Argus Core - ResultsPanel Component Tests
 * ==========================================
 * Comprehensive tests for ResultsPanel component.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Component Tests (P0)
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Test Coverage:
 * - Full, compact, and card variants
 * - Loading states with skeleton
 * - Error states with retry functionality
 * - Failed analysis display
 * - Complete results rendering
 * - Score gauge integration
 * - Verdict badge display
 * - Explanation panel integration
 * - Score breakdown integration
 * - Action buttons (download, share)
 * - Accessibility compliance (WCAG 2.1 AA)
 * 
 * Target: >80% coverage
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, checkAccessibility } from '../utils/test-utils';
import { ResultsPanel, ResultsPanelSkeleton } from '@/components/results/ResultsPanel';
import type { AnalysisDetailResponse, TrustScore, Verdict } from '@/types/analysis';

// ============== MOCKS ==============

// Mock hooks
const mockUseAnalysisDetail = vi.hoisted(() => ({
  analysis: null,
  detail: null,
  isLoading: false,
  isDetailLoading: false,
  error: null,
  refetch: vi.fn(),
  isComplete: false,
  isFailed: false,
}));

const mockUseReport = vi.hoisted(() => ({
  data: null,
  isLoading: false,
  error: null,
}));

vi.mock('@/hooks/useAnalysisDetail', () => ({
  useAnalysisDetail: vi.fn(() => mockUseAnalysisDetail),
  useReport: vi.fn(() => mockUseReport),
}));

// Mock child components
vi.mock('@/components/results/TrustScoreGauge', () => ({
  TrustScoreGauge: ({ score, verdict }: any) => (
    <div data-testid="trust-score-gauge">
      Score: {score} | Verdict: {verdict}
    </div>
  ),
  TrustScoreGaugeSkeleton: () => <div data-testid="gauge-skeleton" />,
}));

vi.mock('@/components/results/VerdictBadge', () => ({
  VerdictBadge: ({ verdict }: any) => (
    <div data-testid="verdict-badge">{verdict}</div>
  ),
  VerdictBadgeHero: ({ verdict }: any) => (
    <div data-testid="verdict-badge-hero">{verdict}</div>
  ),
  VerdictBadgeSkeleton: () => <div data-testid="verdict-skeleton" />,
  getVerdictFromScore: (score: number) => 
    score >= 80 ? 'authentic' : score >= 40 ? 'uncertain' : 'fake',
}));

vi.mock('@/components/results/ScoreBreakdown', () => ({
  ScoreBreakdown: ({ breakdown }: any) => (
    <div data-testid="score-breakdown">
      Breakdown: {JSON.stringify(breakdown)}
    </div>
  ),
  ScoreBreakdownSkeleton: () => <div data-testid="breakdown-skeleton" />,
}));

vi.mock('@/components/results/ExplanationPanel', () => ({
  ExplanationPanel: ({ explanation }: any) => (
    <div data-testid="explanation-panel">
      {explanation?.summary}
    </div>
  ),
  ExplanationPanelSkeleton: () => <div data-testid="explanation-skeleton" />,
}));

// Mock Next.js Link
vi.mock('next/link', () => ({
  default: ({ children, href }: any) => <a href={href}>{children}</a>,
}));

// ============== TEST DATA ==============

const MOCK_ANALYSIS_ID = 'test-analysis-123';

const MOCK_TRUST_SCORE: TrustScore = {
  overall: 85.5,
  confidence: 0.92,
  breakdown: {
    video_spatial: 0.88,
    video_temporal: 0.82,
    audio: 0.90,
    text: 0.75,
    metadata: 0.95,
    weights: {
      video_spatial: 0.30,
      video_temporal: 0.25,
      audio: 0.20,
      text: 0.10,
      metadata: 0.15,
    },
  },
};

const MOCK_DETAIL_RESPONSE: AnalysisDetailResponse = {
  analysis_id: MOCK_ANALYSIS_ID,
  status: 'completed',
  trust_score: MOCK_TRUST_SCORE,
  verdict: 'authentic',
  explanation: {
    summary: 'This media appears to be authentic based on multi-modal analysis.',
    confidence_statement: 'High confidence in authenticity assessment.',
    key_findings: [
      'No deepfake artifacts detected in video frames',
      'Audio analysis shows natural speech patterns',
      'Metadata integrity verified',
    ],
    recommendations: [
      'Consider additional verification for high-stakes decisions',
    ],
    technical_details: 'Full technical analysis available in report.',
  },
  report_url: 'https://example.com/report.pdf',
  created_at: '2026-01-15T10:00:00Z',
  completed_at: '2026-01-15T10:00:45Z',
  input: {
    file_name: 'test-video.mp4',
    file_size: 10485760,
    mime_type: 'video/mp4',
    modality: 'video',
  },
  video_result: {
    spatial_score: 0.88,
    temporal_score: 0.82,
    lipsync_score: 0.90,
    heatmap_urls: ['https://example.com/heatmap1.png'],
    frame_count: 300,
  },
  audio_result: {
    score: 0.90,
    spectrogram_url: 'https://example.com/spectrogram.png',
    duration_seconds: 10.5,
  },
  text_result: undefined,
  metadata_result: {
    score: 0.95,
    c2pa_validated: true,
    exif_integrity: true,
  },
  processing_time_seconds: 45.2,
};

// ============== TEST UTILITIES ==============

function setupMockAnalysisDetail(overrides: Partial<typeof mockUseAnalysisDetail> = {}) {
  Object.assign(mockUseAnalysisDetail, {
    analysis: null,
    detail: null,
    isLoading: false,
    isDetailLoading: false,
    error: null,
    refetch: vi.fn(),
    isComplete: false,
    isFailed: false,
    ...overrides,
  });
}

function setupMockReport(overrides: Partial<typeof mockUseReport> = {}) {
  Object.assign(mockUseReport, {
    data: null,
    isLoading: false,
    error: null,
    ...overrides,
  });
}

// ============== CORE FUNCTIONALITY TESTS ==============

describe('ResultsPanel Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMockAnalysisDetail();
    setupMockReport();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ============== LOADING STATE TESTS ==============

  describe('Loading States', () => {
    it('shows skeleton when loading', () => {
      setupMockAnalysisDetail({ isLoading: true });
      
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel-skeleton')).toBeInTheDocument();
    });

    it('shows skeleton when detail is loading', () => {
      setupMockAnalysisDetail({ 
        isLoading: false, 
        isDetailLoading: true 
      });
      
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel-skeleton')).toBeInTheDocument();
    });

    it('renders skeleton with correct variant', () => {
      setupMockAnalysisDetail({ isLoading: true });
      
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          variant="compact"
        />
      );
      
      // Skeleton should be rendered (exact structure depends on variant)
      expect(screen.queryByTestId('results-panel-skeleton')).toBeInTheDocument();
    });
  });

  // ============== ERROR STATE TESTS ==============

  describe('Error States', () => {
    it('shows error message when fetch fails', () => {
      setupMockAnalysisDetail({ 
        error: new Error('Failed to fetch results'),
        analysis: null,
        detail: null,
      });
      
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel-error')).toBeInTheDocument();
      expect(screen.getByText(/Failed to Load Results/i)).toBeInTheDocument();
      expect(screen.getByText(/Failed to fetch results/i)).toBeInTheDocument();
    });

    it('shows retry button on error', () => {
      const mockRefetch = vi.fn();
      setupMockAnalysisDetail({ 
        error: new Error('Network error'),
        analysis: null,
        detail: null,
        refetch: mockRefetch,
      });
      
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      const retryButton = screen.getByRole('button', { name: /Retry/i });
      expect(retryButton).toBeInTheDocument();
      
      fireEvent.click(retryButton);
      expect(mockRefetch).toHaveBeenCalledTimes(1);
    });
  });

  // ============== FAILED ANALYSIS TESTS ==============

  describe('Failed Analysis State', () => {
    it('shows failed message when analysis failed', () => {
      setupMockAnalysisDetail({ 
        isFailed: true,
        analysis: {
          ...MOCK_DETAIL_RESPONSE,
          status: 'failed',
          explanation: {
            summary: 'Analysis failed due to invalid input',
          },
        },
      });
      
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel-failed')).toBeInTheDocument();
      expect(screen.getByText(/Analysis Failed/i)).toBeInTheDocument();
      expect(screen.getByText(/Analysis failed due to invalid input/i)).toBeInTheDocument();
    });

    it('shows "Try Another File" button on failed analysis', () => {
      setupMockAnalysisDetail({ 
        isFailed: true,
        analysis: {
          ...MOCK_DETAIL_RESPONSE,
          status: 'failed',
        },
      });
      
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      const tryButton = screen.getByRole('button', { name: /Try Another File/i });
      expect(tryButton).toBeInTheDocument();
    });
  });

  // ============== PENDING STATE TESTS ==============

  describe('Pending/Incomplete State', () => {
    it('shows pending message when not complete', () => {
      setupMockAnalysisDetail({ 
        isComplete: false,
        analysis: null,
      });
      
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel-pending')).toBeInTheDocument();
      expect(screen.getByText(/Analysis is still in progress/i)).toBeInTheDocument();
    });
  });

  // ============== COMPLETE STATE TESTS ==============

  describe('Complete State - Full Results', () => {
    beforeEach(() => {
      setupMockAnalysisDetail({ 
        isComplete: true,
        analysis: MOCK_DETAIL_RESPONSE,
        detail: MOCK_DETAIL_RESPONSE,
      });
      setupMockReport({
        data: { reportUrl: 'https://example.com/report.pdf' },
      });
    });

    it('renders results panel with data', () => {
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel')).toBeInTheDocument();
    });

    it('displays trust score gauge', () => {
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('trust-score-gauge')).toBeInTheDocument();
      expect(screen.getByText(/Score: 85.5/)).toBeInTheDocument();
    });

    it('displays verdict badge', () => {
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('verdict-badge')).toBeInTheDocument();
      expect(screen.getByText('authentic')).toBeInTheDocument();
    });

    it('displays explanation when showExplanation is true', () => {
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          showExplanation
        />
      );
      
      expect(screen.getByText(/This media appears to be authentic/i)).toBeInTheDocument();
    });

    it('displays score breakdown when showBreakdown is true', () => {
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          showBreakdown
        />
      );
      
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
    });

    it('hides explanation when showExplanation is false', () => {
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          showExplanation={false}
        />
      );
      
      expect(screen.queryByTestId('explanation-panel')).not.toBeInTheDocument();
    });

    it('hides breakdown when showBreakdown is false', () => {
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          showBreakdown={false}
        />
      );
      
      expect(screen.queryByTestId('score-breakdown')).not.toBeInTheDocument();
    });
  });

  // ============== ACTION BUTTONS TESTS ==============

  describe('Action Buttons', () => {
    beforeEach(() => {
      setupMockAnalysisDetail({ 
        isComplete: true,
        analysis: MOCK_DETAIL_RESPONSE,
        detail: MOCK_DETAIL_RESPONSE,
      });
      setupMockReport({
        data: { reportUrl: 'https://example.com/report.pdf' },
      });
    });

    it('shows action buttons when showActions is true', () => {
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          showActions
        />
      );
      
      expect(screen.getByTestId('results-actions')).toBeInTheDocument();
    });

    it('hides action buttons when showActions is false', () => {
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          showActions={false}
        />
      );
      
      expect(screen.queryByTestId('results-actions')).not.toBeInTheDocument();
    });

    it('calls onDownloadReport when download button is clicked', async () => {
      const mockDownload = vi.fn();
      
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          showActions
          onDownloadReport={mockDownload}
        />
      );
      
      const downloadButton = screen.getByRole('button', { name: /Download Report/i });
      await userEvent.click(downloadButton);
      
      expect(mockDownload).toHaveBeenCalledTimes(1);
    });

    it('calls onShare when share button is clicked', async () => {
      const mockShare = vi.fn();
      
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          showActions
          onShare={mockShare}
        />
      );
      
      const shareButton = screen.getByRole('button', { name: /Share/i });
      await userEvent.click(shareButton);
      
      expect(mockShare).toHaveBeenCalledTimes(1);
    });

    it('shows "New Analysis" button', () => {
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          showActions
        />
      );
      
      const newAnalysisButton = screen.getByRole('button', { name: /New Analysis/i });
      expect(newAnalysisButton).toBeInTheDocument();
    });
  });

  // ============== VARIANT TESTS ==============

  describe('Layout Variants', () => {
    beforeEach(() => {
      setupMockAnalysisDetail({ 
        isComplete: true,
        analysis: MOCK_DETAIL_RESPONSE,
        detail: MOCK_DETAIL_RESPONSE,
      });
    });

    it('renders full variant by default', () => {
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel')).toBeInTheDocument();
      // Full variant should have score and verdict sections
      expect(screen.getByTestId('results-score')).toBeInTheDocument();
    });

    it('renders compact variant correctly', () => {
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          variant="compact"
        />
      );
      
      expect(screen.getByTestId('results-panel')).toBeInTheDocument();
    });

    it('renders card variant correctly', () => {
      renderWithProviders(
        <ResultsPanel 
          analysisId={MOCK_ANALYSIS_ID}
          variant="card"
        />
      );
      
      expect(screen.getByTestId('results-panel')).toBeInTheDocument();
      // Card variant should have a "Details" link
      expect(screen.getByRole('button', { name: /Details/i })).toBeInTheDocument();
    });
  });

  // ============== ACCESSIBILITY TESTS ==============

  describe('Accessibility', () => {
    beforeEach(() => {
      setupMockAnalysisDetail({ 
        isComplete: true,
        analysis: MOCK_DETAIL_RESPONSE,
        detail: MOCK_DETAIL_RESPONSE,
      });
    });

    it('meets WCAG 2.1 AA standards', async () => {
      const { container } = renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      await checkAccessibility(container);
    });

    it('has proper data-testid attributes', () => {
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel')).toBeInTheDocument();
      expect(screen.getByTestId('results-score')).toBeInTheDocument();
      expect(screen.getByTestId('results-verdict')).toBeInTheDocument();
    });

    it('has keyboard-accessible action buttons', async () => {
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} showActions />
      );
      
      const downloadButton = screen.getByRole('button', { name: /Download Report/i });
      const shareButton = screen.getByRole('button', { name: /Share/i });
      
      // Buttons should be focusable
      downloadButton.focus();
      expect(document.activeElement).toBe(downloadButton);
      
      shareButton.focus();
      expect(document.activeElement).toBe(shareButton);
    });
  });

  // ============== RESPONSIVE BEHAVIOR TESTS ==============

  describe('Responsive Behavior', () => {
    beforeEach(() => {
      setupMockAnalysisDetail({ 
        isComplete: true,
        analysis: MOCK_DETAIL_RESPONSE,
        detail: MOCK_DETAIL_RESPONSE,
      });
    });

    it('adapts to mobile viewports', () => {
      global.innerWidth = 375;
      global.dispatchEvent(new Event('resize'));
      
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel')).toBeInTheDocument();
    });

    it('adapts to tablet viewports', () => {
      global.innerWidth = 768;
      global.dispatchEvent(new Event('resize'));
      
      renderWithProviders(
        <ResultsPanel analysisId={MOCK_ANALYSIS_ID} />
      );
      
      expect(screen.getByTestId('results-panel')).toBeInTheDocument();
    });
  });
});

// ============== SKELETON TESTS ==============

describe('ResultsPanelSkeleton', () => {
  it('renders skeleton for full variant', () => {
    renderWithProviders(
      <ResultsPanelSkeleton variant="full" />
    );
    
    expect(screen.getByTestId('results-panel-skeleton')).toBeInTheDocument();
  });

  it('renders skeleton for compact variant', () => {
    renderWithProviders(
      <ResultsPanelSkeleton variant="compact" />
    );
    
    expect(screen.getByTestId('gauge-skeleton')).toBeInTheDocument();
    expect(screen.getByTestId('verdict-skeleton')).toBeInTheDocument();
  });

  it('renders skeleton for card variant', () => {
    const { container } = renderWithProviders(
      <ResultsPanelSkeleton variant="card" />
    );
    
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('shows breakdown skeleton when showBreakdown is true', () => {
    renderWithProviders(
      <ResultsPanelSkeleton 
        variant="full"
        showBreakdown
      />
    );
    
    expect(screen.getByTestId('breakdown-skeleton')).toBeInTheDocument();
  });

  it('shows explanation skeleton when showExplanation is true', () => {
    renderWithProviders(
      <ResultsPanelSkeleton 
        variant="full"
        showExplanation
      />
    );
    
    expect(screen.getByTestId('explanation-skeleton')).toBeInTheDocument();
  });
});
