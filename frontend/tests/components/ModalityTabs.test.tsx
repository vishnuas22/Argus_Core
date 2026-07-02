/**
 * Argus Core - ModalityTabs Component Tests
 * ==========================================
 * Comprehensive tests for ModalityTabs component.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Component Tests (P0)
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Test Coverage:
 * - Tab rendering based on available modalities
 * - Tab navigation and switching
 * - Score badge display
 * - Lazy loading of panel components
 * - Empty state handling
 * - Loading/skeleton states
 * - Keyboard navigation
 * - Accessibility compliance (WCAG 2.1 AA)
 * - Different variants (default, compact, card)
 * - Callback handling (onTabChange)
 * - Default tab selection
 * 
 * Target: >80% coverage
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, checkAccessibility, createMockVideoResult, createMockAudioResult, createMockMetadataResult } from '../utils/test-utils';
import { ModalityTabs, ModalityTabsSkeleton } from '@/components/modality/ModalityTabs';

// ============== MOCKS ==============

// Mock lazy-loaded components
vi.mock('@/components/modality/VideoAnalysisPanel', () => ({
  default: ({ result, analysisId }: any) => (
    <div data-testid="video-panel-mock">
      Video Panel - Analysis ID: {analysisId}, Score: {result.aggregated_score}
    </div>
  ),
}));

vi.mock('@/components/modality/AudioAnalysisPanel', () => ({
  default: ({ result, analysisId }: any) => (
    <div data-testid="audio-panel-mock">
      Audio Panel - Analysis ID: {analysisId}, Score: {result.score}
    </div>
  ),
}));

vi.mock('@/components/modality/MetadataPanel', () => ({
  default: ({ result, analysisId }: any) => (
    <div data-testid="metadata-panel-mock">
      Metadata Panel - Analysis ID: {analysisId}, Score: {result.score}
    </div>
  ),
}));

// ============== TEST DATA ==============

const mockAnalysisId = 'test-analysis-123';

const mockVideoResult = createMockVideoResult({
  aggregated_score: 0.75,
  score: 75,
});

const mockAudioResult = createMockAudioResult({
  score: 0.82,
});

const mockMetadataResult = createMockMetadataResult({
  score: 0.95,
});

// ============== TEST SUITE ==============

describe('ModalityTabs', () => {
  // ============== BASIC RENDERING ==============

  describe('Rendering', () => {
    it('should render with minimum required props', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
        />
      );
      
      await waitFor(() => {
        const tabs = screen.getByTestId('modality-tabs');
        expect(tabs).toBeInTheDocument();
      });
    });

    it('should render only video tab when only video result provided', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
        />
      );
      
      await waitFor(() => {
        expect(screen.getByTestId('modality-tab-video')).toBeInTheDocument();
        expect(screen.queryByTestId('modality-tab-audio')).not.toBeInTheDocument();
        expect(screen.queryByTestId('modality-tab-text')).not.toBeInTheDocument();
        expect(screen.queryByTestId('modality-tab-metadata')).not.toBeInTheDocument();
      });
    });

    it('should render all tabs when all results provided', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
          metadataResult={mockMetadataResult}
        />
      );
      
      await waitFor(() => {
        expect(screen.getByTestId('modality-tab-video')).toBeInTheDocument();
        expect(screen.getByTestId('modality-tab-audio')).toBeInTheDocument();
        expect(screen.getByTestId('modality-tab-metadata')).toBeInTheDocument();
      });
    });

    it('should display tab icons', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        const videoTab = screen.getByTestId('modality-tab-video');
        const audioTab = screen.getByTestId('modality-tab-audio');
        
        expect(videoTab.querySelector('svg')).toBeInTheDocument();
        expect(audioTab.querySelector('svg')).toBeInTheDocument();
      });
    });

    it('should display tab labels on desktop', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        expect(screen.getByText('Video')).toBeInTheDocument();
        expect(screen.getByText('Audio')).toBeInTheDocument();
      });
    });
  });

  // ============== TAB NAVIGATION ==============

  describe('Tab Navigation', () => {
    it('should show first tab as active by default', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        const videoTab = screen.getByTestId('modality-tab-video');
        expect(videoTab).toHaveAttribute('data-state', 'active');
      });
    });

    it('should use defaultTab prop if provided', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
          defaultTab="audio"
        />
      );
      
      await waitFor(() => {
        const audioTab = screen.getByTestId('modality-tab-audio');
        expect(audioTab).toHaveAttribute('data-state', 'active');
      });
    });

    it('should switch tabs on click', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        expect(screen.getByTestId('modality-tab-video')).toHaveAttribute('data-state', 'active');
      });
      
      const audioTab = screen.getByTestId('modality-tab-audio');
      await user.click(audioTab);
      
      await waitFor(() => {
        expect(audioTab).toHaveAttribute('data-state', 'active');
        expect(screen.getByTestId('modality-tab-video')).toHaveAttribute('data-state', 'inactive');
      });
    });

    it('should call onTabChange callback when tab changes', async () => {
      const onTabChange = vi.fn();
      const user = userEvent.setup();
      
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
          onTabChange={onTabChange}
        />
      );
      
      const audioTab = screen.getByTestId('modality-tab-audio');
      await user.click(audioTab);
      
      await waitFor(() => {
        expect(onTabChange).toHaveBeenCalledWith('audio');
      });
    });

    it('should support keyboard navigation', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      // Focus first tab
      const videoTab = screen.getByTestId('modality-tab-video');
      videoTab.focus();
      
      // Navigate with arrow right
      await user.keyboard('{ArrowRight}');
      
      await waitFor(() => {
        const audioTab = screen.getByTestId('modality-tab-audio');
        expect(audioTab).toHaveFocus();
      });
    });
  });

  // ============== SCORE BADGES ==============

  describe('Score Badges', () => {
    it('should display score badges for each tab', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        const videoTab = screen.getByTestId('modality-tab-video');
        const audioTab = screen.getByTestId('modality-tab-audio');
        
        expect(within(videoTab).getByText('75')).toBeInTheDocument();
        expect(within(audioTab).getByText('82')).toBeInTheDocument();
      });
    });

    it('should not display badges in compact variant', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
          variant="compact"
        />
      );
      
      await waitFor(() => {
        const videoTab = screen.getByTestId('modality-tab-video');
        expect(within(videoTab).queryByText('75')).not.toBeInTheDocument();
      });
    });

    it('should apply correct color to high score badge', async () => {
      const highScoreVideo = createMockVideoResult({ aggregated_score: 0.90 });
      
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={highScoreVideo}
        />
      );
      
      await waitFor(() => {
        const badge = screen.getByText('90');
        expect(badge).toHaveClass('bg-green-500/10');
      });
    });

    it('should apply correct color to low score badge', async () => {
      const lowScoreVideo = createMockVideoResult({ aggregated_score: 0.15 });
      
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={lowScoreVideo}
        />
      );
      
      await waitFor(() => {
        const badge = screen.getByText('15');
        expect(badge).toHaveClass('bg-red-500/10');
      });
    });
  });

  // ============== PANEL CONTENT ==============

  describe('Panel Content', () => {
    it('should display video panel content when video tab active', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        expect(screen.getByTestId('video-panel-mock')).toBeInTheDocument();
        expect(screen.getByText(/Video Panel - Analysis ID: test-analysis-123/)).toBeInTheDocument();
      });
    });

    it('should display audio panel when audio tab clicked', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      const audioTab = screen.getByTestId('modality-tab-audio');
      await user.click(audioTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('audio-panel-mock')).toBeInTheDocument();
      });
    });

    it('should lazy load panel components', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      // Initially only video panel should be visible
      await waitFor(() => {
        expect(screen.getByTestId('video-panel-mock')).toBeInTheDocument();
      });
      
      // Click audio tab
      const audioTab = screen.getByTestId('modality-tab-audio');
      await user.click(audioTab);
      
      // Now audio panel should be loaded
      await waitFor(() => {
        expect(screen.getByTestId('audio-panel-mock')).toBeInTheDocument();
      });
    });
  });

  // ============== EMPTY STATE ==============

  describe('Empty State', () => {
    it('should show empty state when no results provided', () => {
      renderWithProviders(
        <ModalityTabs analysisId={mockAnalysisId} />
      );
      
      const emptyState = screen.getByTestId('modality-tabs-empty');
      expect(emptyState).toBeInTheDocument();
      expect(screen.getByText(/No Detailed Analysis Available/)).toBeInTheDocument();
    });

    it('should display info icon in empty state', () => {
      renderWithProviders(
        <ModalityTabs analysisId={mockAnalysisId} />
      );
      
      const emptyState = screen.getByTestId('modality-tabs-empty');
      expect(emptyState.querySelector('svg')).toBeInTheDocument();
    });

    it('should show helpful message in empty state', () => {
      renderWithProviders(
        <ModalityTabs analysisId={mockAnalysisId} />
      );
      
      expect(screen.getByText(/Modality-specific analysis results will appear here/)).toBeInTheDocument();
    });
  });

  // ============== VARIANTS ==============

  describe('Variants', () => {
    it('should render default variant without card wrapper', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          variant="default"
        />
      );
      
      await waitFor(() => {
        expect(screen.getByTestId('modality-tabs')).toBeInTheDocument();
        expect(screen.queryByRole('heading', { name: /Detailed Analysis/ })).not.toBeInTheDocument();
      });
    });

    it('should wrap in card when variant is card', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          variant="card"
        />
      );
      
      await waitFor(() => {
        expect(screen.getByText(/Detailed Analysis/)).toBeInTheDocument();
        // Verify card structure exists
        const tabs = screen.getByTestId('modality-tabs');
        const card = tabs.closest('.rounded-xl');
        expect(card).toBeInTheDocument();
      });
    });
  });

  // ============== GRID LAYOUT ==============

  describe('Grid Layout', () => {
    it('should use single column grid with 1 tab', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
        />
      );
      
      await waitFor(() => {
        const tabsList = screen.getByRole('tablist');
        expect(tabsList).toHaveClass('grid-cols-1');
      });
    });

    it('should use 2-column grid with 2 tabs', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        const tabsList = screen.getByRole('tablist');
        expect(tabsList).toHaveClass('grid-cols-2');
      });
    });

    it('should use 3-column grid with 3 tabs', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
          metadataResult={mockMetadataResult}
        />
      );
      
      await waitFor(() => {
        const tabsList = screen.getByRole('tablist');
        expect(tabsList).toHaveClass('grid-cols-3');
      });
    });
  });

  // ============== CUSTOM CLASSNAME ==============

  describe('Custom ClassName', () => {
    it('should apply custom className', async () => {
      const customClass = 'custom-modality-tabs';
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          className={customClass}
        />
      );
      
      await waitFor(() => {
        const tabs = screen.getByTestId('modality-tabs');
        expect(tabs).toHaveClass(customClass);
      });
    });
  });

  // ============== ACCESSIBILITY ==============

  describe('Accessibility', () => {
    it('should use proper ARIA tabs pattern', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        expect(screen.getByRole('tablist')).toBeInTheDocument();
        expect(screen.getAllByRole('tab')).toHaveLength(2);
      });
    });

    it('should mark active tab appropriately', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        const videoTab = screen.getByTestId('modality-tab-video');
        expect(videoTab).toHaveAttribute('data-state', 'active');
      });
    });

    it('should hide decorative icons from screen readers', async () => {
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
        />
      );
      
      await waitFor(() => {
        const videoTab = screen.getByTestId('modality-tab-video');
        const icon = videoTab.querySelector('svg');
        expect(icon).toHaveAttribute('aria-hidden', 'true');
      });
    });

    it('should be keyboard navigable', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      // Tab to first tab
      await user.tab();
      
      await waitFor(() => {
        const videoTab = screen.getByTestId('modality-tab-video');
        expect(videoTab).toHaveFocus();
      });
    });

    it('should pass basic accessibility checks', async () => {
      const { container } = renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      await waitFor(() => {
        expect(screen.getByTestId('modality-tabs')).toBeInTheDocument();
      });
      
      const result = checkAccessibility(container);
      expect(result.passed).toBe(true);
    });
  });

  // ============== INTEGRATION ==============

  describe('Integration', () => {
    it('should work with all props combined', async () => {
      const onTabChange = vi.fn();
      const user = userEvent.setup();
      
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
          metadataResult={mockMetadataResult}
          defaultTab="audio"
          onTabChange={onTabChange}
          variant="card"
          className="custom-class"
        />
      );
      
      await waitFor(() => {
        // Should start with audio tab active
        expect(screen.getByTestId('modality-tab-audio')).toHaveAttribute('data-state', 'active');
        
        // Should be wrapped in card
        expect(screen.getByText(/Detailed Analysis/)).toBeInTheDocument();
        
        // Should have custom class on card wrapper
        const tabs = screen.getByTestId('modality-tabs');
        const cardWrapper = tabs.closest('.custom-class');
        expect(cardWrapper).toBeInTheDocument();
      });
      
      // Change tab
      const videoTab = screen.getByTestId('modality-tab-video');
      await user.click(videoTab);
      
      await waitFor(() => {
        expect(onTabChange).toHaveBeenCalledWith('video');
      });
    });

    it('should handle rapid tab switching', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <ModalityTabs
          analysisId={mockAnalysisId}
          videoResult={mockVideoResult}
          audioResult={mockAudioResult}
        />
      );
      
      const audioTab = screen.getByTestId('modality-tab-audio');
      const videoTab = screen.getByTestId('modality-tab-video');
      
      // Rapidly switch tabs
      await user.click(audioTab);
      await user.click(videoTab);
      
      await waitFor(() => {
        expect(videoTab).toHaveAttribute('data-state', 'active');
        expect(screen.getByTestId('video-panel-mock')).toBeInTheDocument();
      });
    });
  });
});

// ============== SKELETON LOADER TESTS ==============

describe('ModalityTabsSkeleton', () => {
  it('should render skeleton with default tab count', () => {
    renderWithProviders(<ModalityTabsSkeleton />);
    
    const skeleton = screen.getByTestId('modality-tabs-skeleton');
    expect(skeleton).toBeInTheDocument();
  });

  it('should render correct number of tab skeletons', () => {
    renderWithProviders(<ModalityTabsSkeleton tabCount={4} />);
    
    const skeleton = screen.getByTestId('modality-tabs-skeleton');
    // Check the grid container itself has the correct class
    const grid = skeleton.querySelector('.grid');
    expect(grid).toHaveClass('grid-cols-4');
    
    // Check that skeleton children are rendered
    const skeletons = grid?.querySelectorAll('[class*="rounded-sm"]');
    expect(skeletons).toHaveLength(4);
  });

  it('should render panel skeleton', () => {
    renderWithProviders(<ModalityTabsSkeleton />);
    
    expect(screen.getByTestId('modality-panel-skeleton')).toBeInTheDocument();
  });

  it('should apply custom className', () => {
    const customClass = 'custom-skeleton';
    renderWithProviders(<ModalityTabsSkeleton className={customClass} />);
    
    const skeleton = screen.getByTestId('modality-tabs-skeleton');
    expect(skeleton).toHaveClass(customClass);
  });

  it('should have grid layout matching tab count', () => {
    renderWithProviders(<ModalityTabsSkeleton tabCount={3} />);
    
    const skeleton = screen.getByTestId('modality-tabs-skeleton');
    const grid = skeleton.querySelector('.grid');
    expect(grid).toHaveClass('grid-cols-3');
  });
});

// ============== SNAPSHOTS ==============

describe('Snapshots', () => {
  it('should match snapshot with single modality', async () => {
    const { container } = renderWithProviders(
      <ModalityTabs
        analysisId={mockAnalysisId}
        videoResult={mockVideoResult}
      />
    );
    
    await waitFor(() => {
      expect(screen.getByTestId('modality-tabs')).toBeInTheDocument();
    });
    
    expect(container.firstChild).toMatchSnapshot();
  });

  it('should match snapshot with all modalities', async () => {
    const { container } = renderWithProviders(
      <ModalityTabs
        analysisId={mockAnalysisId}
        videoResult={mockVideoResult}
        audioResult={mockAudioResult}
        metadataResult={mockMetadataResult}
      />
    );
    
    await waitFor(() => {
      expect(screen.getByTestId('modality-tabs')).toBeInTheDocument();
    });
    
    expect(container.firstChild).toMatchSnapshot();
  });

  it('should match snapshot in empty state', () => {
    const { container } = renderWithProviders(
      <ModalityTabs analysisId={mockAnalysisId} />
    );
    
    expect(container.firstChild).toMatchSnapshot();
  });

  it('should match snapshot in card variant', async () => {
    const { container } = renderWithProviders(
      <ModalityTabs
        analysisId={mockAnalysisId}
        videoResult={mockVideoResult}
        variant="card"
      />
    );
    
    await waitFor(() => {
      expect(screen.getByText(/Detailed Analysis/)).toBeInTheDocument();
    });
    
    expect(container.firstChild).toMatchSnapshot();
  });
});
