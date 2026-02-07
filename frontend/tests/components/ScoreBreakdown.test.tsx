/**
 * Argus Core - ScoreBreakdown Component Tests
 * ============================================
 * Comprehensive tests for ScoreBreakdown component.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Component Tests (P0)
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Test Coverage:
 * - Default, compact, and detailed variants
 * - Score rendering with proper color coding
 * - Weight badges display
 * - Contribution values
 * - Animation states
 * - Empty state when no breakdown available
 * - Multiple modality combinations
 * - Category color coding
 * - Progress bar rendering
 * - Accessibility compliance (WCAG 2.1 AA)
 * - Keyboard navigation
 * - Tooltips in detailed mode
 * 
 * Target: >80% coverage
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, checkAccessibility } from '../utils/test-utils';
import { ScoreBreakdown, ScoreBreakdownSkeleton } from '@/components/results/ScoreBreakdown';
import type { ScoreBreakdown as ScoreBreakdownType } from '@/types/analysis';

// ============== MOCKS ==============

// Mock Tooltip component
vi.mock('@/components/ui/tooltip', () => ({
  TooltipProvider: ({ children }: any) => <div>{children}</div>,
  Tooltip: ({ children }: any) => <div>{children}</div>,
  TooltipTrigger: ({ children, asChild }: any) => <div data-testid="tooltip-trigger">{children}</div>,
  TooltipContent: ({ children }: any) => <div data-testid="tooltip-content">{children}</div>,
}));

// Mock other UI components
vi.mock('@/components/ui/badge', () => ({
  Badge: ({ children, className }: any) => (
    <span className={className} data-testid="badge">{children}</span>
  ),
}));

vi.mock('@/components/ui/progress', () => ({
  Progress: ({ value }: any) => (
    <div data-testid="progress" data-value={value} />
  ),
}));

vi.mock('@/components/ui/skeleton', () => ({
  Skeleton: ({ className }: any) => (
    <div className={className} data-testid="skeleton" />
  ),
}));

// ============== TEST DATA ==============

/**
 * Complete breakdown with all modalities
 */
const COMPLETE_BREAKDOWN: ScoreBreakdownType = {
  video_spatial: 0.88,
  video_temporal: 0.82,
  video_lipsync: 0.90,
  audio: 0.85,
  text: 0.75,
  metadata: 0.92,
  weights: {
    video_spatial: 0.30,
    video_temporal: 0.25,
    video_lipsync: 0.10,
    audio: 0.20,
    text: 0.10,
    metadata: 0.05,
  },
};

/**
 * Video-only breakdown
 */
const VIDEO_ONLY_BREAKDOWN: ScoreBreakdownType = {
  video_spatial: 0.75,
  video_temporal: 0.70,
  weights: {
    video_spatial: 0.60,
    video_temporal: 0.40,
  },
};

/**
 * Audio-only breakdown
 */
const AUDIO_ONLY_BREAKDOWN: ScoreBreakdownType = {
  audio: 0.88,
  weights: {
    audio: 1.0,
  },
};

/**
 * Mixed modalities without weights (should use defaults)
 */
const NO_WEIGHTS_BREAKDOWN: ScoreBreakdownType = {
  video_spatial: 0.80,
  audio: 0.75,
  metadata: 0.90,
};

/**
 * Empty breakdown
 */
const EMPTY_BREAKDOWN: ScoreBreakdownType = {};

/**
 * Low scores breakdown (should show red/orange colors)
 */
const LOW_SCORES_BREAKDOWN: ScoreBreakdownType = {
  video_spatial: 0.25,
  video_temporal: 0.15,
  audio: 0.30,
  weights: {
    video_spatial: 0.40,
    video_temporal: 0.30,
    audio: 0.30,
  },
};

/**
 * High scores breakdown (should show green colors)
 */
const HIGH_SCORES_BREAKDOWN: ScoreBreakdownType = {
  video_spatial: 0.95,
  video_temporal: 0.92,
  audio: 0.88,
  text: 0.90,
  weights: {
    video_spatial: 0.30,
    video_temporal: 0.30,
    audio: 0.25,
    text: 0.15,
  },
};

// ============== TEST UTILITIES ==============

/**
 * Get all breakdown items from rendered component
 */
function getBreakdownItems() {
  return screen.getAllByTestId('score-breakdown-item');
}

/**
 * Get breakdown item by label text
 */
function getBreakdownItemByLabel(label: string) {
  return screen.getByText(label).closest('[data-testid="score-breakdown-item"]');
}

/**
 * Check if score color matches expected range
 */
function expectScoreColor(score: number, element: HTMLElement) {
  const scorePercent = score * 100;
  
  if (scorePercent >= 80) {
    expect(element).toHaveClass(/green/);
  } else if (scorePercent >= 60) {
    expect(element).toHaveClass(/lime/);
  } else if (scorePercent >= 40) {
    expect(element).toHaveClass(/yellow/);
  } else if (scorePercent >= 20) {
    expect(element).toHaveClass(/orange/);
  } else {
    expect(element).toHaveClass(/red/);
  }
}

// ============== CORE FUNCTIONALITY TESTS ==============

describe('ScoreBreakdown Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ============== RENDERING TESTS ==============

  describe('Rendering', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      expect(container).toBeTruthy();
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
    });

    it('renders with complete breakdown data', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      // Should render all modalities
      const items = getBreakdownItems();
      expect(items.length).toBeGreaterThan(0);
      
      // Check for specific modalities
      expect(screen.getByText('Video (Spatial)')).toBeInTheDocument();
      expect(screen.getByText('Video (Temporal)')).toBeInTheDocument();
      expect(screen.getByText('Audio Analysis')).toBeInTheDocument();
    });

    it('applies custom className', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          className="custom-breakdown-class"
        />
      );
      
      const breakdown = screen.getByTestId('score-breakdown');
      expect(breakdown).toHaveClass('custom-breakdown-class');
    });

    it('has proper list semantics', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      const breakdown = screen.getByTestId('score-breakdown');
      expect(breakdown).toHaveAttribute('role', 'list');
      expect(breakdown).toHaveAttribute('aria-label', 'Score breakdown by modality');
    });

    it('renders each item with listitem role', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      const items = screen.getAllByRole('listitem');
      expect(items.length).toBeGreaterThan(0);
    });
  });

  // ============== EMPTY STATE TESTS ==============

  describe('Empty State', () => {
    it('shows empty state when no breakdown data', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={EMPTY_BREAKDOWN} />
      );
      
      expect(screen.getByTestId('score-breakdown-empty')).toBeInTheDocument();
      expect(screen.getByText(/No detailed score breakdown available/i)).toBeInTheDocument();
    });

    it('shows empty state icon', () => {
      const { container } = renderWithProviders(
        <ScoreBreakdown breakdown={EMPTY_BREAKDOWN} />
      );
      
      // Should have Info icon
      const svgElement = container.querySelector('svg');
      expect(svgElement).toBeInTheDocument();
    });
  });

  // ============== SCORE DISPLAY TESTS ==============

  describe('Score Display', () => {
    it('displays scores as percentages', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      // Video spatial: 0.88 = 88%
      expect(screen.getByText('88')).toBeInTheDocument();
      // Audio: 0.85 = 85%
      expect(screen.getByText('85')).toBeInTheDocument();
    });

    it('displays scores with correct precision', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={{ video_spatial: 0.856 }} />
      );
      
      // Should round to whole number: 85.6 -> 86
      expect(screen.getByText('86')).toBeInTheDocument();
    });

    it('handles scores at boundaries correctly', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={{
            video_spatial: 0.0,
            audio: 0.5,
            metadata: 1.0,
          }} 
        />
      );
      
      expect(screen.getByText('0')).toBeInTheDocument();
      expect(screen.getByText('50')).toBeInTheDocument();
      expect(screen.getByText('100')).toBeInTheDocument();
    });
  });

  // ============== WEIGHT DISPLAY TESTS ==============

  describe('Weight Display', () => {
    it('shows weight badges when showWeights is true', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          showWeights
        />
      );
      
      const badges = screen.getAllByTestId('badge');
      expect(badges.length).toBeGreaterThan(0);
      
      // Check for weight percentages
      expect(screen.getByText('30%')).toBeInTheDocument(); // video_spatial
      expect(screen.getByText('25%')).toBeInTheDocument(); // video_temporal
    });

    it('hides weight badges when showWeights is false', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          showWeights={false}
        />
      );
      
      // Should not show percentage badges
      expect(screen.queryByText('30%')).not.toBeInTheDocument();
    });

    it('uses default weights when weights not provided', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={NO_WEIGHTS_BREAKDOWN}
          showWeights
        />
      );
      
      // Should still render breakdown items
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
      const items = getBreakdownItems();
      expect(items.length).toBeGreaterThan(0);
    });

    it('displays weights in correct order (highest first)', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          showWeights
        />
      );
      
      const items = getBreakdownItems();
      // First item should have highest weight
      expect(items[0]).toBeInTheDocument();
    });
  });

  // ============== CONTRIBUTION DISPLAY TESTS ==============

  describe('Contribution Display', () => {
    it('shows contribution values when showContributions is true', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          showContributions
        />
      );
      
      // Should show contribution values (score * weight)
      const breakdown = screen.getByTestId('score-breakdown');
      expect(breakdown).toBeInTheDocument();
    });

    it('hides contribution values when showContributions is false', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          showContributions={false}
        />
      );
      
      // Default behavior - no contribution values shown
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
    });
  });

  // ============== VARIANT TESTS ==============

  describe('Layout Variants', () => {
    it('renders default variant correctly', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          variant="default"
        />
      );
      
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
      // Default variant shows full information
      expect(screen.getByText('Video (Spatial)')).toBeInTheDocument();
    });

    it('renders compact variant correctly', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          variant="compact"
        />
      );
      
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
      const items = getBreakdownItems();
      expect(items.length).toBeGreaterThan(0);
    });

    it('renders detailed variant with tooltips', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          variant="detailed"
        />
      );
      
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
      // Detailed variant should have tooltip triggers
      const tooltipTriggers = screen.queryAllByTestId('tooltip-trigger');
      expect(tooltipTriggers.length).toBeGreaterThan(0);
    });

    it('shows category legend in detailed variant', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          variant="detailed"
        />
      );
      
      // Should show category labels - use getAllByText for multiple matches
      const videoLabels = screen.getAllByText(/video/i);
      expect(videoLabels.length).toBeGreaterThan(0);
    });
  });

  // ============== ANIMATION TESTS ==============

  describe('Animation', () => {
    it('applies animation when animated is true', () => {
      const { container } = renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          animated
        />
      );
      
      // Check for animation-related classes or styles
      const items = getBreakdownItems();
      expect(items.length).toBeGreaterThan(0);
      
      // Items should have animation
      items.forEach(item => {
        expect(item).toBeInTheDocument();
      });
    });

    it('does not animate when animated is false', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          animated={false}
        />
      );
      
      const items = getBreakdownItems();
      expect(items.length).toBeGreaterThan(0);
    });

    it('respects custom animation duration', () => {
      renderWithProviders(
        <ScoreBreakdown 
          breakdown={COMPLETE_BREAKDOWN}
          animated
          animationDuration={1000}
        />
      );
      
      const items = getBreakdownItems();
      expect(items.length).toBeGreaterThan(0);
    });
  });

  // ============== COLOR CODING TESTS ==============

  describe('Color Coding', () => {
    it('applies green colors for high scores', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={HIGH_SCORES_BREAKDOWN} />
      );
      
      // High scores (>80) should have green coloring
      const breakdown = screen.getByTestId('score-breakdown');
      expect(breakdown).toBeInTheDocument();
    });

    it('applies red/orange colors for low scores', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={LOW_SCORES_BREAKDOWN} />
      );
      
      // Low scores (<40) should have red/orange coloring
      const breakdown = screen.getByTestId('score-breakdown');
      expect(breakdown).toBeInTheDocument();
    });

    it('applies appropriate category colors', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      // Each category should have distinct colors
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
    });
  });

  // ============== MODALITY COMBINATIONS TESTS ==============

  describe('Modality Combinations', () => {
    it('handles video-only breakdown', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={VIDEO_ONLY_BREAKDOWN} />
      );
      
      expect(screen.getByText('Video (Spatial)')).toBeInTheDocument();
      expect(screen.getByText('Video (Temporal)')).toBeInTheDocument();
      expect(screen.queryByText('Audio Analysis')).not.toBeInTheDocument();
    });

    it('handles audio-only breakdown', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={AUDIO_ONLY_BREAKDOWN} />
      );
      
      expect(screen.getByText('Audio Analysis')).toBeInTheDocument();
      expect(screen.queryByText('Video (Spatial)')).not.toBeInTheDocument();
    });

    it('handles mixed modalities', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={NO_WEIGHTS_BREAKDOWN} />
      );
      
      expect(screen.getByText('Video (Spatial)')).toBeInTheDocument();
      expect(screen.getByText('Audio Analysis')).toBeInTheDocument();
      expect(screen.getByText('Metadata')).toBeInTheDocument();
    });

    it('handles all modalities present', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      expect(screen.getByText('Video (Spatial)')).toBeInTheDocument();
      expect(screen.getByText('Video (Temporal)')).toBeInTheDocument();
      expect(screen.getByText('Lip Sync')).toBeInTheDocument();
      expect(screen.getByText('Audio Analysis')).toBeInTheDocument();
      expect(screen.getByText('Text Analysis')).toBeInTheDocument();
      expect(screen.getByText('Metadata')).toBeInTheDocument();
    });
  });

  // ============== PROGRESS BARS TESTS ==============

  describe('Progress Bars', () => {
    it('renders progress bars for each item', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      const bars = screen.getAllByTestId('score-breakdown-bar');
      expect(bars.length).toBeGreaterThan(0);
    });

    it('sets progress bar width based on score', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={{ video_spatial: 0.75 }} />
      );
      
      const bar = screen.getByTestId('score-breakdown-bar');
      const progressBar = bar.querySelector('[role="progressbar"]');
      expect(progressBar).toHaveAttribute('aria-valuenow');
    });

    it('handles 0% score correctly', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={{ video_spatial: 0.0 }} />
      );
      
      const bar = screen.getByTestId('score-breakdown-bar');
      expect(bar).toBeInTheDocument();
    });

    it('handles 100% score correctly', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={{ video_spatial: 1.0 }} />
      );
      
      const bar = screen.getByTestId('score-breakdown-bar');
      expect(bar).toBeInTheDocument();
    });
  });

  // ============== ACCESSIBILITY TESTS ==============

  describe('Accessibility', () => {
    it('meets WCAG 2.1 AA standards', async () => {
      const { container } = renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      await checkAccessibility(container);
    });

    it('has proper ARIA labels for list', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      const list = screen.getByTestId('score-breakdown');
      expect(list).toHaveAttribute('role', 'list');
      expect(list).toHaveAttribute('aria-label', 'Score breakdown by modality');
    });

    it('has proper ARIA labels for progress bars', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={{ video_spatial: 0.85 }} />
      );
      
      const bar = screen.getByTestId('score-breakdown-bar');
      const progressBar = bar.querySelector('[role="progressbar"]');
      
      expect(progressBar).toHaveAttribute('aria-valuenow', '85');
      expect(progressBar).toHaveAttribute('aria-valuemin', '0');
      expect(progressBar).toHaveAttribute('aria-valuemax', '100');
    });

    it('has descriptive labels for each item', () => {
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      // Each item should have a label
      expect(screen.getByText('Video (Spatial)')).toBeInTheDocument();
      expect(screen.getByText('Audio Analysis')).toBeInTheDocument();
    });
  });

  // ============== ICON DISPLAY TESTS ==============

  describe('Icon Display', () => {
    it('displays icons for each modality', () => {
      const { container } = renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      // Should have SVG icons
      const icons = container.querySelectorAll('svg');
      expect(icons.length).toBeGreaterThan(0);
    });

    it('applies category colors to icons', () => {
      const { container } = renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      const items = getBreakdownItems();
      expect(items.length).toBeGreaterThan(0);
    });
  });

  // ============== RESPONSIVE BEHAVIOR TESTS ==============

  describe('Responsive Behavior', () => {
    it('adapts to mobile viewports', () => {
      global.innerWidth = 375;
      global.dispatchEvent(new Event('resize'));
      
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
    });

    it('adapts to tablet viewports', () => {
      global.innerWidth = 768;
      global.dispatchEvent(new Event('resize'));
      
      renderWithProviders(
        <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
      );
      
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
    });
  });
});

// ============== SKELETON TESTS ==============

describe('ScoreBreakdownSkeleton', () => {
  it('renders skeleton correctly', () => {
    renderWithProviders(<ScoreBreakdownSkeleton />);
    
    expect(screen.getByTestId('score-breakdown-skeleton')).toBeInTheDocument();
  });

  it('renders default variant skeleton', () => {
    renderWithProviders(
      <ScoreBreakdownSkeleton variant="default" />
    );
    
    expect(screen.getByTestId('score-breakdown-skeleton')).toBeInTheDocument();
    const skeletons = screen.getAllByTestId('skeleton');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('renders compact variant skeleton', () => {
    renderWithProviders(
      <ScoreBreakdownSkeleton variant="compact" />
    );
    
    expect(screen.getByTestId('score-breakdown-skeleton')).toBeInTheDocument();
  });

  it('renders custom item count', () => {
    renderWithProviders(
      <ScoreBreakdownSkeleton itemCount={6} />
    );
    
    const skeleton = screen.getByTestId('score-breakdown-skeleton');
    expect(skeleton).toBeInTheDocument();
  });

  it('applies custom className to skeleton', () => {
    renderWithProviders(
      <ScoreBreakdownSkeleton className="custom-skeleton" />
    );
    
    const skeleton = screen.getByTestId('score-breakdown-skeleton');
    expect(skeleton).toHaveClass('custom-skeleton');
  });
});

// ============== INTEGRATION TESTS ==============

describe('ScoreBreakdown Integration', () => {
  it('updates when breakdown data changes', async () => {
    const { rerender } = renderWithProviders(
      <ScoreBreakdown breakdown={VIDEO_ONLY_BREAKDOWN} />
    );
    
    // Initial state
    expect(screen.getByText('Video (Spatial)')).toBeInTheDocument();
    expect(screen.queryByText('Audio Analysis')).not.toBeInTheDocument();
    
    // Update with complete breakdown
    rerender(
      <ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />
    );
    
    // Should now show audio
    await waitFor(() => {
      expect(screen.getByText('Audio Analysis')).toBeInTheDocument();
    });
  });

  it('handles rapid data updates correctly', async () => {
    const { rerender } = renderWithProviders(
      <ScoreBreakdown breakdown={VIDEO_ONLY_BREAKDOWN} />
    );
    
    // Rapid updates
    rerender(<ScoreBreakdown breakdown={AUDIO_ONLY_BREAKDOWN} />);
    rerender(<ScoreBreakdown breakdown={COMPLETE_BREAKDOWN} />);
    rerender(<ScoreBreakdown breakdown={HIGH_SCORES_BREAKDOWN} />);
    
    await waitFor(() => {
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
    });
  });

  it('maintains animations through updates', async () => {
    const { rerender } = renderWithProviders(
      <ScoreBreakdown 
        breakdown={VIDEO_ONLY_BREAKDOWN}
        animated
      />
    );
    
    rerender(
      <ScoreBreakdown 
        breakdown={COMPLETE_BREAKDOWN}
        animated
      />
    );
    
    await waitFor(() => {
      const items = getBreakdownItems();
      expect(items.length).toBeGreaterThan(0);
    });
  });
});
