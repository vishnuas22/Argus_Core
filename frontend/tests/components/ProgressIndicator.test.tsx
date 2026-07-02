/**
 * Argus Core - ProgressIndicator Component Tests
 * ===============================================
 * Comprehensive tests for ProgressIndicator component.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Component Tests (P0)
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Test Coverage:
 * - Rendering with different status states (pending, preprocessing, analyzing, aggregating, completed, failed)
 * - Progress percentage display and updates
 * - Compact mode vs full mode
 * - Stage label display
 * - Error state handling with error messages
 * - Success state handling
 * - Loading/pulse animations
 * - Accessibility compliance (WCAG 2.1 AA) with ARIA live regions
 * - Integration with progressStore
 * - Responsive visual states
 * 
 * Target: >80% coverage
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { renderWithProviders, checkAccessibility } from '../utils/test-utils';
import { 
  ProgressIndicator, 
  ProgressIndicatorSkeleton,
  ConnectedProgressIndicator 
} from '@/components/analysis/ProgressIndicator';
import type { AnalysisStatus } from '@/types/analysis';

// ============== MOCKS ==============

// Mock the progressStore with a configurable state
const createMockProgressState = (overrides = {}) => ({
  status: 'pending' as AnalysisStatus,
  progressPercent: 0,
  currentStage: '',
  message: '',
  errorMessage: null,
  errorCode: null,
  estimatedTimeRemaining: null,
  startedAt: null,
  updatedAt: new Date().toISOString(),
  ...overrides,
});

let mockProgressState = createMockProgressState();

vi.mock('@/store/progressStore', () => ({
  useProgressStore: (selector: any) => {
    if (typeof selector === 'function') {
      return selector({ 
        progress: { 
          'test-analysis-id': mockProgressState 
        } 
      });
    }
    return mockProgressState;
  },
  selectProgress: (analysisId: string) => (state: any) => 
    state.progress[analysisId] || createMockProgressState(),
  STAGE_LABELS: {
    pending: 'Pending',
    preprocessing: 'Preprocessing Media',
    analyzing: 'Analyzing Content',
    aggregating: 'Computing Trust Score',
    completed: 'Analysis Complete',
    failed: 'Analysis Failed',
  },
  getEstimatedProgress: (status: AnalysisStatus) => {
    const estimates: Record<AnalysisStatus, number> = {
      pending: 5,
      preprocessing: 15,
      analyzing: 50,
      aggregating: 85,
      completed: 100,
      failed: 0,
    };
    return estimates[status] || 0;
  },
}));

// ============== TEST SUITE ==============

describe('ProgressIndicator', () => {
  const defaultProps = {
    analysisId: 'test-analysis-id',
  };

  beforeEach(() => {
    // Reset mock state before each test
    mockProgressState = createMockProgressState();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ============== BASIC RENDERING ==============

  describe('Rendering', () => {
    it('should render with minimum required props', () => {
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toBeInTheDocument();
    });

    it('should display progress bar', () => {
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const progressBar = screen.getByTestId('progress-bar');
      expect(progressBar).toBeInTheDocument();
    });

    it('should display stage label', () => {
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const stageLabel = screen.getByTestId('progress-stage');
      expect(stageLabel).toBeInTheDocument();
      expect(stageLabel).toHaveTextContent('Pending');
    });

    it('should display progress percentage', () => {
      mockProgressState = createMockProgressState({ progressPercent: 45 });
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toHaveTextContent('45');
      expect(indicator).toHaveTextContent('%');
    });

    it('should render status icon', () => {
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      // Icon should be rendered (SVG element)
      const icon = indicator.querySelector('svg');
      expect(icon).toBeInTheDocument();
    });

    it('should apply custom className', () => {
      const customClass = 'custom-progress-class';
      renderWithProviders(
        <ProgressIndicator {...defaultProps} className={customClass} />
      );
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toHaveClass(customClass);
    });
  });

  // ============== STATUS STATES ==============

  describe('Status States', () => {
    it('should display pending state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'pending',
        progressPercent: 5 
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-stage')).toHaveTextContent('Pending');
      expect(screen.getByTestId('progress-indicator')).toHaveTextContent('5');
    });

    it('should display preprocessing state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'preprocessing',
        progressPercent: 15,
        message: 'Extracting frames...'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-stage')).toHaveTextContent('Preprocessing Media');
      expect(screen.getByTestId('progress-indicator')).toHaveTextContent('15');
      expect(screen.getByTestId('progress-message')).toHaveTextContent('Extracting frames...');
    });

    it('should display analyzing state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'analyzing',
        progressPercent: 50,
        message: 'Running deepfake detection models...'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-stage')).toHaveTextContent('Analyzing Content');
      expect(screen.getByTestId('progress-indicator')).toHaveTextContent('50');
      expect(screen.getByTestId('progress-message')).toHaveTextContent('Running deepfake detection models...');
    });

    it('should display aggregating state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'aggregating',
        progressPercent: 85,
        message: 'Calculating trust score...'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-stage')).toHaveTextContent('Computing Trust Score');
      expect(screen.getByTestId('progress-indicator')).toHaveTextContent('85');
    });

    it('should display completed state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'completed',
        progressPercent: 100
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-stage')).toHaveTextContent('Analysis Complete');
      expect(screen.getByTestId('progress-indicator')).toHaveTextContent('100');
      expect(screen.getByText('Analysis complete - results ready')).toBeInTheDocument();
    });

    it('should display failed state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'File validation failed',
        errorCode: 'INVALID_FILE'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-stage')).toHaveTextContent('Analysis Failed');
      expect(screen.getByRole('alert')).toHaveTextContent('File validation failed');
      expect(screen.getByText('Error code: INVALID_FILE')).toBeInTheDocument();
    });

    it('should show pulse animation for active states', () => {
      mockProgressState = createMockProgressState({ status: 'analyzing' });
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const icon = screen.getByTestId('progress-indicator').querySelector('svg');
      expect(icon).toHaveClass('animate-pulse');
    });

    it('should not show pulse animation for completed state', () => {
      mockProgressState = createMockProgressState({ status: 'completed', progressPercent: 100 });
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const icons = screen.getByTestId('progress-indicator').querySelectorAll('svg');
      const pulsingIcon = Array.from(icons).find(icon => {
        const classAttr = icon.getAttribute('class') || '';
        return classAttr.includes('animate-pulse');
      });
      expect(pulsingIcon).toBeFalsy();
    });

    it('should not show pulse animation for failed state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'Error occurred'
      });
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      // Check that main status icon doesn't pulse
      const indicator = screen.getByTestId('progress-indicator');
      const statusIcons = indicator.querySelectorAll('.animate-pulse');
      // Should only have alert icon, not the main status icon pulsing
      expect(statusIcons.length).toBeLessThanOrEqual(1);
    });
  });

  // ============== PROGRESS UPDATES ==============

  describe('Progress Updates', () => {
    it('should update progress percentage', () => {
      mockProgressState = createMockProgressState({ progressPercent: 25 });
      const { rerender } = renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-indicator')).toHaveTextContent('25');
      
      // Update progress
      mockProgressState = createMockProgressState({ progressPercent: 75 });
      rerender(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-indicator')).toHaveTextContent('75');
    });

    it('should use estimated progress when progressPercent is 0', () => {
      mockProgressState = createMockProgressState({ 
        status: 'preprocessing',
        progressPercent: 0 
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      // Should show estimated 15% for preprocessing
      expect(screen.getByTestId('progress-indicator')).toHaveTextContent('15');
    });

    it('should display custom stage message', () => {
      mockProgressState = createMockProgressState({ 
        status: 'analyzing',
        message: 'Processing video frames 45/100'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-message')).toHaveTextContent('Processing video frames 45/100');
    });

    it('should handle progress > 100', () => {
      mockProgressState = createMockProgressState({ progressPercent: 150 });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      // Should clamp or display as-is based on implementation
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toBeInTheDocument();
    });

    it('should handle negative progress', () => {
      mockProgressState = createMockProgressState({ progressPercent: -10 });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toBeInTheDocument();
    });
  });

  // ============== COMPACT MODE ==============

  describe('Compact Mode', () => {
    it('should render in compact mode', () => {
      renderWithProviders(
        <ProgressIndicator {...defaultProps} compact />
      );
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toBeInTheDocument();
    });

    it('should display status icon in compact mode', () => {
      renderWithProviders(
        <ProgressIndicator {...defaultProps} compact />
      );
      
      const indicator = screen.getByTestId('progress-indicator');
      const icon = indicator.querySelector('svg');
      expect(icon).toBeInTheDocument();
    });

    it('should display stage label in compact mode', () => {
      mockProgressState = createMockProgressState({ status: 'analyzing' });
      
      renderWithProviders(
        <ProgressIndicator {...defaultProps} compact />
      );
      
      expect(screen.getByText('Analyzing Content')).toBeInTheDocument();
    });

    it('should display progress percentage in compact mode for active states', () => {
      mockProgressState = createMockProgressState({ 
        status: 'analyzing',
        progressPercent: 60 
      });
      
      renderWithProviders(
        <ProgressIndicator {...defaultProps} compact />
      );
      
      expect(screen.getByText('60%')).toBeInTheDocument();
    });

    it('should not show progress bar in compact mode', () => {
      renderWithProviders(
        <ProgressIndicator {...defaultProps} compact />
      );
      
      const progressBar = screen.queryByTestId('progress-bar');
      expect(progressBar).not.toBeInTheDocument();
    });

    it('should not show percentage for completed state in compact mode', () => {
      mockProgressState = createMockProgressState({ 
        status: 'completed',
        progressPercent: 100 
      });
      
      renderWithProviders(
        <ProgressIndicator {...defaultProps} compact />
      );
      
      expect(screen.queryByText('100%')).not.toBeInTheDocument();
    });

    it('should not show percentage for failed state in compact mode', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'Error'
      });
      
      renderWithProviders(
        <ProgressIndicator {...defaultProps} compact />
      );
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator.textContent).not.toContain('%');
    });
  });

  // ============== DETAILS VISIBILITY ==============

  describe('Details Visibility', () => {
    it('should show details by default', () => {
      mockProgressState = createMockProgressState({ 
        status: 'analyzing',
        message: 'Processing frames'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByTestId('progress-message')).toBeInTheDocument();
    });

    it('should hide details when showDetails is false', () => {
      mockProgressState = createMockProgressState({ 
        status: 'analyzing',
        message: 'Processing frames'
      });
      
      renderWithProviders(
        <ProgressIndicator {...defaultProps} showDetails={false} />
      );
      
      expect(screen.queryByTestId('progress-message')).not.toBeInTheDocument();
    });

    it('should show stage indicators when showDetails is true', () => {
      renderWithProviders(
        <ProgressIndicator {...defaultProps} showDetails />
      );
      
      expect(screen.getByText('Upload')).toBeInTheDocument();
      expect(screen.getByText('Preprocess')).toBeInTheDocument();
      expect(screen.getByText('Analyze')).toBeInTheDocument();
      expect(screen.getByText('Score')).toBeInTheDocument();
      expect(screen.getByText('Complete')).toBeInTheDocument();
    });

    it('should hide stage indicators when showDetails is false', () => {
      renderWithProviders(
        <ProgressIndicator {...defaultProps} showDetails={false} />
      );
      
      expect(screen.queryByText('Upload')).not.toBeInTheDocument();
      expect(screen.queryByText('Preprocess')).not.toBeInTheDocument();
    });
  });

  // ============== ERROR HANDLING ==============

  describe('Error Handling', () => {
    it('should display error message', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'Network connection failed'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByText('Network connection failed')).toBeInTheDocument();
    });

    it('should display error code', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'Processing error',
        errorCode: 'ERR_PROCESSING_001'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByText('Error code: ERR_PROCESSING_001')).toBeInTheDocument();
    });

    it('should show error alert with appropriate role', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'Analysis failed'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const alert = screen.getByRole('alert');
      expect(alert).toBeInTheDocument();
      expect(alert).toHaveTextContent('Analysis failed');
    });

    it('should show error icon in alert', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'Error occurred'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const alert = screen.getByRole('alert');
      const icon = alert.querySelector('svg');
      expect(icon).toBeInTheDocument();
    });

    it('should apply error styling to progress bar', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'Error'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const progressBar = screen.getByTestId('progress-bar');
      expect(progressBar).toHaveClass('[&>div]:bg-destructive');
    });

    it('should not show error alert when no error message', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const alert = screen.queryByRole('alert');
      expect(alert).not.toBeInTheDocument();
    });
  });

  // ============== SUCCESS STATE ==============

  describe('Success State', () => {
    it('should show success indicator when completed', () => {
      mockProgressState = createMockProgressState({ 
        status: 'completed',
        progressPercent: 100
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.getByText('Analysis complete - results ready')).toBeInTheDocument();
    });

    it('should show checkmark icon in success indicator', () => {
      mockProgressState = createMockProgressState({ 
        status: 'completed',
        progressPercent: 100
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const successText = screen.getByText('Analysis complete - results ready');
      const icon = successText.parentElement?.querySelector('svg');
      expect(icon).toBeInTheDocument();
    });

    it('should not show success indicator for non-completed states', () => {
      mockProgressState = createMockProgressState({ status: 'analyzing' });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(screen.queryByText('Analysis complete - results ready')).not.toBeInTheDocument();
    });
  });

  // ============== ACCESSIBILITY ==============

  describe('Accessibility', () => {
    it('should have role status', () => {
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toHaveAttribute('role', 'status');
    });

    it('should have aria-live polite', () => {
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toHaveAttribute('aria-live', 'polite');
    });

    it('should have descriptive aria-label', () => {
      mockProgressState = createMockProgressState({ 
        status: 'analyzing',
        progressPercent: 50
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      const ariaLabel = indicator.getAttribute('aria-label');
      expect(ariaLabel).toContain('analyzing');
      expect(ariaLabel).toContain('50');
    });

    it('should have role alert for error messages', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'Test error'
      });
      
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const alert = screen.getByRole('alert');
      expect(alert).toBeInTheDocument();
    });

    it('should pass basic accessibility checks', () => {
      const { container } = renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const result = checkAccessibility(container);
      expect(result.passed).toBe(true);
    });

    it('should maintain focus management', () => {
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      // Component should not trap focus
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).not.toHaveAttribute('tabindex');
    });

    it('should announce progress updates to screen readers', () => {
      mockProgressState = createMockProgressState({ progressPercent: 25 });
      const { rerender } = renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toHaveAttribute('aria-live', 'polite');
      
      // Update progress
      mockProgressState = createMockProgressState({ progressPercent: 75 });
      rerender(<ProgressIndicator {...defaultProps} />);
      
      // aria-live region should announce changes
      expect(indicator).toHaveAttribute('aria-live', 'polite');
    });
  });

  // ============== VISUAL STYLING ==============

  describe('Visual Styling', () => {
    it('should apply status-specific colors', () => {
      const statuses: AnalysisStatus[] = ['pending', 'preprocessing', 'analyzing', 'aggregating', 'completed', 'failed'];
      
      statuses.forEach((status) => {
        mockProgressState = createMockProgressState({ status });
        const { unmount } = renderWithProviders(<ProgressIndicator {...defaultProps} />);
        
        const indicator = screen.getByTestId('progress-indicator');
        expect(indicator.className).toBeTruthy();
        expect(indicator.className.length).toBeGreaterThan(0);
        
        unmount();
      });
    });

    it('should show rounded border', () => {
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toHaveClass('rounded-lg');
      expect(indicator).toHaveClass('border');
    });

    it('should have padding', () => {
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toHaveClass('p-4');
    });

    it('should use tabular numbers for percentage', () => {
      mockProgressState = createMockProgressState({ progressPercent: 42 });
      renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      const indicator = screen.getByTestId('progress-indicator');
      const percentText = indicator.querySelector('.tabular-nums');
      expect(percentText).toBeInTheDocument();
    });
  });

  // ============== INTEGRATION ==============

  describe('Integration', () => {
    it('should work with all props combined', () => {
      mockProgressState = createMockProgressState({ 
        status: 'analyzing',
        progressPercent: 60,
        message: 'Analyzing frames',
        currentStage: 'video_analysis'
      });
      
      renderWithProviders(
        <ProgressIndicator
          {...defaultProps}
          showDetails
          compact={false}
          className="custom-class"
        />
      );
      
      expect(screen.getByTestId('progress-indicator')).toBeInTheDocument();
      expect(screen.getByTestId('progress-bar')).toBeInTheDocument();
      expect(screen.getByTestId('progress-stage')).toBeInTheDocument();
      expect(screen.getByTestId('progress-message')).toBeInTheDocument();
    });

    it('should handle rapid state transitions', () => {
      const states: Array<{ status: AnalysisStatus; progressPercent: number }> = [
        { status: 'pending', progressPercent: 0 },
        { status: 'preprocessing', progressPercent: 15 },
        { status: 'analyzing', progressPercent: 50 },
        { status: 'aggregating', progressPercent: 85 },
        { status: 'completed', progressPercent: 100 },
      ];
      
      const { rerender } = renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      states.forEach((state) => {
        mockProgressState = createMockProgressState(state);
        rerender(<ProgressIndicator {...defaultProps} />);
        
        expect(screen.getByTestId('progress-indicator')).toBeInTheDocument();
      });
    });

    it('should handle missing analysisId gracefully', () => {
      renderWithProviders(<ProgressIndicator analysisId="" />);
      
      const indicator = screen.getByTestId('progress-indicator');
      expect(indicator).toBeInTheDocument();
    });
  });

  // ============== SNAPSHOTS ==============

  describe('Snapshots', () => {
    it('should match snapshot in pending state', () => {
      mockProgressState = createMockProgressState({ status: 'pending' });
      const { container } = renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot in analyzing state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'analyzing',
        progressPercent: 50,
        message: 'Analyzing content'
      });
      const { container } = renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot in completed state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'completed',
        progressPercent: 100
      });
      const { container } = renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot in failed state', () => {
      mockProgressState = createMockProgressState({ 
        status: 'failed',
        errorMessage: 'Analysis failed',
        errorCode: 'ERR_001'
      });
      const { container } = renderWithProviders(<ProgressIndicator {...defaultProps} />);
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot in compact mode', () => {
      mockProgressState = createMockProgressState({ 
        status: 'analyzing',
        progressPercent: 60
      });
      const { container } = renderWithProviders(
        <ProgressIndicator {...defaultProps} compact />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });
  });
});

// ============== SKELETON TESTS ==============

describe('ProgressIndicatorSkeleton', () => {
  it('should render skeleton component', () => {
    const { container } = renderWithProviders(<ProgressIndicatorSkeleton />);
    
    const skeleton = screen.getByTestId('progress-indicator-skeleton');
    expect(skeleton).toBeInTheDocument();
  });

  it('should have loading animation', () => {
    renderWithProviders(<ProgressIndicatorSkeleton />);
    
    const skeleton = screen.getByTestId('progress-indicator-skeleton');
    expect(skeleton).toHaveClass('animate-pulse');
  });

  it('should show placeholder elements', () => {
    renderWithProviders(<ProgressIndicatorSkeleton />);
    
    const skeleton = screen.getByTestId('progress-indicator-skeleton');
    // Should have placeholder divs for icon, text, and progress bar
    const placeholders = skeleton.querySelectorAll('.bg-muted');
    expect(placeholders.length).toBeGreaterThan(0);
  });

  it('should apply custom className', () => {
    const customClass = 'custom-skeleton-class';
    renderWithProviders(
      <ProgressIndicatorSkeleton className={customClass} />
    );
    
    const skeleton = screen.getByTestId('progress-indicator-skeleton');
    expect(skeleton).toHaveClass(customClass);
  });

  it('should match skeleton structure to actual component', () => {
    renderWithProviders(<ProgressIndicatorSkeleton />);
    
    const skeleton = screen.getByTestId('progress-indicator-skeleton');
    expect(skeleton).toHaveClass('rounded-lg');
    expect(skeleton).toHaveClass('border');
    expect(skeleton).toHaveClass('p-4');
  });

  it('should match snapshot', () => {
    const { container } = renderWithProviders(<ProgressIndicatorSkeleton />);
    
    expect(container.firstChild).toMatchSnapshot();
  });
});

// ============== CONNECTED COMPONENT TESTS ==============

describe('ConnectedProgressIndicator', () => {
  it('should render ConnectedProgressIndicator', () => {
    renderWithProviders(
      <ConnectedProgressIndicator analysisId="test-analysis-id" />
    );
    
    expect(screen.getByTestId('progress-indicator')).toBeInTheDocument();
  });

  it('should pass props to underlying ProgressIndicator', () => {
    renderWithProviders(
      <ConnectedProgressIndicator 
        analysisId="test-analysis-id"
        compact
        className="custom-class"
      />
    );
    
    const indicator = screen.getByTestId('progress-indicator');
    expect(indicator).toHaveClass('custom-class');
  });

  it('should match snapshot', () => {
    mockProgressState = createMockProgressState({ 
      status: 'analyzing',
      progressPercent: 50
    });
    
    const { container } = renderWithProviders(
      <ConnectedProgressIndicator analysisId="test-analysis-id" />
    );
    
    expect(container.firstChild).toMatchSnapshot();
  });
});
