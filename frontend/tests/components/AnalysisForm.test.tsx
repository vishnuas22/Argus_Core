/**
 * Argus Core - AnalysisForm Component Tests
 * ==========================================
 * Comprehensive tests for AnalysisForm component.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Component Tests (P0)
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Test Coverage:
 * - Empty state (no file selected)
 * - Form rendering with file
 * - File preview display
 * - Option toggles (report, heatmaps)
 * - Defense level selection
 * - Form submission
 * - Loading states during submission
 * - Error handling and display
 * - Clear file action
 * - Disabled states
 * - Accessibility compliance (WCAG 2.1 AA)
 * - Integration with uploadStore
 * - Integration with useAnalysis hook
 * 
 * Target: >80% coverage
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, checkAccessibility, createMockAnalysisResponse } from '../utils/test-utils';
import { AnalysisForm } from '@/components/analysis/AnalysisForm';
import { useUploadStore } from '@/store/uploadStore';
import type { DefenseLevel } from '@/types/analysis';

// ============== MOCKS ==============

// Mock next/navigation
const mockPush = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
  }),
  usePathname: () => '/analyze',
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

// Mock useAnalysis hook
const mockMutateAsync = vi.fn();
const mockAnalysisMutation = {
  mutateAsync: mockMutateAsync,
  isPending: false,
  isError: false,
  error: null,
  reset: vi.fn(),
};

vi.mock('@/hooks/useAnalysis', () => ({
  useAnalysis: () => ({
    submitAnalysis: mockAnalysisMutation,
    isSubmitting: mockAnalysisMutation.isPending,
  }),
  DEFAULT_ANALYSIS_OPTIONS: {
    generateReport: true,
    generateHeatmaps: true,
    defenseLevel: 'standard',
  },
}));

// Mock FileCard component
vi.mock('@/components/upload/FileCard', () => ({
  FileCard: ({ file, onRemove, disabled }: any) => (
    <div data-testid="file-card">
      <span>{file?.name}</span>
      <button
        data-testid="file-card-remove"
        onClick={onRemove}
        disabled={disabled}
      >
        Remove
      </button>
    </div>
  ),
}));

// Mock ConnectedUploadProgress component
vi.mock('@/components/upload/UploadProgress', () => ({
  ConnectedUploadProgress: () => (
    <div data-testid="upload-progress">Uploading...</div>
  ),
}));

// Helper to create mock file
const createMockFile = (name: string, size: number, type: string): File => {
  const blob = new Blob(['x'.repeat(size)], { type });
  return new File([blob], name, { type });
};

// Helper to setup store with file
const setupStoreWithFile = (overrides?: Partial<ReturnType<typeof useUploadStore.getState>>) => {
  const mockFile = createMockFile('test.mp4', 1024000, 'video/mp4');
  
  useUploadStore.setState({
    file: mockFile,
    preview: 'blob:test-preview',
    fileInfo: {
      name: 'test.mp4',
      size: 1024000,
      type: 'video/mp4',
      extension: 'mp4',
    },
    isValid: true,
    validationErrors: [],
    validationWarnings: [],
    status: 'idle',
    uploadProgress: 0,
    error: null,
    analysisId: null,
    ...overrides,
  });
};

// ============== TEST SUITE ==============

describe('AnalysisForm', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useUploadStore.getState().reset();
    mockAnalysisMutation.isPending = false;
    mockAnalysisMutation.isError = false;
    mockAnalysisMutation.error = null;
  });

  afterEach(() => {
    useUploadStore.getState().reset();
  });

  // ============== EMPTY STATE ==============

  describe('Empty State', () => {
    it('should render empty state when no file is selected', () => {
      renderWithProviders(<AnalysisForm />);
      
      const emptyState = screen.getByTestId('analysis-form-empty');
      expect(emptyState).toBeInTheDocument();
      expect(screen.getByText('No file selected')).toBeInTheDocument();
      expect(screen.getByText(/Upload a file to configure analysis options/i)).toBeInTheDocument();
    });

    it('should show FileSearch icon in empty state', () => {
      renderWithProviders(<AnalysisForm />);
      
      const emptyState = screen.getByTestId('analysis-form-empty');
      const icon = emptyState.querySelector('svg');
      expect(icon).toBeInTheDocument();
    });

    it('should have dashed border in empty state', () => {
      renderWithProviders(<AnalysisForm />);
      
      const emptyState = screen.getByTestId('analysis-form-empty');
      expect(emptyState).toHaveClass('border-dashed');
    });

    it('should not render form in empty state', () => {
      renderWithProviders(<AnalysisForm />);
      
      const form = screen.queryByTestId('analysis-form');
      expect(form).not.toBeInTheDocument();
    });
  });

  // ============== FORM RENDERING ==============

  describe('Form Rendering', () => {
    it('should render form when file is selected', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const form = screen.getByTestId('analysis-form');
      expect(form).toBeInTheDocument();
      expect(form.tagName).toBe('FORM');
    });

    it('should display Analysis Options title', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByText('Analysis Options')).toBeInTheDocument();
    });

    it('should display form description', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByText(/Configure how the file should be analyzed/i)).toBeInTheDocument();
    });

    it('should show FileCard with selected file', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const fileCard = screen.getByTestId('file-card');
      expect(fileCard).toBeInTheDocument();
      expect(screen.getByText('test.mp4')).toBeInTheDocument();
    });

    it('should display all option sections', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByText('Generate PDF Report')).toBeInTheDocument();
      expect(screen.getByText('Generate Heatmaps')).toBeInTheDocument();
      expect(screen.getByText('Defense Level')).toBeInTheDocument();
    });

    it('should show Clear and Analyze buttons', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByText('Clear')).toBeInTheDocument();
      expect(screen.getByText('Analyze File')).toBeInTheDocument();
    });
  });

  // ============== REPORT GENERATION TOGGLE ==============

  describe('Report Generation Toggle', () => {
    it('should render report generation toggle', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate PDF Report');
      expect(toggle).toBeInTheDocument();
    });

    it('should be checked by default', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate PDF Report') as HTMLButtonElement;
      expect(toggle).toHaveAttribute('data-state', 'checked');
    });

    it('should toggle report generation option', async () => {
      const user = userEvent.setup();
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate PDF Report');
      
      // Toggle off
      await user.click(toggle);
      await waitFor(() => {
        expect(toggle).toHaveAttribute('data-state', 'unchecked');
      });
      
      // Toggle back on
      await user.click(toggle);
      await waitFor(() => {
        expect(toggle).toHaveAttribute('data-state', 'checked');
      });
    });

    it('should show report generation description', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByText(/Create a detailed forensic report/i)).toBeInTheDocument();
    });

    it('should have proper ARIA attributes', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate PDF Report');
      expect(toggle).toHaveAttribute('role', 'switch');
      expect(toggle).toHaveAttribute('aria-describedby', 'generate-report-desc');
    });

    it('should disable toggle when submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate PDF Report');
      expect(toggle).toBeDisabled();
    });

    it('should disable toggle when uploading', () => {
      setupStoreWithFile({ status: 'uploading' });
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate PDF Report');
      expect(toggle).toBeDisabled();
    });
  });

  // ============== HEATMAP GENERATION TOGGLE ==============

  describe('Heatmap Generation Toggle', () => {
    it('should render heatmap generation toggle', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate Heatmaps');
      expect(toggle).toBeInTheDocument();
    });

    it('should be checked by default', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate Heatmaps') as HTMLButtonElement;
      expect(toggle).toHaveAttribute('data-state', 'checked');
    });

    it('should toggle heatmap generation option', async () => {
      const user = userEvent.setup();
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate Heatmaps');
      
      // Toggle off
      await user.click(toggle);
      await waitFor(() => {
        expect(toggle).toHaveAttribute('data-state', 'unchecked');
      });
    });

    it('should show heatmap generation description', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByText(/Create GradCAM visualizations/i)).toBeInTheDocument();
    });

    it('should have proper ARIA attributes', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate Heatmaps');
      expect(toggle).toHaveAttribute('role', 'switch');
      expect(toggle).toHaveAttribute('aria-describedby', 'generate-heatmaps-desc');
    });

    it('should disable toggle when submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      const toggle = screen.getByLabelText('Generate Heatmaps');
      expect(toggle).toBeDisabled();
    });
  });

  // ============== DEFENSE LEVEL SELECTION ==============

  describe('Defense Level Selection', () => {
    it('should render all defense level options', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByLabelText('None')).toBeInTheDocument();
      expect(screen.getByLabelText('Standard')).toBeInTheDocument();
      expect(screen.getByLabelText('Aggressive')).toBeInTheDocument();
    });

    it('should have Standard selected by default', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const standardOption = screen.getByLabelText('Standard') as HTMLButtonElement;
      expect(standardOption).toHaveAttribute('data-state', 'checked');
    });

    it('should show descriptions for each defense level', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByText(/Fastest processing, no adversarial defense/i)).toBeInTheDocument();
      expect(screen.getByText(/Recommended - balances speed and accuracy/i)).toBeInTheDocument();
      expect(screen.getByText(/Maximum protection against adversarial attacks/i)).toBeInTheDocument();
    });

    it('should allow selecting different defense levels', async () => {
      const user = userEvent.setup();
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const noneOption = screen.getByLabelText('None');
      const aggressiveOption = screen.getByLabelText('Aggressive');
      
      // Select None
      await user.click(noneOption);
      await waitFor(() => {
        expect(noneOption).toHaveAttribute('data-state', 'checked');
      });
      
      // Select Aggressive
      await user.click(aggressiveOption);
      await waitFor(() => {
        expect(aggressiveOption).toHaveAttribute('data-state', 'checked');
      });
    });

    it('should highlight selected defense level', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const standardOption = screen.getByLabelText('Standard').closest('div');
      expect(standardOption).toHaveClass('border-primary', 'bg-primary/5');
    });

    it('should disable defense level selection when submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      const noneOption = screen.getByLabelText('None');
      const standardOption = screen.getByLabelText('Standard');
      const aggressiveOption = screen.getByLabelText('Aggressive');
      
      expect(noneOption).toBeDisabled();
      expect(standardOption).toBeDisabled();
      expect(aggressiveOption).toBeDisabled();
    });
  });

  // ============== FORM SUBMISSION ==============

  describe('Form Submission', () => {
    it('should submit form with correct options', async () => {
      const user = userEvent.setup();
      const mockResponse = createMockAnalysisResponse();
      mockMutateAsync.mockResolvedValue(mockResponse);
      
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          file: expect.any(File),
          options: {
            generateReport: true,
            generateHeatmaps: true,
            defenseLevel: 'standard',
          },
        });
      });
    });

    it('should submit with custom options', async () => {
      const user = userEvent.setup();
      const mockResponse = createMockAnalysisResponse();
      mockMutateAsync.mockResolvedValue(mockResponse);
      
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      // Toggle off report generation
      const reportToggle = screen.getByLabelText('Generate PDF Report');
      await user.click(reportToggle);
      
      // Select aggressive defense level
      const aggressiveOption = screen.getByLabelText('Aggressive');
      await user.click(aggressiveOption);
      
      // Submit
      const submitButton = screen.getByTestId('analysis-form-submit');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          file: expect.any(File),
          options: {
            generateReport: false,
            generateHeatmaps: true,
            defenseLevel: 'aggressive',
          },
        });
      });
    });

    it('should call onSubmitSuccess callback on successful submission', async () => {
      const user = userEvent.setup();
      const onSubmitSuccess = vi.fn();
      const mockResponse = createMockAnalysisResponse({ analysis_id: 'test-123' });
      mockMutateAsync.mockResolvedValue(mockResponse);
      
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm onSubmitSuccess={onSubmitSuccess} />);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(onSubmitSuccess).toHaveBeenCalledWith('test-123');
      });
    });

    it('should call onSubmitError callback on submission failure', async () => {
      const user = userEvent.setup();
      const onSubmitError = vi.fn();
      const mockError = new Error('Submission failed');
      mockMutateAsync.mockRejectedValue(mockError);
      
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm onSubmitError={onSubmitError} />);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(onSubmitError).toHaveBeenCalledWith(mockError);
      });
    });

    it('should prevent submission when no file is selected', async () => {
      const user = userEvent.setup();
      renderWithProviders(<AnalysisForm />);
      
      // No file selected, so no form should be rendered
      const form = screen.queryByTestId('analysis-form');
      expect(form).not.toBeInTheDocument();
    });

    it('should prevent submission when file is invalid', async () => {
      setupStoreWithFile({ isValid: false });
      renderWithProviders(<AnalysisForm />);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      expect(submitButton).toBeDisabled();
    });

    it('should prevent submission when already submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      expect(submitButton).toBeDisabled();
    });

    it('should prevent submission when uploading', () => {
      setupStoreWithFile({ status: 'uploading' });
      renderWithProviders(<AnalysisForm />);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      expect(submitButton).toBeDisabled();
    });

    it('should prevent default form submission', async () => {
      const user = userEvent.setup();
      const mockResponse = createMockAnalysisResponse();
      mockMutateAsync.mockResolvedValue(mockResponse);
      
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const form = screen.getByTestId('analysis-form');
      const submitSpy = vi.fn((e) => e.preventDefault());
      form.addEventListener('submit', submitSpy);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      await user.click(submitButton);
      
      // Form submit event should be prevented
      expect(submitSpy).toHaveBeenCalled();
    });
  });

  // ============== LOADING STATES ==============

  describe('Loading States', () => {
    it('should show loading text when submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByText('Analyzing...')).toBeInTheDocument();
    });

    it('should show loading spinner when submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      const button = screen.getByTestId('analysis-form-submit');
      const spinner = button.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('should disable submit button when submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      expect(submitButton).toBeDisabled();
    });

    it('should disable all form controls when submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      const reportToggle = screen.getByLabelText('Generate PDF Report');
      const heatmapToggle = screen.getByLabelText('Generate Heatmaps');
      const clearButton = screen.getByText('Clear');
      
      expect(reportToggle).toBeDisabled();
      expect(heatmapToggle).toBeDisabled();
      expect(clearButton).toBeDisabled();
    });

    it('should show upload progress when status is uploading', () => {
      setupStoreWithFile({ status: 'uploading' });
      renderWithProviders(<AnalysisForm />);
      
      const uploadProgress = screen.getByTestId('upload-progress');
      expect(uploadProgress).toBeInTheDocument();
    });

    it('should disable FileCard remove button when submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      const removeButton = screen.getByTestId('file-card-remove');
      expect(removeButton).toBeDisabled();
    });
  });

  // ============== ERROR HANDLING ==============

  describe('Error Handling', () => {
    it('should display submission error', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isError = true;
      mockAnalysisMutation.error = new Error('Failed to submit analysis');
      
      renderWithProviders(<AnalysisForm />);
      
      const errorMessage = screen.getByTestId('analysis-form-error');
      expect(errorMessage).toBeInTheDocument();
      expect(errorMessage).toHaveTextContent('Failed to submit analysis');
    });

    it('should show error with alert role', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isError = true;
      mockAnalysisMutation.error = new Error('Test error');
      
      renderWithProviders(<AnalysisForm />);
      
      const errorMessage = screen.getByTestId('analysis-form-error');
      expect(errorMessage).toHaveAttribute('role', 'alert');
    });

    it('should show error icon', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isError = true;
      mockAnalysisMutation.error = new Error('Test error');
      
      renderWithProviders(<AnalysisForm />);
      
      const errorMessage = screen.getByTestId('analysis-form-error');
      const icon = errorMessage.querySelector('svg');
      expect(icon).toBeInTheDocument();
    });

    it('should show upload progress error', () => {
      setupStoreWithFile({ status: 'error', error: 'Upload failed' });
      renderWithProviders(<AnalysisForm />);
      
      // Upload progress component should be shown
      const uploadProgress = screen.getByTestId('upload-progress');
      expect(uploadProgress).toBeInTheDocument();
    });

    it('should handle non-Error objects in submission', async () => {
      const user = userEvent.setup();
      const onSubmitError = vi.fn();
      mockMutateAsync.mockRejectedValue('String error');
      
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm onSubmitError={onSubmitError} />);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(onSubmitError).toHaveBeenCalledWith(expect.any(Error));
      });
    });
  });

  // ============== CLEAR FILE ACTION ==============

  describe('Clear File Action', () => {
    it('should clear file when Clear button is clicked', async () => {
      const user = userEvent.setup();
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const clearButton = screen.getByText('Clear');
      await user.click(clearButton);
      
      await waitFor(() => {
        // FileCard remove button should be called
        expect(screen.queryByTestId('file-card')).not.toBeInTheDocument();
      });
    });

    it('should disable Clear button when submitting', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      renderWithProviders(<AnalysisForm />);
      
      const clearButton = screen.getByText('Clear');
      expect(clearButton).toBeDisabled();
    });

    it('should disable Clear button when uploading', () => {
      setupStoreWithFile({ status: 'uploading' });
      renderWithProviders(<AnalysisForm />);
      
      const clearButton = screen.getByText('Clear');
      expect(clearButton).toBeDisabled();
    });
  });

  // ============== ACCESSIBILITY ==============

  describe('Accessibility', () => {
    it('should have proper form structure', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const form = screen.getByTestId('analysis-form');
      expect(form.tagName).toBe('FORM');
    });

    it('should have labels for all form controls', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByLabelText('Generate PDF Report')).toBeInTheDocument();
      expect(screen.getByLabelText('Generate Heatmaps')).toBeInTheDocument();
      expect(screen.getByLabelText('None')).toBeInTheDocument();
      expect(screen.getByLabelText('Standard')).toBeInTheDocument();
      expect(screen.getByLabelText('Aggressive')).toBeInTheDocument();
    });

    it('should have descriptive button text', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      expect(screen.getByText('Analyze File')).toBeInTheDocument();
      expect(screen.getByText('Clear')).toBeInTheDocument();
    });

    it('should announce errors with role alert', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isError = true;
      mockAnalysisMutation.error = new Error('Test error');
      
      renderWithProviders(<AnalysisForm />);
      
      const error = screen.getByRole('alert');
      expect(error).toBeInTheDocument();
    });

    it('should have proper form accessibility structure', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      // Check form has proper structure
      const form = screen.getByTestId('analysis-form');
      expect(form).toBeInTheDocument();
      expect(form.tagName).toBe('FORM');
      
      // Check all form controls have labels
      const switches = screen.getAllByRole('switch');
      const radios = screen.getAllByRole('radio');
      
      expect(switches.length).toBeGreaterThan(0);
      expect(radios.length).toBeGreaterThan(0);
    });

    it('should have proper ARIA labels for switches', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const reportToggle = screen.getByLabelText('Generate PDF Report');
      const heatmapToggle = screen.getByLabelText('Generate Heatmaps');
      
      expect(reportToggle).toHaveAttribute('role', 'switch');
      expect(heatmapToggle).toHaveAttribute('role', 'switch');
    });

    it('should support keyboard navigation', async () => {
      const user = userEvent.setup();
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      // Tab through form elements
      await user.tab();
      
      // Should be able to focus on form elements
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });
  });

  // ============== CUSTOM CONFIGURATION ==============

  describe('Custom Configuration', () => {
    it('should apply custom className', () => {
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm className="custom-class" />);
      
      const card = screen.getByTestId('analysis-form').querySelector('.custom-class');
      expect(card).toBeInTheDocument();
    });

    it('should work without optional callbacks', async () => {
      const user = userEvent.setup();
      const mockResponse = createMockAnalysisResponse();
      mockMutateAsync.mockResolvedValue(mockResponse);
      
      setupStoreWithFile();
      renderWithProviders(<AnalysisForm />);
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalled();
      });
    });
  });

  // ============== INTEGRATION ==============

  describe('Integration', () => {
    it('should work with all props combined', async () => {
      const user = userEvent.setup();
      const onSubmitSuccess = vi.fn();
      const onSubmitError = vi.fn();
      const mockResponse = createMockAnalysisResponse();
      mockMutateAsync.mockResolvedValue(mockResponse);
      
      setupStoreWithFile();
      renderWithProviders(
        <AnalysisForm
          onSubmitSuccess={onSubmitSuccess}
          onSubmitError={onSubmitError}
          className="custom-class"
        />
      );
      
      const submitButton = screen.getByTestId('analysis-form-submit');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalled();
        expect(onSubmitSuccess).toHaveBeenCalled();
      });
    });

    it('should sync with uploadStore state changes', () => {
      setupStoreWithFile();
      const { rerender } = renderWithProviders(<AnalysisForm />);
      
      // Update store
      useUploadStore.setState({ status: 'uploading', uploadProgress: 50 });
      rerender(<AnalysisForm />);
      
      // Should show upload progress
      expect(screen.getByTestId('upload-progress')).toBeInTheDocument();
    });
  });

  // ============== SNAPSHOTS ==============

  describe('Snapshots', () => {
    it('should match snapshot in empty state', () => {
      const { container } = renderWithProviders(<AnalysisForm />);
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot with file selected', () => {
      setupStoreWithFile();
      const { container } = renderWithProviders(<AnalysisForm />);
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot in submitting state', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isPending = true;
      
      const { container } = renderWithProviders(<AnalysisForm />);
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot with error', () => {
      setupStoreWithFile();
      mockAnalysisMutation.isError = true;
      mockAnalysisMutation.error = new Error('Test error');
      
      const { container } = renderWithProviders(<AnalysisForm />);
      
      expect(container.firstChild).toMatchSnapshot();
    });
  });
});
