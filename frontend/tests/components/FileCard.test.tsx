/**
 * Argus Core - FileCard Component Tests
 * ======================================
 * Comprehensive tests for FileCard component.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Component Tests (P0)
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Test Coverage:
 * - Basic rendering with different file types
 * - Preview display (image, video, icon fallback)
 * - File metadata display (name, size, type)
 * - Remove button functionality
 * - Upload progress states
 * - Error states
 * - Success states
 * - Disabled state
 * - Accessibility compliance (WCAG 2.1 AA)
 * - Skeleton loading state
 * 
 * Target: >80% coverage
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, checkAccessibility } from '../utils/test-utils';
import { FileCard, FileCardSkeleton } from '@/components/upload/FileCard';

// ============== MOCKS ==============

// Helper to create mock files
const createMockFile = (
  name: string,
  size: number,
  type: string
): File => {
  const blob = new Blob(['x'.repeat(size)], { type });
  return new File([blob], name, { type });
};

// ============== TEST SUITE ==============

describe('FileCard', () => {
  const mockOnRemove = vi.fn();
  const mockFile = createMockFile('test-video.mp4', 10485760, 'video/mp4'); // 10MB
  const mockPreview = 'blob:http://localhost/test-preview';

  beforeEach(() => {
    mockOnRemove.mockClear();
  });

  // ============== BASIC RENDERING ==============

  describe('Rendering', () => {
    it('should render with minimum required props', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const card = screen.getByTestId('file-card');
      expect(card).toBeInTheDocument();
    });

    it('should display filename', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      expect(screen.getByText('test-video.mp4')).toBeInTheDocument();
    });

    it('should display file size', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      expect(screen.getByText(/10/)).toBeInTheDocument(); // Size contains 10
    });

    it('should display file type', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      expect(screen.getByText(/video\/mp4/)).toBeInTheDocument();
    });

    it('should show preview container', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const preview = screen.getByTestId('file-card-preview');
      expect(preview).toBeInTheDocument();
    });

    it('should show remove button', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      expect(removeButton).toBeInTheDocument();
    });

    it('should truncate long filenames', () => {
      const longFile = createMockFile(
        'very-long-filename-that-should-be-truncated-for-display.mp4',
        1024,
        'video/mp4'
      );
      
      renderWithProviders(
        <FileCard file={longFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const filename = screen.getByText(/very-long-filename/);
      expect(filename).toHaveClass('truncate');
    });
  });

  // ============== FILE TYPE ICONS ==============

  describe('File Type Icons', () => {
    it('should show video icon for video files', () => {
      const videoFile = createMockFile('test.mp4', 1024, 'video/mp4');
      
      renderWithProviders(
        <FileCard file={videoFile} preview={null} onRemove={mockOnRemove} />
      );
      
      expect(screen.getByText('video')).toBeInTheDocument();
    });

    it('should show audio icon for audio files', () => {
      const audioFile = createMockFile('test.mp3', 1024, 'audio/mpeg');
      
      renderWithProviders(
        <FileCard file={audioFile} preview={null} onRemove={mockOnRemove} />
      );
      
      expect(screen.getByText('audio')).toBeInTheDocument();
    });

    it('should show image icon for image files', () => {
      const imageFile = createMockFile('test.jpg', 1024, 'image/jpeg');
      
      renderWithProviders(
        <FileCard file={imageFile} preview={null} onRemove={mockOnRemove} />
      );
      
      expect(screen.getByText('image')).toBeInTheDocument();
    });

    it('should show file extension badge', () => {
      const file = createMockFile('test.mp4', 1024, 'video/mp4');
      
      renderWithProviders(
        <FileCard file={file} preview={null} onRemove={mockOnRemove} />
      );
      
      expect(screen.getByText('MP4')).toBeInTheDocument();
    });

    it('should limit extension badge to 4 characters', () => {
      const file = createMockFile('test.mpeg', 1024, 'video/mpeg');
      
      renderWithProviders(
        <FileCard file={file} preview={null} onRemove={mockOnRemove} />
      );
      
      const ext = screen.getByText('MPEG');
      expect(ext.textContent?.length).toBeLessThanOrEqual(4);
    });
  });

  // ============== PREVIEW DISPLAY ==============

  describe('Preview Display', () => {
    it('should show image preview when available', () => {
      const imageFile = createMockFile('test.jpg', 1024, 'image/jpeg');
      
      renderWithProviders(
        <FileCard 
          file={imageFile} 
          preview={mockPreview} 
          onRemove={mockOnRemove} 
        />
      );
      
      const preview = screen.getByTestId('file-card-preview');
      const img = preview.querySelector('img');
      expect(img).toBeInTheDocument();
      expect(img).toHaveAttribute('src', mockPreview);
      expect(img).toHaveAttribute('alt', `Preview of ${imageFile.name}`);
    });

    it('should show video preview when available', () => {
      const videoFile = createMockFile('test.mp4', 1024, 'video/mp4');
      
      renderWithProviders(
        <FileCard 
          file={videoFile} 
          preview={mockPreview} 
          onRemove={mockOnRemove} 
        />
      );
      
      const preview = screen.getByTestId('file-card-preview');
      const video = preview.querySelector('video');
      expect(video).toBeInTheDocument();
      expect(video).toHaveAttribute('src', mockPreview);
      expect(video).toHaveAttribute('muted');
    });

    it('should show icon fallback for audio files', () => {
      const audioFile = createMockFile('test.mp3', 1024, 'audio/mpeg');
      
      renderWithProviders(
        <FileCard file={audioFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const preview = screen.getByTestId('file-card-preview');
      expect(preview).toBeInTheDocument();
      // Should show icon instead of img/video
      expect(preview.querySelector('img')).not.toBeInTheDocument();
      expect(preview.querySelector('video')).not.toBeInTheDocument();
    });

    it('should show icon fallback when preview is null', () => {
      const videoFile = createMockFile('test.mp4', 1024, 'video/mp4');
      
      renderWithProviders(
        <FileCard file={videoFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const preview = screen.getByTestId('file-card-preview');
      // Should show icon instead of video
      expect(preview.querySelector('video')).not.toBeInTheDocument();
    });

    it('should show category badge on preview', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={mockPreview} onRemove={mockOnRemove} />
      );
      
      expect(screen.getByText('video')).toBeInTheDocument();
    });
  });

  // ============== REMOVE BUTTON ==============

  describe('Remove Button', () => {
    it('should call onRemove when clicked', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      await user.click(removeButton);
      
      expect(mockOnRemove).toHaveBeenCalledTimes(1);
    });

    it('should have accessible label', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      expect(removeButton).toHaveAttribute('aria-label', `Remove ${mockFile.name}`);
    });

    it('should be disabled during upload', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={50}
        />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      expect(removeButton).toBeDisabled();
    });

    it('should not be disabled after upload complete', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={100}
        />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      expect(removeButton).not.toBeDisabled();
    });

    it('should be disabled when component is disabled', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          disabled
        />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      expect(removeButton).toBeDisabled();
    });
  });

  // ============== UPLOAD PROGRESS ==============

  describe('Upload Progress', () => {
    it('should show progress bar when uploadProgress is provided', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={50}
        />
      );
      
      const progressBar = screen.getByRole('progressbar');
      expect(progressBar).toBeInTheDocument();
    });

    it('should display upload percentage', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={75}
        />
      );
      
      expect(screen.getByText('Uploading... 75%')).toBeInTheDocument();
    });

    it('should show completion message when progress is 100', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={100}
        />
      );
      
      expect(screen.getByText('Upload complete')).toBeInTheDocument();
    });

    it('should not show progress bar when uploadProgress is undefined', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const progressBar = screen.queryByRole('progressbar');
      expect(progressBar).not.toBeInTheDocument();
    });

    it('should show uploading state for progress < 100', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={0}
        />
      );
      
      expect(screen.getByText('Uploading... 0%')).toBeInTheDocument();
    });

    it('should have accessible progress label', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={50}
        />
      );
      
      const progressBar = screen.getByRole('progressbar');
      expect(progressBar).toHaveAttribute('aria-label', 'Upload progress');
    });
  });

  // ============== ERROR STATES ==============

  describe('Error States', () => {
    it('should display error message', () => {
      const errorMessage = 'File validation failed';
      
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          error={errorMessage}
        />
      );
      
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });

    it('should apply error styling to card', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          error="Error message"
        />
      );
      
      const card = screen.getByTestId('file-card');
      expect(card).toHaveClass('border-destructive');
    });

    it('should show error icon', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          error="Error message"
        />
      );
      
      // Error section should have role="alert"
      const errorAlert = screen.getByRole('alert');
      expect(errorAlert).toBeInTheDocument();
    });

    it('should have role alert for error message', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          error="Error message"
        />
      );
      
      const errorElement = screen.getByRole('alert');
      expect(errorElement).toHaveTextContent('Error message');
    });

    it('should not show success indicator when error present', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={100}
          error="Error message"
        />
      );
      
      expect(screen.queryByText('Ready for analysis')).not.toBeInTheDocument();
    });
  });

  // ============== SUCCESS STATES ==============

  describe('Success States', () => {
    it('should show success indicator when upload complete without error', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={100}
        />
      );
      
      expect(screen.getByText('Ready for analysis')).toBeInTheDocument();
    });

    it('should apply success styling when complete', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={100}
        />
      );
      
      const card = screen.getByTestId('file-card');
      expect(card.className).toContain('green');
    });

    it('should show checkmark icon on success', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={100}
        />
      );
      
      const successText = screen.getByText('Ready for analysis');
      const svgIcon = successText.parentElement?.querySelector('svg');
      expect(svgIcon).toBeInTheDocument();
    });

    it('should not show success when upload incomplete', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={50}
        />
      );
      
      expect(screen.queryByText('Ready for analysis')).not.toBeInTheDocument();
    });
  });

  // ============== DISABLED STATE ==============

  describe('Disabled State', () => {
    it('should apply disabled styling', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          disabled
        />
      );
      
      const card = screen.getByTestId('file-card');
      expect(card).toHaveClass('opacity-60');
      expect(card).toHaveClass('pointer-events-none');
    });

    it('should disable remove button', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          disabled
        />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      expect(removeButton).toBeDisabled();
    });

    it('should not call onRemove when disabled', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          disabled
        />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      await user.click(removeButton);
      
      expect(mockOnRemove).not.toHaveBeenCalled();
    });
  });

  // ============== CUSTOM STYLING ==============

  describe('Custom Styling', () => {
    it('should apply custom className', () => {
      const customClass = 'custom-file-card';
      
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          className={customClass}
        />
      );
      
      const card = screen.getByTestId('file-card');
      expect(card).toHaveClass(customClass);
    });

    it('should merge custom styles with default styles', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          className="custom-class"
        />
      );
      
      const card = screen.getByTestId('file-card');
      expect(card.className).toContain('custom-class');
      expect(card.className.length).toBeGreaterThan('custom-class'.length);
    });
  });

  // ============== ACCESSIBILITY ==============

  describe('Accessibility', () => {
    it('should have accessible filename with title attribute', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const filename = screen.getByText('test-video.mp4');
      expect(filename).toHaveAttribute('title', 'test-video.mp4');
    });

    it('should have accessible remove button', () => {
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      expect(removeButton).toHaveAttribute('aria-label');
    });

    it('should have accessible image preview', () => {
      const imageFile = createMockFile('test.jpg', 1024, 'image/jpeg');
      
      renderWithProviders(
        <FileCard 
          file={imageFile} 
          preview={mockPreview} 
          onRemove={mockOnRemove} 
        />
      );
      
      const img = screen.getByAltText(`Preview of ${imageFile.name}`);
      expect(img).toBeInTheDocument();
    });

    it('should pass basic accessibility checks', () => {
      const { container } = renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const result = checkAccessibility(container);
      expect(result.passed).toBe(true);
    });

    it('should have keyboard accessible remove button', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      const removeButton = screen.getByTestId('file-card-remove');
      await user.tab();
      
      // Button should be focusable via keyboard
      expect(document.activeElement).toBe(removeButton);
    });
  });

  // ============== INTEGRATION ==============

  describe('Integration', () => {
    it('should work with all props combined', () => {
      renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={mockPreview}
          onRemove={mockOnRemove}
          uploadProgress={75}
          error={undefined}
          disabled={false}
          className="custom-class"
        />
      );
      
      expect(screen.getByTestId('file-card')).toBeInTheDocument();
      expect(screen.getByText('test-video.mp4')).toBeInTheDocument();
      expect(screen.getByText('Uploading... 75%')).toBeInTheDocument();
    });

    it('should handle state transitions correctly', async () => {
      const { rerender } = renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null}
          onRemove={mockOnRemove}
          uploadProgress={0}
        />
      );
      
      expect(screen.getByText('Uploading... 0%')).toBeInTheDocument();
      
      // Update to 50%
      rerender(
        <FileCard 
          file={mockFile} 
          preview={null}
          onRemove={mockOnRemove}
          uploadProgress={50}
        />
      );
      
      expect(screen.getByText('Uploading... 50%')).toBeInTheDocument();
      
      // Update to complete
      rerender(
        <FileCard 
          file={mockFile} 
          preview={null}
          onRemove={mockOnRemove}
          uploadProgress={100}
        />
      );
      
      expect(screen.getByText('Upload complete')).toBeInTheDocument();
      expect(screen.getByText('Ready for analysis')).toBeInTheDocument();
    });

    it('should handle different file types consistently', () => {
      const fileTypes = [
        { name: 'test.mp4', type: 'video/mp4', category: 'video' },
        { name: 'test.mp3', type: 'audio/mpeg', category: 'audio' },
        { name: 'test.jpg', type: 'image/jpeg', category: 'image' },
      ];

      fileTypes.forEach(({ name, type, category }) => {
        const file = createMockFile(name, 1024, type);
        const { unmount } = renderWithProviders(
          <FileCard file={file} preview={null} onRemove={mockOnRemove} />
        );
        
        expect(screen.getByText(category)).toBeInTheDocument();
        expect(screen.getByText(name)).toBeInTheDocument();
        
        unmount();
      });
    });
  });

  // ============== SNAPSHOTS ==============

  describe('Snapshots', () => {
    it('should match snapshot in default state', () => {
      const { container } = renderWithProviders(
        <FileCard file={mockFile} preview={null} onRemove={mockOnRemove} />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot with preview', () => {
      const { container } = renderWithProviders(
        <FileCard file={mockFile} preview={mockPreview} onRemove={mockOnRemove} />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot with upload progress', () => {
      const { container } = renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          uploadProgress={50}
        />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot with error', () => {
      const { container } = renderWithProviders(
        <FileCard 
          file={mockFile} 
          preview={null} 
          onRemove={mockOnRemove}
          error="Validation error"
        />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });
  });
});

// ============== SKELETON TESTS ==============

describe('FileCardSkeleton', () => {
  it('should render skeleton component', () => {
    const { container } = renderWithProviders(<FileCardSkeleton />);
    
    expect(container.firstChild).toBeInTheDocument();
  });

  it('should have loading animation', () => {
    const { container } = renderWithProviders(<FileCardSkeleton />);
    
    const skeleton = container.firstChild as HTMLElement;
    expect(skeleton).toHaveClass('animate-pulse');
  });

  it('should apply custom className', () => {
    const customClass = 'custom-skeleton';
    const { container } = renderWithProviders(
      <FileCardSkeleton className={customClass} />
    );
    
    const skeleton = container.firstChild as HTMLElement;
    expect(skeleton).toHaveClass(customClass);
  });

  it('should match snapshot', () => {
    const { container } = renderWithProviders(<FileCardSkeleton />);
    
    expect(container.firstChild).toMatchSnapshot();
  });
});
