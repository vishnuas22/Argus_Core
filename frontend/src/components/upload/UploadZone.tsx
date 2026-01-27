'use client';

/**
 * Argus Core - Upload Zone Component
 * ===================================
 * Drag-and-drop file upload zone with validation feedback.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/upload/UploadZone.tsx
 * 
 * Role: Primary file upload interface. Handles drag-drop and click-to-select.
 * Validates files client-side before allowing upload.
 * 
 * Integration:
 * - Imports: hooks/useFileValidation, store/uploadStore
 * - Outputs: Selected file to parent via onFileSelect callback
 * - Updates: uploadStore with file and validation state
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows validation spinner during processing
 * - Error state: Displays validation errors inline
 * - Empty state: Shows instructions and accepted formats
 * - Accessibility: Keyboard navigation, screen reader support
 * - data-testid: upload-zone, upload-zone-input, upload-zone-error
 */

import { useState, useCallback, useRef, type DragEvent, type ChangeEvent } from 'react';
import { Upload, FileVideo, FileAudio, Image, AlertCircle, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useFileValidation } from '@/hooks/useFileValidation';
import { useUploadStore } from '@/store/uploadStore';
import { MAX_FILE_SIZE_MB, ALL_ACCEPTED_EXTENSIONS } from '@/lib/constants';

// ============== TYPES ==============

export interface UploadZoneProps {
  /** Callback when a valid file is selected */
  onFileSelect?: (file: File) => void;
  /** Maximum file size in MB */
  maxSizeMB?: number;
  /** Accepted MIME types */
  acceptedTypes?: string[];
  /** Disable the upload zone */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============== COMPONENT ==============

export function UploadZone({
  onFileSelect,
  maxSizeMB = MAX_FILE_SIZE_MB,
  acceptedTypes,
  disabled = false,
  className,
}: UploadZoneProps) {
  // Local state
  const [isDragging, setIsDragging] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Store and validation hook
  const { setFile, setValidation, setError } = useUploadStore();
  const { validate, generatePreview, errors, warnings, isValid } = useFileValidation({
    maxSizeMB,
    acceptedTypes,
  });

  /**
   * Process file selection
   */
  const processFile = useCallback(async (file: File) => {
    if (disabled) return;
    
    setIsValidating(true);
    setError(null);

    try {
      // Validate file
      const result = await validate(file);
      
      // Generate preview
      const preview = await generatePreview(file);
      
      // Update store
      setFile(file, preview, result.fileInfo);
      setValidation(result.isValid, result.errors, result.warnings);

      // Notify parent if valid
      if (result.isValid && onFileSelect) {
        onFileSelect(file);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process file');
    } finally {
      setIsValidating(false);
    }
  }, [disabled, validate, generatePreview, setFile, setValidation, setError, onFileSelect]);

  /**
   * Handle drag events
   */
  const handleDragEnter = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled) return;

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  }, [disabled, processFile]);

  /**
   * Handle click to select file
   */
  const handleClick = useCallback(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.click();
    }
  }, [disabled]);

  /**
   * Handle file input change
   */
  const handleFileChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
    // Reset input value to allow re-selecting same file
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, [processFile]);

  /**
   * Handle keyboard activation
   */
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
      e.preventDefault();
      handleClick();
    }
  }, [disabled, handleClick]);

  // Build accept string for input
  const acceptString = ALL_ACCEPTED_EXTENSIONS.join(',');

  return (
    <div
      data-testid="upload-zone"
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload file for analysis"
      aria-disabled={disabled}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className={cn(
        // Base styles
        'relative flex flex-col items-center justify-center',
        'w-full min-h-[280px] rounded-xl border-2 border-dashed',
        'transition-all duration-200 ease-in-out cursor-pointer',
        'focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
        
        // Default state
        !isDragging && !disabled && 'border-border bg-muted/30 hover:bg-muted/50 hover:border-primary/50',
        
        // Dragging state
        isDragging && 'border-primary bg-primary/5 scale-[1.02]',
        
        // Disabled state
        disabled && 'opacity-50 cursor-not-allowed border-muted',
        
        // Error state
        errors.length > 0 && 'border-destructive bg-destructive/5',
        
        className
      )}
    >
      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept={acceptString}
        onChange={handleFileChange}
        disabled={disabled}
        className="hidden"
        data-testid="upload-zone-input"
        aria-hidden="true"
      />

      {/* Content */}
      <div className="flex flex-col items-center gap-4 p-8 text-center">
        {/* Icon */}
        {isValidating ? (
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent" />
        ) : (
          <div className={cn(
            'p-4 rounded-full transition-colors',
            isDragging ? 'bg-primary/10' : 'bg-muted'
          )}>
            <Upload 
              className={cn(
                'h-8 w-8 transition-colors',
                isDragging ? 'text-primary' : 'text-muted-foreground'
              )} 
            />
          </div>
        )}

        {/* Instructions */}
        <div className="space-y-2">
          <p className="text-lg font-medium text-foreground">
            {isDragging ? 'Drop your file here' : 'Drop file or click to upload'}
          </p>
          <p className="text-sm text-muted-foreground">
            Maximum file size: {maxSizeMB}MB
          </p>
        </div>

        {/* Supported formats */}
        <div className="flex items-center gap-6 text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <FileVideo className="h-4 w-4" />
            <span>MP4, WebM, MOV</span>
          </div>
          <div className="flex items-center gap-1.5">
            <FileAudio className="h-4 w-4" />
            <span>MP3, WAV, OGG</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Image className="h-4 w-4" />
            <span>JPG, PNG, WebP</span>
          </div>
        </div>

        {/* Validation errors */}
        {errors.length > 0 && (
          <div 
            className="flex items-center gap-2 text-sm text-destructive mt-2"
            data-testid="upload-zone-error"
            role="alert"
          >
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            <span>{errors[0].message}</span>
          </div>
        )}

        {/* Validation warnings */}
        {warnings.length > 0 && errors.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-500 mt-2">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            <span>{warnings[0].message}</span>
          </div>
        )}

        {/* Success indicator */}
        {isValid && errors.length === 0 && warnings.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-500 mt-2">
            <CheckCircle className="h-4 w-4 flex-shrink-0" />
            <span>File ready for analysis</span>
          </div>
        )}
      </div>

      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 flex items-center justify-center bg-primary/5 rounded-xl pointer-events-none">
          <div className="p-4 rounded-full bg-primary/10">
            <Upload className="h-8 w-8 text-primary animate-bounce" />
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadZone;
