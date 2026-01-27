/**
 * Argus Core - Upload Zone Component
 * ===================================
 * Drag-and-drop file upload zone with validation feedback.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/upload/UploadZone.tsx
 * 
 * Role: Handle file drop/selection, validate files client-side,
 * display feedback, and trigger upload flow.
 * 
 * Integration:
 * - Imports: hooks/useFileValidation, store/uploadStore
 * - Inputs: onFileSelect callback, configuration options
 * - Outputs: Drag-drop UI with visual feedback
 * 
 * Component Contract (P0):
 * - ✅ Props interface defined
 * - ✅ Loading state: Shows spinner during validation
 * - ✅ Error state: Displays validation errors inline
 * - ✅ Empty state: Shows instructions and accepted formats
 * - ✅ Accessibility: Keyboard navigation, screen reader support
 * - ✅ data-testid attributes for testing
 */

'use client';

import React, { useCallback, useState, useRef } from 'react';
import { Upload, FileVideo, FileAudio, Image, X, AlertCircle, CheckCircle } from 'lucide-react';
import { cn, formatFileSize, getMimeCategory } from '@/lib/utils';
import { useFileValidation } from '@/hooks/useFileValidation';
import { useUploadStore } from '@/store/uploadStore';
import { ALL_ACCEPTED_EXTENSIONS, MAX_FILE_SIZE_MB } from '@/lib/constants';

// ============== TYPES ==============

interface UploadZoneProps {
  /** Callback when valid file is selected */
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
}: UploadZoneProps): JSX.Element {
  // State
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Validation hook
  const { validate, isValidating } = useFileValidation({ maxSizeMB, acceptedTypes });
  
  // Store
  const { 
    file, 
    preview, 
    fileInfo, 
    validationResult,
    errorMessage,
    status,
    setFile, 
    clearFile, 
    setValidationResult 
  } = useUploadStore();
  
  /**
   * Handle file selection (from drop or input)
   */
  const handleFileSelect = useCallback(async (selectedFile: File) => {
    if (disabled) return;
    
    // Validate file
    const result = await validate(selectedFile);
    setValidationResult(result);
    
    if (result.isValid && result.fileInfo) {
      // Update store with validated file
      setFile(selectedFile, result.fileInfo.preview ?? null, result.fileInfo);
      
      // Notify parent
      onFileSelect?.(selectedFile);
    }
  }, [disabled, validate, setValidationResult, setFile, onFileSelect]);
  
  /**
   * Handle drag events
   */
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (disabled) return;
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  }, [disabled, handleFileSelect]);
  
  /**
   * Handle input change
   */
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFileSelect(selectedFile);
    }
    // Reset input to allow re-selecting same file
    e.target.value = '';
  }, [handleFileSelect]);
  
  /**
   * Handle click to open file picker
   */
  const handleClick = useCallback(() => {
    if (!disabled && !file) {
      inputRef.current?.click();
    }
  }, [disabled, file]);
  
  /**
   * Handle keyboard activation
   */
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && !disabled && !file) {
      e.preventDefault();
      inputRef.current?.click();
    }
  }, [disabled, file]);
  
  /**
   * Handle remove file
   */
  const handleRemove = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    clearFile();
  }, [clearFile]);
  
  /**
   * Get file type icon
   */
  const getFileIcon = (mimeType: string) => {
    const category = getMimeCategory(mimeType);
    switch (category) {
      case 'video':
        return <FileVideo className="h-12 w-12 text-blue-500" />;
      case 'audio':
        return <FileAudio className="h-12 w-12 text-purple-500" />;
      case 'image':
        return <Image className="h-12 w-12 text-green-500" />;
      default:
        return <Upload className="h-12 w-12 text-muted-foreground" />;
    }
  };
  
  // ============== RENDER ==============
  
  // File selected state
  if (file && fileInfo) {
    return (
      <div
        data-testid="upload-zone-preview"
        className={cn(
          'relative rounded-lg border-2 border-dashed p-6 transition-colors',
          status === 'error' 
            ? 'border-red-500 bg-red-500/5' 
            : 'border-green-500 bg-green-500/5',
          className
        )}
      >
        <div className="flex items-start gap-4">
          {/* Preview or Icon */}
          <div className="flex-shrink-0">
            {preview && getMimeCategory(fileInfo.type) === 'image' ? (
              <img 
                src={preview} 
                alt="Preview" 
                className="h-20 w-20 rounded-md object-cover"
              />
            ) : preview && getMimeCategory(fileInfo.type) === 'video' ? (
              <video 
                src={preview} 
                className="h-20 w-20 rounded-md object-cover"
                muted
              />
            ) : (
              getFileIcon(fileInfo.type)
            )}
          </div>
          
          {/* File Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h4 className="font-medium text-foreground truncate">
                {fileInfo.name}
              </h4>
              {status === 'ready' && (
                <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              {formatFileSize(fileInfo.size)} • {fileInfo.type || 'Unknown type'}
            </p>
            
            {/* Validation Warnings */}
            {validationResult?.warnings && validationResult.warnings.length > 0 && (
              <div className="mt-2 flex items-start gap-1 text-yellow-600">
                <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                <span className="text-xs">{validationResult.warnings[0].message}</span>
              </div>
            )}
            
            {/* Error Message */}
            {errorMessage && (
              <div className="mt-2 flex items-start gap-1 text-red-500">
                <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                <span className="text-xs">{errorMessage}</span>
              </div>
            )}
          </div>
          
          {/* Remove Button */}
          <button
            type="button"
            onClick={handleRemove}
            className="flex-shrink-0 rounded-full p-1 hover:bg-muted transition-colors"
            aria-label="Remove file"
            data-testid="upload-zone-remove"
          >
            <X className="h-5 w-5 text-muted-foreground" />
          </button>
        </div>
      </div>
    );
  }
  
  // Empty/Drop state
  return (
    <div
      data-testid="upload-zone"
      role="button"
      tabIndex={disabled ? -1 : 0}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      aria-label="Upload media file for analysis"
      aria-disabled={disabled}
      className={cn(
        'relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-all cursor-pointer',
        'hover:border-primary hover:bg-primary/5',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
        isDragging && 'border-primary bg-primary/10 scale-[1.02]',
        disabled && 'cursor-not-allowed opacity-50',
        isValidating && 'pointer-events-none',
        errorMessage && 'border-red-500',
        className
      )}
    >
      {/* Hidden File Input */}
      <input
        ref={inputRef}
        type="file"
        accept={ALL_ACCEPTED_EXTENSIONS.join(',')}
        onChange={handleInputChange}
        disabled={disabled}
        className="sr-only"
        data-testid="upload-zone-input"
        aria-hidden="true"
      />
      
      {/* Loading State */}
      {isValidating ? (
        <div className="flex flex-col items-center gap-3">
          <div className="animate-spin rounded-full h-10 w-10 border-2 border-primary border-t-transparent" />
          <p className="text-sm text-muted-foreground">Validating file...</p>
        </div>
      ) : (
        <>
          {/* Icon */}
          <div className={cn(
            'mb-4 rounded-full p-3 transition-colors',
            isDragging ? 'bg-primary/20' : 'bg-muted'
          )}>
            <Upload className={cn(
              'h-8 w-8 transition-colors',
              isDragging ? 'text-primary' : 'text-muted-foreground'
            )} />
          </div>
          
          {/* Instructions */}
          <div className="text-center">
            <p className="text-lg font-medium text-foreground">
              {isDragging ? 'Drop your file here' : 'Drag & drop your media file'}
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              or <span className="text-primary font-medium">click to browse</span>
            </p>
          </div>
          
          {/* Accepted Formats */}
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            <span className="inline-flex items-center gap-1 rounded-full bg-blue-500/10 px-2.5 py-0.5 text-xs text-blue-600">
              <FileVideo className="h-3 w-3" />
              Video
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-purple-500/10 px-2.5 py-0.5 text-xs text-purple-600">
              <FileAudio className="h-3 w-3" />
              Audio
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-green-500/10 px-2.5 py-0.5 text-xs text-green-600">
              <Image className="h-3 w-3" />
              Image
            </span>
          </div>
          
          {/* Size Limit */}
          <p className="mt-3 text-xs text-muted-foreground">
            Max file size: {maxSizeMB}MB
          </p>
          
          {/* Error Display */}
          {errorMessage && (
            <div 
              className="mt-4 flex items-center gap-2 text-red-500"
              data-testid="upload-zone-error"
              role="alert"
            >
              <AlertCircle className="h-4 w-4" />
              <span className="text-sm">{errorMessage}</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default UploadZone;
