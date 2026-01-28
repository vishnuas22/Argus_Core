/**
 * Argus Core - File Card Component
 * =================================
 * Display selected file with preview, metadata, and remove action.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/upload/FileCard.tsx
 * 
 * Role: Show file preview card after selection. Display thumbnail, filename,
 * size, type, and provide remove button.
 * 
 * Integration:
 * - Imports: lib/utils, lib/formatters
 * - Used by: UploadZone.tsx, AnalysisForm.tsx
 * - Store: uploadStore for file state
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton during preview generation
 * - Error state: Shows error icon if preview fails
 * - Accessibility: Keyboard accessible remove button
 * - data-testid: file-card, file-card-remove, file-card-preview
 */

'use client';

import { useMemo } from 'react';
import { X, FileVideo, FileAudio, Image as ImageIcon, FileText, AlertCircle } from 'lucide-react';
import { cn, formatFileSize, getMimeCategory } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import type { FileInfo } from '@/types/analysis';

// ============== TYPES ==============

export interface FileCardProps {
  /** File object */
  file: File;
  /** Preview URL (blob URL for images/videos) */
  preview: string | null;
  /** Callback when remove button is clicked */
  onRemove: () => void;
  /** Upload progress (0-100), shows progress bar if defined */
  uploadProgress?: number;
  /** Error message to display */
  error?: string;
  /** Additional file info from validation */
  fileInfo?: FileInfo | null;
  /** Disable interactions */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============== HELPER COMPONENTS ==============

/**
 * Get appropriate icon for file type
 */
function FileTypeIcon({ mimeType, className }: { mimeType: string; className?: string }) {
  const category = getMimeCategory(mimeType);
  
  switch (category) {
    case 'video':
      return <FileVideo className={cn('text-blue-500', className)} />;
    case 'audio':
      return <FileAudio className={cn('text-purple-500', className)} />;
    case 'image':
      return <ImageIcon className={cn('text-green-500', className)} />;
    case 'text':
      return <FileText className={cn('text-orange-500', className)} />;
    default:
      return <FileText className={cn('text-gray-500', className)} />;
  }
}

// ============== COMPONENT ==============

export function FileCard({
  file,
  preview,
  onRemove,
  uploadProgress,
  error,
  fileInfo,
  disabled = false,
  className,
}: FileCardProps) {
  // Derive file info
  const category = useMemo(() => getMimeCategory(file.type), [file.type]);
  const formattedSize = useMemo(() => formatFileSize(file.size), [file.size]);
  const isUploading = uploadProgress !== undefined && uploadProgress < 100;
  const isComplete = uploadProgress === 100;

  // Get display extension
  const extension = useMemo(() => {
    const ext = file.name.split('.').pop()?.toUpperCase() || '';
    return ext.length <= 4 ? ext : ext.substring(0, 4);
  }, [file.name]);

  return (
    <div
      data-testid="file-card"
      className={cn(
        'relative flex items-start gap-4 p-4 rounded-lg border bg-card',
        'transition-all duration-200',
        error && 'border-destructive bg-destructive/5',
        isComplete && !error && 'border-green-500/50 bg-green-500/5',
        !error && !isComplete && 'border-border',
        disabled && 'opacity-60 pointer-events-none',
        className
      )}
    >
      {/* Preview/Thumbnail */}
      <div 
        data-testid="file-card-preview"
        className={cn(
          'relative flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden',
          'bg-muted flex items-center justify-center'
        )}
      >
        {preview && (category === 'image' || category === 'video') ? (
          <>
            {category === 'image' ? (
              // Image preview
              <img
                src={preview}
                alt={`Preview of ${file.name}`}
                className="w-full h-full object-cover"
              />
            ) : (
              // Video preview (shows first frame)
              <video
                src={preview}
                className="w-full h-full object-cover"
                muted
                playsInline
              />
            )}
          </>
        ) : (
          // Icon fallback for audio/unknown
          <div className="flex flex-col items-center gap-1">
            <FileTypeIcon mimeType={file.type} className="h-8 w-8" />
            <span className="text-[10px] font-medium text-muted-foreground uppercase">
              {extension}
            </span>
          </div>
        )}

        {/* Category badge */}
        <div className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-black/60 text-white uppercase">
          {category}
        </div>
      </div>

      {/* File info */}
      <div className="flex-1 min-w-0 space-y-2">
        {/* Filename */}
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h4 
              className="font-medium text-sm truncate text-foreground"
              title={file.name}
            >
              {file.name}
            </h4>
            <p className="text-xs text-muted-foreground">
              {formattedSize} • {file.type || 'Unknown type'}
            </p>
          </div>

          {/* Remove button */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 flex-shrink-0 text-muted-foreground hover:text-foreground"
            onClick={onRemove}
            disabled={disabled || isUploading}
            data-testid="file-card-remove"
            aria-label={`Remove ${file.name}`}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Upload progress */}
        {uploadProgress !== undefined && (
          <div className="space-y-1">
            <Progress 
              value={uploadProgress} 
              className="h-1.5"
              aria-label="Upload progress"
            />
            <p className="text-xs text-muted-foreground">
              {isComplete ? 'Upload complete' : `Uploading... ${uploadProgress}%`}
            </p>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div 
            className="flex items-center gap-1.5 text-xs text-destructive"
            role="alert"
          >
            <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Success indicator (when complete without error) */}
        {isComplete && !error && (
          <div className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-500">
            <svg 
              className="h-3.5 w-3.5 flex-shrink-0" 
              viewBox="0 0 20 20" 
              fill="currentColor"
            >
              <path 
                fillRule="evenodd" 
                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" 
                clipRule="evenodd" 
              />
            </svg>
            <span>Ready for analysis</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ============== SKELETON ==============

/**
 * Loading skeleton for FileCard
 */
export function FileCardSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'flex items-start gap-4 p-4 rounded-lg border bg-card',
        'animate-pulse',
        className
      )}
    >
      {/* Preview skeleton */}
      <div className="flex-shrink-0 w-20 h-20 rounded-lg bg-muted" />
      
      {/* Content skeleton */}
      <div className="flex-1 space-y-2">
        <div className="h-4 w-3/4 bg-muted rounded" />
        <div className="h-3 w-1/2 bg-muted rounded" />
        <div className="h-1.5 w-full bg-muted rounded" />
      </div>
    </div>
  );
}

export default FileCard;
