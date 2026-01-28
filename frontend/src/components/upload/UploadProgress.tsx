/**
 * Argus Core - Upload Progress Component
 * =======================================
 * Upload progress indicator with stages and estimated time.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/upload/UploadProgress.tsx
 * 
 * Role: Display upload progress with visual feedback. Show upload stage,
 * progress percentage, and provide cancel option.
 * 
 * Integration:
 * - Imports: store/uploadStore, components/ui/progress
 * - Used by: AnalysisForm.tsx, UploadZone.tsx
 * - State: Reads from uploadStore
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Animated progress bar
 * - Error state: Error message with retry option
 * - Accessibility: Progress announcements for screen readers
 * - data-testid: upload-progress, upload-progress-bar, upload-progress-cancel
 */

'use client';

import { useEffect, useState, useCallback } from 'react';
import { Loader2, X, CheckCircle, AlertCircle, Upload } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useUploadStore, type UploadStatus } from '@/store/uploadStore';

// ============== TYPES ==============

export interface UploadProgressProps {
  /** Current upload progress (0-100) */
  progress?: number;
  /** Current upload status */
  status?: UploadStatus;
  /** Error message if status is 'error' */
  error?: string | null;
  /** Callback when cancel is clicked */
  onCancel?: () => void;
  /** Callback when retry is clicked */
  onRetry?: () => void;
  /** Show as compact inline version */
  compact?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============== STAGE CONFIG ==============

interface StageConfig {
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
}

const STAGE_CONFIG: Record<UploadStatus, StageConfig> = {
  idle: {
    label: 'Ready',
    description: 'Select a file to begin',
    icon: Upload,
    color: 'text-muted-foreground',
  },
  validating: {
    label: 'Validating',
    description: 'Checking file format and size...',
    icon: Loader2,
    color: 'text-blue-500',
  },
  uploading: {
    label: 'Uploading',
    description: 'Transferring file to server...',
    icon: Loader2,
    color: 'text-primary',
  },
  processing: {
    label: 'Processing',
    description: 'Server is processing your file...',
    icon: Loader2,
    color: 'text-yellow-500',
  },
  complete: {
    label: 'Complete',
    description: 'Upload successful!',
    icon: CheckCircle,
    color: 'text-green-500',
  },
  error: {
    label: 'Error',
    description: 'Upload failed',
    icon: AlertCircle,
    color: 'text-destructive',
  },
};

// ============== COMPONENT ==============

export function UploadProgress({
  progress: propProgress,
  status: propStatus,
  error: propError,
  onCancel,
  onRetry,
  compact = false,
  className,
}: UploadProgressProps) {
  // Use props or store
  const storeState = useUploadStore();
  const progress = propProgress ?? storeState.uploadProgress;
  const status = propStatus ?? storeState.status;
  const error = propError ?? storeState.error;

  // Get current stage config
  const stage = STAGE_CONFIG[status];
  const StageIcon = stage.icon;
  const isAnimating = status === 'uploading' || status === 'validating' || status === 'processing';
  const isComplete = status === 'complete';
  const isError = status === 'error';

  // ============== COMPACT VERSION ==============

  if (compact) {
    return (
      <div
        data-testid="upload-progress"
        className={cn('flex items-center gap-2', className)}
      >
        <StageIcon 
          className={cn(
            'h-4 w-4',
            stage.color,
            isAnimating && 'animate-spin'
          )} 
        />
        <div className="flex-1 min-w-0">
          <Progress 
            value={progress} 
            className="h-1.5"
            data-testid="upload-progress-bar"
          />
        </div>
        <span className="text-xs text-muted-foreground tabular-nums">
          {progress}%
        </span>
        {onCancel && isAnimating && (
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={onCancel}
            data-testid="upload-progress-cancel"
            aria-label="Cancel upload"
          >
            <X className="h-3 w-3" />
          </Button>
        )}
      </div>
    );
  }

  // ============== FULL VERSION ==============

  return (
    <div
      data-testid="upload-progress"
      className={cn(
        'rounded-lg border bg-card p-4 space-y-4',
        isComplete && 'border-green-500/30 bg-green-500/5',
        isError && 'border-destructive/30 bg-destructive/5',
        className
      )}
      role="status"
      aria-live="polite"
      aria-label={`Upload status: ${stage.label}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={cn(
            'p-2 rounded-full',
            isComplete && 'bg-green-500/10',
            isError && 'bg-destructive/10',
            isAnimating && 'bg-primary/10'
          )}>
            <StageIcon 
              className={cn(
                'h-5 w-5',
                stage.color,
                isAnimating && 'animate-spin'
              )}
            />
          </div>
          <div>
            <h4 className="font-medium text-sm text-foreground">
              {stage.label}
            </h4>
            <p className="text-xs text-muted-foreground">
              {isError && error ? error : stage.description}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {onCancel && isAnimating && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onCancel}
              data-testid="upload-progress-cancel"
            >
              Cancel
            </Button>
          )}
          {onRetry && isError && (
            <Button
              variant="outline"
              size="sm"
              onClick={onRetry}
              data-testid="upload-progress-retry"
            >
              Retry
            </Button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      {status !== 'idle' && status !== 'error' && (
        <div className="space-y-2">
          <Progress 
            value={progress}
            className={cn(
              'h-2',
              isComplete && '[&>div]:bg-green-500'
            )}
            data-testid="upload-progress-bar"
          />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {isComplete ? 'Completed' : `${progress}% uploaded`}
            </span>
            {isAnimating && progress > 0 && progress < 100 && (
              <span className="tabular-nums">
                {/* Estimated time - simplified calculation */}
                {Math.ceil((100 - progress) / 10)}s remaining
              </span>
            )}
          </div>
        </div>
      )}

      {/* Screen reader announcement */}
      <span className="sr-only" aria-live="assertive">
        {isComplete && 'Upload complete'}
        {isError && `Upload failed: ${error || 'Unknown error'}`}
        {isAnimating && `Uploading: ${progress} percent complete`}
      </span>
    </div>
  );
}

// ============== MINIMAL PROGRESS ==============

/**
 * Minimal progress indicator for inline use
 */
export function UploadProgressMinimal({
  progress,
  className,
}: {
  progress: number;
  className?: string;
}) {
  return (
    <div 
      className={cn('flex items-center gap-2', className)}
      data-testid="upload-progress-minimal"
    >
      <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
      <Progress value={progress} className="h-1 flex-1" />
      <span className="text-xs text-muted-foreground tabular-nums min-w-[3ch]">
        {progress}%
      </span>
    </div>
  );
}

// ============== CONNECTED VERSION ==============

/**
 * Upload progress connected to store
 * Use this when you want automatic state management
 */
export function ConnectedUploadProgress({
  onCancel,
  onRetry,
  compact = false,
  className,
}: Omit<UploadProgressProps, 'progress' | 'status' | 'error'>) {
  const { uploadProgress, status, error, reset } = useUploadStore();

  const handleCancel = useCallback(() => {
    reset();
    onCancel?.();
  }, [reset, onCancel]);

  const handleRetry = useCallback(() => {
    reset();
    onRetry?.();
  }, [reset, onRetry]);

  // Don't render in idle state unless there's something to show
  if (status === 'idle') {
    return null;
  }

  return (
    <UploadProgress
      progress={uploadProgress}
      status={status}
      error={error}
      onCancel={handleCancel}
      onRetry={handleRetry}
      compact={compact}
      className={className}
    />
  );
}

export default UploadProgress;
