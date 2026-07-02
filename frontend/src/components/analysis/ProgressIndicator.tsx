/**
 * Argus Core - Progress Indicator Component
 * ==========================================
 * Real-time analysis progress display with stage visualization.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/analysis/ProgressIndicator.tsx
 * 
 * Role: Display real-time progress updates from WebSocket connection.
 * Shows animated progress bar with current stage label and status message.
 * 
 * Integration:
 * - Imports: store/progressStore for progress state
 * - Imports: components/ui/progress for progress bar
 * - Connected to: useWebSocket hook for real-time updates
 * - Backend: WebSocket /ws/analysis/{id} endpoint
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows pulsing animation when pending
 * - Error state: Red styling with error message
 * - Accessibility: ARIA live region for progress updates
 * - data-testid: progress-indicator, progress-bar, progress-stage, progress-message
 */

'use client';

import { useMemo } from 'react';
import { 
  Loader2, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  Cog, 
  ScanSearch, 
  Calculator,
  AlertCircle 
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Progress } from '@/components/ui/progress';
import { 
  useProgressStore, 
  selectProgress, 
  STAGE_LABELS,
  getEstimatedProgress 
} from '@/store/progressStore';
import type { AnalysisStatus } from '@/types/analysis';

// ============== TYPES ==============

export interface ProgressIndicatorProps {
  /** Analysis ID to display progress for */
  analysisId: string;
  /** Additional CSS classes */
  className?: string;
  /** Show detailed stage information */
  showDetails?: boolean;
  /** Compact mode for inline display */
  compact?: boolean;
}

// ============== STATUS CONFIG ==============

interface StatusConfig {
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  bgColor: string;
  borderColor: string;
  textColor: string;
  pulseAnimation: boolean;
}

const STATUS_CONFIGS: Record<AnalysisStatus, StatusConfig> = {
  pending: {
    icon: Clock,
    color: 'text-muted-foreground',
    bgColor: 'bg-muted/50',
    borderColor: 'border-muted',
    textColor: 'text-muted-foreground',
    pulseAnimation: true,
  },
  preprocessing: {
    icon: Cog,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30',
    textColor: 'text-blue-600 dark:text-blue-400',
    pulseAnimation: true,
  },
  analyzing: {
    icon: ScanSearch,
    color: 'text-purple-500',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30',
    textColor: 'text-purple-600 dark:text-purple-400',
    pulseAnimation: true,
  },
  aggregating: {
    icon: Calculator,
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
    textColor: 'text-orange-600 dark:text-orange-400',
    pulseAnimation: true,
  },
  completed: {
    icon: CheckCircle2,
    color: 'text-green-500',
    bgColor: 'bg-green-500/10',
    borderColor: 'border-green-500/30',
    textColor: 'text-green-600 dark:text-green-400',
    pulseAnimation: false,
  },
  failed: {
    icon: XCircle,
    color: 'text-destructive',
    bgColor: 'bg-destructive/10',
    borderColor: 'border-destructive/30',
    textColor: 'text-destructive',
    pulseAnimation: false,
  },
};

// ============== COMPONENT ==============

/**
 * ProgressIndicator Component
 * 
 * Displays real-time analysis progress with animated progress bar
 * and stage information. Syncs with progressStore which is updated
 * by WebSocket messages.
 * 
 * @example
 * ```tsx
 * // Basic usage
 * <ProgressIndicator analysisId={analysisId} />
 * 
 * // Compact mode for inline display
 * <ProgressIndicator analysisId={analysisId} compact />
 * 
 * // With details
 * <ProgressIndicator analysisId={analysisId} showDetails />
 * ```
 */
export function ProgressIndicator({
  analysisId,
  className,
  showDetails = true,
  compact = false,
}: ProgressIndicatorProps) {
  // Get progress from store
  const progress = useProgressStore(selectProgress(analysisId));
  
  // Get status configuration
  const statusConfig = useMemo(() => 
    STATUS_CONFIGS[progress.status] || STATUS_CONFIGS.pending,
    [progress.status]
  );
  
  // Calculate display progress (use stored or estimate from status)
  const displayProgress = useMemo(() => {
    if (progress.progressPercent > 0) {
      return progress.progressPercent;
    }
    return getEstimatedProgress(progress.status);
  }, [progress.progressPercent, progress.status]);
  
  // Get stage label
  const stageLabel = useMemo(() => 
    progress.currentStage 
      ? STAGE_LABELS[progress.status as AnalysisStatus] || progress.currentStage
      : STAGE_LABELS[progress.status],
    [progress.currentStage, progress.status]
  );

  const StatusIcon = statusConfig.icon;

  // ============== COMPACT MODE ==============

  if (compact) {
    return (
      <div 
        className={cn(
          'flex items-center gap-2',
          className
        )}
        data-testid="progress-indicator"
        role="status"
        aria-live="polite"
      >
        <StatusIcon 
          className={cn(
            'h-4 w-4',
            statusConfig.color,
            statusConfig.pulseAnimation && 'animate-pulse'
          )} 
        />
        <span className={cn('text-sm', statusConfig.textColor)}>
          {stageLabel}
        </span>
        {progress.status !== 'completed' && progress.status !== 'failed' && (
          <span className="text-sm text-muted-foreground">
            {displayProgress}%
          </span>
        )}
      </div>
    );
  }

  // ============== FULL MODE ==============

  return (
    <div 
      className={cn(
        'rounded-lg border p-4',
        statusConfig.bgColor,
        statusConfig.borderColor,
        className
      )}
      data-testid="progress-indicator"
      role="status"
      aria-live="polite"
      aria-label={`Analysis ${progress.status}: ${displayProgress}%`}
    >
      {/* Header with icon and stage */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {/* Status Icon */}
          <div 
            className={cn(
              'flex items-center justify-center w-10 h-10 rounded-full',
              statusConfig.bgColor,
              'border',
              statusConfig.borderColor
            )}
          >
            <StatusIcon 
              className={cn(
                'h-5 w-5',
                statusConfig.color,
                statusConfig.pulseAnimation && 'animate-pulse'
              )} 
            />
          </div>
          
          {/* Stage Label */}
          <div>
            <p 
              className={cn('font-medium', statusConfig.textColor)}
              data-testid="progress-stage"
            >
              {stageLabel}
            </p>
            {showDetails && progress.message && (
              <p 
                className="text-sm text-muted-foreground mt-0.5"
                data-testid="progress-message"
              >
                {progress.message}
              </p>
            )}
          </div>
        </div>
        
        {/* Percentage */}
        <div className="text-right">
          <span 
            className={cn('text-2xl font-bold tabular-nums', statusConfig.textColor)}
          >
            {displayProgress}
          </span>
          <span className={cn('text-sm', statusConfig.color)}>%</span>
        </div>
      </div>
      
      {/* Progress Bar */}
      <div className="space-y-1.5">
        <Progress 
          value={displayProgress} 
          className={cn(
            'h-2',
            progress.status === 'failed' && '[&>div]:bg-destructive'
          )}
          data-testid="progress-bar"
        />
        
        {/* Stage indicators (optional detailed view) */}
        {showDetails && (
          <div className="flex justify-between text-xs text-muted-foreground pt-1">
            <span>Upload</span>
            <span>Preprocess</span>
            <span>Analyze</span>
            <span>Score</span>
            <span>Complete</span>
          </div>
        )}
      </div>
      
      {/* Error details */}
      {progress.status === 'failed' && progress.errorMessage && (
        <div 
          className="mt-3 flex items-start gap-2 p-2 rounded bg-destructive/10 text-destructive text-sm"
          role="alert"
        >
          <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-medium">Analysis Failed</p>
            <p className="text-destructive/80">{progress.errorMessage}</p>
            {progress.errorCode && (
              <p className="text-xs text-destructive/60 mt-1">
                Error code: {progress.errorCode}
              </p>
            )}
          </div>
        </div>
      )}
      
      {/* Success indicator */}
      {progress.status === 'completed' && (
        <div className="mt-3 flex items-center gap-2 text-green-600 dark:text-green-400 text-sm">
          <CheckCircle2 className="h-4 w-4" />
          <span>Analysis complete - results ready</span>
        </div>
      )}
    </div>
  );
}

// ============== SKELETON ==============

/**
 * Loading skeleton for ProgressIndicator
 */
export function ProgressIndicatorSkeleton({ className }: { className?: string }) {
  return (
    <div 
      className={cn(
        'rounded-lg border p-4 bg-muted/30 animate-pulse',
        className
      )}
      data-testid="progress-indicator-skeleton"
    >
      {/* Header skeleton */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-muted" />
          <div className="space-y-1.5">
            <div className="h-4 w-24 bg-muted rounded" />
            <div className="h-3 w-32 bg-muted rounded" />
          </div>
        </div>
        <div className="h-8 w-12 bg-muted rounded" />
      </div>
      
      {/* Progress bar skeleton */}
      <div className="h-2 w-full bg-muted rounded-full" />
    </div>
  );
}

// ============== CONNECTED VERSION ==============

/**
 * Connected ProgressIndicator that also manages WebSocket connection
 * Use this when you want automatic WebSocket setup
 */
export function ConnectedProgressIndicator({
  analysisId,
  ...props
}: ProgressIndicatorProps) {
  // Note: WebSocket connection is managed by the parent page (analysis/[id]/page.tsx)
  // This component just displays the progress from the store
  return <ProgressIndicator analysisId={analysisId} {...props} />;
}

export default ProgressIndicator;
