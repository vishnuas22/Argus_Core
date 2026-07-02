/**
 * Argus Core - Analysis Timeline Component
 * =========================================
 * Visual timeline of analysis pipeline stages with status indicators.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/analysis/AnalysisTimeline.tsx
 * 
 * Role: Display horizontal timeline showing all analysis pipeline stages
 * with visual indicators for pending, active, completed, and error states.
 * 
 * Integration:
 * - Imports: store/progressStore for progress state
 * - Connected to: WebSocket updates via progressStore
 * - Backend: Analysis status from /ws/analysis/{id}
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: All stages show pending
 * - Error state: Failed stage highlighted in red
 * - Accessibility: ARIA landmarks and stage descriptions
 * - data-testid: analysis-timeline, timeline-stage-{id}
 */

'use client';

import { useMemo } from 'react';
import { 
  Upload, 
  Cog, 
  ScanSearch, 
  Calculator, 
  CheckCircle2,
  XCircle,
  Circle,
  Loader2
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { 
  useProgressStore, 
  selectProgress 
} from '@/store/progressStore';
import type { AnalysisStatus } from '@/types/analysis';

// ============== TYPES ==============

export interface AnalysisTimelineProps {
  /** Analysis ID to display timeline for */
  analysisId: string;
  /** Additional CSS classes */
  className?: string;
  /** Vertical orientation (default: horizontal) */
  vertical?: boolean;
  /** Show duration estimates */
  showEstimates?: boolean;
  /** Compact mode */
  compact?: boolean;
}

/**
 * Stage status for visual representation
 */
type StageStatus = 'pending' | 'active' | 'completed' | 'error';

/**
 * Pipeline stage definition
 */
interface PipelineStage {
  id: string;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  estimatedDuration: string;
  /** Maps to analysis status values that indicate this stage */
  statusMap: AnalysisStatus[];
}

// ============== PIPELINE STAGES ==============

const PIPELINE_STAGES: PipelineStage[] = [
  {
    id: 'upload',
    label: 'Upload',
    description: 'File uploaded and validated',
    icon: Upload,
    estimatedDuration: '< 5s',
    statusMap: [], // Upload is always completed when we reach this view
  },
  {
    id: 'preprocess',
    label: 'Preprocessing',
    description: 'Extracting frames and audio',
    icon: Cog,
    estimatedDuration: '5-15s',
    statusMap: ['preprocessing'],
  },
  {
    id: 'analyze',
    label: 'Analysis',
    description: 'Running detection models',
    icon: ScanSearch,
    estimatedDuration: '10-30s',
    statusMap: ['analyzing'],
  },
  {
    id: 'aggregate',
    label: 'Scoring',
    description: 'Calculating trust score',
    icon: Calculator,
    estimatedDuration: '2-5s',
    statusMap: ['aggregating'],
  },
  {
    id: 'complete',
    label: 'Complete',
    description: 'Results ready',
    icon: CheckCircle2,
    estimatedDuration: '',
    statusMap: ['completed'],
  },
];

// ============== STATUS HELPERS ==============

/**
 * Determine stage status based on current analysis status
 */
function getStageStatus(
  stageIndex: number,
  currentStatus: AnalysisStatus
): StageStatus {
  // Get the index of the current stage based on status
  const statusToStageIndex: Record<AnalysisStatus, number> = {
    pending: 0,
    preprocessing: 1,
    analyzing: 2,
    aggregating: 3,
    completed: 4,
    failed: -1, // Special case
  };
  
  const currentStageIndex = statusToStageIndex[currentStatus];
  
  // Handle failed status
  if (currentStatus === 'failed') {
    // Mark all incomplete stages as error
    if (stageIndex > 0) return 'error';
    return 'completed'; // Upload was completed
  }
  
  // Compare stage index to current stage
  if (stageIndex < currentStageIndex) {
    return 'completed';
  } else if (stageIndex === currentStageIndex) {
    return 'active';
  } else {
    return 'pending';
  }
}

// ============== STAGE STYLES ==============

interface StageStyles {
  icon: string;
  iconBg: string;
  label: string;
  line: string;
}

const STAGE_STYLES: Record<StageStatus, StageStyles> = {
  pending: {
    icon: 'text-muted-foreground',
    iconBg: 'bg-muted border-muted-foreground/30',
    label: 'text-muted-foreground',
    line: 'bg-muted',
  },
  active: {
    icon: 'text-primary',
    iconBg: 'bg-primary/10 border-primary',
    label: 'text-primary font-medium',
    line: 'bg-primary/30',
  },
  completed: {
    icon: 'text-green-500',
    iconBg: 'bg-green-500/10 border-green-500',
    label: 'text-green-600 dark:text-green-400',
    line: 'bg-green-500',
  },
  error: {
    icon: 'text-destructive',
    iconBg: 'bg-destructive/10 border-destructive',
    label: 'text-destructive',
    line: 'bg-destructive/30',
  },
};

// ============== COMPONENT ==============

/**
 * AnalysisTimeline Component
 * 
 * Displays a visual timeline of the analysis pipeline with status
 * indicators for each stage. Updates in real-time via progressStore.
 * 
 * @example
 * ```tsx
 * // Horizontal timeline (default)
 * <AnalysisTimeline analysisId={analysisId} />
 * 
 * // Vertical timeline
 * <AnalysisTimeline analysisId={analysisId} vertical />
 * 
 * // With duration estimates
 * <AnalysisTimeline analysisId={analysisId} showEstimates />
 * ```
 */
export function AnalysisTimeline({
  analysisId,
  className,
  vertical = false,
  showEstimates = false,
  compact = false,
}: AnalysisTimelineProps) {
  // Get progress from store
  const progress = useProgressStore(selectProgress(analysisId));
  
  // Calculate stage statuses
  const stageStatuses = useMemo(() => 
    PIPELINE_STAGES.map((_, index) => 
      getStageStatus(index, progress.status)
    ),
    [progress.status]
  );

  // ============== VERTICAL TIMELINE ==============

  if (vertical) {
    return (
      <div 
        className={cn('space-y-0', className)}
        data-testid="analysis-timeline"
        role="list"
        aria-label="Analysis pipeline stages"
      >
        {PIPELINE_STAGES.map((stage, index) => {
          const status = stageStatuses[index];
          const styles = STAGE_STYLES[status];
          const isLast = index === PIPELINE_STAGES.length - 1;
          const StageIcon = status === 'active' ? Loader2 : stage.icon;
          
          return (
            <div 
              key={stage.id}
              className="relative flex gap-4"
              data-testid={`timeline-stage-${stage.id}`}
              role="listitem"
            >
              {/* Vertical line */}
              {!isLast && (
                <div 
                  className={cn(
                    'absolute left-5 top-10 w-0.5 h-full -ml-px',
                    styles.line
                  )}
                  aria-hidden="true"
                />
              )}
              
              {/* Icon */}
              <div 
                className={cn(
                  'relative z-10 flex items-center justify-center',
                  'w-10 h-10 rounded-full border-2',
                  styles.iconBg
                )}
              >
                <StageIcon 
                  className={cn(
                    'h-5 w-5',
                    styles.icon,
                    status === 'active' && 'animate-spin'
                  )} 
                />
              </div>
              
              {/* Content */}
              <div className="flex-1 pb-8">
                <p className={cn('font-medium', styles.label)}>
                  {stage.label}
                </p>
                {!compact && (
                  <p className="text-sm text-muted-foreground">
                    {stage.description}
                  </p>
                )}
                {showEstimates && stage.estimatedDuration && status !== 'completed' && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Est: {stage.estimatedDuration}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  // ============== HORIZONTAL TIMELINE ==============

  return (
    <div 
      className={cn('w-full', className)}
      data-testid="analysis-timeline"
      role="list"
      aria-label="Analysis pipeline stages"
    >
      <div className="flex items-center justify-between">
        {PIPELINE_STAGES.map((stage, index) => {
          const status = stageStatuses[index];
          const styles = STAGE_STYLES[status];
          const isLast = index === PIPELINE_STAGES.length - 1;
          const StageIcon = status === 'active' ? Loader2 : 
                           status === 'error' && index > 0 ? XCircle : stage.icon;
          
          return (
            <div 
              key={stage.id}
              className="flex items-center flex-1"
              data-testid={`timeline-stage-${stage.id}`}
              role="listitem"
            >
              {/* Stage indicator */}
              <div className="flex flex-col items-center">
                {/* Icon circle */}
                <div 
                  className={cn(
                    'flex items-center justify-center',
                    'rounded-full border-2 transition-all duration-300',
                    compact ? 'w-8 h-8' : 'w-10 h-10',
                    styles.iconBg
                  )}
                  title={`${stage.label}: ${status}`}
                >
                  <StageIcon 
                    className={cn(
                      compact ? 'h-4 w-4' : 'h-5 w-5',
                      styles.icon,
                      status === 'active' && 'animate-spin'
                    )} 
                  />
                </div>
                
                {/* Label */}
                <div className="mt-2 text-center">
                  <p 
                    className={cn(
                      'text-xs sm:text-sm whitespace-nowrap',
                      styles.label
                    )}
                  >
                    {stage.label}
                  </p>
                  {showEstimates && stage.estimatedDuration && status === 'active' && (
                    <p className="text-xs text-muted-foreground">
                      {stage.estimatedDuration}
                    </p>
                  )}
                </div>
              </div>
              
              {/* Connecting line */}
              {!isLast && (
                <div 
                  className={cn(
                    'flex-1 mx-2',
                    compact ? 'h-0.5' : 'h-1',
                    'rounded-full transition-all duration-500',
                    status === 'completed' ? 'bg-green-500' :
                    status === 'active' ? 'bg-gradient-to-r from-green-500 to-muted' :
                    'bg-muted'
                  )}
                  aria-hidden="true"
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============== SKELETON ==============

/**
 * Loading skeleton for AnalysisTimeline
 */
export function AnalysisTimelineSkeleton({ 
  className,
  vertical = false 
}: { 
  className?: string;
  vertical?: boolean;
}) {
  if (vertical) {
    return (
      <div className={cn('space-y-4', className)} data-testid="analysis-timeline-skeleton">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="flex gap-4">
            <div className="w-10 h-10 rounded-full bg-muted animate-pulse" />
            <div className="flex-1 space-y-2">
              <div className="h-4 w-24 bg-muted rounded animate-pulse" />
              <div className="h-3 w-32 bg-muted rounded animate-pulse" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={cn('w-full', className)} data-testid="analysis-timeline-skeleton">
      <div className="flex items-center justify-between">
        {[1, 2, 3, 4, 5].map((i, index) => (
          <div key={i} className="flex items-center flex-1">
            <div className="flex flex-col items-center">
              <div className="w-10 h-10 rounded-full bg-muted animate-pulse" />
              <div className="mt-2 h-4 w-16 bg-muted rounded animate-pulse" />
            </div>
            {index < 4 && (
              <div className="flex-1 h-1 mx-2 bg-muted rounded-full" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ============== SIMPLE VERSION ==============

/**
 * Simplified timeline showing just icons with tooltips
 */
export function SimpleTimeline({
  analysisId,
  className,
}: Pick<AnalysisTimelineProps, 'analysisId' | 'className'>) {
  const progress = useProgressStore(selectProgress(analysisId));
  
  const stageStatuses = useMemo(() => 
    PIPELINE_STAGES.map((_, index) => 
      getStageStatus(index, progress.status)
    ),
    [progress.status]
  );

  return (
    <div 
      className={cn('flex items-center gap-1', className)}
      data-testid="simple-timeline"
      role="list"
      aria-label="Analysis progress"
    >
      {PIPELINE_STAGES.map((stage, index) => {
        const status = stageStatuses[index];
        const isLast = index === PIPELINE_STAGES.length - 1;
        
        return (
          <div key={stage.id} className="flex items-center" role="listitem">
            <div 
              className={cn(
                'w-2 h-2 rounded-full transition-all duration-300',
                status === 'completed' && 'bg-green-500',
                status === 'active' && 'bg-primary animate-pulse',
                status === 'pending' && 'bg-muted',
                status === 'error' && 'bg-destructive'
              )}
              title={`${stage.label}: ${status}`}
            />
            {!isLast && (
              <div 
                className={cn(
                  'w-4 h-0.5',
                  status === 'completed' ? 'bg-green-500' : 'bg-muted'
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

export default AnalysisTimeline;
