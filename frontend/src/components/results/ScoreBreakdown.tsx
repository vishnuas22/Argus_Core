/**
 * Argus Core - Score Breakdown Component
 * =======================================
 * Visual breakdown of scores per modality with D3 bar chart visualization.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/results/ScoreBreakdown.tsx
 * 
 * Role: Display visual breakdown of scores per modality with weighted contributions.
 * Shows how each analysis dimension (video spatial, temporal, audio, etc.) contributes
 * to the overall Trust Score.
 * 
 * Integration:
 * - Imports: d3, components/ui/*, types/analysis
 * - Used by: ResultsPanel.tsx, analysis/[id]/page.tsx, ModalityTabs.tsx
 * - Backend: Maps to ScoreBreakdown schema from backend/schemas/schemas.py
 * 
 * Features:
 * - Horizontal bar chart showing modality scores
 * - Weight indicators for each modality
 * - Color-coded scores based on verdict thresholds
 * - Interactive hover states for detailed information
 * - Animated transitions on mount and updates
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton bars
 * - Empty state: Shows message when no breakdown available
 * - Accessibility: ARIA labels, keyboard navigation
 * - data-testid: score-breakdown, score-breakdown-item, score-breakdown-bar
 */

'use client';

import React, { useMemo, useId } from 'react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { 
  Tooltip, 
  TooltipContent, 
  TooltipProvider, 
  TooltipTrigger 
} from '@/components/ui/tooltip';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Video, 
  AudioLines, 
  FileSearch2, 
  Clock, 
  Mic2,
  Info,
  type LucideIcon 
} from 'lucide-react';
import type { ScoreBreakdown as ScoreBreakdownType } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for ScoreBreakdown component
 */
export interface ScoreBreakdownProps {
  /** Score breakdown data from API */
  breakdown: ScoreBreakdownType;
  /** Layout variant */
  variant?: 'default' | 'compact' | 'detailed';
  /** Whether to animate bars on mount */
  animated?: boolean;
  /** Animation duration in ms */
  animationDuration?: number;
  /** Whether to show weight badges */
  showWeights?: boolean;
  /** Whether to show contribution values */
  showContributions?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Individual breakdown item data
 */
interface BreakdownItem {
  /** Unique key for the item */
  key: string;
  /** Display label */
  label: string;
  /** Short description */
  description: string;
  /** Score value (0-100) */
  score: number;
  /** Weight multiplier (0-1) */
  weight: number;
  /** Weighted contribution to final score */
  contribution: number;
  /** Icon component */
  icon: LucideIcon;
  /** Category grouping */
  category: 'video' | 'audio' | 'metadata';
}

// ============== CONSTANTS ==============

/**
 * Configuration for each modality type
 * Maps backend field names to display configuration
 */
const MODALITY_CONFIG: Record<string, {
  label: string;
  description: string;
  icon: LucideIcon;
  category: 'video' | 'audio' | 'metadata';
}> = {
  video_spatial: {
    label: 'Video (Spatial)',
    description: 'Per-frame face analysis for deepfake artifacts',
    icon: Video,
    category: 'video',
  },
  video_temporal: {
    label: 'Video (Temporal)',
    description: 'Consistency across frames and motion artifacts',
    icon: Clock,
    category: 'video',
  },
  video_lipsync: {
    label: 'Lip Sync',
    description: 'Audio-visual synchronization analysis',
    icon: Mic2,
    category: 'video',
  },
  audio: {
    label: 'Audio Analysis',
    description: 'Voice cloning and spectral anomaly detection',
    icon: AudioLines,
    category: 'audio',
  },
  metadata: {
    label: 'Metadata',
    description: 'C2PA provenance and EXIF integrity checks',
    icon: FileSearch2,
    category: 'metadata',
  },
};

/**
 * Category colors for visual grouping
 */
const CATEGORY_COLORS: Record<string, string> = {
  video: 'text-blue-500',
  audio: 'text-purple-500',
  metadata: 'text-slate-500',
};

// ============== MAIN COMPONENT ==============

/**
 * ScoreBreakdown Component
 * 
 * Displays a visual breakdown of how each analysis modality contributes
 * to the overall Trust Score. Shows scores as horizontal bars with
 * weight indicators and optional contribution values.
 * 
 * @example
 * ```tsx
 * // Basic usage
 * <ScoreBreakdown breakdown={trustScore.breakdown} />
 * 
 * // Detailed view with all options
 * <ScoreBreakdown 
 *   breakdown={breakdown}
 *   variant="detailed"
 *   showWeights
 *   showContributions
 *   animated
 * />
 * 
 * // Compact for cards
 * <ScoreBreakdown 
 *   breakdown={breakdown}
 *   variant="compact"
 *   showWeights={false}
 * />
 * ```
 */
export function ScoreBreakdown({
  breakdown,
  variant = 'default',
  animated = true,
  animationDuration = 700,
  showWeights = true,
  showContributions = false,
  className,
}: ScoreBreakdownProps) {
  const componentId = useId();
  
  // ============== PROCESS BREAKDOWN DATA ==============
  
  const items = useMemo<BreakdownItem[]>(() => {
    const result: BreakdownItem[] = [];
    
    // Process each possible modality
    const processModality = (key: string, score: number | undefined) => {
      if (score === undefined) return;
      
      const config = MODALITY_CONFIG[key];
      if (!config) return;
      
      const weight = breakdown.weights?.[key] || getDefaultWeight(key);
      
      // Score is 0-1 from backend, convert to 0-100 for display
      const scorePercent = typeof score === 'number' 
        ? (score <= 1 ? score * 100 : score) 
        : 0;
      
      result.push({
        key,
        label: config.label,
        description: config.description,
        score: scorePercent,
        weight,
        contribution: scorePercent * weight,
        icon: config.icon,
        category: config.category,
      });
    };
    
    // Process all modalities
    processModality('video_spatial', breakdown.video_spatial);
    processModality('video_temporal', breakdown.video_temporal);
    processModality('video_lipsync', breakdown.video_lipsync);
    processModality('audio', breakdown.audio);
    processModality('metadata', breakdown.metadata);
    
    // Sort by weight (most important first)
    return result.sort((a, b) => b.weight - a.weight);
  }, [breakdown]);
  
  // ============== EMPTY STATE ==============
  
  if (items.length === 0) {
    return (
      <div 
        className={cn('py-6 text-center', className)}
        data-testid="score-breakdown-empty"
      >
        <Info className="h-8 w-8 mx-auto mb-2 text-muted-foreground/50" />
        <p className="text-sm text-muted-foreground">
          No detailed score breakdown available
        </p>
      </div>
    );
  }
  
  // ============== COMPACT VARIANT ==============
  
  if (variant === 'compact') {
    return (
      <div 
        className={cn('space-y-2', className)}
        data-testid="score-breakdown"
        role="list"
        aria-label="Score breakdown by modality"
      >
        {items.map((item) => (
          <CompactBreakdownItem
            key={item.key}
            item={item}
            animated={animated}
            animationDuration={animationDuration}
          />
        ))}
      </div>
    );
  }
  
  // ============== DEFAULT / DETAILED VARIANT ==============
  
  return (
    <TooltipProvider>
      <div 
        className={cn('space-y-4', className)}
        data-testid="score-breakdown"
        role="list"
        aria-label="Score breakdown by modality"
      >
        {items.map((item, index) => (
          <BreakdownItemRow
            key={item.key}
            item={item}
            variant={variant}
            animated={animated}
            animationDuration={animationDuration}
            animationDelay={index * 100}
            showWeight={showWeights}
            showContribution={showContributions}
            componentId={`${componentId}-${item.key}`}
          />
        ))}
        
        {/* Legend for detailed variant */}
        {variant === 'detailed' && (
          <div className="pt-4 border-t mt-4">
            <p className="text-xs text-muted-foreground mb-2">Categories:</p>
            <div className="flex flex-wrap gap-3 text-xs">
              {Object.entries(CATEGORY_COLORS).map(([category, colorClass]) => {
                const hasCategory = items.some(i => i.category === category);
                if (!hasCategory) return null;
                
                return (
                  <div key={category} className="flex items-center gap-1.5">
                    <div className={cn('w-2 h-2 rounded-full', colorClass.replace('text-', 'bg-'))} />
                    <span className="capitalize text-muted-foreground">{category}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </TooltipProvider>
  );
}

// ============== BREAKDOWN ITEM ROW ==============

interface BreakdownItemRowProps {
  item: BreakdownItem;
  variant: 'default' | 'compact' | 'detailed';
  animated: boolean;
  animationDuration: number;
  animationDelay: number;
  showWeight: boolean;
  showContribution: boolean;
  componentId: string;
}

function BreakdownItemRow({
  item,
  variant,
  animated,
  animationDuration,
  animationDelay,
  showWeight,
  showContribution,
  componentId,
}: BreakdownItemRowProps) {
  const Icon = item.icon;
  const scoreColor = getScoreColor(item.score);
  const barColor = getScoreBarColor(item.score);
  
  return (
    <div
      className={cn(
        'group',
        animated && 'animate-in fade-in slide-in-from-left-2',
      )}
      style={animated ? { 
        animationDuration: `${animationDuration}ms`,
        animationDelay: `${animationDelay}ms`,
        animationFillMode: 'both',
      } : undefined}
      data-testid="score-breakdown-item"
      role="listitem"
      aria-labelledby={`${componentId}-label`}
    >
      {/* Header Row */}
      <div className="flex items-center justify-between mb-1.5">
        {/* Label with Icon */}
        <div className="flex items-center gap-2">
          <Icon 
            className={cn(
              'h-4 w-4 flex-shrink-0',
              CATEGORY_COLORS[item.category] || 'text-muted-foreground'
            )} 
            aria-hidden="true"
          />
          <span 
            id={`${componentId}-label`}
            className="text-sm font-medium"
          >
            {item.label}
          </span>
          
          {/* Tooltip for detailed variant */}
          {variant === 'detailed' && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-3 w-3 text-muted-foreground/50 cursor-help" />
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-xs">
                <p className="text-xs">{item.description}</p>
              </TooltipContent>
            </Tooltip>
          )}
        </div>
        
        {/* Score and Weight */}
        <div className="flex items-center gap-2">
          {showWeight && (
            <Badge 
              variant="outline" 
              className="text-xs font-normal px-1.5 py-0"
            >
              {(item.weight * 100).toFixed(0)}%
            </Badge>
          )}
          
          <span 
            className={cn(
              'text-sm font-semibold tabular-nums min-w-[2.5rem] text-right',
              scoreColor
            )}
            aria-label={`Score: ${item.score.toFixed(0)} out of 100`}
          >
            {item.score.toFixed(0)}
          </span>
          
          {showContribution && (
            <span className="text-xs text-muted-foreground tabular-nums">
              (+{item.contribution.toFixed(1)})
            </span>
          )}
        </div>
      </div>
      
      {/* Progress Bar */}
      <div 
        className="relative h-2 bg-muted rounded-full overflow-hidden"
        data-testid="score-breakdown-bar"
      >
        <div
          className={cn(
            'absolute inset-y-0 left-0 rounded-full transition-all ease-out',
            barColor,
            'group-hover:brightness-110'
          )}
          style={{
            width: animated ? '0%' : `${Math.min(100, Math.max(0, item.score))}%`,
            transitionDuration: `${animationDuration}ms`,
            transitionDelay: `${animationDelay}ms`,
            animation: animated 
              ? `score-bar-fill ${animationDuration}ms ease-out ${animationDelay}ms forwards`
              : undefined,
            ['--target-width' as string]: `${Math.min(100, Math.max(0, item.score))}%`,
          }}
          role="progressbar"
          aria-valuenow={item.score}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
      
      {/* CSS for animation */}
      <style jsx>{`
        @keyframes score-bar-fill {
          from { width: 0%; }
          to { width: var(--target-width); }
        }
      `}</style>
    </div>
  );
}

// ============== COMPACT BREAKDOWN ITEM ==============

interface CompactBreakdownItemProps {
  item: BreakdownItem;
  animated: boolean;
  animationDuration: number;
}

function CompactBreakdownItem({
  item,
  animated,
  animationDuration,
}: CompactBreakdownItemProps) {
  const Icon = item.icon;
  const scoreColor = getScoreColor(item.score);
  
  return (
    <div 
      className="flex items-center gap-2"
      data-testid="score-breakdown-item"
      role="listitem"
    >
      <Icon 
        className={cn(
          'h-3.5 w-3.5 flex-shrink-0',
          CATEGORY_COLORS[item.category] || 'text-muted-foreground'
        )} 
        aria-hidden="true"
      />
      <span className="text-xs flex-1 truncate">{item.label}</span>
      <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full',
            getScoreBarColor(item.score)
          )}
          style={{ 
            width: `${Math.min(100, Math.max(0, item.score))}%`,
            transition: animated ? `width ${animationDuration}ms ease-out` : undefined,
          }}
        />
      </div>
      <span 
        className={cn('text-xs font-medium tabular-nums', scoreColor)}
      >
        {item.score.toFixed(0)}
      </span>
    </div>
  );
}

// ============== SKELETON LOADER ==============

/**
 * Skeleton loader for ScoreBreakdown
 */
export function ScoreBreakdownSkeleton({
  itemCount = 4,
  variant = 'default',
  className,
}: {
  itemCount?: number;
  variant?: 'default' | 'compact' | 'detailed';
  className?: string;
}) {
  if (variant === 'compact') {
    return (
      <div className={cn('space-y-2', className)} data-testid="score-breakdown-skeleton">
        {Array.from({ length: itemCount }).map((_, i) => (
          <div key={i} className="flex items-center gap-2 animate-pulse">
            <Skeleton className="h-3.5 w-3.5 rounded" />
            <Skeleton className="h-3 flex-1" />
            <Skeleton className="h-1.5 w-16 rounded-full" />
            <Skeleton className="h-3 w-6" />
          </div>
        ))}
      </div>
    );
  }
  
  return (
    <div className={cn('space-y-4', className)} data-testid="score-breakdown-skeleton">
      {Array.from({ length: itemCount }).map((_, i) => (
        <div key={i} className="space-y-1.5 animate-pulse">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Skeleton className="h-4 w-4 rounded" />
              <Skeleton className="h-4 w-24" />
            </div>
            <div className="flex items-center gap-2">
              <Skeleton className="h-5 w-10 rounded" />
              <Skeleton className="h-4 w-8" />
            </div>
          </div>
          <Skeleton className="h-2 w-full rounded-full" />
        </div>
      ))}
    </div>
  );
}

// ============== UTILITY FUNCTIONS ==============

/**
 * Get default weight for a modality if not provided
 * Matches SCORE_WEIGHT_* from backend config.py
 */
function getDefaultWeight(key: string): number {
  const defaults: Record<string, number> = {
    video_spatial: 0.30,
    video_temporal: 0.25,
    video_lipsync: 0.10,
    audio: 0.20,
    metadata: 0.15,
  };
  return defaults[key] || 0.1;
}

/**
 * Get text color class based on score
 * Matches verdict thresholds from backend config.py
 */
function getScoreColor(score: number): string {
  if (score >= 80) return 'text-green-600 dark:text-green-400';
  if (score >= 60) return 'text-lime-600 dark:text-lime-400';
  if (score >= 40) return 'text-yellow-600 dark:text-yellow-400';
  if (score >= 20) return 'text-orange-600 dark:text-orange-400';
  return 'text-red-600 dark:text-red-400';
}

/**
 * Get bar background color class based on score
 */
function getScoreBarColor(score: number): string {
  if (score >= 80) return 'bg-green-500';
  if (score >= 60) return 'bg-lime-500';
  if (score >= 40) return 'bg-yellow-500';
  if (score >= 20) return 'bg-orange-500';
  return 'bg-red-500';
}

// ============== EXPORTS ==============

export default ScoreBreakdown;
