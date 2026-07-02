/**
 * Argus Core - Confidence Interval Component
 * ==========================================
 * Displays statistical confidence interval with visual representation.
 * 
 * Implements: XAI_FRONTEND_IMPLEMENTATION.md - Section 4.3 - components/xai/ConfidenceInterval.tsx
 * 
 * Role: Visualize the confidence interval for predictions,
 * showing the range of uncertainty in the AI's decision.
 * 
 * Integration:
 * - Used by: XAIExplanationPanel
 * - Data: confidence_interval from XAI explanation
 */

'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import { Info, Minus, Plus } from 'lucide-react';

// UI Components
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// ============== TYPES ==============

/**
 * Props for ConfidenceInterval component
 */
export interface ConfidenceIntervalProps {
  /** Lower bound of the interval (0-1) */
  lower: number;
  /** Upper bound of the interval (0-1) */
  upper: number;
  /** Label for the interval */
  label?: string;
  /** Description of what the interval represents */
  description?: string;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show as a compact inline display */
  compact?: boolean;
  /** Confidence level (e.g., 95 for 95% CI) */
  confidenceLevel?: number;
}

// ============== MAIN COMPONENT ==============

/**
 * ConfidenceInterval
 * 
 * Displays a confidence interval with visual bar representation.
 * 
 * @example
 * ```tsx
 * <ConfidenceInterval
 *   lower={0.72}
 *   upper={0.88}
 *   label="Prediction Confidence"
 *   confidenceLevel={95}
 * />
 * ```
 */
export function ConfidenceInterval({
  lower,
  upper,
  label = 'Confidence Interval',
  description,
  className,
  compact = false,
  confidenceLevel = 95,
}: ConfidenceIntervalProps): React.ReactElement {
  // Calculate midpoint and range
  const midpoint = (lower + upper) / 2;
  const range = upper - lower;
  const lowerPercent = lower * 100;
  const upperPercent = upper * 100;

  // Determine confidence quality
  const getQualityInfo = () => {
    if (range <= 0.1) {
      return { color: 'text-green-500', label: 'High confidence', bgClass: 'bg-green-500' };
    } else if (range <= 0.2) {
      return { color: 'text-yellow-500', label: 'Moderate confidence', bgClass: 'bg-yellow-500' };
    } else {
      return { color: 'text-destructive', label: 'Low confidence', bgClass: 'bg-destructive' };
    }
  };

  const quality = getQualityInfo();

  // Compact inline display
  if (compact) {
    return (
      <div
        className={cn('inline-flex items-center gap-2', className)}
        data-testid="confidence-interval-compact"
      >
        <span className="text-sm text-muted-foreground">{label}:</span>
        <span className={cn('font-medium', quality.color)}>
          {lowerPercent.toFixed(1)}% - {upperPercent.toFixed(1)}%
        </span>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-3 w-3 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent>
              <p>{confidenceLevel}% confidence interval</p>
              <p className="text-xs text-muted-foreground">{quality.label}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    );
  }

  // Full display
  return (
    <Card className={cn('w-full', className)} data-testid="confidence-interval">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">{label}</CardTitle>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className={cn('flex items-center gap-1', quality.color)}>
                  <span className="text-xs">{quality.label}</span>
                  <Info className="h-3 w-3" />
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Interval width: {(range * 100).toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground">
                  Narrower intervals indicate higher certainty
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        {/* Visual bar */}
        <div className="relative h-8 bg-muted rounded-md overflow-hidden">
          {/* Full range background */}
          <div className="absolute inset-0 flex items-center">
            <div className="w-full h-1 bg-muted-foreground/20" />
          </div>
          
          {/* Confidence interval bar */}
          <div
            className={cn('absolute h-full opacity-30', quality.bgClass)}
            style={{
              left: `${lowerPercent}%`,
              width: `${range * 100}%`,
            }}
          />
          
          {/* Interval markers */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-foreground"
            style={{ left: `${lowerPercent}%` }}
          />
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-foreground"
            style={{ left: `${upperPercent}%` }}
          />
          
          {/* Midpoint marker */}
          <div
            className="absolute top-1 bottom-1 w-1 rounded-full bg-primary"
            style={{ left: `${midpoint * 100}%`, transform: 'translateX(-50%)' }}
          />
          
          {/* Labels */}
          <div
            className="absolute top-1 text-xs font-medium"
            style={{ left: `${lowerPercent}%`, transform: 'translateX(-50%)' }}
          >
            {lowerPercent.toFixed(0)}%
          </div>
          <div
            className="absolute bottom-1 text-xs font-medium"
            style={{ left: `${upperPercent}%`, transform: 'translateX(-50%)' }}
          >
            {upperPercent.toFixed(0)}%
          </div>
        </div>

        {/* Stats */}
        <div className="flex justify-between mt-3 text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <Minus className="h-3 w-3" />
            <span>Lower: {lowerPercent.toFixed(1)}%</span>
          </div>
          <div className="font-medium">
            Mid: {(midpoint * 100).toFixed(1)}%
          </div>
          <div className="flex items-center gap-1 text-muted-foreground">
            <span>Upper: {upperPercent.toFixed(1)}%</span>
            <Plus className="h-3 w-3" />
          </div>
        </div>

        {/* Confidence level badge */}
        <div className="mt-2 text-xs text-muted-foreground text-center">
          {confidenceLevel}% Confidence Interval
        </div>
      </CardContent>
    </Card>
  );
}

// ============== SKELETON COMPONENT ==============

/**
 * Skeleton component for ConfidenceInterval loading state
 */
export function ConfidenceIntervalSkeleton({
  className,
  compact = false,
}: {
  className?: string;
  compact?: boolean;
}): React.ReactElement {
  if (compact) {
    return (
      <div className={cn('inline-flex items-center gap-2', className)}>
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-20" />
      </div>
    );
  }

  return (
    <Card className={cn('w-full', className)} data-testid="confidence-interval-skeleton">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-24" />
        </div>
      </CardHeader>
      <CardContent>
        <Skeleton className="h-8 w-full rounded-md" />
        <div className="flex justify-between mt-3">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-20" />
        </div>
      </CardContent>
    </Card>
  );
}

export default ConfidenceInterval;
