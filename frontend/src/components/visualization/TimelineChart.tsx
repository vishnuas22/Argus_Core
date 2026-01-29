/**
 * Argus Core - Timeline Chart Component
 * ======================================
 * D3.js timeline chart showing temporal analysis with anomaly markers.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/visualization/TimelineChart.tsx
 * 
 * Role: Display per-frame scores over time with anomaly markers and click-to-seek.
 * Uses D3.js for smooth animations and interactive visualization.
 * 
 * Integration:
 * - Imports: lib/d3/timeline.ts (createTimelineChart factory)
 * - Used by: VideoAnalysisPanel.tsx, ModalityTabs.tsx
 * - Backend: Uses per_frame_scores, anomaly_indices from VideoResult
 * 
 * Features:
 * - Area chart showing score progression over time
 * - Anomaly markers with tooltips at low-score frames
 * - Threshold line at configurable level (default 50%)
 * - Click-to-seek frame selection
 * - Hover state with vertical indicator line
 * - Responsive design with automatic resize
 * - Smooth D3 animations
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton for chart loading
 * - Empty state: Shows message when no scores available
 * - Error state: Graceful degradation with fallback
 * - Accessibility: ARIA labels, keyboard interaction planned
 * - data-testid: timeline-chart, timeline-container, timeline-legend
 */

'use client';

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Activity,
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Info,
  type LucideIcon,
} from 'lucide-react';
import {
  createTimelineChart,
  TimelineInstance,
  TimelineConfig,
  TimelineData,
  DEFAULT_TIMELINE_CONFIG,
} from '@/lib/d3/timeline';

// ============== TYPES ==============

/**
 * Props for TimelineChart component
 */
export interface TimelineChartProps {
  /** Array of scores (0-1) per frame */
  scores: number[];
  /** Array of frame indices marked as anomalies */
  anomalyIndices: number[];
  /** Frames per second for time axis calculation */
  fps?: number;
  /** Timestamps in milliseconds (optional, overrides fps calculation) */
  timestamps?: number[];
  /** Callback when frame is selected */
  onFrameSelect?: (index: number) => void;
  /** Currently selected frame index */
  selectedFrame?: number;
  /** Chart height in pixels */
  height?: number;
  /** Show threshold line */
  showThreshold?: boolean;
  /** Threshold value (0-1) */
  threshold?: number;
  /** Show area fill under the line */
  showArea?: boolean;
  /** Enable animations */
  animated?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Compact mode */
  compact?: boolean;
  /** Show title and description */
  showHeader?: boolean;
  /** Custom title */
  title?: string;
  /** Custom description */
  description?: string;
}

// ============== CONSTANTS ==============

const MIN_HEIGHT = 120;
const MAX_HEIGHT = 400;
const DEFAULT_HEIGHT = 200;

// ============== MAIN COMPONENT ==============

/**
 * TimelineChart Component
 * 
 * Displays a D3.js timeline visualization of per-frame analysis scores
 * with anomaly markers and interactive frame selection.
 * 
 * @example
 * ```tsx
 * <TimelineChart
 *   scores={[0.9, 0.85, 0.3, 0.8, ...]}
 *   anomalyIndices={[2, 15, 28]}
 *   fps={30}
 *   onFrameSelect={(index) => setSelectedFrame(index)}
 *   showThreshold={true}
 *   threshold={0.5}
 * />
 * ```
 */
export function TimelineChart({
  scores,
  anomalyIndices,
  fps = 30,
  timestamps,
  onFrameSelect,
  selectedFrame,
  height = DEFAULT_HEIGHT,
  showThreshold = true,
  threshold = 0.5,
  showArea = true,
  animated = true,
  className,
  compact = false,
  showHeader = true,
  title = 'Frame Score Timeline',
  description,
}: TimelineChartProps) {
  // ============== REFS ==============
  
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const timelineRef = useRef<TimelineInstance | null>(null);
  
  // ============== STATE ==============
  
  const [containerWidth, setContainerWidth] = useState(0);
  const [isInitialized, setIsInitialized] = useState(false);
  const [hoveredFrame, setHoveredFrame] = useState<number | null>(null);

  // ============== COMPUTED VALUES ==============

  /**
   * Calculate timestamps from fps if not provided
   */
  const calculatedTimestamps = useMemo(() => {
    if (timestamps && timestamps.length === scores.length) {
      return timestamps;
    }
    return scores.map((_, i) => (i * 1000) / fps);
  }, [timestamps, scores, fps]);

  /**
   * Timeline data for D3 chart
   */
  const timelineData: TimelineData = useMemo(() => ({
    scores,
    anomalyIndices,
    timestamps: calculatedTimestamps,
  }), [scores, anomalyIndices, calculatedTimestamps]);

  /**
   * Statistics for display
   */
  const stats = useMemo(() => {
    if (scores.length === 0) return null;
    
    const average = scores.reduce((a, b) => a + b, 0) / scores.length;
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const belowThreshold = scores.filter(s => s < threshold).length;
    
    return { average, min, max, belowThreshold };
  }, [scores, threshold]);

  /**
   * Current frame info for display
   */
  const currentFrameInfo = useMemo(() => {
    const frameIndex = hoveredFrame ?? selectedFrame;
    if (frameIndex === undefined || frameIndex === null || frameIndex >= scores.length) {
      return null;
    }
    
    return {
      index: frameIndex,
      score: scores[frameIndex],
      timestamp: calculatedTimestamps[frameIndex],
      isAnomaly: anomalyIndices.includes(frameIndex),
    };
  }, [hoveredFrame, selectedFrame, scores, calculatedTimestamps, anomalyIndices]);

  // ============== CHART CONFIG ==============

  const chartConfig: Partial<TimelineConfig> = useMemo(() => ({
    width: containerWidth,
    height: Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, height)),
    fps,
    showThreshold,
    threshold,
    showArea,
    animated,
    marginLeft: compact ? 30 : 45,
    marginBottom: compact ? 25 : 35,
    marginTop: compact ? 15 : 20,
    marginRight: compact ? 15 : 20,
  }), [containerWidth, height, fps, showThreshold, threshold, showArea, animated, compact]);

  // ============== HANDLERS ==============

  const handleFrameSelect = useCallback((index: number) => {
    onFrameSelect?.(index);
  }, [onFrameSelect]);

  const handleResize = useCallback(() => {
    if (containerRef.current) {
      const width = containerRef.current.clientWidth;
      setContainerWidth(width);
    }
  }, []);

  // ============== EFFECTS ==============

  /**
   * Initialize resize observer
   */
  useEffect(() => {
    handleResize();
    
    const resizeObserver = new ResizeObserver(handleResize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    
    return () => {
      resizeObserver.disconnect();
    };
  }, [handleResize]);

  /**
   * Initialize D3 chart
   */
  useEffect(() => {
    if (!svgRef.current || containerWidth === 0 || scores.length === 0) {
      return;
    }
    
    // Clean up existing chart
    if (timelineRef.current) {
      timelineRef.current.destroy();
      timelineRef.current = null;
    }
    
    // Create new chart
    timelineRef.current = createTimelineChart(
      svgRef.current,
      timelineData,
      chartConfig,
      handleFrameSelect
    );
    
    setIsInitialized(true);
    
    return () => {
      if (timelineRef.current) {
        timelineRef.current.destroy();
        timelineRef.current = null;
      }
    };
  }, [containerWidth, scores.length > 0]); // Only reinitialize when width changes or scores become available

  /**
   * Update chart data when scores change
   */
  useEffect(() => {
    if (timelineRef.current && isInitialized) {
      timelineRef.current.update(timelineData);
    }
  }, [timelineData, isInitialized]);

  /**
   * Update selected frame
   */
  useEffect(() => {
    if (timelineRef.current && selectedFrame !== undefined) {
      timelineRef.current.setSelected(selectedFrame);
    }
  }, [selectedFrame]);

  /**
   * Handle resize
   */
  useEffect(() => {
    if (timelineRef.current && containerWidth > 0 && isInitialized) {
      timelineRef.current.resize(containerWidth, chartConfig.height || DEFAULT_HEIGHT);
    }
  }, [containerWidth, chartConfig.height, isInitialized]);

  // ============== EMPTY STATE ==============

  if (scores.length === 0) {
    return (
      <EmptyTimelineChart className={className} compact={compact} />
    );
  }

  // ============== RENDER ==============

  const chartContent = (
    <div className="space-y-3">
      {/* Chart Container */}
      <div
        ref={containerRef}
        className="relative w-full"
        data-testid="timeline-container"
      >
        {/* Loading overlay */}
        {containerWidth === 0 && (
          <div className="absolute inset-0 flex items-center justify-center bg-muted/50 rounded-lg">
            <div className="h-6 w-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </div>
        )}
        
        {/* SVG Chart */}
        <svg
          ref={svgRef}
          className="w-full overflow-visible"
          style={{ height: chartConfig.height }}
          aria-label="Timeline chart showing per-frame analysis scores"
          role="img"
        />
      </div>

      {/* Legend and Stats */}
      {!compact && (
        <TimelineLegend
          stats={stats}
          currentFrameInfo={currentFrameInfo}
          threshold={threshold}
          anomalyCount={anomalyIndices.length}
        />
      )}

      {/* Compact frame info */}
      {compact && currentFrameInfo && (
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>
            Frame {currentFrameInfo.index} at {formatTimestamp(currentFrameInfo.timestamp)}
          </span>
          <Badge
            variant="secondary"
            className={cn(
              'text-xs',
              currentFrameInfo.score < threshold
                ? 'bg-red-500/10 text-red-600 dark:text-red-400'
                : 'bg-green-500/10 text-green-600 dark:text-green-400'
            )}
          >
            {(currentFrameInfo.score * 100).toFixed(1)}%
          </Badge>
        </div>
      )}
    </div>
  );

  // Return with or without header
  if (!showHeader) {
    return (
      <div className={className} data-testid="timeline-chart">
        {chartContent}
      </div>
    );
  }

  return (
    <Card className={className} data-testid="timeline-chart">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-500" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              {title}
            </CardTitle>
          </div>
          {anomalyIndices.length > 0 && (
            <Badge variant="destructive" className="gap-1">
              <AlertTriangle className="h-3 w-3" />
              {anomalyIndices.length} Anomalies
            </Badge>
          )}
        </div>
        {!compact && (
          <CardDescription>
            {description || `${scores.length} frames analyzed at ${fps} FPS • Click to select frame`}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {chartContent}
      </CardContent>
    </Card>
  );
}

// ============== SUB-COMPONENTS ==============

/**
 * Empty state for timeline chart
 */
function EmptyTimelineChart({
  className,
  compact,
}: {
  className?: string;
  compact?: boolean;
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center rounded-lg border border-dashed bg-muted/30',
        compact ? 'p-4' : 'p-8',
        className
      )}
      data-testid="timeline-chart-empty"
    >
      <Activity
        className={cn(
          'text-muted-foreground/40 mb-2',
          compact ? 'h-8 w-8' : 'h-12 w-12'
        )}
        aria-hidden="true"
      />
      <p className={cn(
        'text-muted-foreground font-medium',
        compact && 'text-sm'
      )}>
        No Timeline Data
      </p>
      {!compact && (
        <p className="text-sm text-muted-foreground/70 text-center max-w-xs mt-1">
          Per-frame score timeline will appear here when video analysis is complete.
        </p>
      )}
    </div>
  );
}

/**
 * Legend and statistics display
 */
function TimelineLegend({
  stats,
  currentFrameInfo,
  threshold,
  anomalyCount,
}: {
  stats: { average: number; min: number; max: number; belowThreshold: number } | null;
  currentFrameInfo: {
    index: number;
    score: number;
    timestamp: number;
    isAnomaly: boolean;
  } | null;
  threshold: number;
  anomalyCount: number;
}) {
  return (
    <div
      className="flex flex-wrap items-center justify-between gap-4 p-3 rounded-lg bg-muted/30 border border-dashed"
      data-testid="timeline-legend"
    >
      {/* Legend Items */}
      <div className="flex items-center gap-4 text-xs">
        {/* Score line */}
        <div className="flex items-center gap-1.5">
          <div className="h-0.5 w-4 bg-green-500 rounded" />
          <span className="text-muted-foreground">Score</span>
        </div>
        
        {/* Threshold line */}
        <div className="flex items-center gap-1.5">
          <div className="h-0.5 w-4 border-t-2 border-dashed border-red-500" />
          <span className="text-muted-foreground">Threshold ({(threshold * 100).toFixed(0)}%)</span>
        </div>
        
        {/* Anomaly marker */}
        <div className="flex items-center gap-1.5">
          <div className="h-3 w-3 rounded-full bg-orange-500" />
          <span className="text-muted-foreground">Anomaly ({anomalyCount})</span>
        </div>
      </div>

      {/* Stats or Current Frame Info */}
      {currentFrameInfo ? (
        <div className="flex items-center gap-3 text-xs">
          <span className="text-muted-foreground">
            Frame <span className="font-mono font-medium text-foreground">{currentFrameInfo.index}</span>
          </span>
          <span className="text-muted-foreground">
            Time <span className="font-mono font-medium text-foreground">{formatTimestamp(currentFrameInfo.timestamp)}</span>
          </span>
          <Badge
            variant="secondary"
            className={cn(
              'text-xs',
              currentFrameInfo.score < threshold
                ? 'bg-red-500/10 text-red-600 dark:text-red-400'
                : 'bg-green-500/10 text-green-600 dark:text-green-400',
              currentFrameInfo.isAnomaly && 'ring-1 ring-orange-500'
            )}
          >
            {(currentFrameInfo.score * 100).toFixed(1)}%
            {currentFrameInfo.isAnomaly && ' ⚠'}
          </Badge>
        </div>
      ) : stats ? (
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>
            Avg: <span className="font-mono font-medium text-foreground">{(stats.average * 100).toFixed(1)}%</span>
          </span>
          <span>
            Min: <span className="font-mono font-medium text-foreground">{(stats.min * 100).toFixed(1)}%</span>
          </span>
          <span>
            Max: <span className="font-mono font-medium text-foreground">{(stats.max * 100).toFixed(1)}%</span>
          </span>
          {stats.belowThreshold > 0 && (
            <span className="text-red-500">
              {stats.belowThreshold} below threshold
            </span>
          )}
        </div>
      ) : null}
    </div>
  );
}

// ============== UTILITY FUNCTIONS ==============

/**
 * Format timestamp in ms to human readable string
 */
function formatTimestamp(ms: number): string {
  const totalSeconds = ms / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const milliseconds = Math.floor((totalSeconds % 1) * 1000);
  
  if (minutes > 0) {
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0').substring(0, 1)}`;
  }
  return `${seconds}.${milliseconds.toString().padStart(3, '0').substring(0, 2)}s`;
}

// ============== SKELETON LOADER ==============

/**
 * Skeleton loader for TimelineChart
 */
export function TimelineChartSkeleton({
  className,
  compact = false,
  height = DEFAULT_HEIGHT,
}: {
  className?: string;
  compact?: boolean;
  height?: number;
}) {
  return (
    <Card className={className} data-testid="timeline-chart-skeleton">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Skeleton className="h-5 w-5 rounded" />
            <Skeleton className="h-5 w-40" />
          </div>
          <Skeleton className="h-5 w-24" />
        </div>
        {!compact && <Skeleton className="h-4 w-64 mt-2" />}
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {/* Chart skeleton */}
          <Skeleton
            className="w-full rounded-lg"
            style={{ height: Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, height)) }}
          />
          
          {/* Legend skeleton */}
          {!compact && (
            <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
              <div className="flex items-center gap-4">
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-20" />
                <Skeleton className="h-4 w-16" />
              </div>
              <div className="flex items-center gap-3">
                <Skeleton className="h-4 w-12" />
                <Skeleton className="h-4 w-12" />
                <Skeleton className="h-4 w-12" />
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ============== EXPORTS ==============

export default TimelineChart;
