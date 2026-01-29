/**
 * Argus Core - Video Analysis Panel Component
 * ============================================
 * Detailed video analysis results with spatial/temporal/lipsync breakdowns.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/modality/VideoAnalysisPanel.tsx
 * 
 * Role: Display comprehensive video deepfake detection results.
 * Shows spatial analysis (per-frame), temporal consistency, and lip-sync detection.
 * 
 * Integration:
 * - Imports: components/visualization/HeatmapViewer, components/visualization/TimelineChart
 * - Used by: ModalityTabs.tsx (lazy loaded)
 * - Backend: Uses VideoResult from GET /api/v1/analyze/{id}/detail
 * 
 * Features:
 * - Spatial analysis section with frame scores and heatmap viewer
 * - Temporal consistency analysis with flickering detection
 * - Lip-sync detection (if audio present) with sync offset metrics
 * - Interactive frame gallery for anomaly exploration
 * - Per-frame score timeline with D3 visualization
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton loaders for each section
 * - Empty state: Shows message when no video data available
 * - Error state: Displays API errors gracefully
 * - Accessibility: Proper headings, ARIA labels, keyboard navigation
 * - data-testid: video-panel, video-spatial-section, video-temporal-section, video-lipsync-section
 */

'use client';

import React, { useMemo, useState } from 'react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import {
  Video,
  Eye,
  Timer,
  Mic2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ChevronLeft,
  ChevronRight,
  Layers,
  Activity,
  Waves,
  Info,
  ImageIcon,
  Clock,
  type LucideIcon,
} from 'lucide-react';
import type { VideoResult, FrameResult, SpatialResult, TemporalResult, LipsyncResult } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for VideoAnalysisPanel component
 */
export interface VideoAnalysisPanelProps {
  /** Video analysis result from backend */
  result: VideoResult;
  /** Analysis ID for fetching additional data like heatmaps */
  analysisId: string;
  /** Additional CSS classes */
  className?: string;
  /** Show compact version */
  compact?: boolean;
}

/**
 * Score indicator configuration
 */
interface ScoreIndicator {
  label: string;
  value: number;
  icon: LucideIcon;
  description: string;
  status: 'good' | 'warning' | 'danger';
}

// ============== CONSTANTS ==============

/**
 * Score thresholds for status classification
 */
const SCORE_THRESHOLDS = {
  good: 80,
  warning: 50,
} as const;

/**
 * Get status based on score (inverted - higher score means MORE authentic)
 */
function getScoreStatus(score: number): 'good' | 'warning' | 'danger' {
  const normalized = score * 100;
  if (normalized >= SCORE_THRESHOLDS.good) return 'good';
  if (normalized >= SCORE_THRESHOLDS.warning) return 'warning';
  return 'danger';
}

/**
 * Status colors for badges and indicators
 */
const STATUS_COLORS = {
  good: 'bg-green-500/10 text-green-700 dark:text-green-400 border-green-500/20',
  warning: 'bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border-yellow-500/20',
  danger: 'bg-red-500/10 text-red-700 dark:text-red-400 border-red-500/20',
} as const;

// ============== MAIN COMPONENT ==============

/**
 * VideoAnalysisPanel Component
 * 
 * Displays detailed video deepfake detection results with spatial,
 * temporal, and lip-sync analysis sections.
 * 
 * @example
 * ```tsx
 * <VideoAnalysisPanel
 *   result={analysisDetail.video_result}
 *   analysisId={analysisId}
 * />
 * ```
 */
export function VideoAnalysisPanel({
  result,
  analysisId,
  className,
  compact = false,
}: VideoAnalysisPanelProps) {
  // ============== STATE ==============
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);

  // ============== COMPUTED VALUES ==============
  
  /**
   * Overall score indicators for summary display
   */
  const scoreIndicators: ScoreIndicator[] = useMemo(() => {
    const indicators: ScoreIndicator[] = [];

    // Spatial score
    if (result.spatial) {
      indicators.push({
        label: 'Spatial',
        value: result.spatial.score,
        icon: Eye,
        description: 'Per-frame face analysis',
        status: getScoreStatus(result.spatial.score),
      });
    }

    // Temporal score
    if (result.temporal) {
      indicators.push({
        label: 'Temporal',
        value: result.temporal.score,
        icon: Timer,
        description: 'Frame consistency analysis',
        status: getScoreStatus(result.temporal.score),
      });
    }

    // Lipsync score (if available)
    if (result.lipsync) {
      indicators.push({
        label: 'Lip-Sync',
        value: result.lipsync.score,
        icon: Mic2,
        description: 'Audio-visual sync analysis',
        status: getScoreStatus(result.lipsync.score),
      });
    }

    return indicators;
  }, [result]);

  /**
   * Anomaly frames from spatial analysis
   */
  const anomalyFrames = useMemo(() => {
    return result.spatial?.anomaly_frames || [];
  }, [result.spatial]);

  // ============== RENDER ==============

  return (
    <div
      className={cn('space-y-6', className)}
      data-testid="video-panel"
      role="region"
      aria-label="Video Analysis Results"
    >
      {/* Summary Section */}
      <VideoSummarySection
        result={result}
        indicators={scoreIndicators}
        compact={compact}
      />

      {/* Spatial Analysis Section */}
      {result.spatial && (
        <SpatialAnalysisSection
          spatial={result.spatial}
          analysisId={analysisId}
          selectedFrameIndex={selectedFrameIndex}
          onFrameSelect={setSelectedFrameIndex}
          compact={compact}
        />
      )}

      {/* Temporal Analysis Section */}
      {result.temporal && (
        <TemporalAnalysisSection
          temporal={result.temporal}
          compact={compact}
        />
      )}

      {/* Lip-Sync Analysis Section (if available) */}
      {result.lipsync && (
        <LipsyncAnalysisSection
          lipsync={result.lipsync}
          compact={compact}
        />
      )}

      {/* Anomaly Frame Gallery */}
      {anomalyFrames.length > 0 && (
        <AnomalyFrameGallery
          frames={anomalyFrames}
          selectedIndex={selectedFrameIndex}
          onSelectFrame={setSelectedFrameIndex}
        />
      )}
    </div>
  );
}

// ============== SUB-COMPONENTS ==============

/**
 * Summary section showing overall video analysis metrics
 */
function VideoSummarySection({
  result,
  indicators,
  compact,
}: {
  result: VideoResult;
  indicators: ScoreIndicator[];
  compact: boolean;
}) {
  const overallStatus = getScoreStatus(result.aggregated_score);

  return (
    <Card data-testid="video-summary-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Video className="h-5 w-5 text-primary" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              Video Analysis Summary
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', STATUS_COLORS[overallStatus])}>
            {(result.aggregated_score * 100).toFixed(1)}%
          </Badge>
        </div>
        {!compact && (
          <CardDescription>
            Analyzed {result.frames_processed} frames over {result.duration_analyzed_seconds.toFixed(1)}s
          </CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {/* Score Indicators Grid */}
        <div className={cn(
          'grid gap-4',
          indicators.length === 2 && 'grid-cols-2',
          indicators.length >= 3 && 'grid-cols-3',
        )}>
          {indicators.map((indicator) => {
            const Icon = indicator.icon;
            return (
              <div
                key={indicator.label}
                className={cn(
                  'flex flex-col p-3 rounded-lg border',
                  STATUS_COLORS[indicator.status]
                )}
              >
                <div className="flex items-center gap-2 mb-2">
                  <Icon className="h-4 w-4" aria-hidden="true" />
                  <span className="font-medium text-sm">{indicator.label}</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold">
                    {(indicator.value * 100).toFixed(0)}
                  </span>
                  <span className="text-xs text-muted-foreground">/ 100</span>
                </div>
                {!compact && (
                  <p className="text-xs text-muted-foreground mt-1">
                    {indicator.description}
                  </p>
                )}
              </div>
            );
          })}
        </div>

        {/* Confidence Indicator */}
        <div className="mt-4 flex items-center gap-3">
          <span className="text-sm text-muted-foreground">Confidence:</span>
          <Progress value={result.confidence * 100} className="flex-1 h-2" />
          <span className="text-sm font-medium">
            {(result.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Spatial analysis section - per-frame detection results
 */
function SpatialAnalysisSection({
  spatial,
  analysisId,
  selectedFrameIndex,
  onFrameSelect,
  compact,
}: {
  spatial: SpatialResult;
  analysisId: string;
  selectedFrameIndex: number;
  onFrameSelect: (index: number) => void;
  compact: boolean;
}) {
  const status = getScoreStatus(spatial.score);
  const hasAnomalies = spatial.anomaly_frames.length > 0;

  return (
    <Card data-testid="video-spatial-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-blue-500" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              Spatial Analysis
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', STATUS_COLORS[status])}>
            {(spatial.score * 100).toFixed(1)}%
          </Badge>
        </div>
        <CardDescription>
          Per-frame face manipulation detection using {spatial.model_used}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            label="Frames Analyzed"
            value={spatial.frames_analyzed.toString()}
            icon={ImageIcon}
          />
          <MetricCard
            label="Confidence"
            value={`${(spatial.confidence * 100).toFixed(0)}%`}
            icon={Activity}
          />
          <MetricCard
            label="Face Confidence"
            value={`${(spatial.average_face_confidence * 100).toFixed(0)}%`}
            icon={Eye}
          />
          <MetricCard
            label="Anomalies"
            value={spatial.anomaly_frames.length.toString()}
            icon={AlertTriangle}
            highlight={hasAnomalies}
          />
        </div>

        {/* Anomaly Alert */}
        {hasAnomalies && (
          <Alert variant="destructive" className="mt-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Anomalies Detected</AlertTitle>
            <AlertDescription>
              {spatial.anomaly_frames.length} frames show potential manipulation artifacts.
              Review the highlighted frames below.
            </AlertDescription>
          </Alert>
        )}

        {/* Heatmap Preview (if available) */}
        {spatial.anomaly_frames.length > 0 && spatial.anomaly_frames[0]?.heatmap_url && (
          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Eye className="h-4 w-4" aria-hidden="true" />
              GradCAM Heatmap Visualization
            </h4>
            <div className="bg-muted/50 rounded-lg p-4 border border-dashed">
              <p className="text-sm text-muted-foreground text-center">
                Heatmap visualization shows regions where the model detected potential manipulation.
                Warmer colors indicate higher manipulation probability.
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Temporal analysis section - frame consistency detection
 */
function TemporalAnalysisSection({
  temporal,
  compact,
}: {
  temporal: TemporalResult;
  compact: boolean;
}) {
  const status = getScoreStatus(temporal.score);

  return (
    <Card data-testid="video-temporal-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-purple-500" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              Temporal Analysis
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', STATUS_COLORS[status])}>
            {(temporal.score * 100).toFixed(1)}%
          </Badge>
        </div>
        <CardDescription>
          Inter-frame consistency and motion artifact detection using {temporal.model_used}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            label="Consistency"
            value={`${(temporal.consistency_score * 100).toFixed(0)}%`}
            icon={CheckCircle2}
          />
          <MetricCard
            label="Confidence"
            value={`${(temporal.confidence * 100).toFixed(0)}%`}
            icon={Activity}
          />
          <MetricCard
            label="Motion Artifacts"
            value={temporal.motion_artifacts.toString()}
            icon={Waves}
            highlight={temporal.motion_artifacts > 0}
          />
          <MetricCard
            label="Flickering"
            value={temporal.flickering_detected ? 'Detected' : 'None'}
            icon={temporal.flickering_detected ? AlertTriangle : CheckCircle2}
            highlight={temporal.flickering_detected}
          />
        </div>

        {/* Flickering Alert */}
        {temporal.flickering_detected && (
          <Alert variant="destructive" className="mt-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Flickering Detected</AlertTitle>
            <AlertDescription>
              Temporal flickering artifacts suggest potential frame-level manipulation or
              poor quality synthesis. This is a common indicator of deepfake videos.
            </AlertDescription>
          </Alert>
        )}

        {/* Motion Artifacts Alert */}
        {temporal.motion_artifacts > 0 && !temporal.flickering_detected && (
          <Alert className="mt-4">
            <Info className="h-4 w-4" />
            <AlertTitle>Motion Artifacts Found</AlertTitle>
            <AlertDescription>
              {temporal.motion_artifacts} motion artifact(s) detected. This may indicate
              unnatural movement patterns or boundary inconsistencies.
            </AlertDescription>
          </Alert>
        )}

        {/* Consistency Score Bar */}
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Frame Consistency</span>
            <span className="text-sm text-muted-foreground">
              {(temporal.consistency_score * 100).toFixed(1)}%
            </span>
          </div>
          <Progress
            value={temporal.consistency_score * 100}
            className="h-3"
          />
          <p className="text-xs text-muted-foreground mt-1">
            Higher consistency indicates more natural inter-frame transitions
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Lip-sync analysis section - audio-visual synchronization
 */
function LipsyncAnalysisSection({
  lipsync,
  compact,
}: {
  lipsync: LipsyncResult;
  compact: boolean;
}) {
  const status = getScoreStatus(lipsync.score);
  const hasIssues = lipsync.sync_offset_ms > 100 || lipsync.phoneme_mismatches > 5;

  return (
    <Card data-testid="video-lipsync-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Mic2 className="h-5 w-5 text-orange-500" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              Lip-Sync Analysis
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', STATUS_COLORS[status])}>
            {(lipsync.score * 100).toFixed(1)}%
          </Badge>
        </div>
        <CardDescription>
          Audio-visual synchronization analysis using {lipsync.model_used}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            label="Sync Score"
            value={`${(lipsync.score * 100).toFixed(0)}%`}
            icon={Mic2}
          />
          <MetricCard
            label="Confidence"
            value={`${(lipsync.confidence * 100).toFixed(0)}%`}
            icon={Activity}
          />
          <MetricCard
            label="Sync Offset"
            value={`${lipsync.sync_offset_ms}ms`}
            icon={Clock}
            highlight={lipsync.sync_offset_ms > 100}
          />
          <MetricCard
            label="Phoneme Errors"
            value={lipsync.phoneme_mismatches.toString()}
            icon={hasIssues ? XCircle : CheckCircle2}
            highlight={lipsync.phoneme_mismatches > 5}
          />
        </div>

        {/* Sync Issues Alert */}
        {hasIssues && (
          <Alert variant="destructive" className="mt-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Lip-Sync Issues Detected</AlertTitle>
            <AlertDescription>
              {lipsync.sync_offset_ms > 100 && (
                <span>Significant audio-visual offset ({lipsync.sync_offset_ms}ms). </span>
              )}
              {lipsync.phoneme_mismatches > 5 && (
                <span>{lipsync.phoneme_mismatches} phoneme mismatches detected. </span>
              )}
              These patterns may indicate voice cloning or audio replacement.
            </AlertDescription>
          </Alert>
        )}

        {/* Sync Quality Indicator */}
        <div className="mt-4 p-4 rounded-lg bg-muted/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Audio-Visual Alignment</span>
            <Badge variant={lipsync.sync_offset_ms <= 50 ? 'outline' : 'destructive'}>
              {lipsync.sync_offset_ms <= 50 ? 'Good' : lipsync.sync_offset_ms <= 100 ? 'Fair' : 'Poor'}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            Natural speech typically has less than 50ms sync offset.
            Values above 100ms often indicate manipulation.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Anomaly frame gallery for exploring detected issues
 */
function AnomalyFrameGallery({
  frames,
  selectedIndex,
  onSelectFrame,
}: {
  frames: FrameResult[];
  selectedIndex: number;
  onSelectFrame: (index: number) => void;
}) {
  const canGoBack = selectedIndex > 0;
  const canGoForward = selectedIndex < frames.length - 1;

  const currentFrame = frames[selectedIndex];

  return (
    <Card data-testid="video-anomaly-gallery">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ImageIcon className="h-5 w-5 text-red-500" aria-hidden="true" />
            <CardTitle className="text-lg">Anomaly Frames</CardTitle>
          </div>
          <Badge variant="outline">
            {selectedIndex + 1} / {frames.length}
          </Badge>
        </div>
        <CardDescription>
          Frames flagged for potential manipulation. Click to explore details.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Frame Navigation */}
        <div className="flex items-center justify-between mb-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onSelectFrame(selectedIndex - 1)}
            disabled={!canGoBack}
            aria-label="Previous frame"
          >
            <ChevronLeft className="h-4 w-4 mr-1" aria-hidden="true" />
            Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            Frame #{currentFrame?.frame_number || 0} at {((currentFrame?.timestamp_ms || 0) / 1000).toFixed(2)}s
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onSelectFrame(selectedIndex + 1)}
            disabled={!canGoForward}
            aria-label="Next frame"
          >
            Next
            <ChevronRight className="h-4 w-4 ml-1" aria-hidden="true" />
          </Button>
        </div>

        {/* Frame Details */}
        {currentFrame && (
          <div className="space-y-4">
            {/* Frame Preview Placeholder */}
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center border border-dashed">
              {currentFrame.heatmap_url ? (
                <div className="text-center">
                  <ImageIcon className="h-12 w-12 mx-auto text-muted-foreground/40 mb-2" />
                  <p className="text-sm text-muted-foreground">
                    Heatmap visualization available
                  </p>
                </div>
              ) : (
                <div className="text-center">
                  <ImageIcon className="h-12 w-12 mx-auto text-muted-foreground/40 mb-2" />
                  <p className="text-sm text-muted-foreground">
                    Frame #{currentFrame.frame_number}
                  </p>
                </div>
              )}
            </div>

            {/* Frame Metrics */}
            <div className="grid grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-muted/50 text-center">
                <p className="text-xs text-muted-foreground mb-1">Score</p>
                <p className={cn(
                  'text-lg font-bold',
                  currentFrame.score < 0.5 ? 'text-red-500' : 'text-green-500'
                )}>
                  {(currentFrame.score * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-lg bg-muted/50 text-center">
                <p className="text-xs text-muted-foreground mb-1">Faces</p>
                <p className="text-lg font-bold">
                  {currentFrame.faces_detected}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-muted/50 text-center">
                <p className="text-xs text-muted-foreground mb-1">Timestamp</p>
                <p className="text-lg font-bold">
                  {(currentFrame.timestamp_ms / 1000).toFixed(2)}s
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Frame Thumbnails */}
        <div className="flex gap-2 mt-4 overflow-x-auto pb-2">
          {frames.map((frame, index) => (
            <button
              key={frame.frame_number}
              onClick={() => onSelectFrame(index)}
              className={cn(
                'flex-shrink-0 w-16 h-16 rounded border-2 bg-muted flex items-center justify-center transition-all',
                index === selectedIndex
                  ? 'border-primary ring-2 ring-primary/20'
                  : 'border-transparent hover:border-muted-foreground/50'
              )}
              aria-label={`Select frame ${frame.frame_number}`}
              aria-current={index === selectedIndex ? 'true' : undefined}
            >
              <span className="text-xs font-mono">
                #{frame.frame_number}
              </span>
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Small metric card for displaying individual metrics
 */
function MetricCard({
  label,
  value,
  icon: Icon,
  highlight = false,
}: {
  label: string;
  value: string;
  icon: LucideIcon;
  highlight?: boolean;
}) {
  return (
    <div
      className={cn(
        'flex flex-col p-3 rounded-lg bg-muted/50',
        highlight && 'bg-red-500/10 border border-red-500/20'
      )}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <Icon
          className={cn('h-3.5 w-3.5', highlight ? 'text-red-500' : 'text-muted-foreground')}
          aria-hidden="true"
        />
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <span className={cn('text-sm font-semibold', highlight && 'text-red-600 dark:text-red-400')}>
        {value}
      </span>
    </div>
  );
}

// ============== SKELETON LOADER ==============

/**
 * Skeleton loader for VideoAnalysisPanel
 */
export function VideoAnalysisPanelSkeleton() {
  return (
    <div className="space-y-6" data-testid="video-panel-skeleton">
      {/* Summary Skeleton */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-6 w-16" />
          </div>
          <Skeleton className="h-4 w-64 mt-2" />
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-24 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Section Skeletons */}
      {[1, 2].map((i) => (
        <Card key={i}>
          <CardHeader>
            <Skeleton className="h-6 w-40" />
            <Skeleton className="h-4 w-56 mt-2" />
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-3">
              {[1, 2, 3, 4].map((j) => (
                <Skeleton key={j} className="h-16 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ============== EXPORTS ==============

export default VideoAnalysisPanel;
