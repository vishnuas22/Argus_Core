/**
 * Argus Core - Results Panel Component
 * =====================================
 * Main results container organizing score, verdict, explanation, and actions.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/results/ResultsPanel.tsx
 * 
 * Role: Display complete analysis results in a cohesive layout.
 * Organizes TrustScoreGauge, VerdictBadge, ExplanationPanel, ScoreBreakdown,
 * and action buttons in a responsive grid.
 * 
 * Integration:
 * - Imports: All results components, hooks/useAnalysisDetail
 * - Used by: analysis/[id]/page.tsx
 * - Backend: GET /api/v1/analyze/{id}/detail for full results
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton components while fetching
 * - Error state: Displays error with retry option
 * - Empty state: Shows message when no results available
 * - Accessibility: ARIA labels, keyboard navigation
 * - data-testid: results-panel, results-score, results-verdict, results-actions
 */

'use client';

import React, { useMemo, useCallback } from 'react';
import Link from 'next/link';
import { cn } from '@/lib/utils';
import { 
  Download, 
  Share2, 
  RefreshCw, 
  ExternalLink,
  Info,
  Clock,
  FileText,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Loader2
} from 'lucide-react';

// UI Components
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';

// Result Components
import { TrustScoreGauge, TrustScoreGaugeSkeleton } from './TrustScoreGauge';
import { VerdictBadge, VerdictBadgeHero, VerdictBadgeSkeleton, getVerdictFromScore } from './VerdictBadge';
import { ScoreBreakdown, ScoreBreakdownSkeleton } from './ScoreBreakdown';
import { ExplanationPanel, ExplanationPanelSkeleton } from './ExplanationPanel';

// Hooks
import { useAnalysisDetail, useReport } from '@/hooks/useAnalysisDetail';

// Types
import type { 
  AnalysisDetailResponse, 
  TrustScore, 
  Verdict,
  Explanation,
  VideoResult,
  AudioResult,
  TextResult,
  MetadataResult,
  ScoreBreakdown as ScoreBreakdownType
} from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for ResultsPanel component
 */
export interface ResultsPanelProps {
  /** Analysis ID to fetch and display */
  analysisId: string;
  /** Additional CSS classes */
  className?: string;
  /** Layout variant */
  variant?: 'full' | 'compact' | 'card';
  /** Whether to show action buttons */
  showActions?: boolean;
  /** Whether to show score breakdown */
  showBreakdown?: boolean;
  /** Whether to show explanation */
  showExplanation?: boolean;
  /** Callback when download report is clicked */
  onDownloadReport?: () => void;
  /** Callback when share is clicked */
  onShare?: () => void;
}

/**
 * Props for ResultsPanelContent (internal component with data)
 */
interface ResultsPanelContentProps {
  data: AnalysisDetailResponse;
  variant: 'full' | 'compact' | 'card';
  showActions: boolean;
  showBreakdown: boolean;
  showExplanation: boolean;
  onDownloadReport?: () => void;
  onShare?: () => void;
  className?: string;
}

// ============== MAIN COMPONENT ==============

/**
 * ResultsPanel Component
 * 
 * Main container for displaying analysis results. Fetches data and renders
 * appropriate components based on the analysis status and available data.
 * 
 * @example
 * ```tsx
 * // Full layout with all features
 * <ResultsPanel analysisId={id} variant="full" showActions showBreakdown showExplanation />
 * 
 * // Compact card for lists
 * <ResultsPanel analysisId={id} variant="card" />
 * 
 * // Minimal display
 * <ResultsPanel analysisId={id} variant="compact" showActions={false} />
 * ```
 */
export function ResultsPanel({
  analysisId,
  className,
  variant = 'full',
  showActions = true,
  showBreakdown = true,
  showExplanation = true,
  onDownloadReport,
  onShare,
}: ResultsPanelProps) {
  // Fetch analysis data
  const {
    analysis,
    detail,
    isLoading,
    isDetailLoading,
    error,
    refetch,
    isComplete,
    isFailed,
  } = useAnalysisDetail(analysisId);

  // ============== LOADING STATE ==============

  if (isLoading || isDetailLoading) {
    return (
      <ResultsPanelSkeleton 
        className={className} 
        variant={variant}
        showBreakdown={showBreakdown}
        showExplanation={showExplanation}
      />
    );
  }

  // ============== ERROR STATE ==============

  if (error && !analysis && !detail) {
    return (
      <div 
        className={cn('space-y-4', className)}
        data-testid="results-panel-error"
      >
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Failed to Load Results</AlertTitle>
          <AlertDescription className="mt-2">
            <p>{error.message || 'Unable to fetch analysis results.'}</p>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => refetch()}
              className="mt-3"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  // ============== FAILED STATE ==============

  if (isFailed) {
    return (
      <div 
        className={cn('space-y-4', className)}
        data-testid="results-panel-failed"
      >
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertTitle>Analysis Failed</AlertTitle>
          <AlertDescription className="mt-2">
            <p>
              {analysis?.explanation?.summary || 
               'The analysis could not be completed. Please try again with a different file.'}
            </p>
            <div className="flex gap-2 mt-3">
              <Link href="/analyze">
                <Button size="sm">
                  Try Another File
                </Button>
              </Link>
            </div>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  // ============== NOT COMPLETE STATE ==============

  if (!isComplete || !analysis) {
    return (
      <div 
        className={cn('space-y-4', className)}
        data-testid="results-panel-pending"
      >
        <Card>
          <CardContent className="py-8 text-center">
            <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">
              Analysis is still in progress...
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ============== COMPLETE STATE - RENDER RESULTS ==============

  // Use detail data if available, fallback to basic analysis
  const resultData: AnalysisDetailResponse = detail || {
    ...analysis,
    video_result: undefined,
    audio_result: undefined,
    text_result: undefined,
    metadata_result: undefined,
    processing_time_seconds: undefined,
  };

  return (
    <ResultsPanelContent
      data={resultData}
      variant={variant}
      showActions={showActions}
      showBreakdown={showBreakdown}
      showExplanation={showExplanation}
      onDownloadReport={onDownloadReport}
      onShare={onShare}
      className={className}
    />
  );
}

// ============== CONTENT COMPONENT ==============

/**
 * ResultsPanelContent - Renders the actual results UI
 */
function ResultsPanelContent({
  data,
  variant,
  showActions,
  showBreakdown,
  showExplanation,
  onDownloadReport,
  onShare,
  className,
}: ResultsPanelContentProps) {
  // Extract data
  const trustScore = data.trust_score;
  const verdict = data.verdict || (trustScore ? getVerdictFromScore(trustScore.overall) : 'uncertain');
  const explanation = data.explanation;

  // Get report URL
  const { data: reportData, isLoading: reportLoading } = useReport(data.analysis_id, {
    enabled: showActions && !!data.report_url,
  });

  // ============== HANDLERS ==============

  const handleDownloadReport = useCallback(() => {
    if (onDownloadReport) {
      onDownloadReport();
    } else {
      const url = reportData?.reportUrl || data.report_url;
      if (url) {
        window.open(url, '_blank');
      }
    }
  }, [onDownloadReport, reportData, data.report_url]);

  const handleShare = useCallback(() => {
    if (onShare) {
      onShare();
    } else if (navigator.share) {
      navigator.share({
        title: 'Argus Analysis Result',
        text: `Deepfake analysis: ${verdict}`,
        url: window.location.href,
      }).catch(() => {});
    } else {
      navigator.clipboard.writeText(window.location.href);
    }
  }, [onShare, verdict]);

  // ============== CARD VARIANT ==============

  if (variant === 'card') {
    return (
      <Card 
        className={cn('hover:shadow-md transition-shadow', className)}
        data-testid="results-panel"
      >
        <CardContent className="pt-6">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              {trustScore && (
                <TrustScoreGauge
                  score={trustScore.overall}
                  confidence={trustScore.confidence}
                  verdict={verdict}
                  size={80}
                  animated={false}
                  showConfidence={false}
                />
              )}
              <div>
                <VerdictBadge verdict={verdict} size="md" showIcon />
                {trustScore && (
                  <p className="text-xs text-muted-foreground mt-1">
                    {(trustScore.confidence * 100).toFixed(0)}% confidence
                  </p>
                )}
              </div>
            </div>
            <Link href={`/analysis/${data.analysis_id}`}>
              <Button variant="ghost" size="sm">
                Details
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    );
  }

  // ============== COMPACT VARIANT ==============

  if (variant === 'compact') {
    return (
      <div 
        className={cn('space-y-4', className)}
        data-testid="results-panel"
      >
        <div className="flex items-center justify-between gap-6">
          {/* Score + Verdict */}
          <div className="flex items-center gap-4" data-testid="results-score">
            {trustScore && (
              <TrustScoreGauge
                score={trustScore.overall}
                confidence={trustScore.confidence}
                verdict={verdict}
                size={120}
                showLabel
                showConfidence={false}
              />
            )}
            <div className="space-y-2">
              <VerdictBadge 
                verdict={verdict} 
                size="lg" 
                showIcon 
                showDescription 
              />
            </div>
          </div>

          {/* Actions */}
          {showActions && (
            <div className="flex gap-2" data-testid="results-actions">
              {data.report_url && (
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={handleDownloadReport}
                  disabled={reportLoading}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Report
                </Button>
              )}
              <Button variant="ghost" size="sm" onClick={handleShare}>
                <Share2 className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>

        {/* Summary */}
        {showExplanation && explanation?.summary && (
          <p className="text-sm text-muted-foreground">
            {explanation.summary}
          </p>
        )}
      </div>
    );
  }

  // ============== FULL VARIANT (DEFAULT) ==============

  return (
    <div 
      className={cn('space-y-6', className)}
      data-testid="results-panel"
    >
      {/* Main Results Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Score Gauge + Verdict */}
        <div 
          className="lg:col-span-1 space-y-4"
          data-testid="results-score"
        >
          <Card>
            <CardContent className="pt-6 flex flex-col items-center">
              {trustScore ? (
                <>
                  <TrustScoreGauge
                    score={trustScore.overall}
                    confidence={trustScore.confidence}
                    verdict={verdict}
                    size={200}
                    animated
                    showLabel
                    showConfidence
                  />
                  <div className="mt-4" data-testid="results-verdict">
                    <VerdictBadge 
                      verdict={verdict} 
                      size="lg" 
                      showIcon 
                      animated
                    />
                  </div>
                </>
              ) : (
                <div className="py-8 text-center text-muted-foreground">
                  <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
                  <p>Score not available</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Processing Info */}
          {data.processing_time_seconds && (
            <Card className="bg-muted/30">
              <CardContent className="py-3">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Clock className="h-4 w-4" />
                  <span>
                    Processed in {data.processing_time_seconds.toFixed(1)}s
                  </span>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Center Column: Explanation + Key Findings */}
        <div className="lg:col-span-2 space-y-4">
          {/* Explanation Panel */}
          {showExplanation && explanation && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Info className="h-5 w-5" />
                  Analysis Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Summary */}
                <p className="text-muted-foreground">
                  {explanation.summary}
                </p>

                {/* Confidence Statement */}
                {explanation.confidence_statement && (
                  <div className="p-3 bg-muted/50 rounded-lg">
                    <p className="text-sm">
                      <span className="font-medium">Confidence: </span>
                      {explanation.confidence_statement}
                    </p>
                  </div>
                )}

                {/* Key Findings */}
                {explanation.key_findings && explanation.key_findings.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium mb-2">Key Findings</h4>
                    <ul className="space-y-2">
                      {explanation.key_findings.map((finding, idx) => (
                        <li 
                          key={idx} 
                          className="flex items-start gap-2 text-sm text-muted-foreground"
                        >
                          <CheckCircle2 className="h-4 w-4 mt-0.5 text-primary flex-shrink-0" />
                          <span>{finding}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Recommendations */}
                {explanation.recommendations && explanation.recommendations.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium mb-2">Recommendations</h4>
                    <ul className="space-y-1.5">
                      {explanation.recommendations.map((rec, idx) => (
                        <li 
                          key={idx} 
                          className="flex items-start gap-2 text-sm text-muted-foreground"
                        >
                          <ArrowRight className="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" />
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Score Breakdown */}
          {showBreakdown && trustScore?.breakdown && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-lg">Score Breakdown</CardTitle>
                <CardDescription>
                  Contribution from each analysis modality
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScoreBreakdown breakdown={trustScore.breakdown} />
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Actions Bar */}
      {showActions && (
        <>
          <Separator />
          <div 
            className="flex flex-wrap gap-3 justify-end"
            data-testid="results-actions"
          >
            {data.report_url && (
              <Button 
                onClick={handleDownloadReport}
                disabled={reportLoading}
                className="gap-2"
              >
                {reportLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Download className="h-4 w-4" />
                )}
                Download Report
              </Button>
            )}
            <Button variant="outline" onClick={handleShare} className="gap-2">
              <Share2 className="h-4 w-4" />
              Share
            </Button>
            <Link href="/analyze">
              <Button variant="outline" className="gap-2">
                <FileText className="h-4 w-4" />
                New Analysis
              </Button>
            </Link>
          </div>
        </>
      )}
    </div>
  );
}

// ============== SCORE BREAKDOWN COMPONENT ==============

/**
 * Score Breakdown - Visual display of modality scores
 */
function ScoreBreakdown({ breakdown }: { breakdown: ScoreBreakdownType }) {
  const scores = useMemo(() => {
    const items: Array<{ label: string; score: number; weight: number; key: string }> = [];
    
    if (breakdown.video_spatial !== undefined) {
      items.push({ 
        label: 'Video (Spatial)', 
        score: breakdown.video_spatial * 100, 
        weight: breakdown.weights?.video_spatial || 0.3,
        key: 'video_spatial'
      });
    }
    if (breakdown.video_temporal !== undefined) {
      items.push({ 
        label: 'Video (Temporal)', 
        score: breakdown.video_temporal * 100, 
        weight: breakdown.weights?.video_temporal || 0.25,
        key: 'video_temporal'
      });
    }
    if (breakdown.video_lipsync !== undefined) {
      items.push({ 
        label: 'Lip Sync', 
        score: breakdown.video_lipsync * 100, 
        weight: breakdown.weights?.video_lipsync || 0.1,
        key: 'video_lipsync'
      });
    }
    if (breakdown.audio !== undefined) {
      items.push({ 
        label: 'Audio Analysis', 
        score: breakdown.audio * 100, 
        weight: breakdown.weights?.audio || 0.2,
        key: 'audio'
      });
    }
    if (breakdown.text !== undefined) {
      items.push({ 
        label: 'Text Analysis', 
        score: breakdown.text * 100, 
        weight: breakdown.weights?.text || 0.1,
        key: 'text'
      });
    }
    if (breakdown.metadata !== undefined) {
      items.push({ 
        label: 'Metadata', 
        score: breakdown.metadata * 100, 
        weight: breakdown.weights?.metadata || 0.15,
        key: 'metadata'
      });
    }

    return items;
  }, [breakdown]);

  if (scores.length === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-4">
        No detailed scores available
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {scores.map(({ label, score, weight, key }) => (
        <div key={key} className="space-y-1.5">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">{label}</span>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                {(weight * 100).toFixed(0)}% weight
              </Badge>
              <span 
                className={cn(
                  'font-semibold tabular-nums',
                  getScoreTextColor(score)
                )}
              >
                {score.toFixed(0)}
              </span>
            </div>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div 
              className={cn(
                'h-full rounded-full transition-all duration-700 ease-out',
                getScoreBarColor(score)
              )}
              style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ============== SKELETON COMPONENT ==============

/**
 * Skeleton loader for ResultsPanel
 */
export function ResultsPanelSkeleton({
  className,
  variant = 'full',
  showBreakdown = true,
  showExplanation = true,
}: {
  className?: string;
  variant?: 'full' | 'compact' | 'card';
  showBreakdown?: boolean;
  showExplanation?: boolean;
}) {
  if (variant === 'card') {
    return (
      <Card className={className}>
        <CardContent className="pt-6">
          <div className="flex items-center gap-4">
            <Skeleton className="h-20 w-20 rounded-full" />
            <div className="space-y-2">
              <Skeleton className="h-6 w-24" />
              <Skeleton className="h-4 w-16" />
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (variant === 'compact') {
    return (
      <div className={cn('space-y-4', className)}>
        <div className="flex items-center gap-6">
          <TrustScoreGaugeSkeleton size={120} />
          <VerdictBadgeSkeleton size="lg" />
        </div>
        {showExplanation && <Skeleton className="h-16 w-full" />}
      </div>
    );
  }

  // Full variant
  return (
    <div className={cn('space-y-6', className)} data-testid="results-panel-skeleton">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column */}
        <div className="lg:col-span-1 space-y-4">
          <Card>
            <CardContent className="pt-6 flex flex-col items-center">
              <TrustScoreGaugeSkeleton size={200} />
              <div className="mt-4">
                <VerdictBadgeSkeleton size="lg" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column */}
        <div className="lg:col-span-2 space-y-4">
          {showExplanation && (
            <Card>
              <CardHeader className="pb-3">
                <Skeleton className="h-6 w-40" />
              </CardHeader>
              <CardContent className="space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
                <Skeleton className="h-4 w-4/6" />
                <div className="space-y-2 mt-4">
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-3 w-5/6" />
                </div>
              </CardContent>
            </Card>
          )}

          {showBreakdown && (
            <Card>
              <CardHeader className="pb-3">
                <Skeleton className="h-6 w-36" />
              </CardHeader>
              <CardContent className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="space-y-2">
                    <div className="flex justify-between">
                      <Skeleton className="h-4 w-24" />
                      <Skeleton className="h-4 w-12" />
                    </div>
                    <Skeleton className="h-2 w-full" />
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      <Separator />
      <div className="flex gap-3 justify-end">
        <Skeleton className="h-10 w-36" />
        <Skeleton className="h-10 w-24" />
        <Skeleton className="h-10 w-32" />
      </div>
    </div>
  );
}

// ============== UTILITY FUNCTIONS ==============

/**
 * Get text color class based on score
 */
function getScoreTextColor(score: number): string {
  if (score >= 80) return 'text-green-600 dark:text-green-400';
  if (score >= 60) return 'text-lime-600 dark:text-lime-400';
  if (score >= 40) return 'text-yellow-600 dark:text-yellow-400';
  if (score >= 20) return 'text-orange-600 dark:text-orange-400';
  return 'text-red-600 dark:text-red-400';
}

/**
 * Get bar color class based on score
 */
function getScoreBarColor(score: number): string {
  if (score >= 80) return 'bg-green-500';
  if (score >= 60) return 'bg-lime-500';
  if (score >= 40) return 'bg-yellow-500';
  if (score >= 20) return 'bg-orange-500';
  return 'bg-red-500';
}

export default ResultsPanel;
