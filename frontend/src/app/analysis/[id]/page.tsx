/**
 * Argus Core - Analysis Results Page
 * ===================================
 * Dynamic analysis results page with real-time progress updates via WebSocket.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - app/analysis/[id]/page.tsx
 * 
 * Role: Display analysis progress or results based on current status.
 * Connects to WebSocket for real-time updates during analysis.
 * Shows full results with modality breakdowns when complete.
 * 
 * Integration:
 * - Imports: hooks/useWebSocket, hooks/useAnalysisDetail, components/analysis/*
 * - WebSocket: /ws/analysis/{id} for real-time progress
 * - Backend: GET /api/v1/analyze/{id}, GET /api/v1/analyze/{id}/detail
 * - Store: progressStore for real-time state
 * 
 * Component Contract (P0):
 * - Props interface defined (params.id)
 * - Loading state: Shows skeleton while fetching initial data
 * - Error state: Displays error with retry option
 * - Progress state: Shows ProgressIndicator and AnalysisTimeline
 * - Results state: Shows full results with TrustScoreGauge, VerdictBadge
 * - Accessibility: ARIA live regions for status updates
 * - data-testid: analysis-page, analysis-results, analysis-progress
 */

'use client';

import { useEffect, useMemo, useCallback } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { 
  ArrowLeft, 
  Shield, 
  RefreshCw, 
  Clock, 
  Download,
  Share2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  FileVideo,
  FileAudio,
  FileImage,
  FileText,
  Info
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

// Hooks
import { useWebSocket } from '@/hooks/useWebSocket';
import { useAnalysisDetail } from '@/hooks/useAnalysisDetail';
import { useProgressStore, selectProgress, STAGE_LABELS } from '@/store/progressStore';

// Components
import { ProgressIndicator, ProgressIndicatorSkeleton } from '@/components/analysis/ProgressIndicator';
import { AnalysisTimeline, AnalysisTimelineSkeleton } from '@/components/analysis/AnalysisTimeline';

// Types
import type { AnalysisStatus, Verdict, TrustScore } from '@/types/analysis';

// ============== TYPES ==============

interface AnalysisPageProps {
  params: {
    id: string;
  };
}

// ============== VERDICT CONFIG ==============

interface VerdictConfig {
  label: string;
  description: string;
  color: string;
  bgColor: string;
  borderColor: string;
  icon: React.ComponentType<{ className?: string }>;
}

const VERDICT_CONFIGS: Record<Verdict, VerdictConfig> = {
  authentic: {
    label: 'Authentic',
    description: 'High confidence that this content is authentic',
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-500/10',
    borderColor: 'border-green-500/30',
    icon: CheckCircle2,
  },
  likely_authentic: {
    label: 'Likely Authentic',
    description: 'Content appears authentic with minor concerns',
    color: 'text-lime-600 dark:text-lime-400',
    bgColor: 'bg-lime-500/10',
    borderColor: 'border-lime-500/30',
    icon: CheckCircle2,
  },
  uncertain: {
    label: 'Uncertain',
    description: 'Analysis inconclusive - manual review recommended',
    color: 'text-yellow-600 dark:text-yellow-400',
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/30',
    icon: AlertCircle,
  },
  likely_fake: {
    label: 'Likely Fake',
    description: 'Content shows signs of manipulation',
    color: 'text-orange-600 dark:text-orange-400',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
    icon: AlertCircle,
  },
  fake: {
    label: 'Fake',
    description: 'High confidence that this content is manipulated',
    color: 'text-red-600 dark:text-red-400',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/30',
    icon: XCircle,
  },
};

// ============== HELPER FUNCTIONS ==============

/**
 * Get score color based on value
 */
function getScoreColor(score: number): string {
  if (score >= 80) return 'text-green-600 dark:text-green-400';
  if (score >= 60) return 'text-lime-600 dark:text-lime-400';
  if (score >= 40) return 'text-yellow-600 dark:text-yellow-400';
  if (score >= 20) return 'text-orange-600 dark:text-orange-400';
  return 'text-red-600 dark:text-red-400';
}

/**
 * Format date for display
 */
function formatDate(dateString: string | undefined): string {
  if (!dateString) return 'N/A';
  try {
    return new Date(dateString).toLocaleString();
  } catch {
    return dateString;
  }
}

/**
 * Format duration in seconds
 */
function formatDuration(seconds: number | undefined): string {
  if (seconds === undefined) return 'N/A';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(0);
  return `${mins}m ${secs}s`;
}

// ============== COMPONENT ==============

export default function AnalysisPage({ params }: AnalysisPageProps) {
  const { id: analysisId } = params;
  const router = useRouter();

  // ============== WEBSOCKET CONNECTION ==============
  
  // Connect to WebSocket for real-time updates
  const { 
    isConnected: wsConnected, 
    error: wsError,
    reconnect: wsReconnect 
  } = useWebSocket(analysisId);

  // ============== PROGRESS STORE ==============

  // Get progress from store (updated by WebSocket)
  const progress = useProgressStore(selectProgress(analysisId));

  // ============== API DATA ==============

  // Fetch analysis data
  const {
    analysis,
    detail,
    isLoading,
    isDetailLoading,
    error: apiError,
    refetch,
    isComplete,
    isFailed,
    isInProgress,
  } = useAnalysisDetail(analysisId);

  // ============== COMPUTED STATE ==============

  // Determine current status from both API and WebSocket
  const currentStatus: AnalysisStatus = useMemo(() => {
    // Prefer WebSocket progress if available and more recent
    if (progress.status && progress.status !== 'pending') {
      return progress.status;
    }
    return analysis?.status || 'pending';
  }, [progress.status, analysis?.status]);

  // Get verdict config
  const verdictConfig = useMemo(() => {
    const verdict = analysis?.verdict || detail?.verdict;
    if (verdict) {
      return VERDICT_CONFIGS[verdict];
    }
    return null;
  }, [analysis?.verdict, detail?.verdict]);

  // Get trust score
  const trustScore = useMemo(() => {
    return analysis?.trust_score || detail?.trust_score;
  }, [analysis?.trust_score, detail?.trust_score]);

  // ============== HANDLERS ==============

  const handleRefresh = useCallback(async () => {
    await refetch();
    if (!wsConnected) {
      wsReconnect();
    }
  }, [refetch, wsConnected, wsReconnect]);

  const handleDownloadReport = useCallback(() => {
    const reportUrl = analysis?.report_url || detail?.report_url;
    if (reportUrl) {
      window.open(reportUrl, '_blank');
    }
  }, [analysis?.report_url, detail?.report_url]);

  const handleShare = useCallback(() => {
    if (navigator.share) {
      navigator.share({
        title: 'Argus Analysis Result',
        text: `Deepfake analysis result: ${verdictConfig?.label || 'Pending'}`,
        url: window.location.href,
      }).catch(() => {
        // User cancelled or error
      });
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(window.location.href);
    }
  }, [verdictConfig]);

  // ============== LOADING STATE ==============

  if (isLoading && !progress.status) {
    return (
      <div 
        className="min-h-screen bg-gradient-to-b from-background to-muted/20"
        data-testid="analysis-page"
      >
        <AnalysisPageHeader analysisId={analysisId} />
        <main className="container py-8">
          <div className="mx-auto max-w-4xl space-y-6">
            <AnalysisTimelineSkeleton />
            <ProgressIndicatorSkeleton />
            <Card>
              <CardContent className="py-8">
                <div className="space-y-4 animate-pulse">
                  <div className="h-8 w-48 bg-muted rounded mx-auto" />
                  <div className="h-4 w-64 bg-muted rounded mx-auto" />
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    );
  }

  // ============== ERROR STATE ==============

  if (apiError && !analysis && !progress.status) {
    return (
      <div 
        className="min-h-screen bg-gradient-to-b from-background to-muted/20"
        data-testid="analysis-page"
      >
        <AnalysisPageHeader analysisId={analysisId} />
        <main className="container py-8">
          <div className="mx-auto max-w-2xl">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error Loading Analysis</AlertTitle>
              <AlertDescription className="mt-2">
                <p>{apiError.message || 'Failed to load analysis data.'}</p>
                <div className="flex gap-2 mt-4">
                  <Button variant="outline" size="sm" onClick={handleRefresh}>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Retry
                  </Button>
                  <Link href="/analyze">
                    <Button variant="outline" size="sm">
                      <ArrowLeft className="h-4 w-4 mr-2" />
                      New Analysis
                    </Button>
                  </Link>
                </div>
              </AlertDescription>
            </Alert>
          </div>
        </main>
      </div>
    );
  }

  // ============== PROGRESS STATE ==============

  if (isInProgress || (currentStatus !== 'completed' && currentStatus !== 'failed')) {
    return (
      <div 
        className="min-h-screen bg-gradient-to-b from-background to-muted/20"
        data-testid="analysis-page"
      >
        <AnalysisPageHeader analysisId={analysisId} />
        <main className="container py-8">
          <div className="mx-auto max-w-4xl space-y-6" data-testid="analysis-progress">
            {/* Page Title */}
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold tracking-tight">
                Analysis in Progress
              </h2>
              <p className="text-muted-foreground">
                Your file is being analyzed. This typically takes 15-60 seconds.
              </p>
            </div>

            {/* Timeline */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Pipeline Progress</CardTitle>
              </CardHeader>
              <CardContent>
                <AnalysisTimeline 
                  analysisId={analysisId} 
                  showEstimates
                />
              </CardContent>
            </Card>

            {/* Progress Indicator */}
            <ProgressIndicator 
              analysisId={analysisId} 
              showDetails
            />

            {/* Connection Status */}
            <div className="flex items-center justify-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <div 
                  className={cn(
                    'w-2 h-2 rounded-full',
                    wsConnected ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'
                  )}
                />
                <span>{wsConnected ? 'Connected' : 'Connecting...'}</span>
              </div>
              {wsError && (
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={wsReconnect}
                  className="text-xs"
                >
                  <RefreshCw className="h-3 w-3 mr-1" />
                  Reconnect
                </Button>
              )}
            </div>

            {/* Info Card */}
            <Card className="bg-muted/30">
              <CardContent className="py-4">
                <div className="flex items-start gap-3">
                  <Info className="h-5 w-5 text-muted-foreground flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-muted-foreground">
                    <p>
                      <strong>Did you know?</strong> Argus Core uses multiple AI models 
                      to analyze spatial features, temporal consistency, audio patterns, 
                      and metadata to detect synthetic media manipulation.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    );
  }

  // ============== FAILED STATE ==============

  if (isFailed || currentStatus === 'failed') {
    return (
      <div 
        className="min-h-screen bg-gradient-to-b from-background to-muted/20"
        data-testid="analysis-page"
      >
        <AnalysisPageHeader analysisId={analysisId} />
        <main className="container py-8">
          <div className="mx-auto max-w-4xl space-y-6">
            {/* Timeline showing failure */}
            <Card>
              <CardContent className="py-4">
                <AnalysisTimeline analysisId={analysisId} compact />
              </CardContent>
            </Card>

            {/* Error Alert */}
            <Alert variant="destructive">
              <XCircle className="h-4 w-4" />
              <AlertTitle>Analysis Failed</AlertTitle>
              <AlertDescription className="mt-2">
                <p>
                  {progress.errorMessage || analysis?.explanation?.summary || 
                   'The analysis could not be completed. This may be due to an unsupported file format or processing error.'}
                </p>
                {progress.errorCode && (
                  <p className="text-xs mt-2 opacity-70">
                    Error code: {progress.errorCode}
                  </p>
                )}
                <div className="flex gap-2 mt-4">
                  <Link href="/analyze">
                    <Button size="sm">
                      Try Another File
                    </Button>
                  </Link>
                  <Button variant="outline" size="sm" onClick={handleRefresh}>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh Status
                  </Button>
                </div>
              </AlertDescription>
            </Alert>

            {/* Analysis Info */}
            <Card className="bg-muted/30">
              <CardContent className="py-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Analysis ID</p>
                    <p className="font-mono text-xs">{analysisId}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Started</p>
                    <p>{formatDate(analysis?.created_at)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    );
  }

  // ============== COMPLETED STATE - RESULTS ==============

  return (
    <div 
      className="min-h-screen bg-gradient-to-b from-background to-muted/20"
      data-testid="analysis-page"
    >
      <AnalysisPageHeader analysisId={analysisId} />
      
      <main className="container py-8">
        <div className="mx-auto max-w-4xl space-y-6" data-testid="analysis-results">
          {/* Completed Timeline */}
          <Card>
            <CardContent className="py-4">
              <AnalysisTimeline analysisId={analysisId} compact />
            </CardContent>
          </Card>

          {/* Main Results Card */}
          <Card className={cn(
            'border-2',
            verdictConfig?.borderColor
          )}>
            <CardHeader className={cn('pb-4', verdictConfig?.bgColor)}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {verdictConfig && (
                    <div className={cn(
                      'p-2 rounded-full',
                      verdictConfig.bgColor,
                      'border',
                      verdictConfig.borderColor
                    )}>
                      <verdictConfig.icon className={cn('h-6 w-6', verdictConfig.color)} />
                    </div>
                  )}
                  <div>
                    <CardTitle className={cn('text-2xl', verdictConfig?.color)}>
                      {verdictConfig?.label || 'Analysis Complete'}
                    </CardTitle>
                    <CardDescription className="mt-1">
                      {verdictConfig?.description}
                    </CardDescription>
                  </div>
                </div>
                
                {/* Trust Score */}
                {trustScore && (
                  <div className="text-right">
                    <p className="text-sm text-muted-foreground">Trust Score</p>
                    <p className={cn(
                      'text-4xl font-bold tabular-nums',
                      getScoreColor(trustScore.overall)
                    )}>
                      {trustScore.overall}
                      <span className="text-lg text-muted-foreground">/100</span>
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {(trustScore.confidence * 100).toFixed(0)}% confidence
                    </p>
                  </div>
                )}
              </div>
            </CardHeader>

            <CardContent className="pt-6 space-y-6">
              {/* Explanation */}
              {(analysis?.explanation || detail?.explanation) && (
                <div className="space-y-3">
                  <h3 className="font-semibold flex items-center gap-2">
                    <Info className="h-4 w-4" />
                    Analysis Summary
                  </h3>
                  <p className="text-muted-foreground">
                    {analysis?.explanation?.summary || detail?.explanation?.summary}
                  </p>
                  
                  {/* Key Findings */}
                  {(analysis?.explanation?.key_findings || detail?.explanation?.key_findings) && (
                    <div className="mt-4">
                      <p className="text-sm font-medium mb-2">Key Findings:</p>
                      <ul className="space-y-1.5">
                        {(analysis?.explanation?.key_findings || detail?.explanation?.key_findings || []).map((finding, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                            <span className="text-primary">•</span>
                            {finding}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              <Separator />

              {/* Score Breakdown */}
              {trustScore?.breakdown && (
                <div className="space-y-3">
                  <h3 className="font-semibold">Score Breakdown</h3>
                  <div className="grid gap-3">
                    {trustScore.breakdown.video_spatial !== undefined && (
                      <ScoreRow 
                        label="Video (Spatial)" 
                        score={trustScore.breakdown.video_spatial * 100}
                        icon={FileVideo}
                      />
                    )}
                    {trustScore.breakdown.video_temporal !== undefined && (
                      <ScoreRow 
                        label="Video (Temporal)" 
                        score={trustScore.breakdown.video_temporal * 100}
                        icon={FileVideo}
                      />
                    )}
                    {trustScore.breakdown.audio !== undefined && (
                      <ScoreRow 
                        label="Audio Analysis" 
                        score={trustScore.breakdown.audio * 100}
                        icon={FileAudio}
                      />
                    )}
                    {trustScore.breakdown.text !== undefined && (
                      <ScoreRow 
                        label="Text Analysis" 
                        score={trustScore.breakdown.text * 100}
                        icon={FileText}
                      />
                    )}
                    {trustScore.breakdown.metadata !== undefined && (
                      <ScoreRow 
                        label="Metadata" 
                        score={trustScore.breakdown.metadata * 100}
                        icon={FileImage}
                      />
                    )}
                  </div>
                </div>
              )}

              <Separator />

              {/* Actions */}
              <div className="flex flex-wrap gap-3">
                {(analysis?.report_url || detail?.report_url) && (
                  <Button onClick={handleDownloadReport} className="gap-2">
                    <Download className="h-4 w-4" />
                    Download Report
                  </Button>
                )}
                <Button variant="outline" onClick={handleShare} className="gap-2">
                  <Share2 className="h-4 w-4" />
                  Share
                </Button>
                <Link href="/analyze">
                  <Button variant="outline" className="gap-2">
                    <ArrowLeft className="h-4 w-4" />
                    New Analysis
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>

          {/* Metadata Card */}
          <Card className="bg-muted/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Analysis Details</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Analysis ID</p>
                  <p className="font-mono text-xs truncate" title={analysisId}>
                    {analysisId}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Started</p>
                  <p>{formatDate(analysis?.created_at)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Completed</p>
                  <p>{formatDate(analysis?.completed_at || detail?.completed_at)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Processing Time</p>
                  <p>{formatDuration(detail?.processing_time_seconds)}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Input File Info */}
          {detail?.input && (
            <Card className="bg-muted/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Input File</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Filename</p>
                    <p className="truncate" title={detail.input.original_filename}>
                      {detail.input.original_filename}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Type</p>
                    <p>{detail.input.file_type}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Size</p>
                    <p>{(detail.input.file_size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                  {detail.input.duration_seconds && (
                    <div>
                      <p className="text-muted-foreground">Duration</p>
                      <p>{formatDuration(detail.input.duration_seconds)}</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t mt-auto">
        <div className="container py-6 text-center text-sm text-muted-foreground">
          <p>Argus Core - Multi-Modal Deepfake Detection Platform</p>
        </div>
      </footer>
    </div>
  );
}

// ============== SUB-COMPONENTS ==============

/**
 * Page Header Component
 */
function AnalysisPageHeader({ analysisId }: { analysisId: string }) {
  return (
    <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/analyze">
            <Button variant="ghost" size="sm" className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
          </Link>
          <div className="hidden sm:block">
            <h1 className="text-xl font-semibold">Analysis Results</h1>
            <p className="text-xs text-muted-foreground font-mono">
              {analysisId.slice(0, 8)}...
            </p>
          </div>
        </div>
        
        <Link href="/">
          <div className="flex items-center gap-2 font-bold text-lg">
            <Shield className="h-6 w-6 text-primary" />
            <span className="hidden sm:inline">Argus Core</span>
          </div>
        </Link>
      </div>
    </header>
  );
}

/**
 * Score Row Component for breakdown display
 */
function ScoreRow({ 
  label, 
  score, 
  icon: Icon 
}: { 
  label: string; 
  score: number; 
  icon: React.ComponentType<{ className?: string }>;
}) {
  const scoreColor = getScoreColor(score);
  
  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
      <div className="flex items-center gap-3">
        <Icon className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
          <div 
            className={cn(
              'h-full rounded-full transition-all duration-500',
              score >= 80 ? 'bg-green-500' :
              score >= 60 ? 'bg-lime-500' :
              score >= 40 ? 'bg-yellow-500' :
              score >= 20 ? 'bg-orange-500' : 'bg-red-500'
            )}
            style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
          />
        </div>
        <span className={cn('text-sm font-medium tabular-nums w-12 text-right', scoreColor)}>
          {score.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
