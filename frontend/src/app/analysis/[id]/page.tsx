'use client';

import { useMemo, useCallback } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Eye,
  RefreshCw, 
  Download,
  Share2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  FileVideo,
  FileAudio,
  FileImage,
  Info,
  Clock,
  Hash,
  Activity
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

import { useWebSocket } from '@/hooks/useWebSocket';
import { useAnalysisDetail } from '@/hooks/useAnalysisDetail';
import { useProgressStore, selectProgress } from '@/store/progressStore';

import { ProgressIndicator, ProgressIndicatorSkeleton } from '@/components/analysis/ProgressIndicator';
import { AnalysisTimeline, AnalysisTimelineSkeleton } from '@/components/analysis/AnalysisTimeline';
import { XAIExplanationPanel, XAIAttributionPanel } from '@/components/xai';
import { ChatContainer } from '@/components/chat';
import { ErrorBoundary } from '@/components/errors/ErrorBoundary';

import type { AnalysisStatus, Verdict, TrustScore } from '@/types/analysis';

interface AnalysisPageProps {
  params: {
    id: string;
  };
}

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
    color: 'text-verdict-authentic',
    bgColor: 'bg-verdict-authentic/5',
    borderColor: 'border-verdict-authentic/20',
    icon: CheckCircle2,
  },
  likely_authentic: {
    label: 'Likely Authentic',
    description: 'Content appears authentic with minor concerns',
    color: 'text-verdict-likely-authentic',
    bgColor: 'bg-verdict-likely-authentic/5',
    borderColor: 'border-verdict-likely-authentic/20',
    icon: CheckCircle2,
  },
  uncertain: {
    label: 'Uncertain',
    description: 'Analysis inconclusive — manual review recommended',
    color: 'text-verdict-uncertain',
    bgColor: 'bg-verdict-uncertain/5',
    borderColor: 'border-verdict-uncertain/20',
    icon: AlertCircle,
  },
  likely_fake: {
    label: 'Likely Fake',
    description: 'Content shows signs of manipulation',
    color: 'text-verdict-likely-fake',
    bgColor: 'bg-verdict-likely-fake/5',
    borderColor: 'border-verdict-likely-fake/20',
    icon: AlertCircle,
  },
  fake: {
    label: 'Fake',
    description: 'High confidence that this content is manipulated',
    color: 'text-verdict-fake',
    bgColor: 'bg-verdict-fake/5',
    borderColor: 'border-verdict-fake/20',
    icon: XCircle,
  },
};

function getScoreColor(score: number): string {
  if (score >= 80) return 'text-verdict-authentic';
  if (score >= 60) return 'text-verdict-likely-authentic';
  if (score >= 40) return 'text-verdict-uncertain';
  if (score >= 20) return 'text-verdict-likely-fake';
  return 'text-verdict-fake';
}

function formatDate(dateString: string | undefined): string {
  if (!dateString) return '—';
  try {
    return new Date(dateString).toLocaleString();
  } catch {
    return dateString;
  }
}

function formatDuration(seconds: number | undefined): string {
  if (seconds === undefined) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(0);
  return `${mins}m ${secs}s`;
}

export default function AnalysisPage({ params }: AnalysisPageProps) {
  const { id: analysisId } = params;

  const { 
    isConnected: wsConnected, 
    error: wsError,
    reconnect: wsReconnect 
  } = useWebSocket(analysisId);

  const progress = useProgressStore(selectProgress(analysisId));

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

  const currentStatus: AnalysisStatus = useMemo(() => {
    if (progress.status && progress.status !== 'pending') {
      return progress.status;
    }
    return analysis?.status || 'pending';
  }, [progress.status, analysis?.status]);

  const verdictConfig = useMemo(() => {
    const verdict = analysis?.verdict || detail?.verdict;
    if (verdict) {
      return VERDICT_CONFIGS[verdict];
    }
    return null;
  }, [analysis?.verdict, detail?.verdict]);

  const trustScore = useMemo(() => {
    return analysis?.trust_score || detail?.trust_score;
  }, [analysis?.trust_score, detail?.trust_score]);

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
      }).catch(() => {});
    } else {
      navigator.clipboard.writeText(window.location.href);
    }
  }, [verdictConfig]);

  if (isLoading && !progress.status) {
    return (
      <div className="min-h-screen flex flex-col" data-testid="analysis-page">
        <AnalysisPageHeader analysisId={analysisId} />
        <main className="flex-1">
          <div className="mx-auto max-w-7xl px-6 lg:px-8 py-8 md:py-12">
            <div className="max-w-2xl mx-auto space-y-6">
              <AnalysisTimelineSkeleton />
              <ProgressIndicatorSkeleton />
              <div className="surface rounded-lg p-8">
                <div className="space-y-4 animate-pulse">
                  <div className="h-8 w-48 bg-muted/50 rounded mx-auto" />
                  <div className="h-4 w-64 bg-muted/50 rounded mx-auto" />
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (apiError && !analysis && !progress.status) {
    return (
      <div className="min-h-screen flex flex-col" data-testid="analysis-page">
        <AnalysisPageHeader analysisId={analysisId} />
        <main className="flex-1">
          <div className="mx-auto max-w-7xl px-6 lg:px-8 py-8 md:py-12">
            <div className="max-w-lg mx-auto">
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error Loading Analysis</AlertTitle>
                <AlertDescription className="mt-3 space-y-3">
                  <p className="text-sm">{apiError.message || 'Failed to load analysis data.'}</p>
                  <div className="flex gap-2">
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
          </div>
        </main>
      </div>
    );
  }

  if (isInProgress || (currentStatus !== 'completed' && currentStatus !== 'failed')) {
    return (
      <div className="min-h-screen flex flex-col" data-testid="analysis-page">
        <AnalysisPageHeader analysisId={analysisId} />
        <main className="flex-1">
          <div className="mx-auto max-w-7xl px-6 lg:px-8 py-8 md:py-12">
            <div className="max-w-2xl mx-auto space-y-6" data-testid="analysis-progress">
              <div className="text-center space-y-2">
                <h2 className="text-xl font-semibold tracking-tight">Analysis in Progress</h2>
                <p className="text-sm text-muted-foreground">
                  Your file is being analyzed. This typically takes 15–60 seconds.
                </p>
              </div>

              <div className="surface rounded-lg">
                <div className="px-5 py-4 border-b border-border/50">
                  <p className="text-sm font-medium">Pipeline Progress</p>
                </div>
                <div className="p-5">
                  <AnalysisTimeline analysisId={analysisId} showEstimates />
                </div>
              </div>

              <ProgressIndicator analysisId={analysisId} showDetails />

              <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
                <div className="flex items-center gap-2">
                  <div className={cn('w-1.5 h-1.5 rounded-full', wsConnected ? 'bg-primary' : 'bg-yellow-500 animate-pulse-soft')} />
                  <span>{wsConnected ? 'Connected' : 'Connecting...'}</span>
                </div>
                {wsError && (
                  <Button variant="ghost" size="sm" onClick={wsReconnect} className="text-xs h-7">
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Reconnect
                  </Button>
                )}
              </div>

              <div className="p-4 rounded-lg border border-border/50 bg-muted/10">
                <div className="flex items-start gap-3">
                  <Info className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    <strong>Did you know?</strong> Argus Core uses multiple AI models 
                    to analyze spatial features, temporal consistency, audio patterns, 
                    and metadata to detect synthetic media manipulation.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (isFailed || currentStatus === 'failed') {
    return (
      <div className="min-h-screen flex flex-col" data-testid="analysis-page">
        <AnalysisPageHeader analysisId={analysisId} />
        <main className="flex-1">
          <div className="mx-auto max-w-7xl px-6 lg:px-8 py-8 md:py-12">
            <div className="max-w-2xl mx-auto space-y-6">
              <div className="surface rounded-lg overflow-hidden">
                <div className="p-5">
                  <AnalysisTimeline analysisId={analysisId} compact />
                </div>
              </div>

              <Alert variant="destructive" className="glow-destructive">
                <XCircle className="h-4 w-4" />
                <AlertTitle>Analysis Failed</AlertTitle>
                <AlertDescription className="mt-3 space-y-3">
                  <p className="text-sm">
                    {progress.errorMessage || analysis?.explanation?.summary || 
                     'The analysis could not be completed. This may be due to an unsupported file format or processing error.'}
                  </p>
                  {progress.errorCode && (
                    <p className="text-xs text-destructive/70">
                      Error code: {progress.errorCode}
                    </p>
                  )}
                  <div className="flex gap-2">
                    <Link href="/analyze">
                      <Button size="sm">Try Another File</Button>
                    </Link>
                    <Button variant="outline" size="sm" onClick={handleRefresh}>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Refresh
                    </Button>
                  </div>
                </AlertDescription>
              </Alert>

              <div className="surface rounded-lg p-5">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-2xs text-muted-foreground mb-0.5">Analysis ID</p>
                    <p className="font-mono text-xs text-foreground/70">{analysisId.slice(0, 12)}...</p>
                  </div>
                  <div>
                    <p className="text-2xs text-muted-foreground mb-0.5">Started</p>
                    <p className="text-xs text-foreground/70">{formatDate(analysis?.created_at)}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <ErrorBoundary
      errorTitle="Results Display Error"
      errorMessage="Unable to display the analysis results. The analysis completed successfully, but an error occurred while rendering the results."
    >
      <div className="min-h-screen flex flex-col" data-testid="analysis-page">
        <AnalysisPageHeader analysisId={analysisId} />
        
        <main className="flex-1">
          <div className="mx-auto max-w-7xl px-6 lg:px-8 py-8 md:py-12">
            <div className="max-w-3xl mx-auto space-y-8" data-testid="analysis-results">
              <div className="surface rounded-lg overflow-hidden">
                <div className="p-4">
                  <AnalysisTimeline analysisId={analysisId} compact />
                </div>
              </div>

              <div
                className={cn(
                  'rounded-lg border overflow-hidden',
                  verdictConfig?.borderColor || 'border-border/60'
                )}
              >
                <div className={cn('px-6 py-5', verdictConfig?.bgColor)}>
                  <div className="flex items-start justify-between gap-6">
                    <div className="flex items-start gap-4">
                      <div className={cn('p-2.5 rounded-lg border', verdictConfig?.borderColor, verdictConfig?.bgColor)}>
                        {verdictConfig && (
                          <verdictConfig.icon className={cn('h-5 w-5', verdictConfig.color)} />
                        )}
                      </div>
                      <div>
                        <h2 className={cn('text-lg font-semibold tracking-tight', verdictConfig?.color)}>
                          {verdictConfig?.label || 'Analysis Complete'}
                        </h2>
                        <p className="text-sm text-muted-foreground mt-0.5">
                          {verdictConfig?.description}
                        </p>
                      </div>
                    </div>
                    
                    {trustScore && (
                      <div className="text-right shrink-0">
                        <p className="text-2xs text-muted-foreground mb-1">Trust Score</p>
                        <p className={cn('text-3xl font-semibold tabular-nums tracking-tight', getScoreColor(trustScore.overall))}>
                          {trustScore.overall}
                          <span className="text-sm text-muted-foreground font-normal">/100</span>
                        </p>
                        <p className="text-2xs text-muted-foreground mt-0.5">
                          {(trustScore.confidence * 100).toFixed(0)}% confidence
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="px-6 py-5 space-y-5">
                  {(analysis?.explanation || detail?.explanation) && (
                    <div className="space-y-3">
                      <h3 className="text-xs font-semibold tracking-tight text-muted-foreground uppercase">
                        Analysis Summary
                      </h3>
                      <p className="text-sm text-foreground leading-relaxed">
                        {analysis?.explanation?.summary || detail?.explanation?.summary}
                      </p>
                      
                      {(analysis?.explanation?.key_findings || detail?.explanation?.key_findings) && (
                        <div className="p-4 rounded-lg bg-muted/10 border border-border/50">
                          <p className="text-xs font-medium text-foreground mb-2">Key Findings</p>
                          <div className="space-y-2">
                            {(analysis?.explanation?.key_findings || detail?.explanation?.key_findings || []).map((finding, idx) => (
                              <div key={idx} className="flex items-start gap-2.5 text-xs text-muted-foreground">
                                <div className="w-1.5 h-1.5 rounded-full bg-primary/50 mt-1 shrink-0" />
                                {finding}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {trustScore?.breakdown && (
                    <div className="space-y-3 pt-1">
                      <h3 className="text-xs font-semibold tracking-tight text-muted-foreground uppercase">
                        Score Breakdown
                      </h3>
                      <div className="space-y-2">
                        {trustScore.breakdown.video_spatial !== undefined && (
                          <ScoreRow label="Video (Spatial)" score={trustScore.breakdown.video_spatial * 100} icon={FileVideo} />
                        )}
                        {trustScore.breakdown.video_temporal !== undefined && (
                          <ScoreRow label="Video (Temporal)" score={trustScore.breakdown.video_temporal * 100} icon={FileVideo} />
                        )}
                        {trustScore.breakdown.audio !== undefined && (
                          <ScoreRow label="Audio Analysis" score={trustScore.breakdown.audio * 100} icon={FileAudio} />
                        )}
                        {trustScore.breakdown.metadata !== undefined && (
                          <ScoreRow label="Metadata" score={trustScore.breakdown.metadata * 100} icon={FileImage} />
                        )}
                      </div>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-2 pt-1">
                    {(analysis?.report_url || detail?.report_url) && (
                      <Button size="sm" onClick={handleDownloadReport} className="gap-1.5">
                        <Download className="h-3.5 w-3.5" />
                        Download Report
                      </Button>
                    )}
                    <Button variant="outline" size="sm" onClick={handleShare} className="gap-1.5">
                      <Share2 className="h-3.5 w-3.5" />
                      Share
                    </Button>
                    <Link href="/analyze">
                      <Button variant="outline" size="sm" className="gap-1.5">
                        <ArrowLeft className="h-3.5 w-3.5" />
                        New Analysis
                      </Button>
                    </Link>
                  </div>
                </div>
              </div>

              <div className="surface rounded-lg p-5">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
                  <div>
                    <p className="text-2xs text-muted-foreground mb-1 flex items-center gap-1.5">
                      <Hash className="h-3 w-3" strokeWidth={1.5} />
                      Analysis ID
                    </p>
                    <p className="font-mono text-xs text-foreground/70 truncate" title={analysisId}>
                      {analysisId.slice(0, 12)}...
                    </p>
                  </div>
                  <div>
                    <p className="text-2xs text-muted-foreground mb-1 flex items-center gap-1.5">
                      <Clock className="h-3 w-3" strokeWidth={1.5} />
                      Started
                    </p>
                    <p className="text-xs text-foreground/70">{formatDate(analysis?.created_at)}</p>
                  </div>
                  <div>
                    <p className="text-2xs text-muted-foreground mb-1 flex items-center gap-1.5">
                      <Clock className="h-3 w-3" strokeWidth={1.5} />
                      Completed
                    </p>
                    <p className="text-xs text-foreground/70">{formatDate(analysis?.completed_at || detail?.completed_at)}</p>
                  </div>
                  <div>
                    <p className="text-2xs text-muted-foreground mb-1 flex items-center gap-1.5">
                      <Activity className="h-3 w-3" strokeWidth={1.5} />
                      Processing Time
                    </p>
                    <p className="text-xs text-foreground/70">{formatDuration(detail?.processing_time_seconds)}</p>
                  </div>
                </div>
              </div>

              {detail?.input && (
                <div className="surface rounded-lg p-5">
                  <p className="text-2xs text-muted-foreground mb-3 uppercase tracking-wide font-medium">Input File</p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
                    <div>
                      <p className="text-2xs text-muted-foreground mb-1">Filename</p>
                      <p className="text-xs text-foreground/70 truncate" title={detail.input.original_filename}>
                        {detail.input.original_filename}
                      </p>
                    </div>
                    <div>
                      <p className="text-2xs text-muted-foreground mb-1">Type</p>
                      <p className="text-xs text-foreground/70">{detail.input.file_type}</p>
                    </div>
                    <div>
                      <p className="text-2xs text-muted-foreground mb-1">Size</p>
                      <p className="text-xs text-foreground/70">{(detail.input.file_size / (1024 * 1024)).toFixed(2)} MB</p>
                    </div>
                    {detail.input.duration_seconds && (
                      <div>
                        <p className="text-2xs text-muted-foreground mb-1">Duration</p>
                        <p className="text-xs text-foreground/70">{formatDuration(detail.input.duration_seconds)}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {isComplete && (
                <XAIExplanationPanel
                  analysisId={analysisId}
                  showModalityTabs
                  showEvidenceGallery
                  showReferences
                  showReproducibility
                  defaultExpanded
                />
              )}

              {isComplete && (
                <XAIAttributionPanel
                  xai_attribution={
                    // Extract from the first available modality result.
                    // The backend's ModalityResult now includes xai_attribution,
                    // conformal_prediction_set, and route_to_human fields.
                    (detail?.image_result as any)?.xai_attribution ?? null
                  }
                  conformal_prediction_set={
                    (detail?.image_result as any)?.conformal_prediction_set ?? null
                  }
                  route_to_human={
                    (detail?.image_result as any)?.route_to_human ?? false
                  }
                />
              )}

              {isComplete && (
                <ChatContainer
                  analysisId={analysisId}
                  verdict={detail?.verdict}
                />
              )}
            </div>
          </div>
        </main>

        <footer className="border-t border-border/50">
          <div className="mx-auto max-w-7xl px-6 lg:px-8">
            <div className="py-6 flex items-center justify-between">
              <div className="flex items-center gap-2 text-2xs text-muted-foreground">
                <Eye className="h-3 w-3 text-primary/60" strokeWidth={1.5} />
                <span>Argus Core v1.0</span>
              </div>
              <div className="flex items-center gap-1.5 text-2xs text-muted-foreground">
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse-soft" />
                System Online
              </div>
            </div>
          </div>
        </footer>
      </div>
    </ErrorBoundary>
  );
}

function AnalysisPageHeader({ analysisId }: { analysisId: string }) {
  return (
    <header className="sticky top-0 z-50 border-b border-border/50 bg-background/70 backdrop-blur-lg">
      <div className="mx-auto max-w-7xl px-6 lg:px-8 flex h-14 items-center justify-between">
        <div className="flex items-center gap-3">
          <Link href="/analyze">
            <Button variant="ghost" size="sm" className="gap-1.5 text-xs text-muted-foreground h-8">
              <ArrowLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          </Link>
          <div className="hidden sm:block">
            <h1 className="text-sm font-semibold tracking-tight">
              Analysis Results
            </h1>
            <p className="text-2xs text-muted-foreground font-mono">
              {analysisId.slice(0, 8)}...
            </p>
          </div>
        </div>

        <Link href="/" className="flex items-center gap-2 group">
          <div className="p-1.5 rounded-md bg-primary/10 group-hover:bg-primary/15 transition-colors duration-200">
            <Eye className="w-4 h-4 text-primary" strokeWidth={1.5} />
          </div>
          <span className="hidden sm:inline text-sm font-semibold tracking-tight">
            Argus<span className="text-muted-foreground font-normal">Core</span>
          </span>
        </Link>
      </div>
    </header>
  );
}

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
    <div className="flex items-center justify-between p-2.5 rounded-lg bg-muted/10 border border-border/50 hover:border-primary/20 transition-all duration-200">
      <div className="flex items-center gap-2.5 min-w-0">
        <Icon className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        <span className="text-xs text-muted-foreground truncate">{label}</span>
      </div>
      <div className="flex items-center gap-3 shrink-0">
        <div className="w-20 h-1.5 bg-border rounded-full overflow-hidden">
          <div 
            className={cn(
              'h-full rounded-full transition-all duration-1000 ease-out',
              score >= 80 ? 'bg-verdict-authentic' :
              score >= 60 ? 'bg-verdict-likely-authentic' :
              score >= 40 ? 'bg-verdict-uncertain' :
              score >= 20 ? 'bg-verdict-likely-fake' : 
              'bg-verdict-fake'
            )}
            style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
          />
        </div>
        <span className={cn('text-xs font-semibold tabular-nums w-8 text-right', scoreColor)}>
          {score.toFixed(0)}
        </span>
      </div>
    </div>
  );
}
