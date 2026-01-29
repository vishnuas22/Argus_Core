/**
 * Argus Core - Text Analysis Panel Component
 * ==========================================
 * Detailed text analysis results with AI-generated content detection.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/modality/TextAnalysisPanel.tsx
 * 
 * Role: Display comprehensive AI-generated text detection results.
 * Shows perplexity scores, burstiness analysis, and RADAR model classification.
 * 
 * Integration:
 * - Used by: ModalityTabs.tsx (lazy loaded)
 * - Backend: Uses TextResult from GET /api/v1/analyze/{id}/detail
 *   - ai_probability: float (0-1) - Probability text is AI-generated
 *   - perplexity_score: float - GPT-2 perplexity score
 *   - burstiness_score: float - Sentence length variance
 *   - radar_score: Optional[float] - RADAR classifier score
 * 
 * Backend Analysis (from analyzers/text.py):
 * - Model: RADAR (Robust AI-text Detection via Adversarial leaRning)
 * - Features: Perplexity analysis, burstiness measurement
 * - Detection: Low perplexity = likely AI (too predictable)
 * - Detection: Low burstiness = likely AI (uniform variance)
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton loaders for each section
 * - Empty state: Shows message when no text data available
 * - Error state: Displays API errors gracefully
 * - Accessibility: Proper headings, ARIA labels, keyboard navigation
 * - data-testid: text-panel, text-summary-section, text-perplexity-section, text-burstiness-section
 */

'use client';

import React, { useMemo } from 'react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  FileText,
  Activity,
  Brain,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  BarChart3,
  Sparkles,
  Gauge,
  TrendingDown,
  TrendingUp,
  Info,
  Zap,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { TextResult } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for TextAnalysisPanel component
 */
export interface TextAnalysisPanelProps {
  /** Text analysis result from backend */
  result: TextResult;
  /** Analysis ID for fetching additional data */
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
 * AI probability thresholds for status classification
 * Higher AI probability = more likely synthetic
 */
const AI_THRESHOLDS = {
  good: 0.3,    // < 30% AI probability = likely human
  warning: 0.6, // 30-60% = uncertain
  // > 60% = likely AI
} as const;

/**
 * Perplexity thresholds
 * Lower perplexity = more likely AI (too predictable)
 */
const PERPLEXITY_THRESHOLDS = {
  low: 20,      // Very low = likely AI
  normal: 50,   // Normal range
  high: 100,    // High = likely human
} as const;

/**
 * Burstiness thresholds
 * Lower burstiness = more likely AI (uniform variance)
 */
const BURSTINESS_THRESHOLDS = {
  low: 0.3,     // Very low = likely AI
  normal: 0.5,  // Normal range
  high: 0.7,    // High = likely human
} as const;

/**
 * Get status based on AI probability (lower is better for authenticity)
 */
function getAIProbabilityStatus(probability: number): 'good' | 'warning' | 'danger' {
  if (probability <= AI_THRESHOLDS.good) return 'good';
  if (probability <= AI_THRESHOLDS.warning) return 'warning';
  return 'danger';
}

/**
 * Get status based on perplexity (higher is better)
 */
function getPerplexityStatus(score: number): 'good' | 'warning' | 'danger' {
  if (score >= PERPLEXITY_THRESHOLDS.high) return 'good';
  if (score >= PERPLEXITY_THRESHOLDS.normal) return 'warning';
  return 'danger';
}

/**
 * Get status based on burstiness (higher is better)
 */
function getBurstinessStatus(score: number): 'good' | 'warning' | 'danger' {
  if (score >= BURSTINESS_THRESHOLDS.high) return 'good';
  if (score >= BURSTINESS_THRESHOLDS.normal) return 'warning';
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
 * TextAnalysisPanel Component
 * 
 * Displays detailed AI-generated text detection results with perplexity,
 * burstiness, and RADAR model analysis.
 * 
 * @example
 * ```tsx
 * <TextAnalysisPanel
 *   result={analysisDetail.text_result}
 *   analysisId={analysisId}
 * />
 * ```
 */
export function TextAnalysisPanel({
  result,
  analysisId,
  className,
  compact = false,
}: TextAnalysisPanelProps) {
  // ============== COMPUTED VALUES ==============

  /**
   * Calculate human probability (inverse of AI probability)
   */
  const humanProbability = useMemo(() => {
    return 1 - result.ai_probability;
  }, [result.ai_probability]);

  /**
   * Overall status based on AI probability
   */
  const overallStatus = useMemo(() => {
    return getAIProbabilityStatus(result.ai_probability);
  }, [result.ai_probability]);

  /**
   * Score indicators for summary display
   */
  const scoreIndicators: ScoreIndicator[] = useMemo(() => {
    const indicators: ScoreIndicator[] = [];

    // Human probability (inverse of AI probability)
    indicators.push({
      label: 'Human Origin',
      value: humanProbability,
      icon: Brain,
      description: 'Probability of human authorship',
      status: getAIProbabilityStatus(result.ai_probability),
    });

    // Perplexity indicator
    const perplexityNormalized = Math.min(result.perplexity_score / 150, 1);
    indicators.push({
      label: 'Perplexity',
      value: perplexityNormalized,
      icon: Gauge,
      description: result.perplexity_score < PERPLEXITY_THRESHOLDS.low 
        ? 'Unusually predictable' 
        : 'Natural variance',
      status: getPerplexityStatus(result.perplexity_score),
    });

    // Burstiness indicator
    indicators.push({
      label: 'Burstiness',
      value: result.burstiness_score,
      icon: Activity,
      description: result.burstiness_score < BURSTINESS_THRESHOLDS.low
        ? 'Uniform sentence structure'
        : 'Natural variation',
      status: getBurstinessStatus(result.burstiness_score),
    });

    return indicators;
  }, [result, humanProbability]);

  /**
   * Determine if content is likely AI-generated
   */
  const isLikelyAI = result.ai_probability > 0.5;

  // ============== RENDER ==============

  return (
    <div
      className={cn('space-y-6', className)}
      data-testid="text-panel"
      role="region"
      aria-label="Text Analysis Results"
    >
      {/* Summary Section */}
      <TextSummarySection
        result={result}
        humanProbability={humanProbability}
        indicators={scoreIndicators}
        overallStatus={overallStatus}
        compact={compact}
      />

      {/* Perplexity Analysis Section */}
      <PerplexityAnalysisSection
        perplexityScore={result.perplexity_score}
        compact={compact}
      />

      {/* Burstiness Analysis Section */}
      <BurstinessAnalysisSection
        burstinessScore={result.burstiness_score}
        compact={compact}
      />

      {/* RADAR Model Section (if available) */}
      {result.radar_score !== undefined && result.radar_score !== null && (
        <RadarModelSection
          radarScore={result.radar_score}
          compact={compact}
        />
      )}

      {/* Detection Details Section */}
      <DetectionDetailsSection
        result={result}
        isLikelyAI={isLikelyAI}
        compact={compact}
      />
    </div>
  );
}

// ============== SUB-COMPONENTS ==============

/**
 * Summary section showing overall text analysis metrics
 */
function TextSummarySection({
  result,
  humanProbability,
  indicators,
  overallStatus,
  compact,
}: {
  result: TextResult;
  humanProbability: number;
  indicators: ScoreIndicator[];
  overallStatus: 'good' | 'warning' | 'danger';
  compact: boolean;
}) {
  const isLikelyAI = result.ai_probability > 0.5;

  return (
    <Card data-testid="text-summary-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-primary" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              Text Analysis Summary
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', STATUS_COLORS[overallStatus])}>
            {isLikelyAI 
              ? `${(result.ai_probability * 100).toFixed(1)}% AI` 
              : `${(humanProbability * 100).toFixed(1)}% Human`}
          </Badge>
        </div>
        {!compact && (
          <CardDescription>
            RADAR Model AI-Generated Text Detection Analysis
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

        {/* AI Probability Bar */}
        <div className="mt-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">AI Generation Probability</span>
            <span className={cn(
              'text-sm font-medium',
              result.ai_probability > 0.5 ? 'text-red-500' : 'text-green-500'
            )}>
              {(result.ai_probability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="relative h-3 bg-muted rounded-full overflow-hidden">
            <div
              className={cn(
                'absolute left-0 top-0 h-full rounded-full transition-all duration-500',
                result.ai_probability <= 0.3 ? 'bg-green-500' :
                result.ai_probability <= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
              )}
              style={{ width: `${result.ai_probability * 100}%` }}
            />
            {/* Threshold markers */}
            <div className="absolute left-[30%] top-0 h-full w-px bg-muted-foreground/30" />
            <div className="absolute left-[60%] top-0 h-full w-px bg-muted-foreground/30" />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Human</span>
            <span>Uncertain</span>
            <span>AI-Generated</span>
          </div>
        </div>

        {/* Word Count (if available) */}
        {result.word_count && result.word_count > 0 && (
          <div className="mt-4 p-3 rounded-lg bg-muted/50 flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Words Analyzed</span>
            <span className="font-medium">{result.word_count.toLocaleString()}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Perplexity analysis section
 */
function PerplexityAnalysisSection({
  perplexityScore,
  compact,
}: {
  perplexityScore: number;
  compact: boolean;
}) {
  const status = getPerplexityStatus(perplexityScore);
  const isLow = perplexityScore < PERPLEXITY_THRESHOLDS.low;
  const isNormal = perplexityScore >= PERPLEXITY_THRESHOLDS.normal;

  // Normalize perplexity for display (typical range 0-150)
  const normalizedPerplexity = Math.min((perplexityScore / 150) * 100, 100);

  return (
    <Card data-testid="text-perplexity-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Gauge className="h-5 w-5 text-blue-500" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              Perplexity Analysis
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', STATUS_COLORS[status])}>
            {perplexityScore.toFixed(1)}
          </Badge>
        </div>
        <CardDescription>
          GPT-2 based language model perplexity measurement
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Perplexity Explanation */}
        {isLow ? (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Low Perplexity Detected</AlertTitle>
            <AlertDescription>
              The text shows unusually low perplexity ({perplexityScore.toFixed(1)}), 
              indicating highly predictable patterns. AI-generated text often has 
              perplexity below 20 due to its statistical predictability.
            </AlertDescription>
          </Alert>
        ) : isNormal ? (
          <Alert className="border-green-500/20 bg-green-500/5">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            <AlertTitle className="text-green-700 dark:text-green-400">Normal Perplexity</AlertTitle>
            <AlertDescription className="text-green-600 dark:text-green-300">
              The text exhibits natural language variance typical of human writing.
              Perplexity score of {perplexityScore.toFixed(1)} falls within expected human range.
            </AlertDescription>
          </Alert>
        ) : (
          <Alert className="border-yellow-500/20 bg-yellow-500/5">
            <Info className="h-4 w-4 text-yellow-500" />
            <AlertTitle className="text-yellow-700 dark:text-yellow-400">Moderate Perplexity</AlertTitle>
            <AlertDescription className="text-yellow-600 dark:text-yellow-300">
              Perplexity score of {perplexityScore.toFixed(1)} is in the moderate range.
              This could indicate either human or AI authorship.
            </AlertDescription>
          </Alert>
        )}

        {/* Perplexity Scale */}
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Perplexity Scale</span>
            <span className="text-sm text-muted-foreground">
              {perplexityScore.toFixed(1)} / 150+
            </span>
          </div>
          <div className="relative h-4 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full overflow-hidden">
            <div
              className="absolute top-0 h-full w-1 bg-white shadow-lg transform -translate-x-1/2"
              style={{ left: `${normalizedPerplexity}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span className="flex items-center gap-1">
              <TrendingDown className="h-3 w-3" />
              Low (AI-like)
            </span>
            <span>Medium</span>
            <span className="flex items-center gap-1">
              High (Human-like)
              <TrendingUp className="h-3 w-3" />
            </span>
          </div>
        </div>

        {/* Technical Details */}
        {!compact && (
          <div className="mt-4 p-4 rounded-lg bg-muted/30 border border-dashed">
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Info className="h-4 w-4" aria-hidden="true" />
              What is Perplexity?
            </h4>
            <p className="text-xs text-muted-foreground">
              Perplexity measures how "surprised" a language model is by text. Lower values 
              indicate highly predictable text (often AI-generated), while higher values 
              suggest more creative, unpredictable human writing. GPT-2 is used as the 
              reference model for measuring text predictability.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Burstiness analysis section
 */
function BurstinessAnalysisSection({
  burstinessScore,
  compact,
}: {
  burstinessScore: number;
  compact: boolean;
}) {
  const status = getBurstinessStatus(burstinessScore);
  const isLow = burstinessScore < BURSTINESS_THRESHOLDS.low;
  const isHigh = burstinessScore >= BURSTINESS_THRESHOLDS.high;

  return (
    <Card data-testid="text-burstiness-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-purple-500" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              Burstiness Analysis
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', STATUS_COLORS[status])}>
            {(burstinessScore * 100).toFixed(1)}%
          </Badge>
        </div>
        <CardDescription>
          Sentence length variance and structural diversity
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Burstiness Explanation */}
        {isLow ? (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Low Burstiness Detected</AlertTitle>
            <AlertDescription>
              The text shows uniform sentence structure ({(burstinessScore * 100).toFixed(1)}% variance). 
              AI-generated content often lacks the natural variation in sentence length 
              that characterizes human writing.
            </AlertDescription>
          </Alert>
        ) : isHigh ? (
          <Alert className="border-green-500/20 bg-green-500/5">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            <AlertTitle className="text-green-700 dark:text-green-400">Natural Burstiness</AlertTitle>
            <AlertDescription className="text-green-600 dark:text-green-300">
              The text exhibits natural sentence length variation typical of human writing.
              This "bursty" pattern is characteristic of authentic human-authored content.
            </AlertDescription>
          </Alert>
        ) : (
          <Alert className="border-yellow-500/20 bg-yellow-500/5">
            <Info className="h-4 w-4 text-yellow-500" />
            <AlertTitle className="text-yellow-700 dark:text-yellow-400">Moderate Burstiness</AlertTitle>
            <AlertDescription className="text-yellow-600 dark:text-yellow-300">
              Sentence structure variance is in the moderate range.
              Further analysis recommended for definitive classification.
            </AlertDescription>
          </Alert>
        )}

        {/* Visual Representation */}
        <div className="grid grid-cols-2 gap-4 mt-4">
          {/* AI Pattern */}
          <div className={cn(
            'p-4 rounded-lg border',
            isLow ? 'bg-red-500/5 border-red-500/20' : 'bg-muted/30'
          )}>
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-red-500" aria-hidden="true" />
              AI Pattern
            </h4>
            {/* Uniform bars representing AI text */}
            <div className="space-y-1">
              {[0.6, 0.58, 0.62, 0.59, 0.61].map((height, i) => (
                <div
                  key={i}
                  className="h-2 bg-red-500/40 rounded"
                  style={{ width: `${height * 100}%` }}
                />
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Uniform sentence lengths
            </p>
          </div>

          {/* Human Pattern */}
          <div className={cn(
            'p-4 rounded-lg border',
            isHigh ? 'bg-green-500/5 border-green-500/20' : 'bg-muted/30'
          )}>
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Brain className="h-4 w-4 text-green-500" aria-hidden="true" />
              Human Pattern
            </h4>
            {/* Variable bars representing human text */}
            <div className="space-y-1">
              {[0.3, 0.85, 0.45, 0.95, 0.25].map((height, i) => (
                <div
                  key={i}
                  className="h-2 bg-green-500/40 rounded"
                  style={{ width: `${height * 100}%` }}
                />
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Variable sentence lengths
            </p>
          </div>
        </div>

        {/* Burstiness Progress Bar */}
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Variance Score</span>
            <span className="text-sm text-muted-foreground">
              {(burstinessScore * 100).toFixed(1)}%
            </span>
          </div>
          <Progress value={burstinessScore * 100} className="h-3" />
          <p className="text-xs text-muted-foreground mt-1">
            Higher variance indicates more natural, human-like writing patterns
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * RADAR model analysis section
 */
function RadarModelSection({
  radarScore,
  compact,
}: {
  radarScore: number;
  compact: boolean;
}) {
  const isLikelyAI = radarScore > 0.5;
  const status = radarScore <= 0.3 ? 'good' : radarScore <= 0.6 ? 'warning' : 'danger';

  return (
    <Card data-testid="text-radar-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-orange-500" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              RADAR Model Classification
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', STATUS_COLORS[status])}>
            {isLikelyAI ? 'AI Detected' : 'Human Likely'}
          </Badge>
        </div>
        <CardDescription>
          Adversarial learning-based AI text classifier
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* RADAR Score Display */}
        <div className="flex items-center justify-center">
          <div className={cn(
            'w-32 h-32 rounded-full border-8 flex items-center justify-center',
            status === 'good' ? 'border-green-500 bg-green-500/10' :
            status === 'warning' ? 'border-yellow-500 bg-yellow-500/10' :
            'border-red-500 bg-red-500/10'
          )}>
            <div className="text-center">
              <span className={cn(
                'text-3xl font-bold',
                status === 'good' ? 'text-green-500' :
                status === 'warning' ? 'text-yellow-500' :
                'text-red-500'
              )}>
                {(radarScore * 100).toFixed(0)}
              </span>
              <p className="text-xs text-muted-foreground">AI Score</p>
            </div>
          </div>
        </div>

        {/* Classification Result */}
        <div className={cn(
          'p-4 rounded-lg text-center',
          status === 'good' ? 'bg-green-500/10' :
          status === 'warning' ? 'bg-yellow-500/10' :
          'bg-red-500/10'
        )}>
          {isLikelyAI ? (
            <div className="flex items-center justify-center gap-2">
              <XCircle className="h-5 w-5 text-red-500" aria-hidden="true" />
              <span className="font-medium text-red-700 dark:text-red-400">
                RADAR classifies this text as AI-generated
              </span>
            </div>
          ) : (
            <div className="flex items-center justify-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-500" aria-hidden="true" />
              <span className="font-medium text-green-700 dark:text-green-400">
                RADAR classifies this text as human-written
              </span>
            </div>
          )}
        </div>

        {/* Technical Details */}
        {!compact && (
          <div className="mt-4 p-4 rounded-lg bg-muted/30 border border-dashed">
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Info className="h-4 w-4" aria-hidden="true" />
              About RADAR
            </h4>
            <p className="text-xs text-muted-foreground">
              RADAR (Robust AI-text Detection via Adversarial leaRning) is a classifier 
              trained to distinguish AI-generated text from human text using adversarial 
              training techniques. It's designed to be robust against paraphrasing and 
              other evasion techniques.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Detection details section with technical information
 */
function DetectionDetailsSection({
  result,
  isLikelyAI,
  compact,
}: {
  result: TextResult;
  isLikelyAI: boolean;
  compact: boolean;
}) {
  const confidenceLevel = result.ai_probability > 0.8 || result.ai_probability < 0.2
    ? 'High'
    : 'Moderate';

  return (
    <Card data-testid="text-details-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center gap-2">
          <Info className="h-5 w-5 text-slate-500" aria-hidden="true" />
          <CardTitle className={compact ? 'text-base' : 'text-lg'}>
            Detection Summary
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Key Findings */}
          <div className="p-4 rounded-lg bg-muted/50">
            <h4 className="text-sm font-medium mb-3">Key Findings</h4>
            <ul className="space-y-2">
              <li className="flex items-start gap-2 text-sm">
                {isLikelyAI ? (
                  <XCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                ) : (
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                )}
                <span>
                  AI probability: {(result.ai_probability * 100).toFixed(1)}%
                  {result.ai_probability > 0.5 ? ' (elevated)' : ' (normal)'}
                </span>
              </li>
              <li className="flex items-start gap-2 text-sm">
                {result.perplexity_score < PERPLEXITY_THRESHOLDS.low ? (
                  <XCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                ) : (
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                )}
                <span>
                  Perplexity: {result.perplexity_score.toFixed(1)}
                  {result.perplexity_score < PERPLEXITY_THRESHOLDS.low ? ' (low - AI-like)' : ' (normal)'}
                </span>
              </li>
              <li className="flex items-start gap-2 text-sm">
                {result.burstiness_score < BURSTINESS_THRESHOLDS.low ? (
                  <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                ) : (
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                )}
                <span>
                  Burstiness: {(result.burstiness_score * 100).toFixed(1)}%
                  {result.burstiness_score < BURSTINESS_THRESHOLDS.low ? ' (uniform - AI-like)' : ' (varied)'}
                </span>
              </li>
              {result.radar_score !== undefined && result.radar_score !== null && (
                <li className="flex items-start gap-2 text-sm">
                  {result.radar_score > 0.5 ? (
                    <XCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                  ) : (
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                  )}
                  <span>
                    RADAR score: {(result.radar_score * 100).toFixed(1)}%
                    {result.radar_score > 0.5 ? ' (AI detected)' : ' (human likely)'}
                  </span>
                </li>
              )}
            </ul>
          </div>

          {/* Model Information */}
          {!compact && (
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Detection Model:</span>
                <p className="font-medium">{result.model_used || 'RADAR'}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Confidence Level:</span>
                <p className="font-medium">{confidenceLevel}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Analysis Status:</span>
                <p className={cn(
                  'font-medium',
                  isLikelyAI ? 'text-red-500' : 'text-green-500'
                )}>
                  {isLikelyAI ? 'Likely AI-Generated' : 'Likely Human-Written'}
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Confidence Score:</span>
                <p className="font-medium">{(result.confidence * 100).toFixed(1)}%</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ============== SKELETON LOADER ==============

/**
 * Skeleton loader for TextAnalysisPanel
 */
export function TextAnalysisPanelSkeleton() {
  return (
    <div className="space-y-6" data-testid="text-panel-skeleton">
      {/* Summary Skeleton */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-6 w-24" />
          </div>
          <Skeleton className="h-4 w-64 mt-2" />
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-24 w-full" />
            ))}
          </div>
          <Skeleton className="h-12 w-full mt-4" />
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
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-12 w-full mt-4" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ============== EXPORTS ==============

export default TextAnalysisPanel;
