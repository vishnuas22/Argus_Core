/**
 * Argus Core - Trust Score Gauge Component
 * =========================================
 * D3.js radial gauge displaying Trust Score (0-100) with animated transitions.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/results/TrustScoreGauge.tsx
 * 
 * Role: Display animated radial gauge visualization of Trust Score.
 * Uses D3.js for smooth animations and color transitions based on verdict thresholds.
 * 
 * Integration:
 * - Imports: lib/d3/gauge, types/analysis
 * - Used by: ResultsPanel.tsx, analysis/[id]/page.tsx
 * - Inputs: score (0-100), confidence (0-1), verdict
 * - Outputs: Animated SVG gauge with score display
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton gauge
 * - Animated: Default true, animates on mount and updates
 * - Accessibility: ARIA labels for screen readers
 * - data-testid: trust-score-gauge, gauge-score-value, gauge-confidence
 */

'use client';

import React, { useRef, useEffect, useState, useMemo } from 'react';
import { cn } from '@/lib/utils';
import { 
  createGauge, 
  getScoreColor, 
  getVerdictColor, 
  calculateArcPath,
  GaugeInstance,
  DEFAULT_GAUGE_CONFIG 
} from '@/lib/d3/gauge';
import type { Verdict } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for TrustScoreGauge component
 */
export interface TrustScoreGaugeProps {
  /** Trust score value (0-100) */
  score: number;
  /** Confidence level (0-1) */
  confidence?: number;
  /** Verdict classification */
  verdict?: Verdict;
  /** Size of the gauge in pixels (default: 200) */
  size?: number;
  /** Whether to animate the gauge (default: true) */
  animated?: boolean;
  /** Whether to show the score label in center (default: true) */
  showLabel?: boolean;
  /** Whether to show confidence indicator (default: true) */
  showConfidence?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Animation duration in ms (default: 1000) */
  animationDuration?: number;
}

// ============== CONSTANTS ==============

/**
 * Size presets for different use cases
 */
export const GAUGE_SIZES = {
  sm: 120,
  md: 200,
  lg: 280,
  xl: 360,
} as const;

// ============== COMPONENT ==============

/**
 * TrustScoreGauge Component
 * 
 * Displays an animated radial gauge showing the Trust Score.
 * Color changes based on score thresholds matching verdict classification.
 * 
 * @example
 * ```tsx
 * <TrustScoreGauge
 *   score={85}
 *   confidence={0.92}
 *   verdict="authentic"
 *   size={200}
 *   animated
 * />
 * ```
 */
export function TrustScoreGauge({
  score,
  confidence,
  verdict,
  size = 200,
  animated = true,
  showLabel = true,
  showConfidence = true,
  className,
  animationDuration = 1000,
}: TrustScoreGaugeProps) {
  // Clamp and sanitize score
  const clampedScore = useMemo(() => {
    if (isNaN(score)) return 0;
    if (!isFinite(score)) return score > 0 ? 100 : 0;
    return Math.max(0, Math.min(100, Math.round(score)));
  }, [score]);

  const svgRef = useRef<SVGSVGElement>(null);
  const gaugeRef = useRef<GaugeInstance | null>(null);
  const [displayScore, setDisplayScore] = useState(clampedScore);
  const [isInitialized, setIsInitialized] = useState(false);

  // Calculate dimensions
  const innerRadius = size * 0.3;
  const outerRadius = size * 0.425;

  // Update displayScore immediately if not animated
  useEffect(() => {
    if (!animated) {
      setDisplayScore(clampedScore);
    }
  }, [clampedScore, animated]);

  // Get colors
  const scoreColor = useMemo(() => {
    return verdict ? getVerdictColor(verdict) : getScoreColor(clampedScore);
  }, [clampedScore, verdict]);

  // Initialize D3 gauge
  useEffect(() => {
    if (!svgRef.current) return;

    // Create gauge instance
    gaugeRef.current = createGauge(svgRef.current, clampedScore, {
      width: size,
      height: size,
      innerRadius,
      outerRadius,
      animated,
      animationDuration,
    });

    setIsInitialized(true);

    // Animate display score
    if (animated) {
      const startTime = Date.now();
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / animationDuration, 1);
        // Ease out elastic
        const eased = 1 - Math.pow(1 - progress, 3);
        setDisplayScore(Math.round(clampedScore * eased));
        
        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };
      requestAnimationFrame(animate);
    } else {
      // For non-animated mode (e.g., tests), set immediately
      setDisplayScore(clampedScore);
    }

    return () => {
      gaugeRef.current?.destroy();
      gaugeRef.current = null;
    };
  }, [size, innerRadius, outerRadius, animated, animationDuration]);

  // Update gauge when score changes
  useEffect(() => {
    if (!isInitialized || !gaugeRef.current) return;

    gaugeRef.current.update(clampedScore, verdict);

    // Animate display score update
    if (animated) {
      const currentDisplay = displayScore;
      const startTime = Date.now();
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / animationDuration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        setDisplayScore(Math.round(currentDisplay + (clampedScore - currentDisplay) * eased));
        
        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };
      requestAnimationFrame(animate);
    } else {
      setDisplayScore(clampedScore);
    }
  }, [clampedScore, verdict, isInitialized, animated]);

  // Calculate confidence ring dimensions
  const confidenceRingRadius = outerRadius + 12;
  const confidenceRingWidth = 4;

  return (
    <div
      className={cn(
        'relative inline-flex items-center justify-center',
        className
      )}
      style={{ width: size, height: size }}
      data-testid="trust-score-gauge"
      role="img"
      aria-label={`Trust Score: ${clampedScore} out of 100${verdict ? `, ${verdict.replace('_', ' ')}` : ''}${confidence ? `, ${Math.round(confidence * 100)}% confidence` : ''}`}
    >
      {/* Main D3 Gauge SVG */}
      <svg
        ref={svgRef}
        className="absolute inset-0"
        aria-hidden="true"
      />

      {/* Confidence Ring (optional) */}
      {showConfidence && confidence !== undefined && (
        <svg
          className="absolute inset-0 pointer-events-none"
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          aria-hidden="true"
        >
          {/* Confidence background ring */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={confidenceRingRadius}
            fill="none"
            stroke="currentColor"
            strokeWidth={confidenceRingWidth}
            className="text-muted opacity-20"
            strokeDasharray={`${2 * Math.PI * confidenceRingRadius}`}
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
          />
          {/* Confidence filled ring */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={confidenceRingRadius}
            fill="none"
            stroke={scoreColor}
            strokeWidth={confidenceRingWidth}
            strokeLinecap="round"
            opacity={0.5}
            strokeDasharray={`${2 * Math.PI * confidenceRingRadius * confidence} ${2 * Math.PI * confidenceRingRadius}`}
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
            className="transition-all duration-1000 ease-out"
          />
        </svg>
      )}

      {/* Center Label */}
      {showLabel && (
        <div
          className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none"
          style={{ padding: size * 0.15 }}
        >
          {/* Score Value */}
          <span
            className={cn(
              'font-bold tabular-nums leading-none transition-colors duration-500',
              size >= 200 ? 'text-4xl' : size >= 150 ? 'text-3xl' : 'text-2xl'
            )}
            style={{ color: scoreColor }}
            data-testid="gauge-score-value"
          >
            {displayScore}
          </span>
          
          {/* Out of 100 label */}
          <span
            className={cn(
              'text-muted-foreground mt-1',
              size >= 200 ? 'text-sm' : 'text-xs'
            )}
          >
            / 100
          </span>

          {/* Confidence text */}
          {showConfidence && confidence !== undefined && (
            <span
              className={cn(
                'text-muted-foreground mt-2',
                size >= 200 ? 'text-xs' : 'text-[10px]'
              )}
              data-testid="gauge-confidence"
            >
              {Math.round(confidence * 100)}% confidence
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ============== SKELETON ==============

/**
 * Skeleton loader for TrustScoreGauge
 */
export function TrustScoreGaugeSkeleton({
  size = 200,
  className,
}: {
  size?: number;
  className?: string;
}) {
  const innerRadius = size * 0.3;
  const outerRadius = size * 0.425;
  
  return (
    <div
      className={cn(
        'relative inline-flex items-center justify-center animate-pulse',
        className
      )}
      style={{ width: size, height: size }}
      data-testid="trust-score-gauge-skeleton"
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="opacity-30"
      >
        {/* Background arc */}
        <path
          d={calculateArcPath(100, { innerRadius, outerRadius })}
          fill="currentColor"
          transform={`translate(${size / 2}, ${size / 2})`}
          className="text-muted"
        />
      </svg>
      
      {/* Center skeleton */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div 
          className="bg-muted rounded"
          style={{ width: size * 0.25, height: size * 0.15 }}
        />
        <div 
          className="bg-muted rounded mt-2"
          style={{ width: size * 0.15, height: size * 0.05 }}
        />
      </div>
    </div>
  );
}

// ============== COMPACT VARIANT ==============

/**
 * Compact version of TrustScoreGauge for inline/card use
 */
export function TrustScoreGaugeCompact({
  score,
  confidence,
  verdict,
  className,
}: Pick<TrustScoreGaugeProps, 'score' | 'confidence' | 'verdict' | 'className'>) {
  const scoreColor = verdict ? getVerdictColor(verdict) : getScoreColor(score);
  
  return (
    <div
      className={cn(
        'flex items-center gap-3',
        className
      )}
      data-testid="trust-score-gauge-compact"
    >
      {/* Mini gauge */}
      <div
        className="relative w-12 h-12 rounded-full flex items-center justify-center"
        style={{
          background: `conic-gradient(${scoreColor} ${score * 3.6}deg, hsl(var(--muted)) 0deg)`,
        }}
      >
        <div className="absolute inset-1 rounded-full bg-background flex items-center justify-center">
          <span 
            className="text-sm font-bold tabular-nums"
            style={{ color: scoreColor }}
          >
            {score}
          </span>
        </div>
      </div>
      
      {/* Labels */}
      <div className="flex flex-col">
        <span className="text-sm font-medium">Trust Score</span>
        {confidence !== undefined && (
          <span className="text-xs text-muted-foreground">
            {Math.round(confidence * 100)}% confidence
          </span>
        )}
      </div>
    </div>
  );
}

export default TrustScoreGauge;
