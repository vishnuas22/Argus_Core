/**
 * Argus Core - Verdict Badge Component
 * =====================================
 * Verdict display badge with color coding, icon, and optional description.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/results/VerdictBadge.tsx
 * 
 * Role: Display verdict classification with appropriate visual styling.
 * Provides consistent verdict representation across the application.
 * 
 * Integration:
 * - Imports: components/ui/badge, lucide-react icons
 * - Used by: ResultsPanel.tsx, analysis/[id]/page.tsx, AnalysisCard.tsx
 * - Backend: Maps to Verdict enum from backend/schemas/schemas.py
 * 
 * Verdict Thresholds (from backend config.py):
 * - 80-100: Authentic (green)
 * - 60-79: Likely Authentic (lime)
 * - 40-59: Uncertain (yellow)
 * - 20-39: Likely Fake (orange)
 * - 0-19: Fake (red)
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton badge
 * - Accessibility: ARIA labels, color contrast compliant
 * - data-testid: verdict-badge, verdict-badge-icon, verdict-badge-label
 */

'use client';

import React, { useMemo } from 'react';
import { cn } from '@/lib/utils';
import { 
  ShieldCheck, 
  Shield, 
  AlertTriangle, 
  AlertOctagon, 
  XOctagon,
  HelpCircle,
  type LucideIcon
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import type { Verdict } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for VerdictBadge component
 */
export interface VerdictBadgeProps {
  /** Verdict classification */
  verdict: Verdict;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Whether to show the icon */
  showIcon?: boolean;
  /** Whether to show description text */
  showDescription?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Whether to animate the badge on mount */
  animated?: boolean;
}

/**
 * Configuration for each verdict type
 */
interface VerdictConfig {
  /** Display label */
  label: string;
  /** Short description */
  description: string;
  /** Detailed explanation */
  detailedExplanation: string;
  /** Icon component */
  icon: LucideIcon;
  /** Text/icon color class */
  color: string;
  /** Background color class */
  bgColor: string;
  /** Border color class */
  borderColor: string;
  /** Badge variant color for dark backgrounds */
  badgeBg: string;
  /** Ring/glow color for emphasis */
  ringColor: string;
}

// ============== VERDICT CONFIGURATION ==============

/**
 * Complete configuration for all verdict types
 * Colors are WCAG 2.1 AA compliant for accessibility
 */
export const VERDICT_CONFIG: Record<Verdict, VerdictConfig> = {
  authentic: {
    label: 'Authentic',
    description: 'High confidence authentic content',
    detailedExplanation: 'This content shows strong indicators of being authentic and unmanipulated. All detection models agree with high confidence.',
    icon: ShieldCheck,
    color: 'text-green-700 dark:text-green-400',
    bgColor: 'bg-green-50 dark:bg-green-950/50',
    borderColor: 'border-green-200 dark:border-green-800',
    badgeBg: 'bg-green-500',
    ringColor: 'ring-green-500/30',
  },
  likely_authentic: {
    label: 'Likely Authentic',
    description: 'Content appears authentic with minor concerns',
    detailedExplanation: 'This content appears to be authentic, though some minor anomalies were detected. These could be due to compression, editing software, or natural variations.',
    icon: Shield,
    color: 'text-lime-700 dark:text-lime-400',
    bgColor: 'bg-lime-50 dark:bg-lime-950/50',
    borderColor: 'border-lime-200 dark:border-lime-800',
    badgeBg: 'bg-lime-500',
    ringColor: 'ring-lime-500/30',
  },
  uncertain: {
    label: 'Uncertain',
    description: 'Analysis inconclusive - review recommended',
    detailedExplanation: 'The analysis could not determine authenticity with sufficient confidence. Manual review by an expert is recommended before drawing conclusions.',
    icon: AlertTriangle,
    color: 'text-yellow-700 dark:text-yellow-400',
    bgColor: 'bg-yellow-50 dark:bg-yellow-950/50',
    borderColor: 'border-yellow-200 dark:border-yellow-800',
    badgeBg: 'bg-yellow-500',
    ringColor: 'ring-yellow-500/30',
  },
  likely_fake: {
    label: 'Likely Fake',
    description: 'Content shows manipulation indicators',
    detailedExplanation: 'This content shows signs of synthetic generation or manipulation. Multiple detection models identified concerning patterns, though some authentic elements may be present.',
    icon: AlertOctagon,
    color: 'text-orange-700 dark:text-orange-400',
    bgColor: 'bg-orange-50 dark:bg-orange-950/50',
    borderColor: 'border-orange-200 dark:border-orange-800',
    badgeBg: 'bg-orange-500',
    ringColor: 'ring-orange-500/30',
  },
  fake: {
    label: 'Fake',
    description: 'High confidence manipulated content',
    detailedExplanation: 'This content is almost certainly synthetic or manipulated. Strong deepfake indicators were detected across multiple analysis dimensions.',
    icon: XOctagon,
    color: 'text-red-700 dark:text-red-400',
    bgColor: 'bg-red-50 dark:bg-red-950/50',
    borderColor: 'border-red-200 dark:border-red-800',
    badgeBg: 'bg-red-500',
    ringColor: 'ring-red-500/30',
  },
};

// ============== SIZE CONFIGURATION ==============

/**
 * Size variants configuration
 */
const SIZE_CONFIG = {
  sm: {
    badge: 'text-xs px-2 py-0.5',
    icon: 'h-3 w-3',
    gap: 'gap-1',
    description: 'text-xs',
  },
  md: {
    badge: 'text-sm px-3 py-1',
    icon: 'h-4 w-4',
    gap: 'gap-1.5',
    description: 'text-sm',
  },
  lg: {
    badge: 'text-base px-4 py-1.5',
    icon: 'h-5 w-5',
    gap: 'gap-2',
    description: 'text-base',
  },
} as const;

// ============== MAIN COMPONENT ==============

/**
 * VerdictBadge Component
 * 
 * Displays a verdict classification with appropriate styling and optional icon.
 * Supports multiple size variants and can show detailed description.
 * 
 * @example
 * ```tsx
 * // Basic usage
 * <VerdictBadge verdict="authentic" />
 * 
 * // Large size with description
 * <VerdictBadge 
 *   verdict="likely_fake" 
 *   size="lg" 
 *   showIcon 
 *   showDescription 
 * />
 * 
 * // Minimal badge (no icon)
 * <VerdictBadge verdict="uncertain" size="sm" showIcon={false} />
 * ```
 */
export function VerdictBadge({
  verdict,
  size = 'md',
  showIcon = true,
  showDescription = false,
  className,
  animated = false,
}: VerdictBadgeProps) {
  // Get configuration for this verdict
  const config = useMemo(() => VERDICT_CONFIG[verdict], [verdict]);
  const sizeStyles = SIZE_CONFIG[size];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        'inline-flex flex-col',
        sizeStyles.gap,
        animated && 'animate-in fade-in slide-in-from-bottom-2 duration-500',
        className
      )}
      data-testid="verdict-badge"
      role="status"
      aria-label={`Verdict: ${config.label}. ${config.description}`}
    >
      {/* Main Badge */}
      <Badge
        className={cn(
          'font-semibold border transition-all duration-300',
          sizeStyles.badge,
          sizeStyles.gap,
          config.bgColor,
          config.borderColor,
          config.color,
          'hover:ring-2',
          config.ringColor
        )}
        variant="outline"
      >
        {showIcon && (
          <Icon 
            className={cn(sizeStyles.icon, 'flex-shrink-0')} 
            aria-hidden="true"
            data-testid="verdict-badge-icon"
          />
        )}
        <span data-testid="verdict-badge-label">
          {config.label}
        </span>
      </Badge>

      {/* Description (optional) */}
      {showDescription && (
        <p 
          className={cn(
            'text-muted-foreground',
            sizeStyles.description
          )}
          data-testid="verdict-badge-description"
        >
          {config.description}
        </p>
      )}
    </div>
  );
}

// ============== VARIANT COMPONENTS ==============

/**
 * Compact verdict badge for list views and cards
 */
export function VerdictBadgeCompact({
  verdict,
  className,
}: Pick<VerdictBadgeProps, 'verdict' | 'className'>) {
  const config = VERDICT_CONFIG[verdict];
  const Icon = config.icon;

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 text-xs font-medium',
        config.color,
        className
      )}
      data-testid="verdict-badge-compact"
      title={config.description}
    >
      <Icon className="h-3 w-3" aria-hidden="true" />
      <span>{config.label}</span>
    </span>
  );
}

/**
 * Large verdict display for hero sections
 */
export function VerdictBadgeHero({
  verdict,
  className,
  showExplanation = true,
}: Pick<VerdictBadgeProps, 'verdict' | 'className'> & {
  showExplanation?: boolean;
}) {
  const config = VERDICT_CONFIG[verdict];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        'flex flex-col items-center text-center space-y-3 p-6 rounded-2xl border-2',
        config.bgColor,
        config.borderColor,
        className
      )}
      data-testid="verdict-badge-hero"
      role="status"
      aria-label={`Verdict: ${config.label}`}
    >
      {/* Large Icon */}
      <div 
        className={cn(
          'p-4 rounded-full',
          config.bgColor,
          'border-2',
          config.borderColor
        )}
      >
        <Icon 
          className={cn('h-12 w-12', config.color)} 
          aria-hidden="true"
          strokeWidth={1.5}
        />
      </div>

      {/* Label */}
      <div className="space-y-1">
        <h3 className={cn('text-2xl font-bold', config.color)}>
          {config.label}
        </h3>
        <p className="text-muted-foreground">
          {config.description}
        </p>
      </div>

      {/* Detailed explanation */}
      {showExplanation && (
        <p className="text-sm text-muted-foreground max-w-md">
          {config.detailedExplanation}
        </p>
      )}
    </div>
  );
}

/**
 * Verdict indicator dot (minimal)
 */
export function VerdictDot({
  verdict,
  size = 'md',
  className,
}: {
  verdict: Verdict;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}) {
  const config = VERDICT_CONFIG[verdict];
  
  const dotSize = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4',
  }[size];

  return (
    <span
      className={cn(
        'inline-block rounded-full',
        config.badgeBg,
        dotSize,
        className
      )}
      role="img"
      aria-label={`Verdict: ${config.label}`}
      title={config.label}
      data-testid="verdict-dot"
    />
  );
}

// ============== SKELETON ==============

/**
 * Skeleton loader for VerdictBadge
 */
export function VerdictBadgeSkeleton({
  size = 'md',
  className,
}: {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}) {
  const widths = {
    sm: 'w-16',
    md: 'w-24',
    lg: 'w-32',
  };

  const heights = {
    sm: 'h-5',
    md: 'h-6',
    lg: 'h-8',
  };

  return (
    <div
      className={cn(
        'animate-pulse rounded-full bg-muted',
        widths[size],
        heights[size],
        className
      )}
      data-testid="verdict-badge-skeleton"
    />
  );
}

// ============== UTILITY FUNCTIONS ==============

/**
 * Get verdict from score using backend thresholds
 * Matches VERDICT_THRESHOLD_* from backend config.py
 */
export function getVerdictFromScore(score: number): Verdict {
  if (score >= 80) return 'authentic';
  if (score >= 60) return 'likely_authentic';
  if (score >= 40) return 'uncertain';
  if (score >= 20) return 'likely_fake';
  return 'fake';
}

/**
 * Check if verdict indicates potential manipulation
 */
export function isManipulationVerdict(verdict: Verdict): boolean {
  return verdict === 'likely_fake' || verdict === 'fake';
}

/**
 * Check if verdict indicates authenticity
 */
export function isAuthenticVerdict(verdict: Verdict): boolean {
  return verdict === 'authentic' || verdict === 'likely_authentic';
}

/**
 * Get severity level for verdict (for sorting/prioritization)
 */
export function getVerdictSeverity(verdict: Verdict): number {
  const severityMap: Record<Verdict, number> = {
    fake: 5,
    likely_fake: 4,
    uncertain: 3,
    likely_authentic: 2,
    authentic: 1,
  };
  return severityMap[verdict];
}

export default VerdictBadge;
