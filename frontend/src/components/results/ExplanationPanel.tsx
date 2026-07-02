/**
 * Argus Core - Explanation Panel Component
 * =========================================
 * AI-generated explanation display with key findings, confidence statement,
 * and actionable recommendations.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/results/ExplanationPanel.tsx
 * 
 * Role: Display AI-generated explanations for analysis results. Shows summary,
 * confidence rationale, key findings, and recommendations in an organized layout.
 * 
 * Integration:
 * - Imports: components/ui/*, lucide-react icons
 * - Used by: ResultsPanel.tsx, analysis/[id]/page.tsx
 * - Backend: Maps to Explanation schema from backend/schemas/schemas.py
 * 
 * Features:
 * - Summary text with proper typography
 * - Confidence statement highlight box
 * - Key findings as bullet list with icons
 * - Recommendations section
 * - Collapsible sections for long content
 * - Copy to clipboard functionality
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton paragraphs
 * - Empty state: Shows message when no explanation available
 * - Accessibility: Semantic headings, ARIA labels
 * - data-testid: explanation-panel, explanation-summary, explanation-findings
 */

'use client';

import React, { useState, useCallback, useId } from 'react';
import { cn } from '@/lib/utils';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Collapsible, 
  CollapsibleContent, 
  CollapsibleTrigger 
} from '@/components/ui/collapsible';
import { 
  Info, 
  CheckCircle2, 
  AlertTriangle, 
  ArrowRight, 
  Copy, 
  Check,
  ChevronDown,
  ChevronUp,
  Shield,
  Lightbulb,
  FileText,
  Brain,
  type LucideIcon
} from 'lucide-react';
import type { Explanation, Verdict } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for ExplanationPanel component
 */
export interface ExplanationPanelProps {
  /** Explanation data from API */
  explanation: Explanation;
  /** Verdict for context-aware styling */
  verdict?: Verdict;
  /** Display variant */
  variant?: 'default' | 'compact' | 'expanded';
  /** Whether to show the card wrapper */
  showCard?: boolean;
  /** Whether to show copy button */
  showCopyButton?: boolean;
  /** Whether sections are collapsible */
  collapsible?: boolean;
  /** Initial collapsed state */
  defaultCollapsed?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============== CONSTANTS ==============

/**
 * Icons for different finding types based on content
 */
const FINDING_ICONS: Record<string, LucideIcon> = {
  face: Shield,
  audio: Info,
  temporal: Info,
  metadata: FileText,
  default: CheckCircle2,
};

/**
 * Get appropriate icon for a finding based on its content
 */
function getFindingIcon(finding: string): LucideIcon {
  const lowerFinding = finding.toLowerCase();
  if (lowerFinding.includes('face') || lowerFinding.includes('spatial')) return FINDING_ICONS.face;
  if (lowerFinding.includes('audio') || lowerFinding.includes('voice')) return FINDING_ICONS.audio;
  if (lowerFinding.includes('temporal') || lowerFinding.includes('frame')) return FINDING_ICONS.temporal;
  if (lowerFinding.includes('metadata') || lowerFinding.includes('c2pa')) return FINDING_ICONS.metadata;
  return FINDING_ICONS.default;
}

/**
 * Verdict-aware styling for confidence box
 */
const VERDICT_CONFIDENCE_STYLES: Record<Verdict, string> = {
  authentic: 'bg-green-50 dark:bg-green-950/30 border-green-200 dark:border-green-800',
  likely_authentic: 'bg-lime-50 dark:bg-lime-950/30 border-lime-200 dark:border-lime-800',
  uncertain: 'bg-yellow-50 dark:bg-yellow-950/30 border-yellow-200 dark:border-yellow-800',
  likely_fake: 'bg-orange-50 dark:bg-orange-950/30 border-orange-200 dark:border-orange-800',
  fake: 'bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800',
};

// ============== MAIN COMPONENT ==============

/**
 * ExplanationPanel Component
 * 
 * Displays AI-generated analysis explanation with structured sections
 * for summary, confidence, key findings, and recommendations.
 * 
 * @example
 * ```tsx
 * // Basic usage
 * <ExplanationPanel explanation={data.explanation} />
 * 
 * // With verdict-aware styling
 * <ExplanationPanel 
 *   explanation={explanation}
 *   verdict={verdict}
 *   showCopyButton
 *   collapsible
 * />
 * 
 * // Compact without card
 * <ExplanationPanel 
 *   explanation={explanation}
 *   variant="compact"
 *   showCard={false}
 * />
 * ```
 */
export function ExplanationPanel({
  explanation,
  verdict,
  variant = 'default',
  showCard = true,
  showCopyButton = false,
  collapsible = false,
  defaultCollapsed = false,
  className,
}: ExplanationPanelProps) {
  const componentId = useId();
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const [copied, setCopied] = useState(false);
  
  // ============== HANDLERS ==============
  
  const handleCopyToClipboard = useCallback(async () => {
    const text = [
      explanation.summary,
      explanation.confidence_statement && `\nConfidence: ${explanation.confidence_statement}`,
      explanation.key_findings?.length && `\nKey Findings:\n${explanation.key_findings.map(f => `• ${f}`).join('\n')}`,
      explanation.recommendations?.length && `\nRecommendations:\n${explanation.recommendations.map(r => `• ${r}`).join('\n')}`,
    ].filter(Boolean).join('\n');
    
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [explanation]);
  
  // ============== EMPTY STATE ==============
  
  if (!explanation || (!explanation.summary && !explanation.key_findings?.length)) {
    return (
      <EmptyExplanation showCard={showCard} className={className} />
    );
  }
  
  // ============== CONTENT ==============
  
  const content = (
    <div 
      className="space-y-4"
      data-testid="explanation-panel"
      role="region"
      aria-labelledby={`${componentId}-title`}
    >
      {/* Header with Copy Button */}
      {(variant !== 'compact' || showCopyButton) && (
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" aria-hidden="true" />
            <h3 
              id={`${componentId}-title`}
              className="text-lg font-semibold"
            >
              Analysis Summary
            </h3>
          </div>
          
          {showCopyButton && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopyToClipboard}
              className="h-8 px-2"
              aria-label={copied ? 'Copied to clipboard' : 'Copy explanation to clipboard'}
            >
              {copied ? (
                <Check className="h-4 w-4 text-green-500" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </Button>
          )}
        </div>
      )}
      
      {/* Summary Text */}
      {explanation.summary && (
        <p 
          className="text-muted-foreground leading-relaxed"
          data-testid="explanation-summary"
        >
          {explanation.summary}
        </p>
      )}
      
      {/* Confidence Statement */}
      {explanation.confidence_statement && (
        <ConfidenceBox
          statement={explanation.confidence_statement}
          verdict={verdict}
          componentId={componentId}
        />
      )}
      
      {/* Key Findings - Collapsible if enabled */}
      {explanation.key_findings && explanation.key_findings.length > 0 && (
        <CollapsibleSection
          title="Key Findings"
          icon={CheckCircle2}
          count={explanation.key_findings.length}
          collapsible={collapsible && explanation.key_findings.length > 3}
          defaultOpen={!isCollapsed}
          componentId={`${componentId}-findings`}
        >
          <KeyFindingsList 
            findings={explanation.key_findings}
            variant={variant}
          />
        </CollapsibleSection>
      )}
      
      {/* Recommendations */}
      {explanation.recommendations && explanation.recommendations.length > 0 && (
        <CollapsibleSection
          title="Recommendations"
          icon={Lightbulb}
          count={explanation.recommendations.length}
          collapsible={collapsible && explanation.recommendations.length > 3}
          defaultOpen={!isCollapsed}
          componentId={`${componentId}-recommendations`}
        >
          <RecommendationsList 
            recommendations={explanation.recommendations}
            variant={variant}
          />
        </CollapsibleSection>
      )}
    </div>
  );
  
  // ============== RENDER ==============
  
  if (!showCard) {
    return <div className={className}>{content}</div>;
  }
  
  return (
    <Card className={className}>
      <CardContent className="pt-6">
        {content}
      </CardContent>
    </Card>
  );
}

// ============== SUB-COMPONENTS ==============

/**
 * Confidence statement highlight box
 */
function ConfidenceBox({
  statement,
  verdict,
  componentId,
}: {
  statement: string;
  verdict?: Verdict;
  componentId: string;
}) {
  const boxStyle = verdict 
    ? VERDICT_CONFIDENCE_STYLES[verdict]
    : 'bg-muted/50 border-border';
  
  return (
    <div 
      className={cn(
        'p-4 rounded-lg border',
        boxStyle
      )}
      role="note"
      aria-labelledby={`${componentId}-confidence`}
    >
      <div className="flex items-start gap-3">
        <Shield className="h-5 w-5 mt-0.5 text-primary flex-shrink-0" aria-hidden="true" />
        <div>
          <p 
            id={`${componentId}-confidence`}
            className="text-sm font-medium mb-1"
          >
            Confidence Assessment
          </p>
          <p className="text-sm text-muted-foreground">
            {statement}
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * Collapsible section wrapper
 */
function CollapsibleSection({
  title,
  icon: Icon,
  count,
  collapsible,
  defaultOpen,
  componentId,
  children,
}: {
  title: string;
  icon: LucideIcon;
  count: number;
  collapsible: boolean;
  defaultOpen: boolean;
  componentId: string;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  if (!collapsible) {
    return (
      <div data-testid={`explanation-${title.toLowerCase().replace(' ', '-')}`}>
        <div className="flex items-center gap-2 mb-3">
          <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
          <h4 
            id={componentId}
            className="text-sm font-medium"
          >
            {title}
          </h4>
          <Badge variant="secondary" className="text-xs px-1.5">
            {count}
          </Badge>
        </div>
        {children}
      </div>
    );
  }
  
  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <button
          className="flex items-center gap-2 w-full text-left hover:bg-muted/50 -mx-2 px-2 py-1.5 rounded-md transition-colors"
          aria-expanded={isOpen}
          aria-controls={`${componentId}-content`}
        >
          <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
          <h4 
            id={componentId}
            className="text-sm font-medium flex-1"
          >
            {title}
          </h4>
          <Badge variant="secondary" className="text-xs px-1.5">
            {count}
          </Badge>
          {isOpen ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent 
        id={`${componentId}-content`}
        className="pt-2"
      >
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
}

/**
 * Key findings list
 */
function KeyFindingsList({
  findings,
  variant,
}: {
  findings: string[];
  variant: 'default' | 'compact' | 'expanded';
}) {
  return (
    <ul 
      className="space-y-2"
      data-testid="explanation-findings"
      role="list"
    >
      {findings.map((finding, idx) => {
        const Icon = getFindingIcon(finding);
        
        return (
          <li 
            key={idx} 
            className={cn(
              'flex items-start gap-2.5',
              variant === 'compact' ? 'text-xs' : 'text-sm'
            )}
            role="listitem"
          >
            <Icon 
              className={cn(
                'mt-0.5 text-primary flex-shrink-0',
                variant === 'compact' ? 'h-3 w-3' : 'h-4 w-4'
              )} 
              aria-hidden="true"
            />
            <span className="text-muted-foreground">{finding}</span>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Recommendations list
 */
function RecommendationsList({
  recommendations,
  variant,
}: {
  recommendations: string[];
  variant: 'default' | 'compact' | 'expanded';
}) {
  return (
    <ul 
      className="space-y-2"
      data-testid="explanation-recommendations"
      role="list"
    >
      {recommendations.map((rec, idx) => (
        <li 
          key={idx} 
          className={cn(
            'flex items-start gap-2.5',
            variant === 'compact' ? 'text-xs' : 'text-sm'
          )}
          role="listitem"
        >
          <ArrowRight 
            className={cn(
              'mt-0.5 text-muted-foreground flex-shrink-0',
              variant === 'compact' ? 'h-3 w-3' : 'h-4 w-4'
            )} 
            aria-hidden="true"
          />
          <span className="text-muted-foreground">{rec}</span>
        </li>
      ))}
    </ul>
  );
}

/**
 * Empty state component
 */
function EmptyExplanation({
  showCard,
  className,
}: {
  showCard: boolean;
  className?: string;
}) {
  const content = (
    <div 
      className="py-8 text-center"
      data-testid="explanation-panel-empty"
    >
      <Info className="h-10 w-10 mx-auto mb-3 text-muted-foreground/40" aria-hidden="true" />
      <p className="text-muted-foreground font-medium mb-1">
        No Explanation Available
      </p>
      <p className="text-sm text-muted-foreground/70">
        Detailed analysis explanation will appear here once processing is complete.
      </p>
    </div>
  );
  
  if (!showCard) {
    return <div className={className}>{content}</div>;
  }
  
  return (
    <Card className={className}>
      <CardContent>{content}</CardContent>
    </Card>
  );
}

// ============== SKELETON LOADER ==============

/**
 * Skeleton loader for ExplanationPanel
 */
export function ExplanationPanelSkeleton({
  showCard = true,
  showFindings = true,
  showRecommendations = false,
  className,
}: {
  showCard?: boolean;
  showFindings?: boolean;
  showRecommendations?: boolean;
  className?: string;
}) {
  const content = (
    <div className="space-y-4 animate-pulse" data-testid="explanation-panel-skeleton">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Skeleton className="h-5 w-5" />
        <Skeleton className="h-5 w-36" />
      </div>
      
      {/* Summary */}
      <div className="space-y-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-11/12" />
        <Skeleton className="h-4 w-4/5" />
      </div>
      
      {/* Confidence Box */}
      <Skeleton className="h-20 w-full rounded-lg" />
      
      {/* Key Findings */}
      {showFindings && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 mb-3">
            <Skeleton className="h-4 w-4" />
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-5 w-6 rounded" />
          </div>
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="flex items-start gap-2">
              <Skeleton className="h-4 w-4 mt-0.5" />
              <Skeleton className="h-4 flex-1" />
            </div>
          ))}
        </div>
      )}
      
      {/* Recommendations */}
      {showRecommendations && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 mb-3">
            <Skeleton className="h-4 w-4" />
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-5 w-6 rounded" />
          </div>
          {Array.from({ length: 2 }).map((_, i) => (
            <div key={i} className="flex items-start gap-2">
              <Skeleton className="h-4 w-4 mt-0.5" />
              <Skeleton className="h-4 flex-1" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
  
  if (!showCard) {
    return <div className={className}>{content}</div>;
  }
  
  return (
    <Card className={className}>
      <CardContent className="pt-6">{content}</CardContent>
    </Card>
  );
}

// ============== COMPACT VARIANT ==============

/**
 * Compact explanation display for inline use
 */
export function ExplanationPanelCompact({
  explanation,
  maxFindings = 3,
  className,
}: {
  explanation: Explanation;
  maxFindings?: number;
  className?: string;
}) {
  if (!explanation || !explanation.summary) {
    return null;
  }
  
  const displayFindings = explanation.key_findings?.slice(0, maxFindings) || [];
  const hasMoreFindings = (explanation.key_findings?.length || 0) > maxFindings;
  
  return (
    <div 
      className={cn('space-y-2', className)}
      data-testid="explanation-panel-compact"
    >
      <p className="text-sm text-muted-foreground line-clamp-2">
        {explanation.summary}
      </p>
      
      {displayFindings.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {displayFindings.map((finding, idx) => (
            <Badge 
              key={idx} 
              variant="outline" 
              className="text-xs font-normal"
            >
              {finding.length > 40 ? `${finding.slice(0, 40)}...` : finding}
            </Badge>
          ))}
          {hasMoreFindings && (
            <Badge variant="secondary" className="text-xs">
              +{(explanation.key_findings?.length || 0) - maxFindings} more
            </Badge>
          )}
        </div>
      )}
    </div>
  );
}

// ============== EXPORTS ==============

export default ExplanationPanel;
