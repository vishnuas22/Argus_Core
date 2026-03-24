/**
 * Argus Core - Scientific References Component
 * ============================================
 * Displays scientific citations for XAI methodology.
 * 
 * Implements: XAI_FRONTEND_IMPLEMENTATION.md - Section 4.3 - components/xai/ScientificReferences.tsx
 * 
 * Role: Display peer-reviewed scientific references that support
 * the XAI methodology used in the analysis.
 * 
 * Integration:
 * - Used by: XAIExplanationPanel
 * - Data: scientific_references from XAI explanation
 */

'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import {
  BookOpen,
  ExternalLink,
  FileText,
  Link as LinkIcon,
  Info,
} from 'lucide-react';

// UI Components
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// Types
import type { ScientificReference } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for ScientificReferences component
 */
export interface ScientificReferencesProps {
  /** Array of scientific references */
  references: ScientificReference[];
  /** Title for the section */
  title?: string;
  /** Description of the references */
  description?: string;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show in compact mode */
  compact?: boolean;
  /** Whether to show DOI links */
  showDoiLinks?: boolean;
}

// ============== HELPER FUNCTIONS ==============

/**
 * Format DOI as URL
 */
function getDoiUrl(doi: string): string {
  if (doi.startsWith('http')) return doi;
  return `https://doi.org/${doi}`;
}

/**
 * Get method category badge color
 */
function getMethodCategory(methodName: string): string {
  const method = methodName.toLowerCase();
  if (method.includes('gradcam') || method.includes('attention')) {
    return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
  }
  if (method.includes('dct') || method.includes('frequency')) {
    return 'bg-purple-500/10 text-purple-500 border-purple-500/20';
  }
  if (method.includes('shap') || method.includes('attribution')) {
    return 'bg-green-500/10 text-green-500 border-green-500/20';
  }
  if (method.includes('spectral') || method.includes('spectrogram')) {
    return 'bg-orange-500/10 text-orange-500 border-orange-500/20';
  }
  return 'bg-muted text-muted-foreground';
}

// ============== MAIN COMPONENT ==============

/**
 * ScientificReferences
 * 
 * Displays scientific citations with method names and DOI links.
 * 
 * @example
 * ```tsx
 * <ScientificReferences
 *   references={[
 *     {
 *       method_name: 'GradCAM++',
 *       citation: 'Selvaraju et al., 2019',
 *       doi: '10.1109/ICCV.2019.00000'
 *     }
 *   ]}
 * />
 * ```
 */
export function ScientificReferences({
  references,
  title = 'Scientific References',
  description = 'Peer-reviewed methodologies used in this analysis',
  className,
  compact = false,
  showDoiLinks = true,
}: ScientificReferencesProps): React.ReactElement {
  // Empty state
  if (references.length === 0) {
    return (
      <Card className={cn('w-full', className)} data-testid="scientific-references-empty">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-4 text-center">
            <Info className="h-8 w-8 text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground">
              No scientific references available
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Compact inline display
  if (compact) {
    return (
      <div
        className={cn('inline-flex flex-wrap gap-2', className)}
        data-testid="scientific-references-compact"
      >
        {references.map((ref, index) => (
          <TooltipProvider key={index}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge
                  variant="outline"
                  className={cn('text-xs', getMethodCategory(ref.method_name))}
                >
                  {ref.method_name}
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <p className="font-medium">{ref.method_name}</p>
                <p className="text-xs text-muted-foreground">{ref.citation}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ))}
      </div>
    );
  }

  // Full display
  return (
    <Card className={cn('w-full', className)} data-testid="scientific-references">
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-primary" />
          <div>
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {references.map((ref, index) => (
            <div
              key={index}
              className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg"
              data-testid={`reference-${index}`}
            >
              {/* Method badge */}
              <Badge
                variant="outline"
                className={cn('mt-0.5 flex-shrink-0', getMethodCategory(ref.method_name))}
              >
                {ref.method_name}
              </Badge>

              {/* Citation */}
              <div className="flex-1 min-w-0">
                <p className="text-sm">{ref.citation}</p>
                
                {/* DOI link */}
                {ref.doi && showDoiLinks && (
                  <a
                    href={getDoiUrl(ref.doi)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 mt-1 text-xs text-primary hover:underline"
                    data-testid={`doi-link-${index}`}
                  >
                    <LinkIcon className="h-3 w-3" />
                    DOI: {ref.doi}
                    <ExternalLink className="h-3 w-3" />
                  </a>
                )}
              </div>

              {/* File icon */}
              <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
            </div>
          ))}
        </div>

        {/* Footer info */}
        <div className="mt-4 pt-3 border-t text-xs text-muted-foreground">
          <p>
            These references cite the peer-reviewed research that supports
            the explainability methods used in this analysis. They can be
            used for court submissions and scientific verification.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// ============== SKELETON COMPONENT ==============

/**
 * Skeleton component for ScientificReferences loading state
 */
export function ScientificReferencesSkeleton({
  className,
  compact = false,
  count = 3,
}: {
  className?: string;
  compact?: boolean;
  count?: number;
}): React.ReactElement {
  if (compact) {
    return (
      <div className={cn('inline-flex flex-wrap gap-2', className)}>
        {Array.from({ length: count }).map((_, i) => (
          <Skeleton key={i} className="h-5 w-20" />
        ))}
      </div>
    );
  }

  return (
    <Card className={cn('w-full', className)} data-testid="scientific-references-skeleton">
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <Skeleton className="h-4 w-4" />
          <div className="space-y-1">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-48" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {Array.from({ length: count }).map((_, i) => (
            <div key={i} className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
              <Skeleton className="h-5 w-20" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-3 w-32" />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default ScientificReferences;
