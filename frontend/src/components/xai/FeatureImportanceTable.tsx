/**
 * Argus Core - Feature Importance Table Component
 * ===============================================
 * Displays feature importance scores in a sortable table format.
 * 
 * Implements: XAI_FRONTEND_IMPLEMENTATION.md - Section 4.3 - components/xai/FeatureImportanceTable.tsx
 * 
 * Role: Visualize which features contributed most to the AI's decision,
 * showing contribution direction and confidence levels.
 * 
 * Integration:
 * - Used by: XAIExplanationPanel
 * - Data: FeatureImportance[] from XAI explanation
 */

'use client';

import React, { useMemo, useState } from 'react';
import { cn } from '@/lib/utils';
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  AlertCircle,
  CheckCircle2,
  Minus,
  Info,
} from 'lucide-react';

// UI Components
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// Types
import type { FeatureImportance } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for FeatureImportanceTable component
 */
export interface FeatureImportanceTableProps {
  /** Array of feature importance data */
  features: FeatureImportance[];
  /** Table title */
  title?: string;
  /** Table description */
  description?: string;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show contribution direction column */
  showContributionDirection?: boolean;
  /** Whether to show confidence column */
  showConfidence?: boolean;
  /** Number of top features to highlight */
  highlightTop?: number;
  /** Maximum number of features to display */
  maxFeatures?: number;
  /** Whether to enable sorting */
  sortable?: boolean;
  /** Initial sort field */
  initialSortField?: 'importance' | 'confidence' | 'name';
  /** Initial sort direction */
  initialSortDirection?: 'asc' | 'desc';
}

/**
 * Sort configuration
 */
interface SortConfig {
  field: 'importance' | 'confidence' | 'name';
  direction: 'asc' | 'desc';
}

// ============== HELPER FUNCTIONS ==============

/**
 * Format feature name for display
 */
function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (str) => str.toUpperCase())
    .trim();
}

/**
 * Get contribution direction icon and color
 */
function getContributionInfo(direction: FeatureImportance['contribution_direction']): {
  icon: React.ElementType;
  color: string;
  label: string;
} {
  switch (direction) {
    case 'increases_fake':
      return {
        icon: ArrowUp,
        color: 'text-destructive',
        label: 'Increases fake probability',
      };
    case 'decreases_fake':
      return {
        icon: ArrowDown,
        color: 'text-green-500',
        label: 'Decreases fake probability',
      };
    default:
      return {
        icon: Minus,
        color: 'text-muted-foreground',
        label: 'Neutral contribution',
      };
  }
}

/**
 * Get importance level color
 */
function getImportanceColor(importance: number): string {
  if (importance >= 0.7) return 'bg-destructive';
  if (importance >= 0.5) return 'bg-orange-500';
  if (importance >= 0.3) return 'bg-yellow-500';
  return 'bg-green-500';
}

// ============== MAIN COMPONENT ==============

/**
 * FeatureImportanceTable
 * 
 * Displays feature importance scores in a sortable, interactive table.
 * 
 * @example
 * ```tsx
 * <FeatureImportanceTable
 *   features={features}
 *   showContributionDirection
 *   highlightTop={3}
 * />
 * ```
 */
export function FeatureImportanceTable({
  features,
  title = 'Feature Importance',
  description,
  className,
  showContributionDirection = true,
  showConfidence = true,
  highlightTop = 0,
  maxFeatures,
  sortable = true,
  initialSortField = 'importance',
  initialSortDirection = 'desc',
}: FeatureImportanceTableProps): React.ReactElement {
  // Sort state
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: initialSortField,
    direction: initialSortDirection,
  });

  // Sort features
  const sortedFeatures = useMemo(() => {
    let sorted = [...features];
    
    if (sortConfig.field === 'importance') {
      sorted.sort((a, b) =>
        sortConfig.direction === 'asc'
          ? a.importance_score - b.importance_score
          : b.importance_score - a.importance_score
      );
    } else if (sortConfig.field === 'confidence') {
      sorted.sort((a, b) =>
        sortConfig.direction === 'asc'
          ? a.confidence - b.confidence
          : b.confidence - a.confidence
      );
    } else {
      sorted.sort((a, b) =>
        sortConfig.direction === 'asc'
          ? a.feature_name.localeCompare(b.feature_name)
          : b.feature_name.localeCompare(a.feature_name)
      );
    }

    // Apply max features limit
    if (maxFeatures && maxFeatures > 0) {
      sorted = sorted.slice(0, maxFeatures);
    }

    return sorted;
  }, [features, sortConfig, maxFeatures]);

  // Handle sort toggle
  const handleSort = (field: SortConfig['field']) => {
    if (!sortable) return;
    
    setSortConfig((prev) => ({
      field,
      direction: prev.field === field && prev.direction === 'desc' ? 'asc' : 'desc',
    }));
  };

  // Sort icon component
  const SortIcon = ({ field }: { field: SortConfig['field'] }) => {
    if (!sortable) return null;
    
    if (sortConfig.field !== field) {
      return <ArrowUpDown className="h-4 w-4 ml-1 opacity-50" />;
    }
    
    return sortConfig.direction === 'asc' ? (
      <ArrowUp className="h-4 w-4 ml-1" />
    ) : (
      <ArrowDown className="h-4 w-4 ml-1" />
    );
  };

  // Empty state
  if (features.length === 0) {
    return (
      <Card className={cn('w-full', className)} data-testid="feature-importance-empty">
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center text-center">
            <Info className="h-10 w-10 text-muted-foreground mb-3" />
            <p className="text-muted-foreground">No feature importance data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn('w-full', className)} data-testid="feature-importance-table">
      {(title || description) && (
        <CardHeader className="pb-2">
          {title && <CardTitle className="text-lg">{title}</CardTitle>}
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
      )}
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead
                className={cn('w-12', sortable && 'cursor-pointer')}
                onClick={() => handleSort('name')}
                data-testid="sort-name"
              >
                #
              </TableHead>
              <TableHead
                className={cn(sortable && 'cursor-pointer')}
                onClick={() => handleSort('name')}
                data-testid="sort-feature"
              >
                <div className="flex items-center">
                  Feature
                  <SortIcon field="name" />
                </div>
              </TableHead>
              <TableHead
                className={cn('w-40', sortable && 'cursor-pointer')}
                onClick={() => handleSort('importance')}
                data-testid="sort-importance"
              >
                <div className="flex items-center">
                  Importance
                  <SortIcon field="importance" />
                </div>
              </TableHead>
              {showContributionDirection && (
                <TableHead className="w-32">Direction</TableHead>
              )}
              {showConfidence && (
                <TableHead
                  className={cn('w-24', sortable && 'cursor-pointer')}
                  onClick={() => handleSort('confidence')}
                  data-testid="sort-confidence"
                >
                  <div className="flex items-center">
                    Confidence
                    <SortIcon field="confidence" />
                  </div>
                </TableHead>
              )}
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedFeatures.map((feature, index) => {
              const contributionInfo = getContributionInfo(feature.contribution_direction);
              const ContributionIcon = contributionInfo.icon;
              const isHighlighted = highlightTop > 0 && index < highlightTop;

              return (
                <TableRow
                  key={feature.feature_name}
                  className={cn(isHighlighted && 'bg-muted/50')}
                  data-testid={`feature-row-${feature.feature_name}`}
                >
                  <TableCell className="font-medium">
                    {isHighlighted ? (
                      <Badge variant="secondary" className="w-6 h-6 p-0 flex items-center justify-center">
                        {index + 1}
                      </Badge>
                    ) : (
                      <span className="text-muted-foreground">{index + 1}</span>
                    )}
                  </TableCell>
                  <TableCell>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="font-medium">
                            {formatFeatureName(feature.feature_name)}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>{feature.feature_name}</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </TableCell>
                  <TableCell>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                          {(feature.importance_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress
                        value={feature.importance_score * 100}
                        className="h-2"
                        indicatorClassName={getImportanceColor(feature.importance_score)}
                      />
                    </div>
                  </TableCell>
                  {showContributionDirection && (
                    <TableCell>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex items-center gap-1">
                              <ContributionIcon className={cn('h-4 w-4', contributionInfo.color)} />
                              <span className={cn('text-sm', contributionInfo.color)}>
                                {feature.contribution_direction === 'increases_fake' ? 'Fake' : 'Auth'}
                              </span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{contributionInfo.label}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </TableCell>
                  )}
                  {showConfidence && (
                    <TableCell>
                      <div className="flex items-center gap-1">
                        {feature.confidence >= 0.8 ? (
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                        ) : feature.confidence >= 0.5 ? (
                          <Info className="h-4 w-4 text-yellow-500" />
                        ) : (
                          <AlertCircle className="h-4 w-4 text-muted-foreground" />
                        )}
                        <span className="text-sm">
                          {(feature.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </TableCell>
                  )}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

// ============== SKELETON COMPONENT ==============

/**
 * Skeleton component for FeatureImportanceTable loading state
 */
export function FeatureImportanceSkeleton({
  className,
  rows = 5,
}: {
  className?: string;
  rows?: number;
}): React.ReactElement {
  return (
    <Card className={cn('w-full', className)} data-testid="feature-importance-skeleton">
      <CardHeader className="pb-2">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-4 w-60" />
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {Array.from({ length: rows }).map((_, i) => (
            <div key={i} className="flex items-center gap-4">
              <Skeleton className="h-6 w-6 rounded" />
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-4 w-16" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default FeatureImportanceTable;
