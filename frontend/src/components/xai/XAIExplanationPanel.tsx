/**
 * Argus Core - XAI Explanation Panel Component
 * ============================================
 * Main container for Explainable AI visualizations and evidence.
 * 
 * Implements: XAI_FRONTEND_IMPLEMENTATION.md - Section 4.3 - components/xai/XAIExplanationPanel.tsx
 * 
 * Role: Display comprehensive XAI explanations including feature importance,
 * visual evidence, scientific references, and reproducibility information.
 * 
 * Integration:
 * - Imports: All XAI sub-components, hooks/useXAI, store/xaiStore
 * - Used by: ModalityTabs, analysis/[id]/page.tsx
 * - Backend: GET /api/v1/analyze/{id}/xai
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton components while fetching
 * - Error state: Displays error with retry option
 * - Empty state: Shows message when no XAI data available
 * - Accessibility: ARIA labels, keyboard navigation
 * - data-testid: xai-explanation-panel
 */

'use client';

import React, { useMemo, useCallback } from 'react';
import { cn } from '@/lib/utils';
import {
  Brain,
  AlertCircle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Info,
  Shield,
  BarChart3,
  Image as ImageIcon,
  FileText,
  Volume2,
  Video,
  Loader2,
} from 'lucide-react';

// UI Components
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';

// XAI Sub-components (will be created next)
import { FeatureImportanceTable, FeatureImportanceSkeleton } from './FeatureImportanceTable';
import { ConfidenceInterval, ConfidenceIntervalSkeleton } from './ConfidenceInterval';
import { ReproducibilityHash, ReproducibilityHashSkeleton } from './ReproducibilityHash';
import { ScientificReferences, ScientificReferencesSkeleton } from './ScientificReferences';
import { XAIEvidenceGallery, XAIEvidenceGallerySkeleton } from './XAIEvidenceGallery';

// Hooks
import { useXAI, useFeatureImportance, useReproducibility } from '@/hooks/useXAI';

// Types
import type {
  XAIExplanation,
  ModalityXAI,
  Modality,
  FeatureImportance as FeatureImportanceType,
  VisualEvidence,
  ScientificReference as ScientificReferenceType,
} from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for XAIExplanationPanel component
 */
export interface XAIExplanationPanelProps {
  /** Analysis ID to fetch XAI for */
  analysisId: string;
  /** Additional CSS classes */
  className?: string;
  /** Initial expanded state */
  defaultExpanded?: boolean;
  /** Whether to show modality tabs */
  showModalityTabs?: boolean;
  /** Whether to show evidence gallery */
  showEvidenceGallery?: boolean;
  /** Whether to show scientific references */
  showReferences?: boolean;
  /** Whether to show reproducibility info */
  showReproducibility?: boolean;
  /** Callback when XAI data is loaded */
  onXAILoaded?: (data: ModalityXAI | null) => void;
}

/**
 * Modality tab configuration
 */
interface ModalityTabConfig {
  id: string;
  label: string;
  icon: React.ElementType;
  description: string;
}

// ============== MODALITY TABS CONFIG ==============

const MODALITY_TABS: ModalityTabConfig[] = [
  {
    id: 'image',
    label: 'Image',
    icon: ImageIcon,
    description: 'Spatial analysis and manipulation detection',
  },
  {
    id: 'video',
    label: 'Video',
    icon: Video,
    description: 'Temporal and spatial-temporal analysis',
  },
  {
    id: 'audio',
    label: 'Audio',
    icon: Volume2,
    description: 'Spectral analysis and vocoder detection',
  },
  {
    id: 'text',
    label: 'Text',
    icon: FileText,
    description: 'AI-generated text detection',
  },
];

// ============== MAIN COMPONENT ==============

/**
 * XAIExplanationPanel
 * 
 * Displays comprehensive XAI explanations with modality-specific tabs,
 * feature importance tables, visual evidence, and scientific references.
 * 
 * @example
 * ```tsx
 * <XAIExplanationPanel 
 *   analysisId={analysisId}
 *   showModalityTabs
 *   showEvidenceGallery
 *   showReferences
 * />
 * ```
 */
export function XAIExplanationPanel({
  analysisId,
  className,
  defaultExpanded = true,
  showModalityTabs = true,
  showEvidenceGallery = true,
  showReferences = true,
  showReproducibility = true,
  onXAILoaded,
}: XAIExplanationPanelProps): React.ReactElement {
  // State for collapsible sections
  const [isExpanded, setIsExpanded] = React.useState(defaultExpanded);
  const [activeSection, setActiveSection] = React.useState<string>('features');

  // Fetch XAI data
  const {
    data: xaiData,
    isLoading,
    isFetching,
    error,
    refetch,
    hasXAI,
    activeModality,
    setActiveModality,
    getModalityXAI,
  } = useXAI(analysisId);

  // Get feature importance
  const { features, sortedFeatures, fakeIndicators, authenticIndicators } =
    useFeatureImportance(analysisId);

  // Get reproducibility info
  const { hash, confidenceInterval, modelVersions } = useReproducibility(analysisId);

  // Get current modality XAI
  const currentModalityXAI = useMemo(
    () => getModalityXAI(activeModality),
    [getModalityXAI, activeModality]
  );

  // Notify parent when XAI data loads
  React.useEffect(() => {
    if (onXAILoaded && !isLoading) {
      onXAILoaded(currentModalityXAI);
    }
  }, [currentModalityXAI, isLoading, onXAILoaded]);

  // Handle refetch
  const handleRefetch = useCallback(() => {
    refetch();
  }, [refetch]);

  // ============== LOADING STATE ==============

  if (isLoading) {
    return <XAIExplanationPanelSkeleton className={className} />;
  }

  // ============== ERROR STATE ==============

  if (error) {
    return (
      <Card className={cn('w-full', className)} data-testid="xai-explanation-panel-error">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-destructive" />
            XAI Explanation Unavailable
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error Loading XAI Data</AlertTitle>
            <AlertDescription className="mt-2">
              {error.message || 'Unable to load explainability data. Please try again.'}
            </AlertDescription>
          </Alert>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefetch}
            className="mt-4"
            data-testid="xai-retry-button"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  // ============== EMPTY STATE ==============

  if (!hasXAI || !xaiData) {
    return (
      <Card className={cn('w-full', className)} data-testid="xai-explanation-panel-empty">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-muted-foreground" />
            Explainable AI
          </CardTitle>
          <CardDescription>
            Detailed explanation of how the AI reached its conclusion
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Info className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">
              No explainability data available for this analysis.
            </p>
            <p className="text-sm text-muted-foreground mt-2">
              XAI data may not have been generated during processing.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // ============== MAIN RENDER ==============

  return (
    <Card className={cn('w-full', className)} data-testid="xai-explanation-panel">
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary" />
                <div>
                  <CardTitle>Explainable AI Analysis</CardTitle>
                  <CardDescription>
                    Court-admissible forensic evidence with scientific justification
                  </CardDescription>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {isFetching && !isLoading && (
                  <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                )}
                <Badge variant="outline" className="hidden sm:flex">
                  {sortedFeatures.length} features
                </Badge>
                {isExpanded ? (
                  <ChevronUp className="h-5 w-5 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-5 w-5 text-muted-foreground" />
                )}
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="pt-4">
            {/* Modality Tabs */}
            {showModalityTabs && (
              <Tabs
                value={activeModality}
                onValueChange={(v) => setActiveModality(v as typeof activeModality)}
                className="w-full mb-6"
              >
                <TabsList className="grid grid-cols-4 w-full max-w-lg">
                  {MODALITY_TABS.map((tab) => {
                    const Icon = tab.icon;
                    const hasData = getModalityXAI(tab.id as typeof activeModality) !== null;
                    return (
                      <TabsTrigger
                        key={tab.id}
                        value={tab.id}
                        disabled={!hasData}
                        className="flex items-center gap-1"
                        data-testid={`xai-tab-${tab.id}`}
                      >
                        <Icon className="h-4 w-4" />
                        <span className="hidden sm:inline">{tab.label}</span>
                      </TabsTrigger>
                    );
                  })}
                </TabsList>

                {MODALITY_TABS.map((tab) => (
                  <TabsContent key={tab.id} value={tab.id} className="mt-4">
                    <p className="text-sm text-muted-foreground mb-4">
                      {tab.description}
                    </p>
                  </TabsContent>
                ))}
              </Tabs>
            )}

            {/* Section Navigation */}
            <div className="flex flex-wrap gap-2 mb-6">
              <Button
                variant={activeSection === 'features' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setActiveSection('features')}
                data-testid="xai-section-features"
              >
                <BarChart3 className="h-4 w-4 mr-2" />
                Feature Importance
              </Button>
              {showEvidenceGallery && (
                <Button
                  variant={activeSection === 'evidence' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setActiveSection('evidence')}
                  data-testid="xai-section-evidence"
                >
                  <ImageIcon className="h-4 w-4 mr-2" />
                  Visual Evidence
                </Button>
              )}
              {showReproducibility && (
                <Button
                  variant={activeSection === 'reproducibility' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setActiveSection('reproducibility')}
                  data-testid="xai-section-reproducibility"
                >
                  <Shield className="h-4 w-4 mr-2" />
                  Reproducibility
                </Button>
              )}
              {showReferences && (
                <Button
                  variant={activeSection === 'references' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setActiveSection('references')}
                  data-testid="xai-section-references"
                >
                  <FileText className="h-4 w-4 mr-2" />
                  Scientific References
                </Button>
              )}
            </div>

            {/* Active Section Content */}
            <div className="space-y-6">
              {/* Feature Importance Section */}
              {activeSection === 'features' && (
                <div className="space-y-4">
                  {/* Confidence Interval */}
                  {confidenceInterval && showReproducibility && (
                    <ConfidenceInterval
                      lower={confidenceInterval[0]}
                      upper={confidenceInterval[1]}
                      label="Prediction Confidence Interval"
                    />
                  )}

                  {/* Feature Importance Table */}
                  <FeatureImportanceTable
                    features={sortedFeatures}
                    title="Feature Importance Analysis"
                    description="Features ranked by their contribution to the AI's decision"
                    showContributionDirection
                    highlightTop={3}
                  />

                  {/* Summary Stats */}
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="flex items-center gap-2">
                          <AlertCircle className="h-4 w-4 text-destructive" />
                          <span className="text-sm font-medium">Fake Indicators</span>
                        </div>
                        <p className="text-2xl font-bold mt-2">{fakeIndicators.length}</p>
                        <p className="text-xs text-muted-foreground">
                          Features suggesting manipulation
                        </p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="flex items-center gap-2">
                          <Shield className="h-4 w-4 text-green-500" />
                          <span className="text-sm font-medium">Authentic Indicators</span>
                        </div>
                        <p className="text-2xl font-bold mt-2">{authenticIndicators.length}</p>
                        <p className="text-xs text-muted-foreground">
                          Features supporting authenticity
                        </p>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}

              {/* Visual Evidence Section */}
              {activeSection === 'evidence' && showEvidenceGallery && currentModalityXAI && (
                <XAIEvidenceGallery
                  evidence={currentModalityXAI.explanation?.visual_evidence || []}
                  heatmapUrls={currentModalityXAI.heatmapUrls}
                  overlayUrl={currentModalityXAI.overlayUrl}
                  modality={activeModality as Modality}
                />
              )}

              {/* Reproducibility Section */}
              {activeSection === 'reproducibility' && showReproducibility && (
                <div className="space-y-4">
                  {hash && (
                    <ReproducibilityHash
                      hash={hash}
                      label="Analysis Reproducibility Hash"
                      description="SHA-256 hash of all analysis inputs for forensic verification"
                    />
                  )}
                  {confidenceInterval && (
                    <ConfidenceInterval
                      lower={confidenceInterval[0]}
                      upper={confidenceInterval[1]}
                      label="Statistical Confidence Interval"
                      description="95% confidence interval for the prediction"
                    />
                  )}
                  {Object.keys(modelVersions).length > 0 && (
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Model Versions</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          {Object.entries(modelVersions).map(([model, version]) => (
                            <div
                              key={model}
                              className="flex justify-between items-center text-sm"
                            >
                              <span className="text-muted-foreground">{model}</span>
                              <code className="text-xs bg-muted px-2 py-1 rounded">
                                {version}
                              </code>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}

              {/* Scientific References Section */}
              {activeSection === 'references' && showReferences && (
                <ScientificReferences
                  references={currentModalityXAI?.explanation?.scientific_references || []}
                />
              )}
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

// ============== SKELETON COMPONENT ==============

/**
 * Skeleton component for XAIExplanationPanel loading state
 */
export function XAIExplanationPanelSkeleton({
  className,
}: {
  className?: string;
}): React.ReactElement {
  return (
    <Card className={cn('w-full', className)} data-testid="xai-explanation-panel-skeleton">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Skeleton className="h-5 w-5 rounded-full" />
          <div className="space-y-2">
            <Skeleton className="h-5 w-40" />
            <Skeleton className="h-4 w-60" />
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Tabs skeleton */}
        <div className="flex gap-2">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-9 w-20" />
          ))}
        </div>

        {/* Section buttons skeleton */}
        <div className="flex gap-2">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-9 w-32" />
          ))}
        </div>

        {/* Content skeleton */}
        <FeatureImportanceSkeleton />
        <ConfidenceIntervalSkeleton />
        <ReproducibilityHashSkeleton />
      </CardContent>
    </Card>
  );
}

export default XAIExplanationPanel;
