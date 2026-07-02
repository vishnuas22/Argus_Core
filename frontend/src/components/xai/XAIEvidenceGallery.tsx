/**
 * Argus Core - XAI Evidence Gallery Component
 * ===========================================
 * Displays visual evidence from XAI analysis (heatmaps, spectrograms, etc).
 * 
 * Implements: XAI_FRONTEND_IMPLEMENTATION.md - Section 4.3 - components/xai/XAIEvidenceGallery.tsx
 * 
 * Role: Display visual evidence including GradCAM heatmaps, spectrogram overlays,
 * token highlights, and attention maps in an interactive gallery.
 * 
 * Integration:
 * - Used by: XAIExplanationPanel
 * - Data: visual_evidence, heatmap_urls from XAI explanation
 */

'use client';

import React, { useState, useMemo } from 'react';
import Image from 'next/image';
import { cn } from '@/lib/utils';
import {
  Image as ImageIcon,
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  Grid,
  List,
  Play,
  Pause,
  Volume2,
  Info,
  ExternalLink,
  Download,
  Maximize2,
  Video,
} from 'lucide-react';

// UI Components
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// Types
import type { VisualEvidence, Modality } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for XAIEvidenceGallery component
 */
export interface XAIEvidenceGalleryProps {
  /** Array of visual evidence */
  evidence: VisualEvidence[];
  /** Heatmap URLs from analysis */
  heatmapUrls?: string[];
  /** Overlay URL (combined visualization) */
  overlayUrl?: string | null;
  /** Current modality being displayed */
  modality?: Modality;
  /** Title for the gallery */
  title?: string;
  /** Description */
  description?: string;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show in grid or list view */
  defaultView?: 'grid' | 'list';
  /** Whether to enable fullscreen mode */
  enableFullscreen?: boolean;
}

/**
 * Evidence item with additional metadata
 */
interface EvidenceItem {
  id: string;
  url: string;
  type: VisualEvidence['artifact_type'];
  description: string;
  frameIndex?: number;
  timestamp?: number;
}

// ============== HELPER FUNCTIONS ==============

/**
 * Get evidence type icon
 */
function getEvidenceTypeIcon(type: VisualEvidence['artifact_type']): React.ElementType {
  switch (type) {
    case 'heatmap':
      return ImageIcon;
    case 'spectrogram':
      return Volume2;
    case 'attention_map':
      return ImageIcon;
    case 'overlay':
      return ImageIcon;
    case 'frequency_plot':
      return Volume2;
    case 'temporal_chart':
      return Video;
    default:
      return ImageIcon;
  }
}

/**
 * Get evidence type label
 */
function getEvidenceTypeLabel(type: VisualEvidence['artifact_type']): string {
  switch (type) {
    case 'heatmap':
      return 'GradCAM Heatmap';
    case 'spectrogram':
      return 'Spectrogram';
    case 'attention_map':
      return 'Attention Map';
    case 'overlay':
      return 'Overlay';
    case 'frequency_plot':
      return 'Frequency Plot';
    case 'temporal_chart':
      return 'Temporal Chart';
    default:
      return 'Evidence';
  }
}

/**
 * Get evidence type color
 */
function getEvidenceTypeColor(type: VisualEvidence['artifact_type']): string {
  switch (type) {
    case 'heatmap':
      return 'bg-red-500/10 text-red-500 border-red-500/20';
    case 'spectrogram':
      return 'bg-purple-500/10 text-purple-500 border-purple-500/20';
    case 'attention_map':
      return 'bg-green-500/10 text-green-500 border-green-500/20';
    case 'overlay':
      return 'bg-orange-500/10 text-orange-500 border-orange-500/20';
    case 'frequency_plot':
      return 'bg-cyan-500/10 text-cyan-500 border-cyan-500/20';
    case 'temporal_chart':
      return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
    default:
      return 'bg-muted text-muted-foreground';
  }
}

// ============== MAIN COMPONENT ==============

/**
 * XAIEvidenceGallery
 * 
 * Displays visual evidence in an interactive gallery with grid/list views
 * and fullscreen capabilities.
 * 
 * @example
 * ```tsx
 * <XAIEvidenceGallery
 *   evidence={visualEvidence}
 *   heatmapUrls={heatmapUrls}
 *   modality="image"
 *   enableFullscreen
 * />
 * ```
 */
export function XAIEvidenceGallery({
  evidence,
  heatmapUrls = [],
  overlayUrl,
  modality = 'image',
  title = 'Visual Evidence',
  description = 'AI-generated visualizations explaining the analysis',
  className,
  defaultView = 'grid',
  enableFullscreen = true,
}: XAIEvidenceGalleryProps): React.ReactElement {
  // View state
  const [viewMode, setViewMode] = useState<'grid' | 'list'>(defaultView);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Combine evidence with heatmap URLs
  const allEvidence = useMemo<EvidenceItem[]>(() => {
    const items: EvidenceItem[] = [];

    // Add evidence from the evidence array
    evidence.forEach((e, index) => {
      items.push({
        id: `evidence-${index}`,
        url: e.url,
        type: e.artifact_type,
        description: e.description,
        frameIndex: e.frame_index,
        timestamp: e.timestamp_seconds,
      });
    });

    // Add heatmap URLs
    heatmapUrls.forEach((url, index) => {
      if (!items.some((i) => i.url === url)) {
        items.push({
          id: `heatmap-${index}`,
          url,
          type: 'heatmap',
          description: `Frame ${index + 1} GradCAM visualization`,
          frameIndex: index,
        });
      }
    });

    // Add overlay URL
    if (overlayUrl && !items.some((i) => i.url === overlayUrl)) {
      items.push({
        id: 'overlay',
        url: overlayUrl,
        type: 'heatmap',
        description: 'Combined overlay visualization',
      });
    }

    return items;
  }, [evidence, heatmapUrls, overlayUrl]);

  // Group evidence by type
  const evidenceByType = useMemo(() => {
    const groups: Record<string, EvidenceItem[]> = {};
    allEvidence.forEach((item) => {
      if (!groups[item.type]) {
        groups[item.type] = [];
      }
      groups[item.type].push(item);
    });
    return groups;
  }, [allEvidence]);

  // Navigation handlers
  const handlePrevious = () => {
    setSelectedIndex((prev) => (prev > 0 ? prev - 1 : allEvidence.length - 1));
  };

  const handleNext = () => {
    setSelectedIndex((prev) => (prev < allEvidence.length - 1 ? prev + 1 : 0));
  };

  // Empty state
  if (allEvidence.length === 0) {
    return (
      <Card className={cn('w-full', className)} data-testid="xai-evidence-gallery-empty">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <ImageIcon className="h-12 w-12 text-muted-foreground mb-3" />
            <p className="text-muted-foreground">No visual evidence available</p>
            <p className="text-sm text-muted-foreground mt-1">
              Evidence may not have been generated during analysis
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Selected evidence
  const selectedEvidence = allEvidence[selectedIndex];
  const EvidenceIcon = getEvidenceTypeIcon(selectedEvidence?.type);

  return (
    <Card className={cn('w-full', className)} data-testid="xai-evidence-gallery">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4 text-primary" />
            <div>
              <CardTitle className="text-sm font-medium">{title}</CardTitle>
              <CardDescription>{description}</CardDescription>
            </div>
          </div>
          
          {/* View toggle */}
          <div className="flex items-center gap-1">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={viewMode === 'grid' ? 'default' : 'ghost'}
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setViewMode('grid')}
                    data-testid="view-grid"
                  >
                    <Grid className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Grid view</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={viewMode === 'list' ? 'default' : 'ghost'}
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setViewMode('list')}
                    data-testid="view-list"
                  >
                    <List className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>List view</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {/* Type tabs */}
        {Object.keys(evidenceByType).length > 1 && (
          <Tabs defaultValue={Object.keys(evidenceByType)[0]} className="mb-4">
            <TabsList>
              {Object.entries(evidenceByType).map(([type, items]) => (
                <TabsTrigger key={type} value={type} className="flex items-center gap-1">
                  {React.createElement(getEvidenceTypeIcon(type as VisualEvidence['artifact_type']), {
                    className: 'h-3 w-3',
                  })}
                  {getEvidenceTypeLabel(type as VisualEvidence['artifact_type'])}
                  <Badge variant="secondary" className="ml-1 text-xs">
                    {items.length}
                  </Badge>
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        )}

        {/* Grid view */}
        {viewMode === 'grid' && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {allEvidence.map((item, index) => {
              const ItemIcon = getEvidenceTypeIcon(item.type);
              return (
                <div
                  key={item.id}
                  className={cn(
                    'relative aspect-square rounded-lg overflow-hidden border cursor-pointer transition-all',
                    selectedIndex === index
                      ? 'ring-2 ring-primary'
                      : 'hover:ring-1 hover:ring-primary/50'
                  )}
                  onClick={() => setSelectedIndex(index)}
                  data-testid={`evidence-item-${index}`}
                >
                  {/* Heatmap/evidence image */}
                  <div className="absolute inset-0 bg-muted">
                    <img
                      src={item.url}
                      alt={item.description}
                      className="w-full h-full object-cover"
                      loading="lazy"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                        (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
                      }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center hidden">
                      <ItemIcon className="h-8 w-8 text-muted-foreground" />
                    </div>
                  </div>
                  
                  {/* Type badge */}
                  <Badge
                    variant="outline"
                    className={cn(
                      'absolute top-2 left-2 text-xs',
                      getEvidenceTypeColor(item.type)
                    )}
                  >
                    {item.type}
                  </Badge>

                  {/* Frame/timestamp info */}
                  {(item.frameIndex !== undefined || item.timestamp !== undefined) && (
                    <div className="absolute bottom-2 left-2 text-xs bg-black/50 text-white px-1.5 py-0.5 rounded">
                      {item.frameIndex !== undefined
                        ? `Frame ${item.frameIndex}`
                        : `${item.timestamp?.toFixed(2)}s`}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* List view */}
        {viewMode === 'list' && (
          <div className="space-y-3">
            {allEvidence.map((item, index) => {
              const ItemIcon = getEvidenceTypeIcon(item.type);
              return (
                <div
                  key={item.id}
                  className={cn(
                    'flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all',
                    selectedIndex === index
                      ? 'bg-muted ring-1 ring-primary'
                      : 'hover:bg-muted/50'
                  )}
                  onClick={() => setSelectedIndex(index)}
                  data-testid={`evidence-list-item-${index}`}
                >
                  {/* Thumbnail */}
                  <div className="w-16 h-16 rounded bg-muted flex items-center justify-center flex-shrink-0 overflow-hidden">
                    <img
                      src={item.url}
                      alt={item.description}
                      className="w-full h-full object-cover"
                      loading="lazy"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                    <ItemIcon className="h-6 w-6 text-muted-foreground" />
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className={cn('text-xs', getEvidenceTypeColor(item.type))}
                      >
                        {getEvidenceTypeLabel(item.type)}
                      </Badge>
                      {(item.frameIndex !== undefined || item.timestamp !== undefined) && (
                        <span className="text-xs text-muted-foreground">
                          {item.frameIndex !== undefined
                            ? `Frame ${item.frameIndex}`
                            : `${item.timestamp?.toFixed(2)}s`}
                        </span>
                      )}
                    </div>
                    <p className="text-sm mt-1 truncate">{item.description}</p>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <ExternalLink className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Open in new tab</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Selected evidence detail */}
        {selectedEvidence && (
          <div className="mt-4 p-4 bg-muted/50 rounded-lg">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2">
                <EvidenceIcon className="h-4 w-4 text-primary" />
                <div>
                  <h4 className="font-medium text-sm">
                    {getEvidenceTypeLabel(selectedEvidence.type)}
                  </h4>
                  <p className="text-xs text-muted-foreground">
                    {selectedEvidence.description}
                  </p>
                </div>
              </div>
              
              {/* Navigation */}
              {allEvidence.length > 1 && (
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={handlePrevious}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <span className="text-xs text-muted-foreground px-2">
                    {selectedIndex + 1} / {allEvidence.length}
                  </span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={handleNext}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>

            {/* Fullscreen button */}
            {enableFullscreen && (
              <div className="mt-3 flex justify-end">
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm">
                      <Maximize2 className="h-4 w-4 mr-2" />
                      View Full Size
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-4xl">
                    <DialogHeader>
                      <DialogTitle className="flex items-center gap-2">
                        <EvidenceIcon className="h-5 w-5" />
                        {getEvidenceTypeLabel(selectedEvidence.type)}
                      </DialogTitle>
                      <DialogDescription>
                        {selectedEvidence.description}
                      </DialogDescription>
                    </DialogHeader>
                    <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src={selectedEvidence.url}
                        alt={selectedEvidence.description}
                        className="w-full h-full object-contain"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = 'none';
                          (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
                        }}
                      />
                      <EvidenceIcon className="h-16 w-16 text-muted-foreground hidden" />
                    </div>
                    <div className="flex justify-between">
                      <Button variant="outline" onClick={handlePrevious}>
                        <ChevronLeft className="h-4 w-4 mr-2" />
                        Previous
                      </Button>
                      <Button variant="outline" asChild>
                        <a href={selectedEvidence.url} target="_blank" rel="noopener noreferrer">
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </a>
                      </Button>
                      <Button variant="outline" onClick={handleNext}>
                        Next
                        <ChevronRight className="h-4 w-4 ml-2" />
                      </Button>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            )}
          </div>
        )}

        {/* Summary */}
        <div className="mt-4 pt-3 border-t text-xs text-muted-foreground">
          <p>
            {allEvidence.length} evidence items •{' '}
            {Object.keys(evidenceByType).length} type
            {Object.keys(evidenceByType).length !== 1 ? 's' : ''}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// ============== SKELETON COMPONENT ==============

/**
 * Skeleton component for XAIEvidenceGallery loading state
 */
export function XAIEvidenceGallerySkeleton({
  className,
  count = 6,
}: {
  className?: string;
  count?: number;
}): React.ReactElement {
  return (
    <Card className={cn('w-full', className)} data-testid="xai-evidence-gallery-skeleton">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4" />
            <div className="space-y-1">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-3 w-48" />
            </div>
          </div>
          <div className="flex gap-1">
            <Skeleton className="h-8 w-8" />
            <Skeleton className="h-8 w-8" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {Array.from({ length: count }).map((_, i) => (
            <Skeleton key={i} className="aspect-square rounded-lg" />
          ))}
        </div>
        <Skeleton className="mt-4 h-20 w-full rounded-lg" />
      </CardContent>
    </Card>
  );
}

export default XAIEvidenceGallery;
