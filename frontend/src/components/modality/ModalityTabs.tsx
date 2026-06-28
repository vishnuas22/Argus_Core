/**
 * Argus Core - Modality Tabs Component
 * =====================================
 * Tab navigation for modality-specific analysis panels.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/modality/ModalityTabs.tsx
 * 
 * Role: Provide tabbed interface for switching between different modality analysis panels.
 * Shows available modalities based on what was analyzed and enables lazy loading of panels.
 * 
 * Integration:
 * - Imports: components/ui/tabs, modality panel components
 * - Used by: analysis/[id]/page.tsx, ResultsPanel.tsx
 * - Backend: Uses video_result, audio_result, metadata_result from API
 * 
 * Features:
 * - Dynamic tab visibility based on available modalities
 * - Badge indicators showing scores per modality
 * - Lazy loading of panel content for performance
 * - Keyboard navigation (arrow keys, home/end)
 * - Responsive design for mobile/desktop
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton tabs and content
 * - Empty state: Shows message when no modality data available
 * - Accessibility: ARIA tabs pattern, keyboard navigation
 * - data-testid: modality-tabs, modality-tab-{type}, modality-panel-{type}
 */

'use client';

import React, { useMemo, useState, Suspense, lazy } from 'react';
import { cn } from '@/lib/utils';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  Video,
  AudioLines,
  FileSearch2,
  Info,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type {
  VideoResult,
  AudioResult,
  MetadataResult,
  Modality,
} from '@/types/analysis';

// Lazy load panel components for performance
const VideoAnalysisPanel = lazy(() => import('./VideoAnalysisPanel'));
const AudioAnalysisPanel = lazy(() => import('./AudioAnalysisPanel'));
const MetadataPanel = lazy(() => import('./MetadataPanel'));

// ============== TYPES ==============

/**
 * Props for ModalityTabs component
 */
export interface ModalityTabsProps {
  /** Analysis ID for fetching additional data */
  analysisId: string;
  /** Video analysis result */
  videoResult?: VideoResult;
  /** Audio analysis result */
  audioResult?: AudioResult;
  /** Metadata analysis result */
  metadataResult?: MetadataResult;
  /** Default selected tab */
  defaultTab?: ModalityTabType;
  /** Callback when tab changes */
  onTabChange?: (tab: ModalityTabType) => void;
  /** Additional CSS classes */
  className?: string;
  /** Variant for different layouts */
  variant?: 'default' | 'compact' | 'card';
}

/**
 * Tab types including metadata
 */
export type ModalityTabType = 'video' | 'audio' | 'metadata';

/**
 * Configuration for each modality tab
 */
interface TabConfig {
  id: ModalityTabType;
  label: string;
  shortLabel: string;
  icon: LucideIcon;
  description: string;
}

// ============== CONSTANTS ==============

/**
 * Tab configuration for all modalities
 */
const TAB_CONFIG: TabConfig[] = [
  {
    id: 'video',
    label: 'Video Analysis',
    shortLabel: 'Video',
    icon: Video,
    description: 'Spatial, temporal, and lip-sync analysis results',
  },
  {
    id: 'audio',
    label: 'Audio Analysis',
    shortLabel: 'Audio',
    icon: AudioLines,
    description: 'Voice cloning and spectral anomaly detection',
  },
  {
    id: 'metadata',
    label: 'Metadata',
    shortLabel: 'Meta',
    icon: FileSearch2,
    description: 'C2PA provenance and EXIF integrity checks',
  },
];

// ============== MAIN COMPONENT ==============

/**
 * ModalityTabs Component
 * 
 * Provides tabbed navigation for different modality analysis results.
 * Only shows tabs for modalities that have data available.
 * 
 * @example
 * ```tsx
 * <ModalityTabs
 *   analysisId={id}
 *   videoResult={detail.video_result}
 *   audioResult={detail.audio_result}
 *   metadataResult={detail.metadata_result}
 *   defaultTab="video"
 * />
 * ```
 */
export function ModalityTabs({
  analysisId,
  videoResult,
  audioResult,
  metadataResult,
  defaultTab,
  onTabChange,
  className,
  variant = 'default',
}: ModalityTabsProps) {
  // ============== STATE ==============

  // Determine available tabs based on provided data
  const availableTabs = useMemo(() => {
    const tabs: TabConfig[] = [];
    
    if (videoResult) {
      tabs.push(TAB_CONFIG.find(t => t.id === 'video')!);
    }
    if (audioResult) {
      tabs.push(TAB_CONFIG.find(t => t.id === 'audio')!);
    }
    if (metadataResult) {
      tabs.push(TAB_CONFIG.find(t => t.id === 'metadata')!);
    }
    
    return tabs;
  }, [videoResult, audioResult, metadataResult]);

  // Determine default active tab
  const initialTab = useMemo(() => {
    if (defaultTab && availableTabs.some(t => t.id === defaultTab)) {
      return defaultTab;
    }
    return availableTabs[0]?.id || 'video';
  }, [defaultTab, availableTabs]);

  const [activeTab, setActiveTab] = useState<ModalityTabType>(initialTab);

  // ============== HANDLERS ==============

  const handleTabChange = (value: string) => {
    const tab = value as ModalityTabType;
    setActiveTab(tab);
    onTabChange?.(tab);
  };

  // ============== GET SCORE FOR TAB ==============

  const getScoreForTab = (tabId: ModalityTabType): number | undefined => {
    switch (tabId) {
      case 'video':
        return videoResult?.aggregated_score !== undefined
          ? videoResult.aggregated_score * 100
          : undefined;
      case 'audio':
        return audioResult?.score !== undefined
          ? audioResult.score * 100
          : undefined;
      case 'metadata':
        return metadataResult?.score !== undefined
          ? metadataResult.score * 100
          : undefined;
      default:
        return undefined;
    }
  };

  // ============== EMPTY STATE ==============

  if (availableTabs.length === 0) {
    return (
      <EmptyModalityTabs className={className} />
    );
  }

  // ============== RENDER ==============

  const content = (
    <Tabs
      value={activeTab}
      onValueChange={handleTabChange}
      className={cn('w-full', className)}
      data-testid="modality-tabs"
    >
      {/* Tab List */}
      <TabsList
        className={cn(
          'grid w-full',
          availableTabs.length === 1 && 'grid-cols-1',
          availableTabs.length === 2 && 'grid-cols-2',
          availableTabs.length === 3 && 'grid-cols-3',
          availableTabs.length === 4 && 'grid-cols-4',
        )}
      >
        {availableTabs.map((tab) => {
          const score = getScoreForTab(tab.id);
          const Icon = tab.icon;
          
          return (
            <TabsTrigger
              key={tab.id}
              value={tab.id}
              className="gap-2 relative"
              data-testid={`modality-tab-${tab.id}`}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              <span className="hidden sm:inline">{tab.shortLabel}</span>
              {score !== undefined && variant !== 'compact' && (
                <Badge
                  variant="secondary"
                  className={cn(
                    'ml-1 text-xs px-1.5 py-0',
                    getScoreBadgeColor(score)
                  )}
                >
                  {score.toFixed(0)}
                </Badge>
              )}
            </TabsTrigger>
          );
        })}
      </TabsList>

      {/* Tab Panels */}
      <div className="mt-4">
        {/* Video Panel */}
        {videoResult && (
          <TabsContent
            value="video"
            data-testid="modality-panel-video"
            className="focus-visible:outline-none"
          >
            <Suspense fallback={<PanelSkeleton />}>
              <VideoAnalysisPanel
                result={videoResult}
                analysisId={analysisId}
              />
            </Suspense>
          </TabsContent>
        )}

        {/* Audio Panel */}
        {audioResult && (
          <TabsContent
            value="audio"
            data-testid="modality-panel-audio"
            className="focus-visible:outline-none"
          >
            <Suspense fallback={<PanelSkeleton />}>
              <AudioAnalysisPanel
                result={audioResult}
                analysisId={analysisId}
              />
            </Suspense>
          </TabsContent>
        )}

        {/* Metadata Panel */}
        {metadataResult && (
          <TabsContent
            value="metadata"
            data-testid="modality-panel-metadata"
            className="focus-visible:outline-none"
          >
            <Suspense fallback={<PanelSkeleton />}>
              <MetadataPanel
                result={metadataResult}
                analysisId={analysisId}
              />
            </Suspense>
          </TabsContent>
        )}
      </div>
    </Tabs>
  );

  // Wrap in card if variant is 'card'
  if (variant === 'card') {
    return (
      <Card className={className}>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Detailed Analysis</CardTitle>
        </CardHeader>
        <CardContent>{content}</CardContent>
      </Card>
    );
  }

  return content;
}

// ============== SUB-COMPONENTS ==============

/**
 * Empty state when no modality data is available
 */
function EmptyModalityTabs({ className }: { className?: string }) {
  return (
    <div
      className={cn('py-8 text-center', className)}
      data-testid="modality-tabs-empty"
    >
      <Info className="h-10 w-10 mx-auto mb-3 text-muted-foreground/40" aria-hidden="true" />
      <p className="text-muted-foreground font-medium mb-1">
        No Detailed Analysis Available
      </p>
      <p className="text-sm text-muted-foreground/70">
        Modality-specific analysis results will appear here once processing is complete.
      </p>
    </div>
  );
}

/**
 * Panel skeleton loader
 */
function PanelSkeleton() {
  return (
    <div className="space-y-4 animate-pulse" data-testid="modality-panel-skeleton">
      <div className="flex items-center gap-4">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-6 w-20" />
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
      <Skeleton className="h-48 w-full" />
    </div>
  );
}

// ============== SKELETON LOADER ==============

/**
 * Skeleton loader for ModalityTabs
 */
export function ModalityTabsSkeleton({
  tabCount = 3,
  className,
}: {
  tabCount?: number;
  className?: string;
}) {
  return (
    <div className={cn('space-y-4', className)} data-testid="modality-tabs-skeleton">
      {/* Tabs skeleton */}
      <div
        className={cn(
          'grid gap-1 h-10 bg-muted rounded-md p-1',
          tabCount === 2 && 'grid-cols-2',
          tabCount === 3 && 'grid-cols-3',
          tabCount === 4 && 'grid-cols-4',
        )}
      >
        {Array.from({ length: tabCount }).map((_, i) => (
          <Skeleton key={i} className="h-full rounded-sm" />
        ))}
      </div>
      
      {/* Content skeleton */}
      <PanelSkeleton />
    </div>
  );
}

// ============== UTILITY FUNCTIONS ==============

/**
 * Get badge color based on score
 */
function getScoreBadgeColor(score: number): string {
  if (score >= 80) return 'bg-green-500/10 text-green-700 dark:text-green-400';
  if (score >= 60) return 'bg-lime-500/10 text-lime-700 dark:text-lime-400';
  if (score >= 40) return 'bg-yellow-500/10 text-yellow-700 dark:text-yellow-400';
  if (score >= 20) return 'bg-orange-500/10 text-orange-700 dark:text-orange-400';
  return 'bg-red-500/10 text-red-700 dark:text-red-400';
}

// ============== EXPORTS ==============

export default ModalityTabs;
