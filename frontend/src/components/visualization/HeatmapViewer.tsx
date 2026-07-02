/**
 * Argus Core - Heatmap Viewer Component
 * ======================================
 * GradCAM heatmap overlay viewer with zoom and frame navigation.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/visualization/HeatmapViewer.tsx
 * 
 * Role: Display GradCAM heatmap overlays showing where the model detected anomalies.
 * Supports toggling between original frame and heatmap, zoom functionality, and frame navigation.
 * 
 * Integration:
 * - Used by: VideoAnalysisPanel.tsx
 * - Backend: Uses heatmap URLs from GET /api/v1/analyze/{id}/heatmaps
 *   - Presigned MinIO URLs for heatmap images
 *   - Original frame URLs for comparison
 * 
 * Features:
 * - Heatmap overlay with opacity control
 * - Toggle between original/heatmap views
 * - Zoom in/out with pan support
 * - Frame navigation for multi-frame analysis
 * - Color scale legend for heatmap intensity
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton for image loading
 * - Empty state: Shows placeholder when no heatmaps available
 * - Error state: Shows error message with retry option
 * - Accessibility: Keyboard navigation, ARIA labels
 * - data-testid: heatmap-viewer, heatmap-image, heatmap-controls, heatmap-nav
 */

'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Skeleton } from '@/components/ui/skeleton';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Download,
  Maximize2,
  ImageIcon,
  AlertTriangle,
  Layers,
} from 'lucide-react';

// ============== TYPES ==============

/**
 * Props for HeatmapViewer component
 */
export interface HeatmapViewerProps {
  /** Array of heatmap image URLs */
  heatmapUrls: string[];
  /** Array of original frame URLs (same order as heatmaps) */
  originalFrameUrls?: string[];
  /** Currently selected frame index */
  selectedIndex?: number;
  /** Callback when frame index changes */
  onIndexChange?: (index: number) => void;
  /** Show overlay toggle (default: true) */
  showOverlayToggle?: boolean;
  /** Initial overlay visibility (default: true) */
  initialOverlayVisible?: boolean;
  /** Frame timestamps in milliseconds (for display) */
  timestamps?: number[];
  /** Frame scores (for display) */
  scores?: number[];
  /** Additional CSS classes */
  className?: string;
  /** Compact mode */
  compact?: boolean;
}

/**
 * Zoom/pan transform state
 */
interface TransformState {
  scale: number;
  translateX: number;
  translateY: number;
}

// ============== CONSTANTS ==============

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;
const ZOOM_STEP = 0.5;

/**
 * Color scale for heatmap intensity legend
 */
const HEATMAP_COLOR_SCALE = [
  { color: '#0000FF', label: 'Low' },
  { color: '#00FFFF', label: '' },
  { color: '#00FF00', label: 'Medium' },
  { color: '#FFFF00', label: '' },
  { color: '#FF0000', label: 'High' },
];

// ============== MAIN COMPONENT ==============

/**
 * HeatmapViewer Component
 * 
 * Displays GradCAM heatmap overlays for visualizing model attention areas.
 * Warmer colors indicate regions where the model detected higher manipulation probability.
 * 
 * @example
 * ```tsx
 * <HeatmapViewer
 *   heatmapUrls={['/heatmap1.png', '/heatmap2.png']}
 *   originalFrameUrls={['/frame1.png', '/frame2.png']}
 *   selectedIndex={0}
 *   onIndexChange={(i) => setSelected(i)}
 *   timestamps={[0, 1000, 2000]}
 *   scores={[0.2, 0.8, 0.5]}
 * />
 * ```
 */
export function HeatmapViewer({
  heatmapUrls,
  originalFrameUrls,
  selectedIndex = 0,
  onIndexChange,
  showOverlayToggle = true,
  initialOverlayVisible = true,
  timestamps,
  scores,
  className,
  compact = false,
}: HeatmapViewerProps) {
  // ============== STATE ==============

  const [currentIndex, setCurrentIndex] = useState(selectedIndex);
  const [showOverlay, setShowOverlay] = useState(initialOverlayVisible);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [transform, setTransform] = useState<TransformState>({
    scale: 1,
    translateX: 0,
    translateY: 0,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // ============== SYNC PROPS ==============

  useEffect(() => {
    setCurrentIndex(selectedIndex);
  }, [selectedIndex]);

  // ============== HANDLERS ==============

  const handleIndexChange = useCallback((newIndex: number) => {
    if (newIndex >= 0 && newIndex < heatmapUrls.length) {
      setCurrentIndex(newIndex);
      onIndexChange?.(newIndex);
      setIsLoading(true);
      setHasError(false);
    }
  }, [heatmapUrls.length, onIndexChange]);

  const handleZoomIn = useCallback(() => {
    setTransform(prev => ({
      ...prev,
      scale: Math.min(prev.scale + ZOOM_STEP, MAX_ZOOM),
    }));
  }, []);

  const handleZoomOut = useCallback(() => {
    setTransform(prev => ({
      ...prev,
      scale: Math.max(prev.scale - ZOOM_STEP, MIN_ZOOM),
    }));
  }, []);

  const handleReset = useCallback(() => {
    setTransform({ scale: 1, translateX: 0, translateY: 0 });
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (transform.scale > 1) {
      setIsDragging(true);
      dragStartRef.current = {
        x: e.clientX - transform.translateX,
        y: e.clientY - transform.translateY,
      };
    }
  }, [transform.scale, transform.translateX, transform.translateY]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging) {
      setTransform(prev => ({
        ...prev,
        translateX: e.clientX - dragStartRef.current.x,
        translateY: e.clientY - dragStartRef.current.y,
      }));
    }
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleImageLoad = useCallback(() => {
    setIsLoading(false);
    setHasError(false);
  }, []);

  const handleImageError = useCallback(() => {
    setIsLoading(false);
    setHasError(true);
  }, []);

  // ============== COMPUTED VALUES ==============

  const hasMultipleFrames = heatmapUrls.length > 1;
  const currentHeatmapUrl = heatmapUrls[currentIndex];
  const currentOriginalUrl = originalFrameUrls?.[currentIndex];
  const currentTimestamp = timestamps?.[currentIndex];
  const currentScore = scores?.[currentIndex];
  const canGoBack = currentIndex > 0;
  const canGoForward = currentIndex < heatmapUrls.length - 1;

  // ============== EMPTY STATE ==============

  if (heatmapUrls.length === 0) {
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center p-8 rounded-lg border border-dashed bg-muted/30',
          className
        )}
        data-testid="heatmap-viewer-empty"
      >
        <ImageIcon className="h-12 w-12 text-muted-foreground/40 mb-3" aria-hidden="true" />
        <p className="text-muted-foreground font-medium mb-1">No Heatmaps Available</p>
        <p className="text-sm text-muted-foreground/70 text-center max-w-xs">
          GradCAM heatmap visualizations will appear here when anomaly frames are detected.
        </p>
      </div>
    );
  }

  // ============== RENDER ==============

  return (
    <div
      className={cn('space-y-4', className)}
      data-testid="heatmap-viewer"
    >
      {/* Header with controls */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-primary" aria-hidden="true" />
          <span className="font-medium">GradCAM Heatmap</span>
          {currentScore !== undefined && (
            <Badge
              variant="secondary"
              className={cn(
                'ml-1 font-mono',
                currentScore < 0.5 ? 'bg-red-500/10 text-red-700 dark:text-red-400' : 'bg-green-500/10 text-green-700 dark:text-green-400'
              )}
            >
              {(currentScore * 100).toFixed(1)}%
            </Badge>
          )}
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-1" data-testid="heatmap-controls">
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8"
            onClick={handleZoomOut}
            disabled={transform.scale <= MIN_ZOOM}
            aria-label="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-xs text-muted-foreground w-12 text-center">
            {Math.round(transform.scale * 100)}%
          </span>
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8"
            onClick={handleZoomIn}
            disabled={transform.scale >= MAX_ZOOM}
            aria-label="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8"
            onClick={handleReset}
            disabled={transform.scale === 1 && transform.translateX === 0 && transform.translateY === 0}
            aria-label="Reset zoom"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Overlay toggle (if enabled and original frames available) */}
      {showOverlayToggle && currentOriginalUrl && (
        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
          <div className="flex items-center gap-2">
            <Switch
              id="overlay-toggle"
              checked={showOverlay}
              onCheckedChange={setShowOverlay}
            />
            <Label htmlFor="overlay-toggle" className="text-sm cursor-pointer">
              {showOverlay ? 'Showing Heatmap Overlay' : 'Showing Original Frame'}
            </Label>
          </div>
          {showOverlay && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Opacity:</span>
              <Slider
                value={[overlayOpacity * 100]}
                onValueChange={(v) => setOverlayOpacity(v[0] / 100)}
                min={20}
                max={100}
                step={10}
                className="w-24"
                aria-label="Overlay opacity"
              />
              <span className="text-xs text-muted-foreground w-8">
                {Math.round(overlayOpacity * 100)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* Image Container */}
      <div
        ref={containerRef}
        className={cn(
          'relative aspect-video bg-black rounded-lg overflow-hidden cursor-grab',
          isDragging && 'cursor-grabbing',
          transform.scale === 1 && 'cursor-default'
        )}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        data-testid="heatmap-image"
      >
        {/* Loading state */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-muted/80 z-10">
            <div className="flex flex-col items-center gap-2">
              <div className="h-8 w-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-muted-foreground">Loading heatmap...</span>
            </div>
          </div>
        )}

        {/* Error state */}
        {hasError && (
          <div className="absolute inset-0 flex items-center justify-center bg-muted z-10">
            <div className="flex flex-col items-center gap-2 text-center p-4">
              <AlertTriangle className="h-10 w-10 text-yellow-500" />
              <p className="text-sm font-medium">Failed to load heatmap</p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setIsLoading(true);
                  setHasError(false);
                }}
              >
                Retry
              </Button>
            </div>
          </div>
        )}

        {/* Original frame (background) */}
        {currentOriginalUrl && !showOverlay && (
          <img
            src={currentOriginalUrl}
            alt={`Original frame ${currentIndex + 1}`}
            className="absolute inset-0 w-full h-full object-contain"
            style={{
              transform: `scale(${transform.scale}) translate(${transform.translateX / transform.scale}px, ${transform.translateY / transform.scale}px)`,
              transformOrigin: 'center',
            }}
            onLoad={handleImageLoad}
            onError={handleImageError}
            draggable={false}
          />
        )}

        {/* Heatmap image */}
        {currentHeatmapUrl && (showOverlay || !currentOriginalUrl) && (
          <img
            src={currentHeatmapUrl}
            alt={`GradCAM heatmap frame ${currentIndex + 1}`}
            className="absolute inset-0 w-full h-full object-contain"
            style={{
              transform: `scale(${transform.scale}) translate(${transform.translateX / transform.scale}px, ${transform.translateY / transform.scale}px)`,
              transformOrigin: 'center',
              opacity: currentOriginalUrl && showOverlay ? overlayOpacity : 1,
            }}
            onLoad={handleImageLoad}
            onError={handleImageError}
            draggable={false}
          />
        )}

        {/* Frame info overlay */}
        <div className="absolute bottom-2 left-2 flex items-center gap-2">
          {currentTimestamp !== undefined && (
            <Badge variant="secondary" className="bg-black/60 text-white">
              {formatTimestamp(currentTimestamp)}
            </Badge>
          )}
        </div>

        {/* Frame counter */}
        {hasMultipleFrames && (
          <Badge
            variant="secondary"
            className="absolute top-2 right-2 bg-black/60 text-white"
          >
            {currentIndex + 1} / {heatmapUrls.length}
          </Badge>
        )}
      </div>

      {/* Navigation (if multiple frames) */}
      {hasMultipleFrames && (
        <div
          className="flex items-center justify-between"
          data-testid="heatmap-nav"
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleIndexChange(currentIndex - 1)}
            disabled={!canGoBack}
            aria-label="Previous frame"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Previous
          </Button>

          {/* Frame thumbnails (max 5 visible) */}
          <div className="flex gap-1 overflow-x-auto max-w-[50%] py-1">
            {heatmapUrls.slice(0, 7).map((_, index) => (
              <button
                key={index}
                onClick={() => handleIndexChange(index)}
                className={cn(
                  'w-8 h-8 rounded border-2 bg-muted flex items-center justify-center text-xs font-mono transition-all flex-shrink-0',
                  index === currentIndex
                    ? 'border-primary ring-2 ring-primary/20'
                    : 'border-transparent hover:border-muted-foreground/50'
                )}
                aria-label={`Go to frame ${index + 1}`}
                aria-current={index === currentIndex ? 'true' : undefined}
              >
                {index + 1}
              </button>
            ))}
            {heatmapUrls.length > 7 && (
              <span className="flex items-center text-xs text-muted-foreground px-2">
                +{heatmapUrls.length - 7}
              </span>
            )}
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() => handleIndexChange(currentIndex + 1)}
            disabled={!canGoForward}
            aria-label="Next frame"
          >
            Next
            <ChevronRight className="h-4 w-4 ml-1" />
          </Button>
        </div>
      )}

      {/* Color scale legend */}
      {!compact && (
        <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30 border border-dashed">
          <span className="text-xs text-muted-foreground font-medium">Intensity Scale:</span>
          <div className="flex items-center gap-1 flex-1">
            <div
              className="h-4 flex-1 rounded"
              style={{
                background: `linear-gradient(to right, ${HEATMAP_COLOR_SCALE.map(c => c.color).join(', ')})`,
              }}
            />
          </div>
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ============== UTILITY FUNCTIONS ==============

/**
 * Format timestamp in ms to MM:SS.mmm
 */
function formatTimestamp(ms: number): string {
  const totalSeconds = ms / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const milliseconds = Math.floor((totalSeconds % 1) * 1000);
  
  if (minutes > 0) {
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
  }
  return `${seconds}.${milliseconds.toString().padStart(3, '0')}s`;
}

// ============== SKELETON LOADER ==============

/**
 * Skeleton loader for HeatmapViewer
 */
export function HeatmapViewerSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn('space-y-4', className)} data-testid="heatmap-viewer-skeleton">
      {/* Header */}
      <div className="flex items-center justify-between">
        <Skeleton className="h-6 w-40" />
        <div className="flex gap-1">
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-12" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
        </div>
      </div>

      {/* Toggle */}
      <Skeleton className="h-12 w-full" />

      {/* Image */}
      <Skeleton className="aspect-video w-full rounded-lg" />

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <Skeleton className="h-9 w-24" />
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-8 w-8" />
          ))}
        </div>
        <Skeleton className="h-9 w-24" />
      </div>

      {/* Legend */}
      <Skeleton className="h-12 w-full" />
    </div>
  );
}

// ============== EXPORTS ==============

export default HeatmapViewer;
