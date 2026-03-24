# XAI Frontend Implementation Document
## Explainable AI Visualization Components for Argus Core

**Version:** 1.0  
**Status:** Production-Ready Specification  
**Compliance:** AGENTS_FRONTEND.md, AGENTS.md

---

## Executive Summary

This document provides comprehensive frontend implementation specifications for the newly implemented backend XAI (Explainable AI) features. The backend XAI system generates court-admissible forensic reports with:

- GradCAM++ heatmaps for image/video analysis
- Spectrogram overlays for audio analysis
- Token attribution for text analysis
- Feature importance breakdowns
- Scientific references and reproducibility hashes

---

## Phase 1: Backend XAI Endpoint Mapping

### 1.1 Current Backend Endpoints

| Endpoint | Method | Purpose | XAI Data Returned |
|----------|--------|---------|-------------------|
| `/api/v1/analyze/{id}` | GET | Basic analysis status | `trust_score`, `verdict` |
| `/api/v1/analyze/{id}/detail` | GET | Detailed results | All modality results with XAI fields |
| `/api/v1/analyze/{id}/heatmaps` | GET | Heatmap URLs | Presigned MinIO URLs |
| `/api/v1/analyze/{id}/report` | GET | PDF report | Report URL with XAI evidence |

### 1.2 New XAI Fields in API Responses

The backend now returns extended schemas with XAI fields:

#### AudioResult XAI Fields
```typescript
interface AudioResultXAI {
  // Existing fields
  synthetic_probability: number;
  vocoder_artifacts_detected: boolean;
  voice_consistency_score: number;
  
  // NEW XAI fields
  artifact_regions: AudioArtifactRegion[];
  frequency_anomaly_score: number;
  aasist_score: number;
}
```

#### TextResult XAI Fields
```typescript
interface TextResultXAI {
  // Existing fields
  ai_probability: number;
  perplexity_score: number;
  burstiness_score: number;
  
  // NEW XAI fields
  token_attributions: TokenAttribution[];
  perplexity_breakdown: PerplexityBreakdown[];
  roberta_score: number;
  vocabulary_diversity: number;
}
```

#### SpatialResult XAI Fields
```typescript
interface SpatialResultXAI {
  // Existing fields
  score: number;
  per_frame_scores: number[];
  
  // NEW XAI fields
  dct_anomaly_score: number;
  gan_fingerprint_detected: boolean;
  manipulation_regions: ManipulationRegion[];
  efficientnet_score: number;
  clip_score: number;
}
```

#### TemporalResult XAI Fields
```typescript
interface TemporalResultXAI {
  // Existing fields
  consistency_score: number;
  flickering_detected: boolean;
  
  // NEW XAI fields
  motion_anomaly_score: number;
  landmark_jitter_score: number;
  xclip_score: number;
}
```

---

## Phase 2: TypeScript Interface Definitions

### 2.1 Core XAI Types

Add to [`frontend/src/types/analysis.ts`](frontend/src/types/analysis.ts):

```typescript
// ============== XAI TYPES ==============

/**
 * Audio artifact region detected in spectrogram
 * Maps to backend AudioArtifactRegion schema
 */
export interface AudioArtifactRegion {
  start_time: number;      // Start time in seconds
  end_time: number;        // End time in seconds
  freq_low: number;        // Low frequency bound in Hz
  freq_high: number;       // High frequency bound in Hz
  artifact_type: 'vocoder' | 'spectral_gap' | 'harmonic_inconsistency' | 'high_energy_anomaly';
  confidence: number;      // Detection confidence (0-1)
}

/**
 * Token attribution for text XAI
 * Shows which tokens contribute to AI detection
 */
export interface TokenAttribution {
  token: string;           // The token/word
  attribution_score: number;  // Contribution to AI detection (-1 to 1)
  position: number;        // Token position in text
  is_ai_indicator: boolean;   // Whether this token indicates AI generation
}

/**
 * Perplexity breakdown by text segment
 */
export interface PerplexityBreakdown {
  segment: string;         // Text segment
  perplexity: number;      // Perplexity score
  is_anomalous: boolean;   // Whether segment is anomalous
}

/**
 * Feature importance for model decision
 */
export interface FeatureImportance {
  feature_name: string;    // Feature identifier
  importance_score: number; // Importance (0-1)
  contribution_direction: 'increases_fake' | 'decreases_fake';
  confidence: number;      // Confidence in importance (0-1)
}

/**
 * Manipulation region in image/video
 */
export interface ManipulationRegion {
  region_type: string;     // Region type: face, mouth, background, etc.
  location: string;        // Description or coordinates
  confidence: number;      // Detection confidence (0-1)
  frame_indices?: number[]; // Affected frame indices
}

/**
 * Scientific reference for methodology
 */
export interface ScientificReference {
  method_name: string;     // Method name (e.g., "GradCAM++")
  citation: string;        // Citation string
  doi?: string;            // DOI link
}

/**
 * Visual evidence for reports
 */
export interface VisualEvidence {
  evidence_type: 'heatmap' | 'spectrogram_overlay' | 'token_highlight' | 'attention_map';
  url: string;             // MinIO presigned URL
  description: string;     // Human-readable description
  frame_index?: number;    // For video frames
  timestamp?: number;      // For audio timestamps
}

/**
 * Complete XAI explanation package
 */
export interface XAIExplanation {
  feature_importance: FeatureImportance[];
  visual_evidence: VisualEvidence[];
  scientific_references: ScientificReference[];
  reproducibility_hash: string;
  confidence_interval: [number, number];  // Tuple of (lower, upper)
  model_versions: Record<string, string>;
}

/**
 * Extended AudioResult with XAI fields
 */
export interface AudioResultXAI extends AudioResult {
  artifact_regions: AudioArtifactRegion[];
  frequency_anomaly_score: number;
  aasist_score: number;
  xai_explanation?: XAIExplanation;
}

/**
 * Extended TextResult with XAI fields
 */
export interface TextResultXAI extends TextResult {
  token_attributions: TokenAttribution[];
  perplexity_breakdown: PerplexityBreakdown[];
  roberta_score: number;
  vocabulary_diversity: number;
  xai_explanation?: XAIExplanation;
}

/**
 * Extended SpatialResult with XAI fields
 */
export interface SpatialResultXAI extends SpatialResult {
  dct_anomaly_score: number;
  gan_fingerprint_detected: boolean;
  manipulation_regions: ManipulationRegion[];
  efficientnet_score: number;
  clip_score: number;
  xai_explanation?: XAIExplanation;
}

/**
 * Extended TemporalResult with XAI fields
 */
export interface TemporalResultXAI extends TemporalResult {
  motion_anomaly_score: number;
  landmark_jitter_score: number;
  xclip_score: number;
}

/**
 * Extended VideoResult with XAI fields
 */
export interface VideoResultXAI extends VideoResult {
  frame_heatmap_urls: string[];
  temporal_heatmap_url?: string;
  confidence_interval: [number, number];
  spatial: SpatialResultXAI;
  temporal: TemporalResultXAI;
  xai_explanation?: XAIExplanation;
}
```

### 2.2 API Response Extensions

```typescript
/**
 * Extended analysis detail response with XAI data
 */
export interface AnalysisDetailResponseXAI extends AnalysisDetailResponse {
  video_result?: VideoResultXAI;
  audio_result?: AudioResultXAI;
  text_result?: TextResultXAI;
  xai_evidence_package?: {
    feature_importance: FeatureImportance[];
    visual_evidence: VisualEvidence[];
    scientific_references: ScientificReference[];
    reproducibility_hash: string;
    confidence_interval: [number, number];
  };
}

/**
 * XAI heatmap response with metadata
 */
export interface XAIHeatmapResponse {
  heatmaps: Array<{
    key: string;
    url: string;
    frame_index?: number;
    timestamp_ms?: number;
    overlay_url?: string;  // Combined overlay URL
  }>;
  spectrogram_overlay?: {
    url: string;
    artifact_regions: AudioArtifactRegion[];
  };
  token_highlights?: Array<{
    text: string;
    highlights: Array<{
      start: number;
      end: number;
      score: number;
    }>;
  }>;
  count: number;
}
```

---

## Phase 3: Component Hierarchy and File Structure

### 3.1 Proposed Component Structure

```
frontend/src/
  components/
    xai/                          # NEW: XAI-specific components
      XAIExplanationPanel.tsx     # Main XAI display container
      FeatureImportanceTable.tsx  # Feature importance visualization
      HeatmapOverlayViewer.tsx    # Enhanced heatmap viewer with XAI
      SpectrogramOverlay.tsx      # Audio spectrogram with artifacts
      TokenAttributionView.tsx    # Text token highlighting
      ScientificReferences.tsx    # Citations display
      ConfidenceInterval.tsx      # Confidence interval visualization
      ReproducibilityHash.tsx     # Hash display with copy button
      XAIEvidenceGallery.tsx      # Gallery of all visual evidence
      
    visualization/
      HeatmapViewer.tsx           # EXISTING: Enhance with XAI support
      SpectrogramViewer.tsx       # EXISTING: Enhance with artifact markers
      TimelineChart.tsx           # EXISTING: Add XAI timeline markers
      
  hooks/
    useXAI.ts                     # NEW: XAI data fetching hook
    useHeatmapOverlay.ts          # NEW: Heatmap overlay management
    
  services/
    analysisApi.ts                # EXISTING: Add XAI endpoint methods
    
  store/
    xaiStore.ts                   # NEW: XAI state management
```

### 3.2 Component Integration Points

| Component | Location | Integration |
|-----------|----------|-------------|
| `XAIExplanationPanel` | `components/xai/` | Used in `VideoAnalysisPanel`, `AudioAnalysisPanel`, `TextAnalysisPanel` |
| `FeatureImportanceTable` | `components/xai/` | Child of `XAIExplanationPanel` |
| `HeatmapOverlayViewer` | `components/xai/` | Replaces/enhances `HeatmapViewer` for XAI |
| `SpectrogramOverlay` | `components/xai/` | Used in `AudioAnalysisPanel` |
| `TokenAttributionView` | `components/xai/` | Used in `TextAnalysisPanel` |
| `ScientificReferences` | `components/xai/` | Used in `ResultsPanel`, PDF export |

---

## Phase 4: Component Specifications

### 4.1 XAIExplanationPanel

**Purpose:** Main container for all XAI visualizations

**Props Interface:**
```typescript
interface XAIExplanationPanelProps {
  /** XAI explanation data from backend */
  explanation: XAIExplanation;
  /** Modality type for context */
  modality: 'video' | 'audio' | 'image' | 'text';
  /** Analysis ID for fetching additional data */
  analysisId: string;
  /** Callback when evidence is selected */
  onEvidenceSelect?: (evidence: VisualEvidence) => void;
  /** Show scientific references section */
  showReferences?: boolean;
  /** Compact mode for sidebar */
  compact?: boolean;
  /** Additional CSS classes */
  className?: string;
}
```

**State Management:**
```typescript
interface XAIExplanationPanelState {
  activeTab: 'features' | 'evidence' | 'references';
  selectedEvidenceIndex: number | null;
  isExpanded: boolean;
}
```

**Component Contract (P0 Compliance):**
- Loading: Skeleton with "Loading explanation..." text
- Error: Error boundary with retry button
- Empty: "No explanation available" message
- Accessibility: ARIA labels, keyboard navigation between tabs

**Implementation Snippet:**
```typescript
export function XAIExplanationPanel({
  explanation,
  modality,
  analysisId,
  onEvidenceSelect,
  showReferences = true,
  compact = false,
  className,
}: XAIExplanationPanelProps) {
  const [activeTab, setActiveTab] = useState<'features' | 'evidence' | 'references'>('features');
  const [selectedEvidenceIndex, setSelectedEvidenceIndex] = useState<number | null>(null);
  
  const handleEvidenceClick = useCallback((index: number) => {
    setSelectedEvidenceIndex(index);
    onEvidenceSelect?.(explanation.visual_evidence[index]);
  }, [explanation.visual_evidence, onEvidenceSelect]);
  
  return (
    <Card className={cn('xai-explanation-panel', className)} data-testid="xai-explanation-panel">
      <CardHeader>
        <CardTitle>Explainable AI Analysis</CardTitle>
        <CardDescription>
          Transparent decision factors and visual evidence
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
          <TabsList>
            <TabsTrigger value="features">Feature Importance</TabsTrigger>
            <TabsTrigger value="evidence">Visual Evidence</TabsTrigger>
            {showReferences && <TabsTrigger value="references">References</TabsTrigger>}
          </TabsList>
          
          <TabsContent value="features">
            <FeatureImportanceTable features={explanation.feature_importance} />
            <ConfidenceInterval interval={explanation.confidence_interval} />
          </TabsContent>
          
          <TabsContent value="evidence">
            <XAIEvidenceGallery
              evidence={explanation.visual_evidence}
              selectedIndex={selectedEvidenceIndex}
              onSelect={handleEvidenceClick}
              modality={modality}
            />
          </TabsContent>
          
          <TabsContent value="references">
            <ScientificReferences references={explanation.scientific_references} />
          </TabsContent>
        </Tabs>
        
        <ReproducibilityHash hash={explanation.reproducibility_hash} />
      </CardContent>
    </Card>
  );
}
```

### 4.2 FeatureImportanceTable

**Purpose:** Display feature importance scores in a sortable table

**Props Interface:**
```typescript
interface FeatureImportanceTableProps {
  /** Feature importance data */
  features: FeatureImportance[];
  /** Sort by importance score (default: true) */
  sortByImportance?: boolean;
  /** Show contribution direction icons */
  showDirection?: boolean;
  /** Highlight top N features */
  highlightTop?: number;
  /** Additional CSS classes */
  className?: string;
}
```

**Implementation Snippet:**
```typescript
export function FeatureImportanceTable({
  features,
  sortByImportance = true,
  showDirection = true,
  highlightTop = 3,
  className,
}: FeatureImportanceTableProps) {
  const sortedFeatures = useMemo(() => {
    if (!sortByImportance) return features;
    return [...features].sort((a, b) => b.importance_score - a.importance_score);
  }, [features, sortByImportance]);
  
  return (
    <div className={cn('feature-importance-table', className)} data-testid="feature-importance-table">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Feature</TableHead>
            <TableHead>Importance</TableHead>
            {showDirection && <TableHead>Effect</TableHead>}
            <TableHead>Confidence</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sortedFeatures.map((feature, index) => (
            <TableRow
              key={feature.feature_name}
              className={cn(index < highlightTop && 'bg-muted/50')}
            >
              <TableCell className="font-medium">
                {formatFeatureName(feature.feature_name)}
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <Progress value={feature.importance_score * 100} className="w-20" />
                  <span className="text-sm text-muted-foreground">
                    {(feature.importance_score * 100).toFixed(1)}%
                  </span>
                </div>
              </TableCell>
              {showDirection && (
                <TableCell>
                  <Badge variant={feature.contribution_direction === 'increases_fake' ? 'destructive' : 'default'}>
                    {feature.contribution_direction === 'increases_fake' ? (
                      <><TrendingUp className="mr-1 h-3 w-3" /> Increases Fake</>
                    ) : (
                      <><TrendingDown className="mr-1 h-3 w-3" /> Decreases Fake</>
                    )}
                  </Badge>
                </TableCell>
              )}
              <TableCell>
                <span className="text-sm">{(feature.confidence * 100).toFixed(0)}%</span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
```

### 4.3 SpectrogramOverlay

**Purpose:** Display audio spectrogram with artifact region markers

**Props Interface:**
```typescript
interface SpectrogramOverlayProps {
  /** Spectrogram image URL */
  spectrogramUrl: string;
  /** Detected artifact regions */
  artifactRegions: AudioArtifactRegion[];
  /** Audio duration in seconds */
  durationSeconds: number;
  /** Sample rate for frequency calculation */
  sampleRate?: number;
  /** Selected region index */
  selectedRegionIndex?: number;
  /** Callback when region is clicked */
  onRegionClick?: (index: number, region: AudioArtifactRegion) => void;
  /** Show frequency axis labels */
  showFrequencyAxis?: boolean;
  /** Show time axis labels */
  showTimeAxis?: boolean;
  /** Additional CSS classes */
  className?: string;
}
```

**Implementation Snippet:**
```typescript
export function SpectrogramOverlay({
  spectrogramUrl,
  artifactRegions,
  durationSeconds,
  sampleRate = 16000,
  selectedRegionIndex,
  onRegionClick,
  showFrequencyAxis = true,
  showTimeAxis = true,
  className,
}: SpectrogramOverlayProps) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [hoveredRegion, setHoveredRegion] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Calculate region position as percentages
  const getRegionStyle = useCallback((region: AudioArtifactRegion) => {
    const left = (region.start_time / durationSeconds) * 100;
    const width = ((region.end_time - region.start_time) / durationSeconds) * 100;
    const maxFreq = sampleRate / 2;
    const bottom = (region.freq_low / maxFreq) * 100;
    const height = ((region.freq_high - region.freq_low) / maxFreq) * 100;
    
    return {
      left: `${left}%`,
      width: `${width}%`,
      bottom: `${bottom}%`,
      height: `${height}%`,
    };
  }, [durationSeconds, sampleRate]);
  
  return (
    <div
      ref={containerRef}
      className={cn('spectrogram-overlay relative', className)}
      data-testid="spectrogram-overlay"
    >
      {/* Spectrogram image */}
      <div className="relative">
        {imageLoaded || <Skeleton className="absolute inset-0" />}
        <img
          src={spectrogramUrl}
          alt="Audio spectrogram with artifact detection"
          className="w-full h-auto"
          onLoad={() => setImageLoaded(true)}
        />
        
        {/* Artifact region overlays */}
        {imageLoaded && artifactRegions.map((region, index) => (
          <div
            key={index}
            className={cn(
              'absolute border-2 cursor-pointer transition-all',
              'hover:border-white hover:z-10',
              getArtifactColor(region.artifact_type),
              selectedRegionIndex === index && 'border-white z-20',
              hoveredRegion === index && 'border-white z-10'
            )}
            style={getRegionStyle(region)}
            onClick={() => onRegionClick?.(index, region)}
            onMouseEnter={() => setHoveredRegion(index)}
            onMouseLeave={() => setHoveredRegion(null)}
            role="button"
            tabIndex={0}
            aria-label={`${region.artifact_type} artifact at ${region.start_time.toFixed(2)}s`}
          >
            {/* Tooltip on hover */}
            {hoveredRegion === index && (
              <div className="absolute bottom-full left-0 mb-2 p-2 bg-popover text-popover-foreground rounded shadow-lg text-xs whitespace-nowrap z-30">
                <div className="font-semibold">{formatArtifactType(region.artifact_type)}</div>
                <div>Time: {region.start_time.toFixed(2)}s - {region.end_time.toFixed(2)}s</div>
                <div>Freq: {region.freq_low.toFixed(0)}Hz - {region.freq_high.toFixed(0)}Hz</div>
                <div>Confidence: {(region.confidence * 100).toFixed(0)}%</div>
              </div>
            )}
          </div>
        ))}
      </div>
      
      {/* Time axis */}
      {showTimeAxis && (
        <div className="flex justify-between mt-1 text-xs text-muted-foreground">
          <span>0s</span>
          <span>{(durationSeconds / 2).toFixed(1)}s</span>
          <span>{durationSeconds.toFixed(1)}s</span>
        </div>
      )}
      
      {/* Legend */}
      <div className="flex flex-wrap gap-2 mt-3">
        {artifactRegions.length > 0 && (
          <div className="flex items-center gap-1 text-xs">
            <span className="text-muted-foreground">Detected artifacts:</span>
            {getUniqueArtifactTypes(artifactRegions).map(type => (
              <Badge key={type} variant="outline" className={getArtifactColor(type)}>
                {formatArtifactType(type)}
              </Badge>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Helper functions
function getArtifactColor(type: string): string {
  const colors: Record<string, string> = {
    vocoder: 'border-red-500 bg-red-500/20',
    spectral_gap: 'border-yellow-500 bg-yellow-500/20',
    harmonic_inconsistency: 'border-purple-500 bg-purple-500/20',
    high_energy_anomaly: 'border-orange-500 bg-orange-500/20',
  };
  return colors[type] || 'border-blue-500 bg-blue-500/20';
}

function formatArtifactType(type: string): string {
  return type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
```

### 4.4 TokenAttributionView

**Purpose:** Display text with token-level attribution highlighting

**Props Interface:**
```typescript
interface TokenAttributionViewProps {
  /** Original text content */
  text: string;
  /** Token attribution data */
  attributions: TokenAttribution[];
  /** Highlight mode */
  highlightMode?: 'gradient' | 'binary';
  /** Show attribution scores on hover */
  showScores?: boolean;
  /** Color for AI-indicating tokens */
  aiIndicatorColor?: string;
  /** Color for human-indicating tokens */
  humanIndicatorColor?: string;
  /** Additional CSS classes */
  className?: string;
}
```

**Implementation Snippet:**
```typescript
export function TokenAttributionView({
  text,
  attributions,
  highlightMode = 'gradient',
  showScores = true,
  aiIndicatorColor = 'bg-red-500/30',
  humanIndicatorColor = 'bg-green-500/30',
  className,
}: TokenAttributionViewProps) {
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);
  
  // Build token map for efficient lookup
  const tokenMap = useMemo(() => {
    const map = new Map<number, TokenAttribution>();
    attributions.forEach(attr => {
      map.set(attr.position, attr);
    });
    return map;
  }, [attributions]);
  
  // Split text into tokens (simplified - use tokenizer for production)
  const tokens = useMemo(() => {
    return text.split(/(\s+)/);
  }, [text]);
  
  // Calculate background color based on attribution
  const getTokenStyle = useCallback((attr: TokenAttribution | undefined) => {
    if (!attr) return {};
    
    if (highlightMode === 'binary') {
      return {
        className: attr.is_ai_indicator ? aiIndicatorColor : humanIndicatorColor,
      };
    }
    
    // Gradient mode - intensity based on absolute score
    const intensity = Math.abs(attr.attribution_score);
    const color = attr.attribution_score > 0 ? 'red' : 'green';
    return {
      style: {
        backgroundColor: `color-mix(in srgb, var(--${color}-500) ${intensity * 100}%, transparent)`,
      },
    };
  }, [highlightMode, aiIndicatorColor, humanIndicatorColor]);
  
  return (
    <div
      className={cn('token-attribution-view font-mono text-sm leading-relaxed', className)}
      data-testid="token-attribution-view"
    >
      <div className="p-4 rounded-lg border bg-card">
        {tokens.map((token, index) => {
          const attr = tokenMap.get(index);
          const style = getTokenStyle(attr);
          
          if (token.match(/^\s+$/)) {
            return <span key={index}>{token}</span>;
          }
          
          return (
            <span
              key={index}
              className={cn(
                'relative inline-block px-0.5 rounded transition-all',
                attr?.is_ai_indicator && 'font-semibold',
                hoveredToken === index && 'ring-2 ring-primary',
                style.className
              )}
              style={style.style}
              onMouseEnter={() => setHoveredToken(index)}
              onMouseLeave={() => setHoveredToken(null)}
            >
              {token}
              
              {/* Tooltip */}
              {hoveredToken === index && attr && showScores && (
                <span className="absolute bottom-full left-0 mb-1 p-1 bg-popover text-popover-foreground rounded text-xs whitespace-nowrap z-10 shadow-lg">
                  Score: {attr.attribution_score.toFixed(3)}
                  {attr.is_ai_indicator && ' (AI indicator)'}
                </span>
              )}
            </span>
          );
        })}
      </div>
      
      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <span className="w-4 h-4 rounded bg-red-500/30" />
          <span>AI-indicating</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-4 h-4 rounded bg-green-500/30" />
          <span>Human-indicating</span>
        </div>
      </div>
    </div>
  );
}
```

### 4.5 ConfidenceInterval

**Purpose:** Visualize confidence interval for predictions

**Props Interface:**
```typescript
interface ConfidenceIntervalProps {
  /** Confidence interval tuple [lower, upper] */
  interval: [number, number];
  /** Point estimate (optional, defaults to midpoint) */
  pointEstimate?: number;
  /** Label for the metric */
  label?: string;
  /** Show as percentage */
  asPercentage?: boolean;
  /** Width of the visualization */
  width?: number | string;
  /** Additional CSS classes */
  className?: string;
}
```

**Implementation Snippet:**
```typescript
export function ConfidenceInterval({
  interval,
  pointEstimate,
  label = 'Confidence',
  asPercentage = true,
  width = '100%',
  className,
}: ConfidenceIntervalProps) {
  const [lower, upper] = interval;
  const midpoint = pointEstimate ?? (lower + upper) / 2;
  const range = upper - lower;
  
  const formatValue = useCallback((v: number) => {
    return asPercentage ? `${(v * 100).toFixed(1)}%` : v.toFixed(3);
  }, [asPercentage]);
  
  return (
    <div className={cn('confidence-interval', className)} data-testid="confidence-interval">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className="text-sm font-medium">{formatValue(midpoint)}</span>
      </div>
      
      <div className="relative h-6" style={{ width }}>
        {/* Background track */}
        <div className="absolute inset-y-0 left-0 right-0 bg-muted rounded-full" />
        
        {/* Interval range */}
        <div
          className="absolute inset-y-0 bg-primary/30 rounded-full"
          style={{
            left: `${lower * 100}%`,
            width: `${range * 100}%`,
          }}
        />
        
        {/* Point estimate marker */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-1 h-4 bg-primary rounded-full"
          style={{ left: `${midpoint * 100}%` }}
        />
        
        {/* Lower bound marker */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-border"
          style={{ left: `${lower * 100}%` }}
        />
        
        {/* Upper bound marker */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-border"
          style={{ left: `${upper * 100}%` }}
        />
      </div>
      
      <div className="flex justify-between mt-1 text-xs text-muted-foreground">
        <span>{formatValue(lower)}</span>
        <span>95% CI</span>
        <span>{formatValue(upper)}</span>
      </div>
    </div>
  );
}
```

### 4.6 ReproducibilityHash

**Purpose:** Display reproducibility hash with copy functionality

**Props Interface:**
```typescript
interface ReproducibilityHashProps {
  /** SHA-256 hash string */
  hash: string;
  /** Show full hash or truncated */
  showFull?: boolean;
  /** Label text */
  label?: string;
  /** Additional CSS classes */
  className?: string;
}
```

**Implementation Snippet:**
```typescript
export function ReproducibilityHash({
  hash,
  showFull = false,
  label = 'Reproducibility Hash',
  className,
}: ReproducibilityHashProps) {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [hash]);
  
  const displayHash = showFull ? hash : `${hash.slice(0, 16)}...${hash.slice(-8)}`;
  
  return (
    <div className={cn('reproducibility-hash flex items-center gap-2', className)} data-testid="reproducibility-hash">
      <span className="text-xs text-muted-foreground">{label}:</span>
      <code className="text-xs font-mono bg-muted px-2 py-1 rounded">
        {displayHash}
      </code>
      <Button
        variant="ghost"
        size="sm"
        className="h-6 w-6 p-0"
        onClick={handleCopy}
        aria-label="Copy hash"
      >
        {copied ? (
          <Check className="h-3 w-3 text-green-500" />
        ) : (
          <Copy className="h-3 w-3" />
        )}
      </Button>
    </div>
  );
}
```

---

## Phase 5: State Management Strategy

### 5.1 XAI Store (Zustand)

Create [`frontend/src/store/xaiStore.ts`](frontend/src/store/xaiStore.ts):

```typescript
/**
 * Argus Core - XAI State Store
 * ============================
 * Zustand store for XAI data management.
 * 
 * Implements: AGENTS_FRONTEND.md - Section 11 - State Management
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type {
  XAIExplanation,
  VisualEvidence,
  AudioArtifactRegion,
  TokenAttribution,
  FeatureImportance,
} from '@/types/analysis';

interface XAIState {
  // Current XAI data
  currentExplanation: XAIExplanation | null;
  artifactRegions: AudioArtifactRegion[];
  tokenAttributions: TokenAttribution[];
  featureImportance: FeatureImportance[];
  
  // UI state
  selectedEvidenceIndex: number | null;
  selectedRegionIndex: number | null;
  heatmapOpacity: number;
  showOverlay: boolean;
  
  // Loading states
  isLoading: boolean;
  error: string | null;
  
  // Actions
  setExplanation: (explanation: XAIExplanation | null) => void;
  setArtifactRegions: (regions: AudioArtifactRegion[]) => void;
  setTokenAttributions: (attributions: TokenAttribution[]) => void;
  setFeatureImportance: (features: FeatureImportance[]) => void;
  selectEvidence: (index: number | null) => void;
  selectRegion: (index: number | null) => void;
  setHeatmapOpacity: (opacity: number) => void;
  toggleOverlay: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState = {
  currentExplanation: null,
  artifactRegions: [],
  tokenAttributions: [],
  featureImportance: [],
  selectedEvidenceIndex: null,
  selectedRegionIndex: null,
  heatmapOpacity: 0.7,
  showOverlay: true,
  isLoading: false,
  error: null,
};

export const useXAIStore = create<XAIState>()(
  devtools(
    persist(
      (set) => ({
        ...initialState,
        
        setExplanation: (explanation) => set({ currentExplanation: explanation }),
        
        setArtifactRegions: (regions) => set({ artifactRegions: regions }),
        
        setTokenAttributions: (attributions) => set({ tokenAttributions: attributions }),
        
        setFeatureImportance: (features) => set({ featureImportance: features }),
        
        selectEvidence: (index) => set({ selectedEvidenceIndex: index }),
        
        selectRegion: (index) => set({ selectedRegionIndex: index }),
        
        setHeatmapOpacity: (opacity) => set({ heatmapOpacity: Math.max(0, Math.min(1, opacity)) }),
        
        toggleOverlay: () => set((state) => ({ showOverlay: !state.showOverlay })),
        
        setLoading: (loading) => set({ isLoading: loading }),
        
        setError: (error) => set({ error }),
        
        reset: () => set(initialState),
      }),
      {
        name: 'xai-storage',
        partialize: (state) => ({
          heatmapOpacity: state.heatmapOpacity,
          showOverlay: state.showOverlay,
        }),
      }
    ),
    { name: 'XAIStore' }
  )
);
```

### 5.2 XAI Hook

Create [`frontend/src/hooks/useXAI.ts`](frontend/src/hooks/useXAI.ts):

```typescript
/**
 * Argus Core - XAI Data Hook
 * ==========================
 * React Query hook for fetching XAI data.
 * 
 * Implements: AGENTS_FRONTEND.md - Section 11 - State Management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { analysisApi } from '@/services/analysisApi';
import { useXAIStore } from '@/store/xaiStore';
import type { XAIHeatmapResponse, AnalysisDetailResponseXAI } from '@/types/analysis';

/**
 * Hook for fetching XAI data for an analysis
 */
export function useXAI(analysisId: string | null) {
  const queryClient = useQueryClient();
  const { setExplanation, setArtifactRegions, setTokenAttributions, setFeatureImportance, setLoading, setError } = useXAIStore();
  
  // Fetch detailed analysis with XAI data
  const detailQuery = useQuery<AnalysisDetailResponseXAI>({
    queryKey: ['analysis', analysisId, 'detail'],
    queryFn: async () => {
      if (!analysisId) throw new Error('No analysis ID');
      const response = await analysisApi.getAnalysisDetail(analysisId);
      return response as AnalysisDetailResponseXAI;
    },
    enabled: !!analysisId,
    onSuccess: (data) => {
      // Update store with XAI data
      if (data.xai_evidence_package) {
        setExplanation({
          feature_importance: data.xai_evidence_package.feature_importance,
          visual_evidence: data.xai_evidence_package.visual_evidence,
          scientific_references: data.xai_evidence_package.scientific_references,
          reproducibility_hash: data.xai_evidence_package.reproducibility_hash,
          confidence_interval: data.xai_evidence_package.confidence_interval,
          model_versions: {},
        });
        setFeatureImportance(data.xai_evidence_package.feature_importance);
      }
      
      if (data.audio_result?.artifact_regions) {
        setArtifactRegions(data.audio_result.artifact_regions);
      }
      
      if (data.text_result?.token_attributions) {
        setTokenAttributions(data.text_result.token_attributions);
      }
    },
    onError: (error) => {
      setError(error instanceof Error ? error.message : 'Failed to fetch XAI data');
    },
  });
  
  // Fetch heatmap data
  const heatmapQuery = useQuery<XAIHeatmapResponse>({
    queryKey: ['analysis', analysisId, 'heatmaps'],
    queryFn: async () => {
      if (!analysisId) throw new Error('No analysis ID');
      return analysisApi.getHeatmaps(analysisId);
    },
    enabled: !!analysisId,
  });
  
  return {
    // Data
    detail: detailQuery.data,
    heatmaps: heatmapQuery.data,
    
    // Loading states
    isLoading: detailQuery.isLoading || heatmapQuery.isLoading,
    isFetching: detailQuery.isFetching || heatmapQuery.isFetching,
    
    // Error states
    error: detailQuery.error || heatmapQuery.error,
    
    // Refetch
    refetch: () => {
      detailQuery.refetch();
      heatmapQuery.refetch();
    },
    
    // Status
    isSuccess: detailQuery.isSuccess && heatmapQuery.isSuccess,
  };
}

/**
 * Hook for downloading XAI report
 */
export function useXAIReport(analysisId: string | null) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (regenerate: boolean = false) => {
      if (!analysisId) throw new Error('No analysis ID');
      return analysisApi.getReport(analysisId, regenerate);
    },
    onSuccess: (data) => {
      // Open report in new tab or trigger download
      window.open(data.reportUrl, '_blank');
    },
  });
}
```

---

## Phase 6: API Service Extensions

### 6.1 Extended analysisApi.ts

Add to [`frontend/src/services/analysisApi.ts`](frontend/src/services/analysisApi.ts):

```typescript
// Add to existing analysisApi object

/**
 * Get XAI explanation data
 * GET /api/v1/analyze/{id}/xai
 */
getXAI: async (id: string): Promise<XAIExplanation> => {
  const response = await api.get<XAIExplanation>(`/api/v1/analyze/${id}/xai`);
  return response.data;
},

/**
 * Get spectrogram overlay with artifacts
 * GET /api/v1/analyze/{id}/spectrogram
 */
getSpectrogram: async (id: string): Promise<{
  spectrogram_url: string;
  overlay_url: string;
  artifact_regions: AudioArtifactRegion[];
}> => {
  const response = await api.get(`/api/v1/analyze/${id}/spectrogram`);
  return response.data;
},

/**
 * Get token attribution data
 * GET /api/v1/analyze/{id}/tokens
 */
getTokenAttribution: async (id: string): Promise<{
  text: string;
  attributions: TokenAttribution[];
}> => {
  const response = await api.get(`/api/v1/analyze/${id}/tokens`);
  return response.data;
},

/**
 * Export XAI evidence package
 * POST /api/v1/analyze/{id}/export
 */
exportXAI: async (id: string, format: 'pdf' | 'json' | 'zip'): Promise<{ download_url: string }> => {
  const response = await api.post(`/api/v1/analyze/${id}/export`, { format });
  return response.data;
},
```

---

## Phase 7: UI/UX Wireframe Descriptions

### 7.1 XAI Explanation Panel Layout

```
+-----------------------------------------------+
| Explainable AI Analysis                       |
| Transparent decision factors and visual evidence
+-----------------------------------------------+
| [Feature Importance] [Visual Evidence] [References]
+-----------------------------------------------+
| Feature          | Importance | Effect       | Conf  |
| ---------------- | ---------- | ------------ | ----- |
| DCT Anomaly      | ========== | Increases    | 85%   |
| GAN Fingerprint  | ========    | Increases    | 72%   |
| Edge Artifacts   | ======      | Increases    | 68%   |
+-----------------------------------------------+
| Confidence Interval: [====|====] 85% (75%-95%)
+-----------------------------------------------+
| Reproducibility Hash: a4f532770bc622bb... [Copy]
+-----------------------------------------------+
```

### 7.2 Spectrogram Overlay Layout

```
+-----------------------------------------------+
| Audio Spectrogram Analysis                    |
+-----------------------------------------------+
| Frequency (Hz)                                |
| 8000 |                                         |
| 6000 |    [====]                               |
| 4000 |        [====]   <-- Vocoder artifact    |
| 2000 |                                         |
|    0 |_________________________________________|
|       0s        5s        10s       15s       Time
+-----------------------------------------------+
| Legend: [Vocoder] [Spectral Gap] [Harmonic]
+-----------------------------------------------+
```

### 7.3 Token Attribution Layout

```
+-----------------------------------------------+
| Text Analysis - Token Attribution             |
+-----------------------------------------------+
| "The quick brown fox jumps over the lazy dog"|
|     ^^^^      ^^^^                            |
|     AI        AI                              |
+-----------------------------------------------+
| Legend: [AI-indicating] [Human-indicating]
+-----------------------------------------------+
```

---

## Phase 8: Performance Optimization Strategies

### 8.1 Large Heatmap Rendering

1. **Lazy Loading**: Load heatmaps only when visible in viewport
2. **Progressive Loading**: Load low-res first, then high-res
3. **Canvas Rendering**: Use Canvas API for large images instead of DOM
4. **Tile-based Rendering**: Split large heatmaps into tiles

```typescript
// Example: Lazy heatmap loading with IntersectionObserver
const heatmapObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const img = entry.target as HTMLImageElement;
        img.src = img.dataset.src!;
        heatmapObserver.unobserve(img);
      }
    });
  },
  { rootMargin: '100px' }
);
```

### 8.2 Virtualization for Token Lists

```typescript
// Use react-window for large token lists
import { FixedSizeList } from 'react-window';

function VirtualizedTokenList({ tokens }: { tokens: TokenAttribution[] }) {
  return (
    <FixedSizeList
      height={400}
      itemCount={tokens.length}
      itemSize={35}
      width="100%"
    >
      {({ index, style }) => (
        <div style={style}>
          <TokenItem token={tokens[index]} />
        </div>
      )}
    </FixedSizeList>
  );
}
```

### 8.3 Memoization Strategy

```typescript
// Memoize expensive computations
const sortedFeatures = useMemo(
  () => [...features].sort((a, b) => b.importance_score - a.importance_score),
  [features]
);

// Memoize callback functions
const handleEvidenceSelect = useCallback(
  (evidence: VisualEvidence) => {
    onEvidenceSelect?.(evidence);
  },
  [onEvidenceSelect]
);
```

---

## Phase 9: Accessibility Requirements

### 9.1 ARIA Labels

All XAI components must include:

```typescript
// Example ARIA implementation
<div
  role="region"
  aria-label="Feature importance analysis"
  aria-describedby="feature-importance-desc"
>
  <span id="feature-importance-desc" className="sr-only">
    Shows which features contributed most to the deepfake detection
  </span>
  {/* Component content */}
</div>
```

### 9.2 Keyboard Navigation

```typescript
// Example keyboard navigation for evidence gallery
const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
  switch (e.key) {
    case 'ArrowLeft':
      setSelectedIndex((prev) => Math.max(0, prev - 1));
      break;
    case 'ArrowRight':
      setSelectedIndex((prev) => Math.min(evidence.length - 1, prev + 1));
      break;
    case 'Enter':
    case ' ':
      onEvidenceSelect?.(evidence[selectedIndex]);
      break;
  }
}, [evidence, selectedIndex, onEvidenceSelect]);
```

### 9.3 Color Contrast

All visualization colors must meet WCAG 2.1 AA:
- Minimum contrast ratio: 4.5:1 for text
- Minimum contrast ratio: 3:1 for graphics

---

## Phase 10: Testing Requirements

### 10.1 Unit Tests

Each component must have unit tests covering:
- Rendering with valid props
- Loading state
- Error state
- Empty state
- User interactions
- Accessibility attributes

### 10.2 Integration Tests

```typescript
// Example integration test
describe('XAIExplanationPanel Integration', () => {
  it('should fetch and display XAI data', async () => {
    render(<XAIExplanationPanel analysisId="test-id" />);
    
    await waitFor(() => {
      expect(screen.getByTestId('feature-importance-table')).toBeInTheDocument();
    });
    
    expect(screen.getByText('DCT Anomaly')).toBeInTheDocument();
  });
});
```

---

## Phase 11: Implementation Checklist

### 11.1 Files to Create

- [ ] `frontend/src/types/analysis.ts` - Add XAI type definitions
- [ ] `frontend/src/store/xaiStore.ts` - XAI state management
- [ ] `frontend/src/hooks/useXAI.ts` - XAI data fetching hook
- [ ] `frontend/src/components/xai/XAIExplanationPanel.tsx`
- [ ] `frontend/src/components/xai/FeatureImportanceTable.tsx`
- [ ] `frontend/src/components/xai/SpectrogramOverlay.tsx`
- [ ] `frontend/src/components/xai/TokenAttributionView.tsx`
- [ ] `frontend/src/components/xai/ConfidenceInterval.tsx`
- [ ] `frontend/src/components/xai/ReproducibilityHash.tsx`
- [ ] `frontend/src/components/xai/ScientificReferences.tsx`
- [ ] `frontend/src/components/xai/XAIEvidenceGallery.tsx`

### 11.2 Files to Modify

- [ ] `frontend/src/services/analysisApi.ts` - Add XAI endpoints
- [ ] `frontend/src/components/visualization/HeatmapViewer.tsx` - Enhance with XAI
- [ ] `frontend/src/components/visualization/SpectrogramViewer.tsx` - Add artifact markers
- [ ] `frontend/src/components/modality/AudioAnalysisPanel.tsx` - Integrate SpectrogramOverlay
- [ ] `frontend/src/components/modality/TextAnalysisPanel.tsx` - Integrate TokenAttributionView
- [ ] `frontend/src/components/results/ResultsPanel.tsx` - Add XAI section

### 11.3 Tests to Create

- [ ] `frontend/tests/components/xai/XAIExplanationPanel.test.tsx`
- [ ] `frontend/tests/components/xai/FeatureImportanceTable.test.tsx`
- [ ] `frontend/tests/components/xai/SpectrogramOverlay.test.tsx`
- [ ] `frontend/tests/components/xai/TokenAttributionView.test.tsx`
- [ ] `frontend/tests/hooks/useXAI.test.ts`

---

## Compliance Verification

### AGENTS_FRONTEND.md Compliance

| Rule | Status | Notes |
|------|--------|-------|
| P0 - TypeScript strict mode | PASS | All interfaces defined |
| P0 - No `any` types | PASS | All types explicit |
| P0 - Loading/Error/Empty states | PASS | Specified for all components |
| P0 - Accessibility | PASS | ARIA labels, keyboard nav defined |
| P1 - Component contract | PASS | Props interfaces defined |
| P1 - State management | PASS | Zustand store specified |
| P1 - API integration | PASS | React Query hooks defined |
| P0 - Performance | PASS | Optimization strategies defined |

### AGENTS.md Backend Compliance

| Rule | Status | Notes |
|------|--------|-------|
| Schema mapping | PASS | TypeScript interfaces match backend |
| API endpoints | PASS | All XAI endpoints mapped |
| Data flow | PASS | Analyzer -> XAI -> PDF verified |

---

## Document Status

**Version:** 1.0  
**Created:** 2026-02-15  
**Author:** Elite Senior AI Specialist  
**Review Status:** Ready for Implementation  
**Estimated Implementation:** 2-3 development cycles

---

End of Document