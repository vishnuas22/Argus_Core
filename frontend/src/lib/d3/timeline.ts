/**
 * Argus Core - D3 Timeline Chart Factory
 * =======================================
 * D3.js timeline chart for displaying temporal analysis with anomaly markers.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - lib/d3/timeline.ts
 * 
 * Role: Factory function to create interactive timeline visualization using D3.js.
 * Shows per-frame scores over time with anomaly markers and click-to-seek.
 * 
 * Integration:
 * - Imports: d3
 * - Used by: components/visualization/TimelineChart.tsx
 * - Inputs: Container SVGElement, scores array, anomaly indices, config
 * - Outputs: Renders timeline visualization into container
 * 
 * Features:
 * - Area chart showing score progression
 * - Anomaly markers with hover tooltips
 * - Threshold line at 50%
 * - Click to select frame
 * - Responsive design
 * - Smooth animations
 */

import * as d3 from 'd3';

// ============== TYPES ==============

/**
 * Configuration options for timeline chart
 */
export interface TimelineConfig {
  /** Width of the chart container */
  width: number;
  /** Height of the chart container */
  height: number;
  /** Left margin for axis */
  marginLeft?: number;
  /** Right margin */
  marginRight?: number;
  /** Top margin */
  marginTop?: number;
  /** Bottom margin for axis */
  marginBottom?: number;
  /** Animation duration in ms */
  animationDuration?: number;
  /** Whether to animate the chart */
  animated?: boolean;
  /** Frames per second for time axis */
  fps?: number;
  /** Show threshold line */
  showThreshold?: boolean;
  /** Threshold value (0-1) */
  threshold?: number;
  /** Color for normal scores */
  normalColor?: string;
  /** Color for low scores (below threshold) */
  dangerColor?: string;
  /** Color for anomaly markers */
  anomalyColor?: string;
  /** Line stroke width */
  strokeWidth?: number;
  /** Show area fill */
  showArea?: boolean;
  /** Background color */
  backgroundColor?: string;
}

/**
 * Data for timeline chart
 */
export interface TimelineData {
  /** Array of scores (0-1) per frame */
  scores: number[];
  /** Array of frame indices marked as anomalies */
  anomalyIndices: number[];
  /** Optional timestamps in ms for each frame */
  timestamps?: number[];
}

/**
 * Timeline chart instance
 */
export interface TimelineInstance {
  /** Update chart data */
  update: (data: TimelineData) => void;
  /** Set selected frame */
  setSelected: (index: number | null) => void;
  /** Resize chart */
  resize: (width: number, height: number) => void;
  /** Destroy chart and clean up */
  destroy: () => void;
  /** Get current data */
  getData: () => TimelineData;
}

// ============== CONSTANTS ==============

/**
 * Default timeline configuration
 */
export const DEFAULT_TIMELINE_CONFIG: Required<TimelineConfig> = {
  width: 600,
  height: 200,
  marginLeft: 40,
  marginRight: 20,
  marginTop: 20,
  marginBottom: 30,
  animationDuration: 750,
  animated: true,
  fps: 30,
  showThreshold: true,
  threshold: 0.5,
  normalColor: '#22c55e',
  dangerColor: '#ef4444',
  anomalyColor: '#f97316',
  strokeWidth: 2,
  showArea: true,
  backgroundColor: 'transparent',
};

// ============== MAIN FUNCTION ==============

/**
 * Create a D3 timeline chart
 * 
 * @param container - SVG element to render into
 * @param data - Timeline data with scores and anomalies
 * @param config - Configuration options
 * @param onFrameSelect - Callback when frame is clicked
 * @returns Timeline instance with update and destroy methods
 * 
 * @example
 * ```tsx
 * const svgRef = useRef<SVGSVGElement>(null);
 * const timelineRef = useRef<TimelineInstance | null>(null);
 * 
 * useEffect(() => {
 *   if (svgRef.current) {
 *     timelineRef.current = createTimelineChart(
 *       svgRef.current,
 *       { scores: [0.9, 0.8, 0.3, ...], anomalyIndices: [2, 5, 8] },
 *       { width: 600, height: 200 },
 *       (index) => console.log('Selected frame:', index)
 *     );
 *   }
 *   return () => timelineRef.current?.destroy();
 * }, []);
 * ```
 */
export function createTimelineChart(
  container: SVGSVGElement,
  data: TimelineData,
  config: Partial<TimelineConfig> = {},
  onFrameSelect?: (index: number) => void
): TimelineInstance {
  // Merge with defaults
  const cfg: Required<TimelineConfig> = { ...DEFAULT_TIMELINE_CONFIG, ...config };
  
  // Current data and state
  let currentData = { ...data };
  let selectedIndex: number | null = null;
  
  // Calculate chart dimensions
  const chartWidth = cfg.width - cfg.marginLeft - cfg.marginRight;
  const chartHeight = cfg.height - cfg.marginTop - cfg.marginBottom;
  
  // Create D3 selection
  const svg = d3.select(container)
    .attr('width', cfg.width)
    .attr('height', cfg.height)
    .attr('viewBox', `0 0 ${cfg.width} ${cfg.height}`)
    .style('background', cfg.backgroundColor);
  
  // Clear any existing content
  svg.selectAll('*').remove();
  
  // Add definitions for gradients and filters
  const defs = svg.append('defs');
  
  // Gradient for area fill
  const gradient = defs.append('linearGradient')
    .attr('id', 'timeline-area-gradient')
    .attr('x1', '0%')
    .attr('y1', '0%')
    .attr('x2', '0%')
    .attr('y2', '100%');
  
  gradient.append('stop')
    .attr('offset', '0%')
    .attr('stop-color', cfg.normalColor)
    .attr('stop-opacity', 0.4);
  
  gradient.append('stop')
    .attr('offset', '100%')
    .attr('stop-color', cfg.normalColor)
    .attr('stop-opacity', 0.05);
  
  // Glow filter
  const filter = defs.append('filter')
    .attr('id', 'timeline-glow')
    .attr('x', '-50%')
    .attr('y', '-50%')
    .attr('width', '200%')
    .attr('height', '200%');
  
  filter.append('feGaussianBlur')
    .attr('stdDeviation', '2')
    .attr('result', 'coloredBlur');
  
  const feMerge = filter.append('feMerge');
  feMerge.append('feMergeNode').attr('in', 'coloredBlur');
  feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
  
  // Create main group with margins
  const g = svg.append('g')
    .attr('transform', `translate(${cfg.marginLeft}, ${cfg.marginTop})`)
    .attr('class', 'timeline-group');
  
  // Create scales
  const x = d3.scaleLinear()
    .domain([0, Math.max(1, currentData.scores.length - 1)])
    .range([0, chartWidth]);
  
  const y = d3.scaleLinear()
    .domain([0, 1])
    .range([chartHeight, 0]);
  
  // Create line generator
  const line = d3.line<number>()
    .x((_, i) => x(i))
    .y(d => y(d))
    .curve(d3.curveMonotoneX);
  
  // Create area generator
  const area = d3.area<number>()
    .x((_, i) => x(i))
    .y0(chartHeight)
    .y1(d => y(d))
    .curve(d3.curveMonotoneX);
  
  // Draw clip path for chart area
  svg.append('defs')
    .append('clipPath')
    .attr('id', 'timeline-clip')
    .append('rect')
    .attr('width', chartWidth)
    .attr('height', chartHeight);
  
  // Add X axis
  const xAxisGroup = g.append('g')
    .attr('class', 'x-axis')
    .attr('transform', `translate(0, ${chartHeight})`)
    .call(d3.axisBottom(x)
      .ticks(Math.min(10, currentData.scores.length))
      .tickFormat((d) => {
        const frameNum = d as number;
        if (currentData.timestamps && currentData.timestamps[frameNum]) {
          return `${(currentData.timestamps[frameNum] / 1000).toFixed(1)}s`;
        }
        return `${frameNum}`;
      })
    );
  
  xAxisGroup.selectAll('text')
    .attr('fill', 'currentColor')
    .attr('opacity', 0.6)
    .style('font-size', '10px');
  
  xAxisGroup.selectAll('line, path')
    .attr('stroke', 'currentColor')
    .attr('opacity', 0.2);
  
  // Add Y axis
  const yAxisGroup = g.append('g')
    .attr('class', 'y-axis')
    .call(d3.axisLeft(y)
      .ticks(5)
      .tickFormat(d => `${(d as number) * 100}%`)
    );
  
  yAxisGroup.selectAll('text')
    .attr('fill', 'currentColor')
    .attr('opacity', 0.6)
    .style('font-size', '10px');
  
  yAxisGroup.selectAll('line, path')
    .attr('stroke', 'currentColor')
    .attr('opacity', 0.2);
  
  // Add threshold line
  const thresholdLine = g.append('line')
    .attr('class', 'threshold-line')
    .attr('x1', 0)
    .attr('x2', chartWidth)
    .attr('y1', y(cfg.threshold))
    .attr('y2', y(cfg.threshold))
    .attr('stroke', cfg.dangerColor)
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '4,4')
    .attr('opacity', cfg.showThreshold ? 0.5 : 0);
  
  // Add threshold label
  g.append('text')
    .attr('class', 'threshold-label')
    .attr('x', chartWidth - 4)
    .attr('y', y(cfg.threshold) - 4)
    .attr('text-anchor', 'end')
    .attr('fill', cfg.dangerColor)
    .attr('font-size', '10px')
    .attr('opacity', cfg.showThreshold ? 0.7 : 0)
    .text('Threshold');
  
  // Add area fill
  const areaPath = g.append('path')
    .datum(currentData.scores)
    .attr('class', 'timeline-area')
    .attr('fill', 'url(#timeline-area-gradient)')
    .attr('d', area)
    .attr('opacity', cfg.showArea ? 1 : 0);
  
  // Add score line
  const linePath = g.append('path')
    .datum(currentData.scores)
    .attr('class', 'timeline-line')
    .attr('fill', 'none')
    .attr('stroke', cfg.normalColor)
    .attr('stroke-width', cfg.strokeWidth)
    .attr('d', line)
    .attr('filter', 'url(#timeline-glow)');
  
  // Initial animation
  if (cfg.animated) {
    const totalLength = linePath.node()?.getTotalLength() || 0;
    
    linePath
      .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
      .attr('stroke-dashoffset', totalLength)
      .transition()
      .duration(cfg.animationDuration)
      .ease(d3.easeQuadOut)
      .attr('stroke-dashoffset', 0);
    
    areaPath
      .attr('opacity', 0)
      .transition()
      .delay(cfg.animationDuration / 2)
      .duration(cfg.animationDuration / 2)
      .attr('opacity', cfg.showArea ? 1 : 0);
  }
  
  // Add anomaly markers
  const anomalyMarkers = g.append('g')
    .attr('class', 'anomaly-markers');
  
  const drawAnomalies = () => {
    anomalyMarkers.selectAll('.anomaly-marker').remove();
    
    const markers = anomalyMarkers.selectAll('.anomaly-marker')
      .data(currentData.anomalyIndices.filter(i => i < currentData.scores.length))
      .enter()
      .append('g')
      .attr('class', 'anomaly-marker')
      .attr('transform', d => `translate(${x(d)}, ${y(currentData.scores[d])})`)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        setSelected(d);
        onFrameSelect?.(d);
      });
    
    // Marker circle
    markers.append('circle')
      .attr('r', 0)
      .attr('fill', cfg.anomalyColor)
      .attr('stroke', 'white')
      .attr('stroke-width', 2)
      .attr('filter', 'url(#timeline-glow)')
      .transition()
      .delay((_, i) => cfg.animated ? cfg.animationDuration + i * 50 : 0)
      .duration(300)
      .ease(d3.easeElasticOut)
      .attr('r', 6);
    
    // Tooltip on hover
    markers
      .append('title')
      .text(d => `Frame ${d}\nScore: ${(currentData.scores[d] * 100).toFixed(1)}%`);
  };
  
  drawAnomalies();
  
  // Add interactive overlay for frame selection
  const overlay = g.append('rect')
    .attr('class', 'interaction-overlay')
    .attr('width', chartWidth)
    .attr('height', chartHeight)
    .attr('fill', 'transparent')
    .style('cursor', 'crosshair');
  
  // Selection indicator
  const selectionLine = g.append('line')
    .attr('class', 'selection-line')
    .attr('y1', 0)
    .attr('y2', chartHeight)
    .attr('stroke', 'currentColor')
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '4,2')
    .attr('opacity', 0);
  
  const selectionDot = g.append('circle')
    .attr('class', 'selection-dot')
    .attr('r', 4)
    .attr('fill', 'currentColor')
    .attr('stroke', 'white')
    .attr('stroke-width', 2)
    .attr('opacity', 0);
  
  // Handle click on overlay
  overlay.on('click', (event) => {
    const [mouseX] = d3.pointer(event);
    const frameIndex = Math.round(x.invert(mouseX));
    const clampedIndex = Math.max(0, Math.min(currentData.scores.length - 1, frameIndex));
    setSelected(clampedIndex);
    onFrameSelect?.(clampedIndex);
  });
  
  // Handle hover
  overlay.on('mousemove', (event) => {
    const [mouseX] = d3.pointer(event);
    const frameIndex = Math.round(x.invert(mouseX));
    const clampedIndex = Math.max(0, Math.min(currentData.scores.length - 1, frameIndex));
    
    if (clampedIndex >= 0 && clampedIndex < currentData.scores.length) {
      const score = currentData.scores[clampedIndex];
      selectionLine
        .attr('x1', x(clampedIndex))
        .attr('x2', x(clampedIndex))
        .attr('opacity', 0.3);
    }
  });
  
  overlay.on('mouseleave', () => {
    if (selectedIndex === null) {
      selectionLine.attr('opacity', 0);
    }
  });
  
  // ============== INSTANCE METHODS ==============
  
  /**
   * Set selected frame
   */
  const setSelected = (index: number | null) => {
    selectedIndex = index;
    
    if (index !== null && index >= 0 && index < currentData.scores.length) {
      const score = currentData.scores[index];
      
      selectionLine
        .transition()
        .duration(200)
        .attr('x1', x(index))
        .attr('x2', x(index))
        .attr('opacity', 0.6);
      
      selectionDot
        .transition()
        .duration(200)
        .attr('cx', x(index))
        .attr('cy', y(score))
        .attr('opacity', 1);
    } else {
      selectionLine.transition().duration(200).attr('opacity', 0);
      selectionDot.transition().duration(200).attr('opacity', 0);
    }
  };
  
  /**
   * Update chart with new data
   */
  const update = (newData: TimelineData) => {
    currentData = { ...newData };
    
    // Update scale domain
    x.domain([0, Math.max(1, currentData.scores.length - 1)]);
    
    // Update line
    linePath
      .datum(currentData.scores)
      .transition()
      .duration(cfg.animationDuration)
      .attr('d', line);
    
    // Update area
    areaPath
      .datum(currentData.scores)
      .transition()
      .duration(cfg.animationDuration)
      .attr('d', area);
    
    // Update x axis
    xAxisGroup
      .transition()
      .duration(cfg.animationDuration)
      .call(d3.axisBottom(x)
        .ticks(Math.min(10, currentData.scores.length))
        .tickFormat((d) => {
          const frameNum = d as number;
          if (currentData.timestamps && currentData.timestamps[frameNum]) {
            return `${(currentData.timestamps[frameNum] / 1000).toFixed(1)}s`;
          }
          return `${frameNum}`;
        })
      );
    
    // Redraw anomalies
    drawAnomalies();
    
    // Update selection if still valid
    if (selectedIndex !== null && selectedIndex >= currentData.scores.length) {
      setSelected(null);
    } else if (selectedIndex !== null) {
      setSelected(selectedIndex);
    }
  };
  
  /**
   * Resize chart
   */
  const resize = (width: number, height: number) => {
    cfg.width = width;
    cfg.height = height;
    
    const newChartWidth = width - cfg.marginLeft - cfg.marginRight;
    const newChartHeight = height - cfg.marginTop - cfg.marginBottom;
    
    svg.attr('width', width).attr('height', height)
       .attr('viewBox', `0 0 ${width} ${height}`);
    
    x.range([0, newChartWidth]);
    y.range([newChartHeight, 0]);
    
    // Update elements
    xAxisGroup.attr('transform', `translate(0, ${newChartHeight})`);
    thresholdLine.attr('x2', newChartWidth).attr('y1', y(cfg.threshold)).attr('y2', y(cfg.threshold));
    overlay.attr('width', newChartWidth).attr('height', newChartHeight);
    selectionLine.attr('y2', newChartHeight);
    
    update(currentData);
  };
  
  /**
   * Destroy chart and clean up
   */
  const destroy = () => {
    svg.selectAll('*').remove();
  };
  
  /**
   * Get current data
   */
  const getData = () => ({ ...currentData });
  
  return { update, setSelected, resize, destroy, getData };
}

// ============== UTILITY FUNCTIONS ==============

/**
 * Generate sample timeline data for testing
 */
export function generateSampleTimelineData(
  frameCount: number = 100,
  anomalyCount: number = 5
): TimelineData {
  const scores: number[] = [];
  const anomalyIndices: number[] = [];
  const timestamps: number[] = [];
  
  // Generate scores with some variation
  for (let i = 0; i < frameCount; i++) {
    const base = 0.7 + Math.random() * 0.2;
    const noise = Math.sin(i / 10) * 0.1;
    scores.push(Math.max(0, Math.min(1, base + noise)));
    timestamps.push(i * (1000 / 30)); // Assume 30fps
  }
  
  // Add anomalies with low scores
  const step = Math.floor(frameCount / (anomalyCount + 1));
  for (let i = 1; i <= anomalyCount; i++) {
    const index = step * i + Math.floor(Math.random() * step * 0.5);
    if (index < frameCount) {
      anomalyIndices.push(index);
      scores[index] = 0.2 + Math.random() * 0.2;
    }
  }
  
  return { scores, anomalyIndices, timestamps };
}

export default createTimelineChart;
