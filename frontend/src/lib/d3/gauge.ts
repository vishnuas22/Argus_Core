/**
 * Argus Core - D3 Gauge Chart Factory
 * ====================================
 * D3.js radial gauge chart for displaying Trust Score (0-100).
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - lib/d3/gauge.ts
 * 
 * Role: Factory function to create animated radial gauge using D3.js.
 * Renders SVG arc with color-coded score display and smooth animations.
 * 
 * Integration:
 * - Imports: d3
 * - Used by: components/results/TrustScoreGauge.tsx
 * - Inputs: Container SVGElement, score, config options
 * - Outputs: Renders gauge visualization into container
 * 
 * Color Scale (matching verdict thresholds from backend config.py):
 * - 80-100: Green (#22c55e) - Authentic
 * - 60-79: Lime (#84cc16) - Likely Authentic
 * - 40-59: Yellow (#eab308) - Uncertain
 * - 20-39: Orange (#f97316) - Likely Fake
 * - 0-19: Red (#ef4444) - Fake
 */

import * as d3 from 'd3';
import type { Verdict } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Configuration options for gauge chart
 */
export interface GaugeConfig {
  /** Width of the gauge container */
  width: number;
  /** Height of the gauge container */
  height: number;
  /** Inner radius of the arc (creates donut effect) */
  innerRadius: number;
  /** Outer radius of the arc */
  outerRadius: number;
  /** Start angle in radians (default: -π/2 = top) */
  startAngle?: number;
  /** End angle in radians (default: π/2 = bottom) */
  endAngle?: number;
  /** Animation duration in ms */
  animationDuration?: number;
  /** Whether to animate the gauge */
  animated?: boolean;
  /** Corner radius for arc ends */
  cornerRadius?: number;
  /** Background arc color */
  backgroundColor?: string;
  /** Custom color override (uses color scale if not provided) */
  color?: string;
  /** Padding between arc and container */
  padding?: number;
}

/**
 * Gauge chart instance returned by createGauge
 */
export interface GaugeInstance {
  /** Update the gauge score with animation */
  update: (newScore: number, newVerdict?: Verdict) => void;
  /** Destroy the gauge and clean up */
  destroy: () => void;
  /** Get current score */
  getScore: () => number;
}

// ============== CONSTANTS ==============

/**
 * Default gauge configuration
 */
export const DEFAULT_GAUGE_CONFIG: Required<GaugeConfig> = {
  width: 200,
  height: 200,
  innerRadius: 60,
  outerRadius: 85,
  startAngle: -Math.PI * 0.75,  // -135 degrees
  endAngle: Math.PI * 0.75,      // 135 degrees
  animationDuration: 1000,
  animated: true,
  cornerRadius: 4,
  backgroundColor: 'hsl(var(--muted))',
  color: '',
  padding: 10,
};

/**
 * Color scale based on verdict thresholds
 * Matches VERDICT_THRESHOLD_* values from backend config.py
 */
export const SCORE_COLOR_SCALE = d3.scaleThreshold<number, string>()
  .domain([20, 40, 60, 80])
  .range([
    '#ef4444',  // 0-19: Red (Fake)
    '#f97316',  // 20-39: Orange (Likely Fake)
    '#eab308',  // 40-59: Yellow (Uncertain)
    '#84cc16',  // 60-79: Lime (Likely Authentic)
    '#22c55e',  // 80-100: Green (Authentic)
  ]);

/**
 * Verdict to color mapping
 */
export const VERDICT_COLORS: Record<Verdict, string> = {
  authentic: '#22c55e',
  likely_authentic: '#84cc16',
  uncertain: '#eab308',
  likely_fake: '#f97316',
  fake: '#ef4444',
};

// ============== MAIN FUNCTION ==============

/**
 * Create a D3 radial gauge chart
 * 
 * @param container - SVG element to render into
 * @param score - Initial score (0-100)
 * @param config - Configuration options
 * @returns Gauge instance with update and destroy methods
 * 
 * @example
 * ```tsx
 * const svgRef = useRef<SVGSVGElement>(null);
 * const gaugeRef = useRef<GaugeInstance | null>(null);
 * 
 * useEffect(() => {
 *   if (svgRef.current) {
 *     gaugeRef.current = createGauge(svgRef.current, score, {
 *       width: 200,
 *       height: 200,
 *       animated: true
 *     });
 *   }
 *   return () => gaugeRef.current?.destroy();
 * }, []);
 * 
 * // Update score
 * useEffect(() => {
 *   gaugeRef.current?.update(newScore);
 * }, [newScore]);
 * ```
 */
export function createGauge(
  container: SVGSVGElement,
  score: number,
  config: Partial<GaugeConfig> = {}
): GaugeInstance {
  // Merge with defaults
  const cfg: Required<GaugeConfig> = { ...DEFAULT_GAUGE_CONFIG, ...config };
  
  // Clamp score to 0-100
  let currentScore = Math.max(0, Math.min(100, score));
  
  // Calculate center
  const centerX = cfg.width / 2;
  const centerY = cfg.height / 2;
  
  // Calculate angle range
  const angleRange = cfg.endAngle - cfg.startAngle;
  
  // Get color based on score
  const getColor = (s: number) => cfg.color || SCORE_COLOR_SCALE(s);
  
  // Create D3 selection
  const svg = d3.select(container)
    .attr('width', cfg.width)
    .attr('height', cfg.height)
    .attr('viewBox', `0 0 ${cfg.width} ${cfg.height}`);
  
  // Clear any existing content
  svg.selectAll('*').remove();
  
  // Create main group centered in container
  const g = svg.append('g')
    .attr('transform', `translate(${centerX}, ${centerY})`)
    .attr('class', 'gauge-group');
  
  // Create arc generator for background
  const backgroundArc = d3.arc<void>()
    .innerRadius(cfg.innerRadius)
    .outerRadius(cfg.outerRadius)
    .startAngle(cfg.startAngle)
    .endAngle(cfg.endAngle)
    .cornerRadius(cfg.cornerRadius);
  
  // Draw background arc
  g.append('path')
    .attr('class', 'gauge-background')
    .attr('d', backgroundArc())
    .attr('fill', cfg.backgroundColor)
    .attr('opacity', 0.3);
  
  // Create arc generator for score
  const scoreArcGenerator = d3.arc<{ score: number }>()
    .innerRadius(cfg.innerRadius)
    .outerRadius(cfg.outerRadius)
    .startAngle(cfg.startAngle)
    .endAngle(d => cfg.startAngle + (d.score / 100) * angleRange)
    .cornerRadius(cfg.cornerRadius);
  
  // Draw score arc
  const scoreArc = g.append('path')
    .attr('class', 'gauge-score')
    .datum({ score: cfg.animated ? 0 : currentScore })
    .attr('d', scoreArcGenerator)
    .attr('fill', getColor(currentScore));
  
  // Animate if enabled
  if (cfg.animated) {
    scoreArc
      .transition()
      .duration(cfg.animationDuration)
      .ease(d3.easeElasticOut.amplitude(1).period(0.4))
      .attrTween('d', function() {
        const interpolate = d3.interpolate(0, currentScore);
        return function(t: number) {
          return scoreArcGenerator({ score: interpolate(t) }) || '';
        };
      });
  }
  
  // Add gradient definition for enhanced visual
  const defs = svg.append('defs');
  
  // Glow filter for score arc
  const filter = defs.append('filter')
    .attr('id', 'gauge-glow')
    .attr('x', '-50%')
    .attr('y', '-50%')
    .attr('width', '200%')
    .attr('height', '200%');
  
  filter.append('feGaussianBlur')
    .attr('stdDeviation', '3')
    .attr('result', 'coloredBlur');
  
  const feMerge = filter.append('feMerge');
  feMerge.append('feMergeNode').attr('in', 'coloredBlur');
  feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
  
  // Apply glow to score arc
  scoreArc.attr('filter', 'url(#gauge-glow)');
  
  // Add tick marks
  const tickCount = 5;  // 0, 25, 50, 75, 100
  const ticks = g.append('g')
    .attr('class', 'gauge-ticks')
    .selectAll('.tick')
    .data(d3.range(tickCount + 1))
    .enter()
    .append('g')
    .attr('class', 'tick')
    .attr('transform', (d) => {
      const angle = cfg.startAngle + (d / tickCount) * angleRange;
      const x = Math.sin(angle) * (cfg.outerRadius + 8);
      const y = -Math.cos(angle) * (cfg.outerRadius + 8);
      return `translate(${x}, ${y})`;
    });
  
  ticks.append('circle')
    .attr('r', 2)
    .attr('fill', 'currentColor')
    .attr('opacity', 0.3);
  
  // Add score indicator needle/dot
  const indicatorRadius = (cfg.innerRadius + cfg.outerRadius) / 2;
  const indicator = g.append('circle')
    .attr('class', 'gauge-indicator')
    .attr('r', 6)
    .attr('fill', getColor(currentScore))
    .attr('stroke', 'white')
    .attr('stroke-width', 2)
    .attr('filter', 'url(#gauge-glow)');
  
  // Position indicator at score
  const positionIndicator = (s: number, animate = false) => {
    const angle = cfg.startAngle + (s / 100) * angleRange;
    const x = Math.sin(angle) * indicatorRadius;
    const y = -Math.cos(angle) * indicatorRadius;
    
    if (animate) {
      indicator
        .transition()
        .duration(cfg.animationDuration)
        .ease(d3.easeElasticOut.amplitude(1).period(0.4))
        .attr('cx', x)
        .attr('cy', y)
        .attr('fill', getColor(s));
    } else {
      indicator
        .attr('cx', x)
        .attr('cy', y);
    }
  };
  
  // Set initial position
  if (cfg.animated) {
    indicator.attr('cx', Math.sin(cfg.startAngle) * indicatorRadius)
             .attr('cy', -Math.cos(cfg.startAngle) * indicatorRadius);
    
    setTimeout(() => positionIndicator(currentScore, true), 50);
  } else {
    positionIndicator(currentScore, false);
  }
  
  // ============== INSTANCE METHODS ==============
  
  /**
   * Update gauge to new score
   */
  const update = (newScore: number, newVerdict?: Verdict) => {
    const clampedScore = Math.max(0, Math.min(100, newScore));
    const previousScore = currentScore;
    currentScore = clampedScore;
    
    const color = newVerdict ? VERDICT_COLORS[newVerdict] : getColor(clampedScore);
    
    // Animate arc
    scoreArc
      .transition()
      .duration(cfg.animationDuration)
      .ease(d3.easeElasticOut.amplitude(1).period(0.4))
      .attrTween('d', function() {
        const interpolate = d3.interpolate(previousScore, clampedScore);
        return function(t: number) {
          return scoreArcGenerator({ score: interpolate(t) }) || '';
        };
      })
      .attr('fill', color);
    
    // Update indicator
    positionIndicator(clampedScore, true);
    indicator.transition()
      .duration(cfg.animationDuration)
      .attr('fill', color);
  };
  
  /**
   * Destroy gauge and clean up
   */
  const destroy = () => {
    svg.selectAll('*').remove();
  };
  
  /**
   * Get current score
   */
  const getScore = () => currentScore;
  
  return { update, destroy, getScore };
}

// ============== UTILITY FUNCTIONS ==============

/**
 * Get color for a given score
 */
export function getScoreColor(score: number): string {
  return SCORE_COLOR_SCALE(Math.max(0, Math.min(100, score)));
}

/**
 * Get color for a given verdict
 */
export function getVerdictColor(verdict: Verdict): string {
  return VERDICT_COLORS[verdict];
}

/**
 * Calculate arc path for given score
 * Useful for static SVG rendering without full D3 setup
 */
export function calculateArcPath(
  score: number,
  config: Partial<Pick<GaugeConfig, 'innerRadius' | 'outerRadius' | 'startAngle' | 'endAngle' | 'cornerRadius'>> = {}
): string {
  const cfg = {
    innerRadius: 60,
    outerRadius: 85,
    startAngle: -Math.PI * 0.75,
    endAngle: Math.PI * 0.75,
    cornerRadius: 4,
    ...config,
  };
  
  const angleRange = cfg.endAngle - cfg.startAngle;
  const clampedScore = Math.max(0, Math.min(100, score));
  
  const arc = d3.arc<void>()
    .innerRadius(cfg.innerRadius)
    .outerRadius(cfg.outerRadius)
    .startAngle(cfg.startAngle)
    .endAngle(cfg.startAngle + (clampedScore / 100) * angleRange)
    .cornerRadius(cfg.cornerRadius);
  
  return arc() || '';
}

export default createGauge;
