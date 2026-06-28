/**
 * Argus Core - Forensic Report Types
 * ====================================
 * TypeScript interfaces for the ForensicReport component
 * that visualizes the "Reasoning Path" (why it thinks it's a fake).
 *
 * Implements: FORENSIC_VULNERABILITY_REPORT.md - Integration Fix
 *
 * Role: Type-safe contract for multi-pass forensic analysis results.
 * Enables the UI to display artifact-level evidence, verification status,
 * and the complete reasoning path from detection to self-critique.
 */

// ============== ARTIFACT EVIDENCE ==============

/**
 * Single detected forensic artifact with justification.
 */
export interface ArtifactEvidence {
  /** Type of artifact detected (e.g., "dct_high_frequency_anomaly") */
  artifact_type: string;
  /** Location in the media where artifact was found */
  location?: string;
  /** Confidence score [0, 1] */
  confidence: number;
  /** Human-readable description of the artifact */
  description: string;
  /** Additional supporting data (numeric values, coordinates, etc.) */
  supporting_data?: Record<string, unknown>;
}

// ============== PASS RESULTS ==============

/**
 * Results from the initial detection pass (Pass 1).
 */
export interface PassOneResult {
  /** Per-modality manipulation scores [0, 1] */
  modality_scores: Record<string, number>;
  /** Per-modality confidence scores [0, 1] */
  modality_confidences: Record<string, number>;
  /** All detected artifacts before self-critique */
  artifacts: ArtifactEvidence[];
  /** Fused manipulation score across all modalities */
  fused_score: number;
  /** Fused confidence across all modalities */
  fused_confidence: number;
}

/**
 * Results from the self-critique pass (Pass 2).
 */
export interface PassTwoResult {
  /** Artifacts that passed verification */
  verified_artifacts: ArtifactEvidence[];
  /** Artifacts that were retracted (insufficient evidence) */
  retracted_artifacts: ArtifactEvidence[];
  /** Confidence adjustment from verification ratio [-0.1, 0.1] */
  confidence_adjustment: number;
  /** Complete reasoning path as human-readable text */
  reasoning: string;
  /** Final manipulation score after self-critique */
  final_score: number;
  /** Final confidence after self-critique */
  final_confidence: number;
}

// ============== FORENSIC REPORT ==============

/**
 * Complete forensic analysis report with full reasoning path.
 *
 * This is the primary data structure consumed by the ForensicReport
 * component to visualize why the system believes content is manipulated.
 */
export interface ForensicReport {
  /** Unique analysis identifier */
  analysis_id: string;
  /** Initial detection pass results */
  pass_one: PassOneResult;
  /** Self-critique pass results */
  pass_two: PassTwoResult;
  /** SHA-256 hash for chain-of-custody verification */
  reproducibility_hash: string;
  /** Versions of all models used in analysis */
  model_versions: Record<string, string>;
  /** Total processing time in seconds */
  processing_time_seconds: number;
}

// ============== UI COMPONENT PROPS ==============

/**
 * Props for the ForensicReport visualization component.
 */
export interface ForensicReportProps {
  /** The forensic report data to display */
  report: ForensicReport;
  /** Whether to show the detailed reasoning path */
  showReasoning?: boolean;
  /** Whether to show retracted artifacts */
  showRetracted?: boolean;
  /** Callback when user clicks on an artifact */
  onArtifactClick?: (artifact: ArtifactEvidence) => void;
  /** Additional CSS class names */
  className?: string;
}

/**
 * Props for the ArtifactCard sub-component.
 */
export interface ArtifactCardProps {
  /** The artifact to display */
  artifact: ArtifactEvidence;
  /** Verification status */
  status: 'verified' | 'retracted' | 'pending';
  /** Whether the card is expanded */
  isExpanded: boolean;
  /** Toggle expansion callback */
  onToggle: () => void;
}

/**
 * Props for the ReasoningPath sub-component.
 */
export interface ReasoningPathProps {
  /** The reasoning text from Pass 2 */
  reasoning: string;
  /** Number of verified artifacts */
  verifiedCount: number;
  /** Number of retracted artifacts */
  retractedCount: number;
  /** Final confidence score */
  finalConfidence: number;
}

// ============== HELPER FUNCTIONS ==============

/**
 * Get the severity level for an artifact based on confidence.
 */
export function getArtifactSeverity(confidence: number): 'critical' | 'high' | 'medium' | 'low' {
  if (confidence >= 0.8) return 'critical';
  if (confidence >= 0.6) return 'high';
  if (confidence >= 0.4) return 'medium';
  return 'low';
}

/**
 * Get the display color for an artifact severity level.
 */
export function getSeverityColor(severity: 'critical' | 'high' | 'medium' | 'low'): string {
  const colors = {
    critical: 'text-red-500',
    high: 'text-orange-500',
    medium: 'text-yellow-500',
    low: 'text-green-500',
  };
  return colors[severity];
}

/**
 * Format the reasoning path into displayable steps.
 */
export function formatReasoningSteps(reasoning: string): string[] {
  if (!reasoning) return [];
  return reasoning.split('; ').filter((step) => step.length > 0);
}

/**
 * Compute the verification rate from a forensic report.
 */
export function computeVerificationRate(report: ForensicReport): number {
  const total = report.pass_one.artifacts.length;
  if (total === 0) return 0;
  return report.pass_two.verified_artifacts.length / total;
}
