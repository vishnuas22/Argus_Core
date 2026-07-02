/**
 * Argus Core - XAI Components Index
 * ==================================
 * Exports all XAI (Explainable AI) components.
 * 
 * Components:
 * - XAIExplanationPanel: Main container for XAI visualizations
 * - FeatureImportanceTable: Feature importance scores table
 * - ConfidenceInterval: Statistical confidence interval display
 * - ReproducibilityHash: Cryptographic hash for reproducibility
 * - ScientificReferences: Scientific citations display
 * - XAIEvidenceGallery: Visual evidence gallery
 */

// Main panel
export { XAIExplanationPanel, XAIExplanationPanelSkeleton } from './XAIExplanationPanel';

// Feature importance
export { FeatureImportanceTable, FeatureImportanceSkeleton } from './FeatureImportanceTable';

// Confidence interval
export { ConfidenceInterval, ConfidenceIntervalSkeleton } from './ConfidenceInterval';

// Reproducibility hash
export { ReproducibilityHash, ReproducibilityHashSkeleton } from './ReproducibilityHash';

// Scientific references
export { ScientificReferences, ScientificReferencesSkeleton } from './ScientificReferences';

// Evidence gallery
export { XAIEvidenceGallery, XAIEvidenceGallerySkeleton } from './XAIEvidenceGallery';

// Iteration 5: XAI attribution panel (Eigen-CAM heatmap + conformal badge)
export { XAIAttributionPanel } from './XAIAttributionPanel';

// Types
export type { FeatureImportanceTableProps } from './FeatureImportanceTable';
export type { ConfidenceIntervalProps } from './ConfidenceInterval';
export type { ReproducibilityHashProps } from './ReproducibilityHash';
export type { ScientificReferencesProps } from './ScientificReferences';
export type { XAIEvidenceGalleryProps } from './XAIEvidenceGallery';
export type { XAIExplanationPanelProps } from './XAIExplanationPanel';
