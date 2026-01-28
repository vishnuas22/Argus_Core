/**
 * Argus Core - Analysis Form Component
 * =====================================
 * Analysis options form with submit action.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/analysis/AnalysisForm.tsx
 * 
 * Role: Collect analysis options (report generation, heatmaps, defense level).
 * Submit analysis request and handle loading/error states.
 * 
 * Integration:
 * - Imports: hooks/useAnalysis, store/uploadStore, components/ui/*
 * - Inputs: File from uploadStore, form options
 * - Outputs: Triggers analysis submission, navigates on success
 * - Backend: POST /api/v1/analyze
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Disable form, show spinner on submit
 * - Error state: Display server errors
 * - Accessibility: Form labels, error announcements
 * - data-testid: analysis-form, analysis-form-submit, analysis-form-error
 */

'use client';

import { useState, useCallback } from 'react';
import { Loader2, FileSearch, Shield, FileText, Image, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { useAnalysis, DEFAULT_ANALYSIS_OPTIONS } from '@/hooks/useAnalysis';
import { useUploadStore } from '@/store/uploadStore';
import { FileCard } from '@/components/upload/FileCard';
import { ConnectedUploadProgress } from '@/components/upload/UploadProgress';
import type { AnalysisOptions, DefenseLevel } from '@/types/analysis';

// ============== TYPES ==============

export interface AnalysisFormProps {
  /** Optional callback after successful submission */
  onSubmitSuccess?: (analysisId: string) => void;
  /** Optional callback on submission error */
  onSubmitError?: (error: Error) => void;
  /** Additional CSS classes */
  className?: string;
}

// ============== DEFENSE LEVEL CONFIG ==============

interface DefenseLevelOption {
  value: DefenseLevel;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

const DEFENSE_LEVELS: DefenseLevelOption[] = [
  {
    value: 'none',
    label: 'None',
    description: 'Fastest processing, no adversarial defense',
    icon: Shield,
  },
  {
    value: 'standard',
    label: 'Standard',
    description: 'Recommended - balances speed and accuracy',
    icon: Shield,
  },
  {
    value: 'aggressive',
    label: 'Aggressive',
    description: 'Maximum protection against adversarial attacks',
    icon: Shield,
  },
];

// ============== COMPONENT ==============

export function AnalysisForm({
  onSubmitSuccess,
  onSubmitError,
  className,
}: AnalysisFormProps) {
  // Store state
  const { file, preview, fileInfo, isValid, clearFile, status, error } = useUploadStore();
  
  // Analysis mutation
  const { submitAnalysis, isSubmitting } = useAnalysis();

  // Form state
  const [options, setOptions] = useState<AnalysisOptions>(DEFAULT_ANALYSIS_OPTIONS);

  // Derived state
  const isUploading = status === 'uploading' || status === 'processing';
  const canSubmit = file && isValid && !isSubmitting && !isUploading;

  /**
   * Handle form submission
   */
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file || !canSubmit) return;

    try {
      const result = await submitAnalysis.mutateAsync({
        file,
        options,
      });
      
      onSubmitSuccess?.(result.analysis_id);
    } catch (err) {
      onSubmitError?.(err instanceof Error ? err : new Error('Submission failed'));
    }
  }, [file, canSubmit, options, submitAnalysis, onSubmitSuccess, onSubmitError]);

  /**
   * Update option
   */
  const updateOption = useCallback(<K extends keyof AnalysisOptions>(
    key: K,
    value: AnalysisOptions[K]
  ) => {
    setOptions(prev => ({ ...prev, [key]: value }));
  }, []);

  // ============== NO FILE STATE ==============

  if (!file) {
    return (
      <Card 
        className={cn('border-dashed', className)}
        data-testid="analysis-form-empty"
      >
        <CardContent className="flex flex-col items-center justify-center py-12 text-center">
          <FileSearch className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="font-medium text-lg">No file selected</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Upload a file to configure analysis options
          </p>
        </CardContent>
      </Card>
    );
  }

  // ============== FORM ==============

  return (
    <form onSubmit={handleSubmit} data-testid="analysis-form">
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-lg">Analysis Options</CardTitle>
          <CardDescription>
            Configure how the file should be analyzed for deepfake detection
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* File preview */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Selected File</Label>
            <FileCard
              file={file}
              preview={preview}
              fileInfo={fileInfo}
              onRemove={clearFile}
              uploadProgress={isUploading ? useUploadStore.getState().uploadProgress : undefined}
              error={status === 'error' ? error || undefined : undefined}
              disabled={isSubmitting || isUploading}
            />
          </div>

          {/* Upload progress */}
          {(isUploading || status === 'error') && (
            <ConnectedUploadProgress />
          )}

          {/* Report generation toggle */}
          <div className="flex items-center justify-between rounded-lg border p-4">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-muted-foreground" />
                <Label 
                  htmlFor="generate-report" 
                  className="font-medium cursor-pointer"
                >
                  Generate PDF Report
                </Label>
              </div>
              <p className="text-xs text-muted-foreground">
                Create a detailed forensic report with findings and evidence
              </p>
            </div>
            <Switch
              id="generate-report"
              checked={options.generateReport}
              onCheckedChange={(checked) => updateOption('generateReport', checked)}
              disabled={isSubmitting || isUploading}
              aria-describedby="generate-report-desc"
            />
          </div>

          {/* Heatmap generation toggle */}
          <div className="flex items-center justify-between rounded-lg border p-4">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2">
                <Image className="h-4 w-4 text-muted-foreground" />
                <Label 
                  htmlFor="generate-heatmaps" 
                  className="font-medium cursor-pointer"
                >
                  Generate Heatmaps
                </Label>
              </div>
              <p className="text-xs text-muted-foreground">
                Create GradCAM visualizations showing detected manipulation regions
              </p>
            </div>
            <Switch
              id="generate-heatmaps"
              checked={options.generateHeatmaps}
              onCheckedChange={(checked) => updateOption('generateHeatmaps', checked)}
              disabled={isSubmitting || isUploading}
              aria-describedby="generate-heatmaps-desc"
            />
          </div>

          {/* Defense level selection */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-muted-foreground" />
              <Label className="font-medium">Defense Level</Label>
            </div>
            <RadioGroup
              value={options.defenseLevel}
              onValueChange={(value) => updateOption('defenseLevel', value as DefenseLevel)}
              disabled={isSubmitting || isUploading}
              className="grid gap-3"
            >
              {DEFENSE_LEVELS.map((level) => (
                <div
                  key={level.value}
                  className={cn(
                    'flex items-center space-x-3 rounded-lg border p-4 cursor-pointer',
                    'transition-colors hover:bg-muted/50',
                    options.defenseLevel === level.value && 'border-primary bg-primary/5'
                  )}
                >
                  <RadioGroupItem 
                    value={level.value} 
                    id={`defense-${level.value}`} 
                  />
                  <div className="flex-1 space-y-0.5">
                    <Label 
                      htmlFor={`defense-${level.value}`}
                      className="font-medium cursor-pointer"
                    >
                      {level.label}
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      {level.description}
                    </p>
                  </div>
                </div>
              ))}
            </RadioGroup>
          </div>

          {/* Error display */}
          {submitAnalysis.error && (
            <div 
              className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 text-destructive text-sm"
              role="alert"
              data-testid="analysis-form-error"
            >
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span>{submitAnalysis.error.message || 'Failed to submit analysis'}</span>
            </div>
          )}
        </CardContent>

        <CardFooter className="flex justify-between border-t pt-6">
          <Button
            type="button"
            variant="outline"
            onClick={clearFile}
            disabled={isSubmitting || isUploading}
          >
            Clear
          </Button>
          <Button
            type="submit"
            disabled={!canSubmit}
            className="min-w-[140px]"
            data-testid="analysis-form-submit"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <FileSearch className="mr-2 h-4 w-4" />
                Analyze File
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
    </form>
  );
}

export default AnalysisForm;
