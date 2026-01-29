/**
 * Argus Core - Metadata Panel Component
 * =====================================
 * Detailed metadata analysis results with C2PA provenance and EXIF integrity checks.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - components/modality/MetadataPanel.tsx
 * 
 * Role: Display comprehensive metadata analysis results.
 * Shows C2PA Content Credentials verification, EXIF anomaly detection, and file structure integrity.
 * 
 * Integration:
 * - Used by: ModalityTabs.tsx (lazy loaded)
 * - Backend: Uses MetadataResult from GET /api/v1/analyze/{id}/detail
 *   - c2pa: C2PAManifest - Content Credentials data
 *   - exif_anomalies: List[str] - Detected EXIF anomalies
 *   - file_structure_valid: bool - File structure integrity check
 * 
 * Backend Analysis (from analyzers/metadata.py):
 * - C2PA v2.3 specification for content authenticity verification
 * - EXIF metadata consistency analysis
 * - File container structure validation
 * 
 * Component Contract (P0):
 * - Props interface defined
 * - Loading state: Shows skeleton loaders for each section
 * - Empty state: Shows message when no metadata available
 * - Error state: Displays validation errors gracefully
 * - Accessibility: Proper headings, ARIA labels, keyboard navigation
 * - data-testid: metadata-panel, metadata-c2pa-section, metadata-exif-section, metadata-structure-section
 */

'use client';

import React, { useMemo } from 'react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import {
  FileSearch2,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  ShieldQuestion,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  FileCode,
  FileWarning,
  Calendar,
  User,
  Award,
  Hash,
  Clock,
  MapPin,
  Camera,
  Settings2,
  type LucideIcon,
} from 'lucide-react';
import type { MetadataResult, C2PAData, EXIFData } from '@/types/analysis';

// ============== TYPES ==============

/**
 * Props for MetadataPanel component
 */
export interface MetadataPanelProps {
  /** Metadata analysis result from backend */
  result: MetadataResult;
  /** Analysis ID for fetching additional data */
  analysisId: string;
  /** Additional CSS classes */
  className?: string;
  /** Show compact version */
  compact?: boolean;
}

// ============== CONSTANTS ==============

/**
 * Status colors for badges and indicators
 */
const STATUS_COLORS = {
  good: 'bg-green-500/10 text-green-700 dark:text-green-400 border-green-500/20',
  warning: 'bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border-yellow-500/20',
  danger: 'bg-red-500/10 text-red-700 dark:text-red-400 border-red-500/20',
  neutral: 'bg-slate-500/10 text-slate-700 dark:text-slate-400 border-slate-500/20',
} as const;

/**
 * C2PA verification status configurations
 */
const C2PA_STATUS = {
  valid: {
    icon: ShieldCheck,
    label: 'Verified',
    color: 'text-green-500',
    bgColor: 'bg-green-500/10',
    borderColor: 'border-green-500/20',
  },
  invalid: {
    icon: ShieldX,
    label: 'Invalid',
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/20',
  },
  notPresent: {
    icon: ShieldQuestion,
    label: 'Not Present',
    color: 'text-slate-500',
    bgColor: 'bg-slate-500/10',
    borderColor: 'border-slate-500/20',
  },
  unknown: {
    icon: ShieldAlert,
    label: 'Unknown',
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/20',
  },
} as const;

// ============== MAIN COMPONENT ==============

/**
 * MetadataPanel Component
 * 
 * Displays detailed metadata analysis results including C2PA Content Credentials,
 * EXIF anomaly detection, and file structure integrity checks.
 * 
 * @example
 * ```tsx
 * <MetadataPanel
 *   result={analysisDetail.metadata_result}
 *   analysisId={analysisId}
 * />
 * ```
 */
export function MetadataPanel({
  result,
  analysisId,
  className,
  compact = false,
}: MetadataPanelProps) {
  // ============== COMPUTED VALUES ==============

  /**
   * Determine overall metadata health
   */
  const overallStatus = useMemo(() => {
    const hasC2PAIssues = result.c2pa.has_manifest && !result.c2pa.is_valid;
    const hasExifIssues = result.exif.anomalies.length > 0;
    const hasStructureIssues = !result.file_integrity.structure_valid;

    if (hasC2PAIssues || hasStructureIssues) return 'danger';
    if (hasExifIssues) return 'warning';
    if (result.c2pa.has_manifest && result.c2pa.is_valid) return 'good';
    return 'neutral';
  }, [result]);

  /**
   * Get overall status label
   */
  const statusLabel = useMemo(() => {
    switch (overallStatus) {
      case 'good': return 'Verified';
      case 'warning': return 'Anomalies Found';
      case 'danger': return 'Issues Detected';
      default: return 'No Provenance';
    }
  }, [overallStatus]);

  // ============== RENDER ==============

  return (
    <div
      className={cn('space-y-6', className)}
      data-testid="metadata-panel"
      role="region"
      aria-label="Metadata Analysis Results"
    >
      {/* Summary Section */}
      <MetadataSummarySection
        result={result}
        overallStatus={overallStatus}
        statusLabel={statusLabel}
        compact={compact}
      />

      {/* C2PA Content Credentials Section */}
      <C2PASection
        c2pa={result.c2pa}
        compact={compact}
      />

      {/* EXIF Analysis Section */}
      <EXIFSection
        exif={result.exif}
        compact={compact}
      />

      {/* File Structure Section */}
      <FileStructureSection
        fileIntegrity={result.file_integrity}
        compact={compact}
      />
    </div>
  );
}

// ============== SUB-COMPONENTS ==============

/**
 * Summary section showing overall metadata analysis
 */
function MetadataSummarySection({
  result,
  overallStatus,
  statusLabel,
  compact,
}: {
  result: MetadataResult;
  overallStatus: 'good' | 'warning' | 'danger' | 'neutral';
  statusLabel: string;
  compact: boolean;
}) {
  const statusConfig = overallStatus === 'good' ? STATUS_COLORS.good :
    overallStatus === 'warning' ? STATUS_COLORS.warning :
    overallStatus === 'danger' ? STATUS_COLORS.danger : STATUS_COLORS.neutral;

  const totalIssues = result.exif.anomalies.length + 
    (result.file_integrity.suspicious_markers?.length || 0) +
    (!result.file_integrity.structure_valid ? 1 : 0) +
    (result.c2pa.has_manifest && !result.c2pa.is_valid ? 1 : 0);

  return (
    <Card data-testid="metadata-summary-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileSearch2 className="h-5 w-5 text-primary" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              Metadata Analysis Summary
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', statusConfig)}>
            {statusLabel}
          </Badge>
        </div>
        {!compact && (
          <CardDescription>
            Content provenance and file integrity verification
          </CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {/* Status Overview Grid */}
        <div className="grid grid-cols-3 gap-4">
          {/* C2PA Status */}
          <StatusCard
            label="C2PA Credentials"
            status={result.c2pa.has_manifest 
              ? (result.c2pa.is_valid ? 'good' : 'danger') 
              : 'neutral'}
            value={result.c2pa.has_manifest 
              ? (result.c2pa.is_valid ? 'Valid' : 'Invalid') 
              : 'Not Present'}
            icon={result.c2pa.has_manifest 
              ? (result.c2pa.is_valid ? ShieldCheck : ShieldX) 
              : ShieldQuestion}
          />

          {/* EXIF Status */}
          <StatusCard
            label="EXIF Metadata"
            status={result.exif.anomalies.length === 0 ? 'good' : 'warning'}
            value={result.exif.anomalies.length === 0 
              ? 'Clean' 
              : `${result.exif.anomalies.length} Issue${result.exif.anomalies.length > 1 ? 's' : ''}`}
            icon={result.exif.anomalies.length === 0 ? CheckCircle2 : AlertTriangle}
          />

          {/* File Structure Status */}
          <StatusCard
            label="File Structure"
            status={result.file_integrity.structure_valid ? 'good' : 'danger'}
            value={result.file_integrity.structure_valid ? 'Valid' : 'Invalid'}
            icon={result.file_integrity.structure_valid ? CheckCircle2 : XCircle}
          />
        </div>

        {/* Total Issues Alert */}
        {totalIssues > 0 && (
          <Alert variant="destructive" className="mt-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Metadata Issues Detected</AlertTitle>
            <AlertDescription>
              {totalIssues} potential issue{totalIssues > 1 ? 's' : ''} found in metadata analysis.
              Review the detailed sections below for more information.
            </AlertDescription>
          </Alert>
        )}

        {/* All Clear Alert */}
        {totalIssues === 0 && result.c2pa.has_manifest && result.c2pa.is_valid && (
          <Alert className="mt-4 border-green-500/20 bg-green-500/5">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            <AlertTitle className="text-green-700 dark:text-green-400">Metadata Verified</AlertTitle>
            <AlertDescription className="text-green-600 dark:text-green-300">
              Content credentials are valid and no metadata anomalies were detected.
              This file has verified provenance.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * C2PA Content Credentials section
 */
function C2PASection({
  c2pa,
  compact,
}: {
  c2pa: C2PAData;
  compact: boolean;
}) {
  const statusKey = c2pa.has_manifest 
    ? (c2pa.is_valid ? 'valid' : 'invalid') 
    : 'notPresent';
  const status = C2PA_STATUS[statusKey];
  const StatusIcon = status.icon;

  return (
    <Card data-testid="metadata-c2pa-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <StatusIcon className={cn('h-5 w-5', status.color)} aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              C2PA Content Credentials
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', status.bgColor, status.borderColor, 'border')}>
            {status.label}
          </Badge>
        </div>
        <CardDescription>
          Coalition for Content Provenance and Authenticity (C2PA v2.3)
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {c2pa.has_manifest ? (
          <>
            {/* Manifest Status */}
            {c2pa.is_valid ? (
              <Alert className="border-green-500/20 bg-green-500/5">
                <ShieldCheck className="h-4 w-4 text-green-500" />
                <AlertTitle className="text-green-700 dark:text-green-400">
                  Valid Content Credentials
                </AlertTitle>
                <AlertDescription className="text-green-600 dark:text-green-300">
                  This file contains a verified C2PA manifest that authenticates its origin
                  and edit history. The digital signature has been validated.
                </AlertDescription>
              </Alert>
            ) : (
              <Alert variant="destructive">
                <ShieldX className="h-4 w-4" />
                <AlertTitle>Invalid Content Credentials</AlertTitle>
                <AlertDescription>
                  A C2PA manifest was found but failed validation. This could indicate
                  tampering, corruption, or an expired/revoked certificate.
                </AlertDescription>
              </Alert>
            )}

            {/* Manifest Details */}
            <div className="grid gap-3 mt-4">
              {c2pa.claim_generator && (
                <DetailRow
                  icon={Settings2}
                  label="Claim Generator"
                  value={c2pa.claim_generator}
                />
              )}
              {c2pa.signature_info?.issuer && (
                <DetailRow
                  icon={User}
                  label="Issuer"
                  value={c2pa.signature_info.issuer}
                />
              )}
              {c2pa.signature_info?.valid_from && (
                <DetailRow
                  icon={Calendar}
                  label="Valid From"
                  value={new Date(c2pa.signature_info.valid_from).toLocaleDateString()}
                />
              )}
              {c2pa.signature_info?.valid_to && (
                <DetailRow
                  icon={Clock}
                  label="Valid Until"
                  value={new Date(c2pa.signature_info.valid_to).toLocaleDateString()}
                />
              )}
            </div>

            {/* Assertions */}
            {c2pa.assertions && c2pa.assertions.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                  <Award className="h-4 w-4" aria-hidden="true" />
                  Assertions ({c2pa.assertions.length})
                </h4>
                <div className="space-y-2">
                  {c2pa.assertions.slice(0, 5).map((assertion, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 p-2 rounded-lg bg-muted/50 text-sm"
                    >
                      <CheckCircle2 className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
                      <span className="truncate">{assertion}</span>
                    </div>
                  ))}
                  {c2pa.assertions.length > 5 && (
                    <p className="text-xs text-muted-foreground text-center">
                      +{c2pa.assertions.length - 5} more assertions
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Ingredients */}
            {c2pa.ingredients && c2pa.ingredients.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                  <FileCode className="h-4 w-4" aria-hidden="true" />
                  Ingredients ({c2pa.ingredients.length})
                </h4>
                <div className="space-y-2">
                  {c2pa.ingredients.slice(0, 3).map((ingredient, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 p-2 rounded-lg bg-muted/50 text-sm"
                    >
                      <Hash className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
                      <span className="truncate font-mono text-xs">{ingredient}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-8">
            <ShieldQuestion className="h-12 w-12 mx-auto text-muted-foreground/40 mb-3" />
            <h4 className="font-medium text-muted-foreground mb-1">
              No C2PA Manifest Found
            </h4>
            <p className="text-sm text-muted-foreground/70 max-w-md mx-auto">
              This file does not contain C2PA Content Credentials. Many authentic files
              may not have C2PA metadata, especially if created before widespread adoption.
            </p>
          </div>
        )}

        {/* Technical Info */}
        {!compact && (
          <div className="mt-4 p-4 rounded-lg bg-muted/30 border border-dashed">
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Info className="h-4 w-4" aria-hidden="true" />
              About C2PA
            </h4>
            <p className="text-xs text-muted-foreground">
              C2PA (Coalition for Content Provenance and Authenticity) is an open standard
              for certifying the source and history of media content. It uses cryptographic
              signatures to verify that content hasn't been modified since creation.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * EXIF metadata analysis section
 */
function EXIFSection({
  exif,
  compact,
}: {
  exif: EXIFData;
  compact: boolean;
}) {
  const hasAnomalies = exif.anomalies.length > 0;
  const status = hasAnomalies ? 'warning' : 'good';

  return (
    <Card data-testid="metadata-exif-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Camera className="h-5 w-5 text-blue-500" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              EXIF Metadata Analysis
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', STATUS_COLORS[status])}>
            {hasAnomalies ? `${exif.anomalies.length} Anomalies` : 'Clean'}
          </Badge>
        </div>
        <CardDescription>
          Exchangeable Image File Format metadata examination
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Anomalies Alert */}
        {hasAnomalies ? (
          <Alert className="border-yellow-500/20 bg-yellow-500/5">
            <AlertTriangle className="h-4 w-4 text-yellow-500" />
            <AlertTitle className="text-yellow-700 dark:text-yellow-400">
              EXIF Anomalies Detected
            </AlertTitle>
            <AlertDescription className="text-yellow-600 dark:text-yellow-300">
              {exif.anomalies.length} potential anomal{exif.anomalies.length > 1 ? 'ies' : 'y'} found 
              in EXIF metadata. This may indicate editing or manipulation.
            </AlertDescription>
          </Alert>
        ) : (
          <Alert className="border-green-500/20 bg-green-500/5">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            <AlertTitle className="text-green-700 dark:text-green-400">
              EXIF Metadata Consistent
            </AlertTitle>
            <AlertDescription className="text-green-600 dark:text-green-300">
              No anomalies detected in EXIF metadata. Timestamps and device information
              appear consistent and unmodified.
            </AlertDescription>
          </Alert>
        )}

        {/* Anomalies List */}
        {hasAnomalies && (
          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2">Detected Anomalies</h4>
            <div className="space-y-2">
              {exif.anomalies.map((anomaly, index) => (
                <div
                  key={index}
                  className="flex items-start gap-2 p-3 rounded-lg bg-yellow-500/5 border border-yellow-500/20"
                >
                  <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                  <span className="text-sm">{anomaly}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* EXIF Details */}
        <Separator className="my-4" />
        <h4 className="text-sm font-medium mb-3">Extracted Metadata</h4>
        <div className="grid gap-3">
          {exif.camera_make && (
            <DetailRow
              icon={Camera}
              label="Camera Make"
              value={exif.camera_make}
            />
          )}
          {exif.camera_model && (
            <DetailRow
              icon={Camera}
              label="Camera Model"
              value={exif.camera_model}
            />
          )}
          {exif.software && (
            <DetailRow
              icon={Settings2}
              label="Software"
              value={exif.software}
            />
          )}
          {exif.datetime_original && (
            <DetailRow
              icon={Calendar}
              label="Date Taken"
              value={new Date(exif.datetime_original).toLocaleString()}
            />
          )}
          {exif.gps_location && (
            <DetailRow
              icon={MapPin}
              label="GPS Location"
              value={`${exif.gps_location.latitude.toFixed(6)}, ${exif.gps_location.longitude.toFixed(6)}`}
            />
          )}
          {!exif.camera_make && !exif.camera_model && !exif.software && !exif.datetime_original && (
            <div className="text-center py-4 text-sm text-muted-foreground">
              No EXIF metadata found in this file
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * File structure integrity section
 */
function FileStructureSection({
  fileIntegrity,
  compact,
}: {
  fileIntegrity: MetadataResult['file_integrity'];
  compact: boolean;
}) {
  const isValid = fileIntegrity.structure_valid && fileIntegrity.hash_verified;
  const hasSuspiciousMarkers = fileIntegrity.suspicious_markers && fileIntegrity.suspicious_markers.length > 0;

  return (
    <Card data-testid="metadata-structure-section">
      <CardHeader className={compact ? 'pb-2' : undefined}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileCode className="h-5 w-5 text-purple-500" aria-hidden="true" />
            <CardTitle className={compact ? 'text-base' : 'text-lg'}>
              File Structure Integrity
            </CardTitle>
          </div>
          <Badge className={cn('font-mono', isValid ? STATUS_COLORS.good : STATUS_COLORS.danger)}>
            {isValid ? 'Valid' : 'Issues Found'}
          </Badge>
        </div>
        <CardDescription>
          Container format and binary structure verification
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Integrity Status */}
        <div className="grid grid-cols-2 gap-4">
          <div className={cn(
            'p-4 rounded-lg border',
            fileIntegrity.structure_valid ? 'bg-green-500/5 border-green-500/20' : 'bg-red-500/5 border-red-500/20'
          )}>
            <div className="flex items-center gap-2 mb-1">
              {fileIntegrity.structure_valid ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" aria-hidden="true" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500" aria-hidden="true" />
              )}
              <span className="text-sm font-medium">Structure</span>
            </div>
            <p className={cn(
              'text-lg font-bold',
              fileIntegrity.structure_valid ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
            )}>
              {fileIntegrity.structure_valid ? 'Valid' : 'Invalid'}
            </p>
          </div>

          <div className={cn(
            'p-4 rounded-lg border',
            fileIntegrity.hash_verified ? 'bg-green-500/5 border-green-500/20' : 'bg-yellow-500/5 border-yellow-500/20'
          )}>
            <div className="flex items-center gap-2 mb-1">
              {fileIntegrity.hash_verified ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" aria-hidden="true" />
              ) : (
                <AlertTriangle className="h-4 w-4 text-yellow-500" aria-hidden="true" />
              )}
              <span className="text-sm font-medium">Hash</span>
            </div>
            <p className={cn(
              'text-lg font-bold',
              fileIntegrity.hash_verified ? 'text-green-600 dark:text-green-400' : 'text-yellow-600 dark:text-yellow-400'
            )}>
              {fileIntegrity.hash_verified ? 'Verified' : 'Unverified'}
            </p>
          </div>
        </div>

        {/* Structure Issues Alert */}
        {!fileIntegrity.structure_valid && (
          <Alert variant="destructive">
            <XCircle className="h-4 w-4" />
            <AlertTitle>Invalid File Structure</AlertTitle>
            <AlertDescription>
              The file container structure does not conform to expected format specifications.
              This may indicate file corruption, manipulation, or an unsupported format variant.
            </AlertDescription>
          </Alert>
        )}

        {/* Suspicious Markers */}
        {hasSuspiciousMarkers && (
          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <FileWarning className="h-4 w-4 text-yellow-500" aria-hidden="true" />
              Suspicious Markers ({fileIntegrity.suspicious_markers!.length})
            </h4>
            <div className="space-y-2">
              {fileIntegrity.suspicious_markers!.map((marker, index) => (
                <div
                  key={index}
                  className="flex items-start gap-2 p-3 rounded-lg bg-yellow-500/5 border border-yellow-500/20"
                >
                  <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                  <span className="text-sm font-mono">{marker}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* All Clear */}
        {isValid && !hasSuspiciousMarkers && (
          <Alert className="border-green-500/20 bg-green-500/5">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            <AlertTitle className="text-green-700 dark:text-green-400">
              File Structure Verified
            </AlertTitle>
            <AlertDescription className="text-green-600 dark:text-green-300">
              The file structure is valid and matches expected format specifications.
              No suspicious binary markers or embedded content detected.
            </AlertDescription>
          </Alert>
        )}

        {/* Technical Info */}
        {!compact && (
          <div className="mt-4 p-4 rounded-lg bg-muted/30 border border-dashed">
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Info className="h-4 w-4" aria-hidden="true" />
              Structure Analysis
            </h4>
            <p className="text-xs text-muted-foreground">
              File structure analysis verifies that the binary container (MP4, MOV, AVI, etc.)
              conforms to format specifications. Invalid structures may indicate file corruption,
              steganographic embedding, or post-processing manipulation.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ============== HELPER COMPONENTS ==============

/**
 * Status card for summary grid
 */
function StatusCard({
  label,
  status,
  value,
  icon: Icon,
}: {
  label: string;
  status: 'good' | 'warning' | 'danger' | 'neutral';
  value: string;
  icon: LucideIcon;
}) {
  const statusConfig = status === 'good' ? STATUS_COLORS.good :
    status === 'warning' ? STATUS_COLORS.warning :
    status === 'danger' ? STATUS_COLORS.danger : STATUS_COLORS.neutral;

  return (
    <div className={cn('flex flex-col p-3 rounded-lg border', statusConfig)}>
      <div className="flex items-center gap-2 mb-2">
        <Icon className="h-4 w-4" aria-hidden="true" />
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <span className="text-sm font-semibold">{value}</span>
    </div>
  );
}

/**
 * Detail row for displaying key-value pairs
 */
function DetailRow({
  icon: Icon,
  label,
  value,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-3 p-2 rounded-lg bg-muted/30">
      <Icon className="h-4 w-4 text-muted-foreground flex-shrink-0" aria-hidden="true" />
      <span className="text-sm text-muted-foreground">{label}:</span>
      <span className="text-sm font-medium truncate">{value}</span>
    </div>
  );
}

// ============== SKELETON LOADER ==============

/**
 * Skeleton loader for MetadataPanel
 */
export function MetadataPanelSkeleton() {
  return (
    <div className="space-y-6" data-testid="metadata-panel-skeleton">
      {/* Summary Skeleton */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-6 w-20" />
          </div>
          <Skeleton className="h-4 w-64 mt-2" />
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Section Skeletons */}
      {[1, 2, 3].map((i) => (
        <Card key={i}>
          <CardHeader>
            <Skeleton className="h-6 w-40" />
            <Skeleton className="h-4 w-56 mt-2" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-20 w-full" />
            <div className="grid gap-3 mt-4">
              {[1, 2, 3].map((j) => (
                <Skeleton key={j} className="h-10 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ============== EXPORTS ==============

export default MetadataPanel;
