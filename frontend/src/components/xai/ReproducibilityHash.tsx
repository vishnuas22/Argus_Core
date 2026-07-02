/**
 * Argus Core - Reproducibility Hash Component
 * ===========================================
 * Displays cryptographic hash for forensic reproducibility verification.
 * 
 * Implements: XAI_FRONTEND_IMPLEMENTATION.md - Section 4.3 - components/xai/ReproducibilityHash.tsx
 * 
 * Role: Display the SHA-256 hash that ensures analysis reproducibility
 * for court-admissible evidence and forensic verification.
 * 
 * Integration:
 * - Used by: XAIExplanationPanel
 * - Data: reproducibility_hash from XAI explanation
 */

'use client';

import React, { useState, useCallback } from 'react';
import { cn } from '@/lib/utils';
import {
  Copy,
  Check,
  Shield,
  ExternalLink,
  Info,
  Hash as HashIcon,
} from 'lucide-react';

// UI Components
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// ============== TYPES ==============

/**
 * Props for ReproducibilityHash component
 */
export interface ReproducibilityHashProps {
  /** SHA-256 hash string */
  hash: string;
  /** Label for the hash */
  label?: string;
  /** Description of what the hash represents */
  description?: string;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show in compact mode */
  compact?: boolean;
  /** Whether to show copy button */
  showCopyButton?: boolean;
  /** Whether to show verification info */
  showVerificationInfo?: boolean;
  /** Callback when hash is copied */
  onCopy?: (hash: string) => void;
}

// ============== HELPER FUNCTIONS ==============

/**
 * Format hash for display (truncate middle)
 */
function formatHash(hash: string, maxLength: number = 16): string {
  if (hash.length <= maxLength) return hash;
  const start = hash.slice(0, maxLength / 2);
  const end = hash.slice(-maxLength / 2);
  return `${start}...${end}`;
}

/**
 * Validate SHA-256 hash format
 */
function isValidSHA256(hash: string): boolean {
  return /^[a-f0-9]{64}$/i.test(hash);
}

// ============== MAIN COMPONENT ==============

/**
 * ReproducibilityHash
 * 
 * Displays a cryptographic hash with copy functionality and verification info.
 * 
 * @example
 * ```tsx
 * <ReproducibilityHash
 *   hash="a1b2c3d4e5f6..."
 *   label="Analysis Hash"
 *   showCopyButton
 * />
 * ```
 */
export function ReproducibilityHash({
  hash,
  label = 'Reproducibility Hash',
  description,
  className,
  compact = false,
  showCopyButton = true,
  showVerificationInfo = true,
  onCopy,
}: ReproducibilityHashProps): React.ReactElement {
  const [copied, setCopied] = useState(false);

  // Validate hash
  const isValid = isValidSHA256(hash);

  // Handle copy
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(hash);
      setCopied(true);
      onCopy?.(hash);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy hash:', err);
    }
  }, [hash, onCopy]);

  // Compact inline display
  if (compact) {
    return (
      <div
        className={cn('inline-flex items-center gap-2', className)}
        data-testid="reproducibility-hash-compact"
      >
        <HashIcon className="h-3 w-3 text-muted-foreground" />
        <code className="text-xs bg-muted px-2 py-0.5 rounded font-mono">
          {formatHash(hash)}
        </code>
        {showCopyButton && (
          <Button
            variant="ghost"
            size="icon"
            className="h-5 w-5"
            onClick={handleCopy}
            data-testid="copy-hash-button"
          >
            {copied ? (
              <Check className="h-3 w-3 text-green-500" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        )}
      </div>
    );
  }

  // Full display
  return (
    <Card className={cn('w-full', className)} data-testid="reproducibility-hash">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-primary" />
            <CardTitle className="text-sm font-medium">{label}</CardTitle>
          </div>
          {isValid && (
            <Badge variant="outline" className="text-xs">
              SHA-256
            </Badge>
          )}
        </div>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        {/* Hash display */}
        <div className="flex items-center gap-2 p-3 bg-muted rounded-md">
          <HashIcon className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <code className="flex-1 text-xs font-mono break-all select-all">
            {hash}
          </code>
          {showCopyButton && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 flex-shrink-0"
                    onClick={handleCopy}
                    data-testid="copy-hash-button"
                  >
                    {copied ? (
                      <Check className="h-4 w-4 text-green-500" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {copied ? 'Copied!' : 'Copy hash'}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>

        {/* Verification info */}
        {showVerificationInfo && (
          <div className="mt-3 space-y-2">
            <div className="flex items-center gap-2 text-sm">
              {isValid ? (
                <>
                  <Check className="h-4 w-4 text-green-500" />
                  <span className="text-green-500">Valid SHA-256 format</span>
                </>
              ) : (
                <>
                  <Info className="h-4 w-4 text-yellow-500" />
                  <span className="text-yellow-500">Non-standard hash format</span>
                </>
              )}
            </div>
            
            <div className="text-xs text-muted-foreground space-y-1">
              <p>
                This hash uniquely identifies this analysis and can be used to:
              </p>
              <ul className="list-disc list-inside space-y-0.5 ml-2">
                <li>Verify analysis integrity</li>
                <li>Reproduce results with same inputs</li>
                <li>Provide court-admissible evidence chain</li>
              </ul>
            </div>
          </div>
        )}

        {/* Copy status */}
        {copied && (
          <div className="mt-2 text-xs text-green-500 flex items-center gap-1">
            <Check className="h-3 w-3" />
            Hash copied to clipboard
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ============== SKELETON COMPONENT ==============

/**
 * Skeleton component for ReproducibilityHash loading state
 */
export function ReproducibilityHashSkeleton({
  className,
  compact = false,
}: {
  className?: string;
  compact?: boolean;
}): React.ReactElement {
  if (compact) {
    return (
      <div className={cn('inline-flex items-center gap-2', className)}>
        <Skeleton className="h-3 w-3" />
        <Skeleton className="h-4 w-24" />
      </div>
    );
  }

  return (
    <Card className={cn('w-full', className)} data-testid="reproducibility-hash-skeleton">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4" />
            <Skeleton className="h-4 w-32" />
          </div>
          <Skeleton className="h-5 w-16" />
        </div>
      </CardHeader>
      <CardContent>
        <Skeleton className="h-12 w-full rounded-md" />
        <div className="mt-3 space-y-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </div>
      </CardContent>
    </Card>
  );
}

export default ReproducibilityHash;
