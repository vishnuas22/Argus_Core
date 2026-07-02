/**
 * Argus Core - Analysis Page Loading Skeleton
 * ============================================
 * Next.js loading UI for the analysis results page.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - app/analysis/[id]/loading.tsx
 * 
 * Role: Display loading skeleton while analysis page data is being fetched.
 * Uses Next.js Suspense boundaries for automatic streaming.
 * 
 * Integration:
 * - Next.js Suspense: Automatically displayed during page load
 * - Backend: While waiting for GET /api/v1/analyze/{id}
 * - Components: Mirrors structure of analysis/[id]/page.tsx
 * 
 * Component Contract (P0):
 * - Matches visual structure of actual page
 * - Provides visual feedback during loading
 * - Accessible: Uses ARIA busy state
 * - data-testid: analysis-loading-skeleton
 */

import { Shield } from 'lucide-react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';

/**
 * Analysis Page Loading Skeleton
 * 
 * Displayed automatically by Next.js when the analysis page is loading.
 * Mirrors the structure of the actual page for a seamless transition.
 */
export default function AnalysisLoading() {
  return (
    <div 
      className="min-h-screen bg-gradient-to-b from-background to-muted/20"
      data-testid="analysis-loading-skeleton"
      role="status"
      aria-busy="true"
      aria-label="Loading analysis results"
    >
      {/* Header Skeleton */}
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          {/* Left side */}
          <div className="flex items-center gap-4">
            <Skeleton className="h-9 w-20 rounded-md" />
            <div className="hidden sm:block space-y-1.5">
              <Skeleton className="h-5 w-32" />
              <Skeleton className="h-3 w-20" />
            </div>
          </div>
          
          {/* Logo */}
          <div className="flex items-center gap-2 font-bold text-lg">
            <Shield className="h-6 w-6 text-primary" />
            <span className="hidden sm:inline text-muted-foreground">Argus Core</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container py-8">
        <div className="mx-auto max-w-4xl space-y-6">
          {/* Timeline Skeleton */}
          <Card>
            <CardContent className="py-4">
              <TimelineSkeleton />
            </CardContent>
          </Card>

          {/* Progress Indicator Skeleton */}
          <ProgressSkeleton />

          {/* Results Card Skeleton */}
          <Card>
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Skeleton className="h-12 w-12 rounded-full" />
                  <div className="space-y-2">
                    <Skeleton className="h-6 w-48" />
                    <Skeleton className="h-4 w-64" />
                  </div>
                </div>
                <div className="text-right space-y-1">
                  <Skeleton className="h-4 w-20 ml-auto" />
                  <Skeleton className="h-10 w-24" />
                  <Skeleton className="h-3 w-16 ml-auto" />
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="pt-4 space-y-6">
              {/* Summary Skeleton */}
              <div className="space-y-3">
                <Skeleton className="h-5 w-36" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-4/5" />
                <Skeleton className="h-4 w-3/4" />
              </div>

              {/* Divider */}
              <div className="border-t" />

              {/* Score Breakdown Skeleton */}
              <div className="space-y-3">
                <Skeleton className="h-5 w-32" />
                <div className="space-y-3">
                  {[1, 2, 3, 4].map((i) => (
                    <ScoreRowSkeleton key={i} />
                  ))}
                </div>
              </div>

              {/* Divider */}
              <div className="border-t" />

              {/* Actions Skeleton */}
              <div className="flex gap-3">
                <Skeleton className="h-10 w-36" />
                <Skeleton className="h-10 w-24" />
                <Skeleton className="h-10 w-32" />
              </div>
            </CardContent>
          </Card>

          {/* Metadata Card Skeleton */}
          <Card className="bg-muted/30">
            <CardHeader className="pb-2">
              <Skeleton className="h-5 w-32" />
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="space-y-1.5">
                    <Skeleton className="h-3 w-20" />
                    <Skeleton className="h-4 w-24" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t mt-auto">
        <div className="container py-6 text-center">
          <Skeleton className="h-4 w-64 mx-auto" />
        </div>
      </footer>
    </div>
  );
}

/**
 * Timeline Skeleton Component
 * Horizontal progress timeline skeleton
 */
function TimelineSkeleton() {
  return (
    <div className="flex items-center justify-between">
      {[1, 2, 3, 4, 5].map((i, index) => (
        <div key={i} className="flex items-center flex-1">
          <div className="flex flex-col items-center">
            <Skeleton className="w-10 h-10 rounded-full" />
            <Skeleton className="mt-2 h-3 w-16" />
          </div>
          {index < 4 && (
            <Skeleton className="flex-1 h-1 mx-2 rounded-full" />
          )}
        </div>
      ))}
    </div>
  );
}

/**
 * Progress Indicator Skeleton Component
 */
function ProgressSkeleton() {
  return (
    <div className="rounded-lg border p-4 bg-muted/30">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <Skeleton className="w-10 h-10 rounded-full" />
          <div className="space-y-1.5">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-48" />
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Skeleton className="h-8 w-12" />
          <Skeleton className="h-4 w-4" />
        </div>
      </div>
      
      {/* Progress bar */}
      <Skeleton className="h-2 w-full rounded-full" />
      
      {/* Stage labels */}
      <div className="flex justify-between pt-2">
        {['Upload', 'Preprocess', 'Analyze', 'Score', 'Complete'].map((_, i) => (
          <Skeleton key={i} className="h-3 w-12" />
        ))}
      </div>
    </div>
  );
}

/**
 * Score Row Skeleton Component
 */
function ScoreRowSkeleton() {
  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
      <div className="flex items-center gap-3">
        <Skeleton className="h-4 w-4" />
        <Skeleton className="h-4 w-28" />
      </div>
      <div className="flex items-center gap-2">
        <Skeleton className="w-24 h-2 rounded-full" />
        <Skeleton className="h-4 w-10" />
      </div>
    </div>
  );
}
