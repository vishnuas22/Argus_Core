/**
 * Argus Core - Analyze Page
 * =========================
 * Main analysis upload page with file upload zone and analysis options form.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - app/analyze/page.tsx
 * 
 * Role: Primary entry point for users to upload media files for deepfake analysis.
 * Contains UploadZone for file selection and AnalysisForm for configuration options.
 * 
 * Integration:
 * - Imports: components/upload/UploadZone, components/analysis/AnalysisForm
 * - State: uploadStore for file state management
 * - Navigation: Redirects to /analysis/[id] after successful submission
 * 
 * Component Contract (P0):
 * - Client component for file handling
 * - Loading state: Shows during file validation
 * - Error state: Displays validation/upload errors
 * - Empty state: Shows upload instructions
 * - Accessibility: Keyboard navigation, screen reader support
 * - data-testid: analyze-page, analyze-page-title
 */

'use client';

import { useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Shield, Zap, FileSearch, Upload } from 'lucide-react';
import Link from 'next/link';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { UploadZone } from '@/components/upload/UploadZone';
import { AnalysisForm } from '@/components/analysis/AnalysisForm';
import { useUploadStore } from '@/store/uploadStore';

// ============== FEATURE HIGHLIGHTS ==============

interface FeatureItem {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
}

const FEATURES: FeatureItem[] = [
  {
    icon: Shield,
    title: 'Multi-Modal Detection',
    description: 'Analyze video, audio, image, and text for manipulation indicators',
  },
  {
    icon: Zap,
    title: 'Real-Time Analysis',
    description: 'Track progress with live updates as your file is processed',
  },
  {
    icon: FileSearch,
    title: 'Detailed Reports',
    description: 'Get comprehensive forensic reports with evidence and findings',
  },
];

// ============== COMPONENT ==============

export default function AnalyzePage() {
  const router = useRouter();
  const { file, reset } = useUploadStore();

  // Reset store on unmount to clean up any stale state
  useEffect(() => {
    return () => {
      // Only reset if navigating away (not to analysis page)
      // The store will be reset by successful submission anyway
    };
  }, []);

  /**
   * Handle successful submission - navigates to analysis page
   */
  const handleSubmitSuccess = useCallback((analysisId: string) => {
    // Navigation is handled by the mutation in useAnalysis hook
    // This callback can be used for additional tracking/analytics
    console.log(`Analysis submitted: ${analysisId}`);
  }, []);

  /**
   * Handle submission error
   */
  const handleSubmitError = useCallback((error: Error) => {
    console.error('Submission failed:', error);
    // Error display is handled by AnalysisForm
  }, []);

  /**
   * Handle file selection from UploadZone
   */
  const handleFileSelect = useCallback((selectedFile: File) => {
    // File is already set in store by UploadZone
    console.log(`File selected: ${selectedFile.name}`);
  }, []);

  return (
    <div 
      className="min-h-screen bg-gradient-to-b from-background to-muted/20"
      data-testid="analyze-page"
    >
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm" className="gap-2">
                <ArrowLeft className="h-4 w-4" />
                Back
              </Button>
            </Link>
            <div className="hidden sm:block">
              <h1 
                className="text-xl font-semibold"
                data-testid="analyze-page-title"
              >
                Analyze Media
              </h1>
            </div>
          </div>
          
          <Link href="/">
            <div className="flex items-center gap-2 font-bold text-lg">
              <Shield className="h-6 w-6 text-primary" />
              <span>Argus Core</span>
            </div>
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main className="container py-8">
        <div className="mx-auto max-w-4xl space-y-8">
          {/* Page Header */}
          <div className="text-center space-y-2">
            <h2 className="text-3xl font-bold tracking-tight">
              Deepfake Detection Analysis
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Upload a video, audio, or image file to analyze for synthetic media manipulation 
              using advanced multi-modal AI detection.
            </p>
          </div>

          {/* Main Card with Upload and Form */}
          <Card className="border-2">
            <CardHeader className="text-center pb-2">
              <CardTitle className="flex items-center justify-center gap-2 text-xl">
                <Upload className="h-5 w-5" />
                Upload Media File
              </CardTitle>
              <CardDescription>
                Drag and drop or click to select a file for analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Upload Zone */}
              <UploadZone 
                onFileSelect={handleFileSelect}
                className={cn(
                  file && 'border-primary/50 bg-primary/5'
                )}
              />

              {/* Analysis Form (visible after file selection) */}
              {file && (
                <div className="pt-4 border-t">
                  <AnalysisForm
                    onSubmitSuccess={handleSubmitSuccess}
                    onSubmitError={handleSubmitError}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          {/* Features Section */}
          <div className="grid gap-4 md:grid-cols-3">
            {FEATURES.map((feature) => (
              <Card 
                key={feature.title}
                className="bg-card/50 hover:bg-card transition-colors"
              >
                <CardContent className="pt-6">
                  <div className="flex flex-col items-center text-center space-y-3">
                    <div className="p-3 rounded-full bg-primary/10">
                      <feature.icon className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="font-semibold">{feature.title}</h3>
                    <p className="text-sm text-muted-foreground">
                      {feature.description}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Info Cards */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* Supported Formats */}
            <Card className="bg-muted/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Supported Formats</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="font-medium text-muted-foreground mb-1">Video</p>
                    <p>MP4, WebM, MOV, AVI</p>
                  </div>
                  <div>
                    <p className="font-medium text-muted-foreground mb-1">Audio</p>
                    <p>MP3, WAV, OGG, FLAC</p>
                  </div>
                  <div>
                    <p className="font-medium text-muted-foreground mb-1">Image</p>
                    <p>JPEG, PNG, WebP</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Analysis Info */}
            <Card className="bg-muted/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Analysis Details</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="text-sm space-y-1.5 text-muted-foreground">
                  <li>• Maximum file size: 500MB</li>
                  <li>• Maximum video duration: 5 minutes</li>
                  <li>• Processing time: 15-60 seconds typical</li>
                  <li>• Results include confidence scores and explanations</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t mt-auto">
        <div className="container py-6 text-center text-sm text-muted-foreground">
          <p>
            Argus Core - Multi-Modal Deepfake Detection Platform
          </p>
          <p className="mt-1">
            Powered by advanced AI models for video, audio, and image analysis
          </p>
        </div>
      </footer>
    </div>
  );
}
