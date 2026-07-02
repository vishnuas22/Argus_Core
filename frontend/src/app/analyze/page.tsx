'use client';

import { useCallback } from 'react';
import Link from 'next/link';
import { ArrowLeft, Eye, Upload, Shield, Zap, FileSearch } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { UploadZone } from '@/components/upload/UploadZone';
import { AnalysisForm } from '@/components/analysis/AnalysisForm';
import { useUploadStore } from '@/store/uploadStore';

const GUIDES = [
  {
    icon: Shield,
    title: 'Supported Formats',
    items: ['MP4, WebM, MOV — Video', 'MP3, WAV, FLAC — Audio', 'JPEG, PNG, WebP — Image'],
  },
  {
    icon: Zap,
    title: 'Limits',
    items: ['Max file size: 500MB', 'Max video duration: 10 min', 'Typical processing: 15-60s'],
  },
  {
    icon: FileSearch,
    title: 'Results Include',
    items: ['Confidence score & verdict', 'GradCAM heatmap overlays', 'Court-admissible PDF report'],
  },
];

export default function AnalyzePage() {
  const { file, setFile } = useUploadStore();

  const handleSubmitSuccess = useCallback((analysisId: string) => {
    window.location.href = `/analysis/${analysisId}`;
  }, []);

  const handleSubmitError = useCallback((error: Error) => {
    console.error('Submission failed:', error);
  }, []);

  const handleFileSelect = useCallback((selectedFile: File) => {
    setFile(selectedFile);
  }, [setFile]);

  return (
    <div className="min-h-screen flex flex-col" data-testid="analyze-page">
      <header className="sticky top-0 z-50 border-b border-border/50 bg-background/70 backdrop-blur-lg">
        <div className="mx-auto max-w-7xl px-6 lg:px-8 flex h-14 items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/">
              <Button variant="ghost" size="sm" className="gap-1.5 text-xs text-muted-foreground h-8">
                <ArrowLeft className="h-3.5 w-3.5" />
                Back
              </Button>
            </Link>
          </div>
          <Link href="/" className="flex items-center gap-2 group">
            <div className="p-1.5 rounded-md bg-primary/10 group-hover:bg-primary/15 transition-colors duration-200">
              <Eye className="w-4 h-4 text-primary" strokeWidth={1.5} />
            </div>
            <span className="text-sm font-semibold tracking-tight">
              Argus<span className="text-muted-foreground font-normal">Core</span>
            </span>
          </Link>
        </div>
      </header>

      <main className="flex-1">
        <div className="mx-auto max-w-7xl px-6 lg:px-8 py-8 md:py-12">
          <div className="max-w-2xl mx-auto space-y-8">
            <div className="text-center space-y-3">
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border border-border bg-muted/30 text-2xs text-muted-foreground font-medium tracking-wide">
                <Upload className="h-3 w-3 text-primary" strokeWidth={1.5} />
                Upload & Analyze
              </div>
              <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">
                Analyze Media
              </h1>
              <p className="text-sm text-muted-foreground max-w-lg mx-auto leading-relaxed">
                Upload a video, audio, or image file for deepfake manipulation analysis using
                multi-modal AI detection.
              </p>
            </div>

            <div className="surface rounded-lg overflow-hidden">
              <div className="px-6 py-4 border-b border-border/50 flex items-center gap-2">
                <div className="p-1.5 rounded-md bg-primary/10">
                  <Upload className="h-3.5 w-3.5 text-primary" strokeWidth={1.5} />
                </div>
                <div>
                  <p className="text-sm font-medium">Upload Media File</p>
                  <p className="text-2xs text-muted-foreground">Drag and drop or click to select</p>
                </div>
              </div>
              <div className="p-6">
                <UploadZone
                  onFileSelect={handleFileSelect}
                  className="border-2 border-dashed border-border hover:border-primary/30 transition-all duration-300 rounded-lg bg-background/50"
                />
                {file && (
                  <div className="mt-6 pt-6 border-t border-border/50 animate-fade-in-up">
                    <AnalysisForm
                      onSubmitSuccess={handleSubmitSuccess}
                      onSubmitError={handleSubmitError}
                    />
                  </div>
                )}
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              {GUIDES.map((guide) => (
                <div
                  key={guide.title}
                  className="p-4 rounded-lg border border-border/50 bg-card/50"
                >
                  <div className="flex items-center gap-2 mb-3">
                    <guide.icon className="h-3.5 w-3.5 text-primary" strokeWidth={1.5} />
                    <span className="text-xs font-semibold tracking-tight">{guide.title}</span>
                  </div>
                  <div className="space-y-1.5">
                    {guide.items.map((item) => (
                      <div key={item} className="flex items-center gap-2 text-2xs text-muted-foreground">
                        <div className="w-1 h-1 rounded-full bg-primary/40 shrink-0" />
                        <span>{item}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      <footer className="border-t border-border/50">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="py-6 flex items-center justify-between">
            <div className="flex items-center gap-2 text-2xs text-muted-foreground">
              <Eye className="h-3 w-3 text-primary/60" strokeWidth={1.5} />
              <span>Argus Core v1.0</span>
            </div>
            <div className="flex items-center gap-1.5 text-2xs text-muted-foreground">
              <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse-soft" />
              System Ready
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
