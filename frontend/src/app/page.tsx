/**
 * Argus Core - Landing Page
 * =========================
 * Landing page with hero section, feature overview, and call-to-action.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - app/page.tsx
 * 
 * Role: Primary entry point for the application. Showcases platform capabilities
 * and provides clear call-to-action to start analysis.
 * 
 * Integration:
 * - Navigation: Links to /analyze for file upload
 * - Static content with client-side animations
 * 
 * Component Contract (P0):
 * - Responsive layout (mobile-first)
 * - Accessible navigation
 * - Performance optimized (no heavy JS)
 * - data-testid: landing-page, hero-section, cta-button
 */

import Link from 'next/link';
import { 
  Shield, 
  FileVideo, 
  FileAudio, 
  Image, 
  Zap, 
  Lock, 
  BarChart3,
  ChevronRight,
  CheckCircle,
  ArrowRight
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

// ============== FEATURE DATA ==============

const FEATURES = [
  {
    icon: FileVideo,
    title: 'Video Analysis',
    description: 'Detect face swaps, lip-sync manipulation, and temporal inconsistencies in video content.',
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
  },
  {
    icon: FileAudio,
    title: 'Audio Analysis',
    description: 'Identify voice cloning, spectral anomalies, and synthetic speech patterns.',
    color: 'text-purple-500',
    bgColor: 'bg-purple-500/10',
  },
  {
    icon: Image,
    title: 'Image Analysis',
    description: 'Analyze images for GAN artifacts, manipulation traces, and provenance verification.',
    color: 'text-green-500',
    bgColor: 'bg-green-500/10',
  },
  {
    icon: BarChart3,
    title: 'Trust Scoring',
    description: 'Get comprehensive authenticity scores with confidence levels and detailed explanations.',
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/10',
  },
];

const BENEFITS = [
  'Multi-modal analysis combining video, audio, and metadata',
  'Real-time progress tracking with WebSocket updates',
  'GradCAM heatmaps showing manipulation regions',
  'C2PA provenance verification support',
  'Detailed PDF forensic reports',
  'Advanced adversarial attack protection',
];

// ============== COMPONENT ==============

export default function LandingPage() {
  return (
    <div 
      className="min-h-screen bg-gradient-to-b from-background via-background to-muted/30"
      data-testid="landing-page"
    >
      {/* Navigation Header */}
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl">
            <Shield className="h-7 w-7 text-primary" />
            <span>Argus Core</span>
          </div>
          
          <nav className="flex items-center gap-4">
            <Link href="/analyze">
              <Button data-testid="nav-analyze-btn">
                Start Analysis
                <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section 
        className="container py-20 md:py-32"
        data-testid="hero-section"
      >
        <div className="mx-auto max-w-4xl text-center space-y-8">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border bg-muted/50 text-sm">
            <Zap className="h-4 w-4 text-yellow-500" />
            <span>Advanced AI-Powered Detection</span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
            Multi-Modal{' '}
            <span className="text-primary">Deepfake Detection</span>{' '}
            Platform
          </h1>

          {/* Subheadline */}
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Analyze videos, audio, and images for synthetic media manipulation 
            using state-of-the-art AI models. Get comprehensive authenticity 
            scores with detailed forensic reports.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Link href="/analyze">
              <Button 
                size="lg" 
                className="text-lg px-8 h-14 gap-2"
                data-testid="cta-button"
              >
                <Shield className="h-5 w-5" />
                Analyze Media Now
                <ArrowRight className="h-5 w-5" />
              </Button>
            </Link>
            <Button 
              variant="outline" 
              size="lg" 
              className="text-lg px-8 h-14"
              asChild
            >
              <a href="#features">Learn More</a>
            </Button>
          </div>

          {/* Trust indicators */}
          <div className="flex flex-wrap items-center justify-center gap-6 pt-8 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <Lock className="h-4 w-4 text-green-500" />
              <span>Secure Processing</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-yellow-500" />
              <span>Fast Analysis</span>
            </div>
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-blue-500" />
              <span>Detailed Reports</span>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="container py-16 md:py-24">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Comprehensive Detection Capabilities
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Argus Core combines multiple detection modalities to provide 
            the most accurate deepfake analysis available.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((feature) => (
            <Card 
              key={feature.title}
              className="group hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
            >
              <CardHeader>
                <div className={`p-3 rounded-lg ${feature.bgColor} w-fit mb-2`}>
                  <feature.icon className={`h-6 w-6 ${feature.color}`} />
                </div>
                <CardTitle className="text-xl">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base">
                  {feature.description}
                </CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Benefits Section */}
      <section className="container py-16 md:py-24">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Why Choose Argus Core?
            </h2>
            <p className="text-muted-foreground text-lg mb-8">
              Our platform provides enterprise-grade deepfake detection with 
              advanced features designed for accuracy and reliability.
            </p>
            
            <ul className="space-y-4">
              {BENEFITS.map((benefit) => (
                <li key={benefit} className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                  <span>{benefit}</span>
                </li>
              ))}
            </ul>

            <div className="mt-8">
              <Link href="/analyze">
                <Button size="lg" className="gap-2">
                  Get Started
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>

          {/* Stats Card */}
          <Card className="bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20">
            <CardContent className="p-8">
              <div className="grid grid-cols-2 gap-8">
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary">500MB</div>
                  <div className="text-sm text-muted-foreground mt-1">Max File Size</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary">5 min</div>
                  <div className="text-sm text-muted-foreground mt-1">Max Video Duration</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary">4+</div>
                  <div className="text-sm text-muted-foreground mt-1">Detection Models</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary">Real-time</div>
                  <div className="text-sm text-muted-foreground mt-1">Progress Updates</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container py-16 md:py-24">
        <Card className="bg-gradient-to-r from-primary/10 via-primary/5 to-background border-primary/20">
          <CardContent className="p-8 md:p-12 text-center">
            <h2 className="text-2xl md:text-3xl font-bold mb-4">
              Ready to Detect Deepfakes?
            </h2>
            <p className="text-muted-foreground mb-8 max-w-xl mx-auto">
              Upload your media file and get comprehensive analysis results in seconds.
              Our advanced AI models will analyze video, audio, and metadata for manipulation indicators.
            </p>
            <Link href="/analyze">
              <Button size="lg" className="text-lg px-8 h-14 gap-2">
                <Shield className="h-5 w-5" />
                Start Free Analysis
                <ArrowRight className="h-5 w-5" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      </section>

      {/* Footer */}
      <footer className="border-t">
        <div className="container py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 font-semibold">
              <Shield className="h-5 w-5 text-primary" />
              <span>Argus Core</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Multi-Modal Deepfake Detection Platform • Powered by Advanced AI
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
