'use client';

import Link from 'next/link';
import {
  Shield,
  Zap,
  BarChart3,
  ChevronRight,
  ArrowRight,
  Eye,
  Scan,
  Activity,
  Radio,
  FileSearch,
  Layers,
  Microscope,
  Fingerprint,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { useInView } from 'react-intersection-observer';

const FEATURES = [
  {
    icon: Scan,
    title: 'Frame Analysis',
    description:
      'Per-frame deepfake detection using ViT-based neural networks with GradCAM visualizations of manipulated regions.',
    hoverBorder: 'hover:border-[hsl(185_68%_43%_/_0.3)]',
    hoverGlow: 'group-hover:shadow-[0_0_20px_-8px_hsl(185_68%_43%_/_0.15)]',
  },
  {
    icon: Radio,
    title: 'Temporal Forensics',
    description:
      'X-CLIP temporal analysis detecting frame-to-frame inconsistencies and lip-sync manipulation across video sequences.',
    hoverBorder: 'hover:border-[hsl(240_60%_60%_/_0.3)]',
    hoverGlow: 'group-hover:shadow-[0_0_20px_-8px_hsl(240_60%_60%_/_0.12)]',
  },
  {
    icon: Activity,
    title: 'Audio Analysis',
    description:
      'Spectral anomaly detection and voice cloning identification using Wav2Vec2 embeddings and frequency analysis.',
    hoverBorder: 'hover:border-[hsl(35_80%_55%_/_0.3)]',
    hoverGlow: 'group-hover:shadow-[0_0_20px_-8px_hsl(35_80%_55%_/_0.12)]',
  },
  {
    icon: BarChart3,
    title: 'Multi-Modal Fusion',
    description:
      'Confidence-weighted ensemble scoring across all modalities for calibrated, explainable results with maximum accuracy.',
    hoverBorder: 'hover:border-[hsl(155_55%_45%_/_0.3)]',
    hoverGlow: 'group-hover:shadow-[0_0_20px_-8px_hsl(155_55%_45%_/_0.12)]',
  },
  {
    icon: Microscope,
    title: 'Forensic Reports',
    description:
      'Comprehensive PDF forensic reports with evidence chains, methodology documentation, and court-admissible findings.',
    hoverBorder: 'hover:border-[hsl(210_70%_55%_/_0.3)]',
    hoverGlow: 'group-hover:shadow-[0_0_20px_-8px_hsl(210_70%_55%_/_0.12)]',
  },
  {
    icon: Fingerprint,
    title: 'Provenance Analysis',
    description:
      'C2PA content provenance verification, metadata forensics, and manipulation artifact detection at pixel level.',
    hoverBorder: 'hover:border-[hsl(340_65%_50%_/_0.3)]',
    hoverGlow: 'group-hover:shadow-[0_0_20px_-8px_hsl(340_65%_50%_/_0.12)]',
  },
];

const STATS = [
  { value: '500MB', label: 'Max File Size' },
  { value: '10 min', label: 'Max Duration' },
  { value: '6+', label: 'Detection Models' },
  { value: '<5s', label: 'Time to Analyze' },
];

function FadeSection({
  children,
  className,
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  const [ref, inView] = useInView({ threshold: 0.05, triggerOnce: true });

  return (
    <div
      ref={ref}
      className={cn(
        'transition-all duration-700 ease-out',
        inView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6',
        className
      )}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="text-xs text-muted-foreground hover:text-foreground transition-colors duration-200"
    >
      {children}
    </Link>
  );
}

export default function LandingPage() {
  return (
    <div className="min-h-screen" data-testid="landing-page">
      <header className="sticky top-0 z-50 border-b border-border/50 bg-background/70 backdrop-blur-lg">
        <div className="mx-auto max-w-7xl px-6 lg:px-8 flex h-14 items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="p-1.5 rounded-md bg-primary/10 group-hover:bg-primary/15 transition-colors duration-200">
              <Eye className="w-4 h-4 text-primary" strokeWidth={1.5} />
            </div>
            <span className="text-sm font-semibold tracking-tight">
              Argus<span className="text-muted-foreground font-normal">Core</span>
            </span>
          </Link>
          <nav className="flex items-center gap-6">
            <NavLink href="#features">Capabilities</NavLink>
            <Link href="/analyze">
              <Button size="sm" className="gap-1.5 text-xs h-8">
                Launch Analysis
                <ChevronRight className="h-3 w-3" strokeWidth={2} />
              </Button>
            </Link>
          </nav>
        </div>
      </header>

      <section className="relative pt-24 pb-28 md:pt-32 md:pb-40" data-testid="hero-section">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="max-w-4xl mx-auto text-center">
            <FadeSection className="mb-8">
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border border-border bg-muted/30 text-2xs text-muted-foreground font-medium tracking-wide">
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse-soft" />
                Multi-Modal Deepfake Detection
              </div>
            </FadeSection>

            <FadeSection delay={80}>
              <h1 className="text-5xl sm:text-6xl lg:text-7xl font-semibold tracking-tight leading-[1.04] mb-6">
                Detect Synthetic Media
                <br />
                <span className="text-[hsl(185_68%_55%)]">with Forensic Precision</span>
              </h1>
            </FadeSection>

            <FadeSection delay={160}>
              <p className="text-base sm:text-lg text-muted-foreground leading-relaxed max-w-2xl mx-auto mb-10">
                Analyze videos, audio, and images for AI-generated manipulation using
                state-of-the-art multi-modal detection. Every result includes full
                explainability and court-admissible evidence chains.
              </p>
            </FadeSection>

            <FadeSection delay={240}>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mb-12">
                <Link href="/analyze" className="w-full sm:w-auto">
                  <Button className="gap-2 w-full sm:w-auto h-10 px-6" data-testid="cta-button">
                    <Shield className="h-4 w-4" strokeWidth={1.5} />
                    Start Analysis
                    <ArrowRight className="h-4 w-4" strokeWidth={1.5} />
                  </Button>
                </Link>
                <Button variant="outline" className="gap-2 w-full sm:w-auto h-10 px-6" asChild>
                  <a href="#features">
                    <Layers className="h-4 w-4" strokeWidth={1.5} />
                    Explore Capabilities
                  </a>
                </Button>
              </div>
            </FadeSection>

            <FadeSection delay={320}>
              <div className="inline-flex items-center gap-6 text-xs text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <div className="w-1 h-1 rounded-full bg-primary/60" />
                  Secure Processing
                </span>
                <span className="flex items-center gap-1.5">
                  <div className="w-1 h-1 rounded-full bg-primary/60" />
                  Real-Time Analysis
                </span>
                <span className="flex items-center gap-1.5">
                  <div className="w-1 h-1 rounded-full bg-primary/60" />
                  Court-Admissible Reports
                </span>
              </div>
            </FadeSection>
          </div>
        </div>
      </section>

      <section className="border-t border-b border-border/50 bg-muted/10">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4">
            {STATS.map((stat, i) => (
              <div
                key={stat.label}
                className={cn(
                  'py-8 md:py-10 text-center',
                  i < STATS.length - 1 && 'border-r border-border/50'
                )}
              >
                <div className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground mb-1">
                  {stat.value}
                </div>
                <div className="text-xs text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="features" className="py-24 md:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <FadeSection className="text-center mb-16 max-w-3xl mx-auto">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border border-border bg-muted/30 text-2xs text-muted-foreground font-medium tracking-wide mb-5">
              <Layers className="h-3 w-3" strokeWidth={1.5} />
              Detection Capabilities
            </div>
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-3">
              Multi-Modal Detection Engine
            </h2>
            <p className="text-muted-foreground text-base leading-relaxed">
              Each modality is analyzed by specialized deep learning models, then fused into a
              single calibrated trust score with full explainability.
            </p>
          </FadeSection>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {FEATURES.map((feature, i) => (
              <FadeSection key={feature.title} delay={i * 60}>
                <Card
                  className={cn(
                    'group h-full border-border/60 transition-all duration-300 cursor-default',
                    feature.hoverBorder,
                    feature.hoverGlow
                  )}
                >
                  <CardHeader>
                    <div className="p-2.5 rounded-lg bg-primary/5 border border-border/50 w-fit mb-2 group-hover:bg-primary/10 transition-colors duration-200">
                      <feature.icon className="h-5 w-5 text-primary" strokeWidth={1.5} />
                    </div>
                    <CardTitle className="text-sm font-semibold tracking-tight">
                      {feature.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <CardDescription className="text-xs leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              </FadeSection>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24 md:py-28 border-t border-border/50 bg-muted/10">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="max-w-2xl mx-auto text-center">
            <FadeSection>
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border border-border bg-muted/30 text-2xs text-muted-foreground font-medium tracking-wide mb-5">
                <Zap className="h-3 w-3" strokeWidth={1.5} />
                Get Started
              </div>
              <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
                Ready to Detect Deepfakes?
              </h2>
              <p className="text-muted-foreground text-base leading-relaxed mb-8 max-w-lg mx-auto">
                Upload your media file and get comprehensive analysis results in seconds. Our
                advanced AI models analyze video, audio, and metadata for manipulation indicators.
              </p>
              <Link href="/analyze">
                <Button className="gap-2 h-10 px-6">
                  <Eye className="h-4 w-4" strokeWidth={1.5} />
                  Launch Analysis Suite
                  <ArrowRight className="h-4 w-4" strokeWidth={1.5} />
                </Button>
              </Link>
            </FadeSection>
          </div>
        </div>
      </section>

      <footer className="border-t border-border/50">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="py-6 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="p-1 rounded-md bg-primary/10">
                <Eye className="w-3.5 h-3.5 text-primary" strokeWidth={1.5} />
              </div>
              <span className="text-xs font-semibold tracking-tight">
                Argus<span className="text-muted-foreground font-normal">Core</span>
              </span>
              <span className="text-2xs text-muted-foreground/50 mx-2">/</span>
              <span className="text-2xs text-muted-foreground">v1.0</span>
            </div>
            <div className="flex items-center gap-1.5 text-2xs text-muted-foreground">
              <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse-soft" />
              System Online
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
