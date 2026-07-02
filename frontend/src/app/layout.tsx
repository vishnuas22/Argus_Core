import './globals.css';
import type { Metadata, Viewport } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { QueryProvider } from '@/providers/QueryProvider';
import { ThemeProvider } from '@/providers/ThemeProvider';
import { AuthProvider } from '@/providers/AuthProvider';
import { ErrorBoundary } from '@/components/errors/ErrorBoundary';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains',
  display: 'swap',
});

export const metadata: Metadata = {
  title: {
    default: 'Argus Core',
    template: '%s | Argus Core',
  },
  description:
    'Multi-Modal Deepfake Detection Platform — Analyze videos, audio, and images for synthetic media manipulation using advanced AI.',
  keywords: [
    'deepfake detection',
    'media verification',
    'AI analysis',
    'synthetic media',
    'forensics',
    'authentication',
  ],
  authors: [{ name: 'Argus Core Team' }],
  creator: 'Argus Core',
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3000'),
  openGraph: {
    type: 'website',
    locale: 'en_US',
    siteName: 'Argus Core',
    title: 'Argus Core — Deepfake Detection Platform',
    description:
      'Analyze videos, audio, and images for synthetic media manipulation using advanced AI.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Argus Core — Deepfake Detection Platform',
    description:
      'Analyze videos, audio, and images for synthetic media manipulation using advanced AI.',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0b1120' },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} dark`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-background font-sans antialiased selection:bg-primary/25">
        <div className="fixed inset-0 bg-ambient pointer-events-none z-0" />
        <ErrorBoundary>
          <QueryProvider>
            <AuthProvider>
              <div className="relative z-10">{children}</div>
            </AuthProvider>
          </QueryProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
