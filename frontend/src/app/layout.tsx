/**
 * Argus Core - Root Layout
 * ========================
 * Root layout wrapping all pages with providers, fonts, and metadata.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - app/layout.tsx
 * 
 * Role: Initialize providers, fonts, metadata. Wrap all pages with
 * QueryProvider for data fetching and ThemeProvider for dark mode.
 * 
 * Integration:
 * - Imports: providers/QueryProvider, providers/ThemeProvider
 * - Inputs: children: React.ReactNode
 * - Outputs: Complete HTML document structure
 */

import './globals.css';
import type { Metadata, Viewport } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { QueryProvider } from '@/providers/QueryProvider';
import { ThemeProvider } from '@/providers/ThemeProvider';

// Primary font for UI text
const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

// Monospace font for code/technical content
const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

// Metadata for SEO and social sharing
export const metadata: Metadata = {
  title: {
    default: 'Argus Core',
    template: '%s | Argus Core',
  },
  description: 'Multi-Modal Deepfake Detection Platform - Analyze videos, audio, and images for synthetic media manipulation using advanced AI.',
  keywords: ['deepfake detection', 'media verification', 'AI analysis', 'synthetic media', 'authenticity'],
  authors: [{ name: 'Argus Core Team' }],
  creator: 'Argus Core',
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3000'),
  openGraph: {
    type: 'website',
    locale: 'en_US',
    siteName: 'Argus Core',
    title: 'Argus Core - Deepfake Detection Platform',
    description: 'Analyze videos, audio, and images for synthetic media manipulation using advanced AI.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Argus Core - Deepfake Detection Platform',
    description: 'Analyze videos, audio, and images for synthetic media manipulation using advanced AI.',
  },
  robots: {
    index: true,
    follow: true,
  },
};

// Viewport configuration
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' },
  ],
};

/**
 * RootLayout component
 * Wraps all pages with necessary providers and configuration
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}): JSX.Element {
  return (
    <html 
      lang="en" 
      className={`${inter.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-background font-sans antialiased">
        <ThemeProvider defaultTheme="system">
          <QueryProvider>
            {/* Main application content */}
            <div className="relative flex min-h-screen flex-col">
              {children}
            </div>
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
