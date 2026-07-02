/**
 * Argus Core - Utility Functions
 * ==============================
 * General utilities used across the application.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - lib/utils.ts
 * 
 * Role: Provide common utility functions. Class name merging, formatters, helpers.
 */

import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { Verdict, AnalysisStatus } from '@/types/analysis';

/**
 * Merge class names with Tailwind conflict resolution
 * Standard shadcn/ui utility
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Format file size to human readable string
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Format duration in seconds to human readable string
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  
  if (minutes < 60) {
    return remainingSeconds > 0
      ? `${minutes}m ${remainingSeconds}s`
      : `${minutes}m`;
  }
  
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  
  return `${hours}h ${remainingMinutes}m`;
}

/**
 * Format date to relative time string
 */
export function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  
  if (diffSeconds < 60) {
    return 'Just now';
  }
  
  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) {
    return `${diffMinutes}m ago`;
  }
  
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) {
    return `${diffHours}h ago`;
  }
  
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) {
    return `${diffDays}d ago`;
  }
  
  return date.toLocaleDateString();
}

/**
 * Format date to ISO string without timezone
 */
export function formatDateTime(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format percentage with specified decimals
 */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${value.toFixed(decimals)}%`;
}

/**
 * Format trust score for display
 */
export function formatTrustScore(score: number): string {
  return Math.round(score).toString();
}

/**
 * Get color class based on verdict
 */
export function getVerdictColor(verdict: Verdict): string {
  const colors: Record<Verdict, string> = {
    authentic: 'text-green-500',
    likely_authentic: 'text-lime-500',
    uncertain: 'text-yellow-500',
    likely_fake: 'text-orange-500',
    fake: 'text-red-500',
  };
  return colors[verdict];
}

/**
 * Get background color class based on verdict
 */
export function getVerdictBgColor(verdict: Verdict): string {
  const colors: Record<Verdict, string> = {
    authentic: 'bg-green-500',
    likely_authentic: 'bg-lime-500',
    uncertain: 'bg-yellow-500',
    likely_fake: 'bg-orange-500',
    fake: 'bg-red-500',
  };
  return colors[verdict];
}

/**
 * Get status color class
 */
export function getStatusColor(status: AnalysisStatus): string {
  const colors: Record<AnalysisStatus, string> = {
    pending: 'text-gray-500',
    preprocessing: 'text-blue-500',
    analyzing: 'text-indigo-500',
    aggregating: 'text-purple-500',
    completed: 'text-green-500',
    failed: 'text-red-500',
  };
  return colors[status];
}

/**
 * Get human readable status label
 */
export function getStatusLabel(status: AnalysisStatus): string {
  const labels: Record<AnalysisStatus, string> = {
    pending: 'Pending',
    preprocessing: 'Preprocessing',
    analyzing: 'Analyzing',
    aggregating: 'Scoring',
    completed: 'Completed',
    failed: 'Failed',
  };
  return labels[status];
}

/**
 * Get human readable verdict label
 */
export function getVerdictLabel(verdict: Verdict): string {
  const labels: Record<Verdict, string> = {
    authentic: 'Authentic',
    likely_authentic: 'Likely Authentic',
    uncertain: 'Uncertain',
    likely_fake: 'Likely Fake',
    fake: 'Fake',
  };
  return labels[verdict];
}

/**
 * Get score color based on value (0-100)
 */
export function getScoreColor(score: number): string {
  if (score >= 80) return 'text-green-500';
  if (score >= 60) return 'text-lime-500';
  if (score >= 40) return 'text-yellow-500';
  if (score >= 20) return 'text-orange-500';
  return 'text-red-500';
}

/**
 * Get score gradient color for D3
 */
export function getScoreGradientColor(score: number): string {
  if (score >= 80) return '#22c55e'; // green-500
  if (score >= 60) return '#84cc16'; // lime-500
  if (score >= 40) return '#eab308'; // yellow-500
  if (score >= 20) return '#f97316'; // orange-500
  return '#ef4444'; // red-500
}

/**
 * Truncate text with ellipsis
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}

/**
 * Generate unique ID
 */
export function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

/**
 * Delay execution
 */
export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Check if value is defined and not null
 */
export function isDefined<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

/**
 * Clamp number between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Get file extension from filename
 */
export function getFileExtension(filename: string): string {
  return filename.split('.').pop()?.toLowerCase() ?? '';
}

/**
 * Get MIME type category
 */
export function getMimeCategory(mimeType: string): 'video' | 'audio' | 'image' | 'text' | 'unknown' {
  if (mimeType.startsWith('video/')) return 'video';
  if (mimeType.startsWith('audio/')) return 'audio';
  if (mimeType.startsWith('image/')) return 'image';
  if (mimeType.startsWith('text/')) return 'text';
  return 'unknown';
}

/**
 * Check if file type is supported
 */
export function isSupportedFileType(mimeType: string): boolean {
  const supportedTypes = [
    'video/mp4',
    'video/webm',
    'video/quicktime',
    'video/x-msvideo',
    'audio/mpeg',
    'audio/wav',
    'audio/ogg',
    'image/jpeg',
    'image/png',
    'image/webp',
  ];
  return supportedTypes.some((type) => mimeType.startsWith(type.split('/')[0]));
}
