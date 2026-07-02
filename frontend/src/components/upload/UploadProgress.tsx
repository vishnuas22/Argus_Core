'use client';

import { AlertCircle, CheckCircle2, Loader2, Upload } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useUploadStore } from '@/store/uploadStore';
import type { UploadStatus } from '@/store/uploadStore';

interface UploadProgressProps {
  status: UploadStatus;
  progress: number;
  error: string | null;
  className?: string;
}

const STATUS_CONFIG: Record<UploadStatus, {
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
}> = {
  idle: { label: 'Ready', icon: Upload, color: 'text-muted-foreground' },
  validating: { label: 'Validating...', icon: Loader2, color: 'text-blue-500' },
  uploading: { label: 'Uploading...', icon: Loader2, color: 'text-primary' },
  processing: { label: 'Processing...', icon: Loader2, color: 'text-indigo-500' },
  complete: { label: 'Complete', icon: CheckCircle2, color: 'text-green-500' },
  error: { label: 'Error', icon: AlertCircle, color: 'text-destructive' },
};

export function UploadProgress({
  status,
  progress,
  error,
  className,
}: UploadProgressProps) {
  const config = STATUS_CONFIG[status];
  const Icon = config.icon;
  const isActive = status === 'uploading' || status === 'processing' || status === 'validating';

  return (
    <div
      className={cn('space-y-2 rounded-lg border border-border bg-card p-4', className)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon
            className={cn('h-4 w-4', config.color, isActive && 'animate-spin')}
          />
          <span className="text-sm font-medium">{config.label}</span>
        </div>
        {status !== 'idle' && status !== 'error' && (
          <span className="text-sm text-muted-foreground">{Math.round(progress)}%</span>
        )}
      </div>

      <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full transition-all duration-500 ease-out',
            status === 'error' ? 'bg-destructive' : 'bg-primary'
          )}
          style={{ width: `${progress}%` }}
        />
      </div>

      {error && (
        <div
          role="alert"
          className="flex items-center gap-1.5 text-xs text-destructive"
        >
          <AlertCircle className="h-3 w-3" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

export function ConnectedUploadProgress({ className }: { className?: string }) {
  const status = useUploadStore((s) => s.status);
  const progress = useUploadStore((s) => s.uploadProgress);
  const error = useUploadStore((s) => s.error);

  if (status === 'idle') return null;

  return (
    <UploadProgress
      status={status}
      progress={progress}
      error={error}
      className={className}
    />
  );
}

export default UploadProgress;
