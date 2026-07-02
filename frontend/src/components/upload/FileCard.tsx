'use client';

import { Film, Music, ImageIcon, X, CheckCircle2, AlertCircle, Upload } from 'lucide-react';
import { cn, formatFileSize } from '@/lib/utils';
import type { FileInfo } from '@/types/analysis';

interface FileCardProps {
  file: File;
  preview: string | null;
  onRemove: () => void;
  fileInfo?: FileInfo | null;
  uploadProgress?: number;
  error?: string;
  disabled?: boolean;
  className?: string;
}

function getFileCategory(type: string): string {
  if (type.startsWith('video/')) return 'video';
  if (type.startsWith('audio/')) return 'audio';
  if (type.startsWith('image/')) return 'image';
  return 'unknown';
}

function getExtension(name: string): string {
  const ext = name.split('.').pop()?.toUpperCase() ?? '';
  return ext.slice(0, 4);
}

export function FileCard({
  file,
  preview,
  onRemove,
  fileInfo,
  uploadProgress,
  error,
  disabled = false,
  className,
}: FileCardProps) {
  const displayType = fileInfo?.type || file.type;
  const displayName = fileInfo?.name || file.name;
  const displaySize = fileInfo?.size ?? file.size;
  const displayExt = fileInfo?.extension || getExtension(file.name);
  const category = getFileCategory(displayType);
  const CategoryIcon = category === 'video' ? Film : category === 'audio' ? Music : category === 'image' ? ImageIcon : Upload;
  const ext = displayExt.slice(0, 4).toUpperCase();
  const isUploading = uploadProgress !== undefined && uploadProgress < 100;
  const isComplete = uploadProgress === 100;
  const filename = displayName;

  return (
    <div
      data-testid="file-card"
      className={cn(
        'relative flex items-center gap-3 rounded-lg border p-3 transition-all duration-200',
        error ? 'border-destructive' : 'border-border',
        isComplete && !error && 'border-green-500/50 bg-green-500/5',
        disabled && 'opacity-60 pointer-events-none',
        className
      )}
    >
      {/* Preview */}
      <div
        data-testid="file-card-preview"
        className="relative flex h-14 w-14 shrink-0 items-center justify-center rounded-md bg-muted overflow-hidden"
      >
        {preview && category === 'image' ? (
          <img
            src={preview}
            alt={`Preview of ${displayName}`}
            className="h-full w-full object-cover"
          />
        ) : preview && category === 'video' ? (
          <video
            src={preview}
            muted
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex flex-col items-center gap-0.5">
            <CategoryIcon className="h-5 w-5 text-muted-foreground" strokeWidth={1.5} />
            <span className="text-[10px] font-medium text-muted-foreground leading-none">{category}</span>
          </div>
        )}
        {/* Extension badge */}
        {(!preview || category === 'audio') && (
          <span className="absolute bottom-0.5 right-0.5 rounded bg-background/80 px-1 py-0.5 text-[9px] font-semibold text-foreground leading-none">
            {ext}
          </span>
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0 space-y-1">
        <p
          className="text-sm font-medium truncate"
          title={filename}
        >
          {filename}
        </p>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>{displayType}</span>
          <span className="text-border">|</span>
          <span>{formatFileSize(displaySize)}</span>
        </div>

        {/* Upload progress */}
        {uploadProgress !== undefined && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">
                {isComplete ? 'Upload complete' : `Uploading... ${Math.round(uploadProgress)}%`}
              </span>
              {isComplete && !error && (
                <span className="flex items-center gap-1 text-green-600 font-medium">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Ready for analysis
                </span>
              )}
            </div>
            <div
              role="progressbar"
              aria-label="Upload progress"
              aria-valuenow={Math.round(uploadProgress)}
              aria-valuemin={0}
              aria-valuemax={100}
              className="h-1.5 w-full rounded-full bg-muted overflow-hidden"
            >
              <div
                className={cn(
                  'h-full rounded-full transition-all duration-300',
                  error ? 'bg-destructive' : 'bg-primary'
                )}
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div
            role="alert"
            className="flex items-center gap-1 text-xs text-destructive"
          >
            <AlertCircle className="h-3 w-3" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Remove */}
      <button
        type="button"
        data-testid="file-card-remove"
        aria-label={`Remove ${filename}`}
        onClick={(e) => {
          e.stopPropagation();
          if (!disabled && !isUploading) onRemove();
        }}
        disabled={disabled || isUploading}
        className={cn(
          'flex h-7 w-7 shrink-0 items-center justify-center rounded-md transition-colors',
          'text-muted-foreground hover:bg-muted hover:text-foreground',
          'disabled:pointer-events-none disabled:opacity-50',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
        )}
      >
        <X className="h-4 w-4" strokeWidth={1.5} />
      </button>
    </div>
  );
}

export function FileCardSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'flex items-center gap-3 rounded-lg border border-border p-3 animate-pulse',
        className
      )}
    >
      <div className="h-14 w-14 shrink-0 rounded-md bg-muted" />
      <div className="flex-1 space-y-2">
        <div className="h-4 w-40 rounded bg-muted" />
        <div className="h-3 w-24 rounded bg-muted" />
      </div>
      <div className="h-7 w-7 rounded-md bg-muted" />
    </div>
  );
}

export default FileCard;
