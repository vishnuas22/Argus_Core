'use client';

import { useCallback, useRef, useState, useId } from 'react';
import { Upload, Loader2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { MAX_FILE_SIZE_MB, ALL_ACCEPTED_TYPES } from '@/lib/constants';

interface UploadZoneProps {
  onFileSelect?: (file: File) => void;
  disabled?: boolean;
  maxSizeMB?: number;
  acceptedTypes?: string[];
  className?: string;
}

export function UploadZone({
  onFileSelect,
  disabled = false,
  maxSizeMB = MAX_FILE_SIZE_MB,
  acceptedTypes = ALL_ACCEPTED_TYPES,
  className,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const id = useId();

  const acceptedString = acceptedTypes.join(',');

  const processFile = useCallback(async (file: File) => {
    setIsValidating(true);
    setError(null);

    const maxBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxBytes) {
      setError(`File size exceeds ${maxSizeMB}MB limit`);
      setIsValidating(false);
      return;
    }

    const isAccepted = acceptedTypes.some(t => file.type.startsWith(t.split('/')[0]));
    if (!isAccepted) {
      setError('Unsupported file type. Accepted: video, audio, image');
      setIsValidating(false);
      return;
    }

    setIsValidating(false);
    onFileSelect?.(file);
  }, [maxSizeMB, acceptedTypes, onFileSelect]);

  const handleClick = useCallback(() => {
    if (disabled) return;
    inputRef.current?.click();
  }, [disabled]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, [processFile]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (disabled) return;

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  }, [disabled, processFile]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (disabled) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  }, [disabled, handleClick]);

  return (
    <div
      data-testid="upload-zone"
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload file for analysis"
      aria-disabled={disabled}
      onKeyDown={handleKeyDown}
      onClick={handleClick}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className={cn(
        'relative flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-8 text-center transition-all duration-300',
        'hover:border-primary/30 hover:bg-muted/20',
        'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
        isDragging && [
          'border-primary scale-[1.02] bg-primary/5',
          'shadow-lg shadow-primary/10',
        ],
        disabled && 'opacity-50 cursor-not-allowed hover:border-border hover:bg-transparent',
        !disabled && 'cursor-pointer',
        className
      )}
    >
      <input
        ref={inputRef}
        id={id}
        type="file"
        accept={acceptedString}
        onChange={handleFileChange}
        onClick={(e) => { e.stopPropagation(); }}
        className="hidden"
        data-testid="upload-zone-input"
        aria-hidden="true"
        disabled={disabled}
      />

      {isValidating ? (
        <Loader2
          className="h-10 w-10 text-muted-foreground animate-spin"
          strokeWidth={1.5}
        />
      ) : (
        <div className={cn('transition-transform duration-300', isDragging && 'animate-bounce')}>
          <Upload
            className="h-10 w-10 text-muted-foreground"
            strokeWidth={1.5}
          />
        </div>
      )}

      <div className="space-y-1.5">
        {isValidating ? (
          <p className="text-sm font-medium">Validating file...</p>
        ) : isDragging ? (
          <>
            <p className="text-sm font-medium text-primary">Drop your file here</p>
            <p className="text-2xs text-muted-foreground">Release to start upload</p>
          </>
        ) : (
          <>
            <p className="text-sm font-medium">Drop file or click to upload</p>
            <p className="text-2xs text-muted-foreground">
              Maximum file size: {maxSizeMB}MB
            </p>
          </>
        )}
      </div>

      <div className="flex gap-3 text-2xs text-muted-foreground">
        <span>MP4, WebM, MOV</span>
        <span className="text-border">|</span>
        <span>MP3, WAV, OGG</span>
        <span className="text-border">|</span>
        <span>JPG, PNG, WebP</span>
      </div>

      {error && (
        <div
          data-testid="upload-zone-error"
          role="alert"
          className="flex items-center gap-1.5 text-xs text-destructive"
        >
          <AlertCircle className="h-3.5 w-3.5" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

export default UploadZone;
