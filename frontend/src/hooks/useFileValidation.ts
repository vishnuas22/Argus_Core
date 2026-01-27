/**
 * Argus Core - File Validation Hook
 * ==================================
 * Client-side file validation with preview generation.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - hooks/useFileValidation.ts
 * 
 * Role: Validate files before upload. Check size, type, magic bytes.
 * Generate previews for supported formats.
 * 
 * Integration:
 * - Used by: components/upload/UploadZone
 * - Imports: lib/constants, types/analysis
 * - Updates: store/uploadStore
 */

import { useCallback, useState } from 'react';
import {
  MAX_FILE_SIZE_BYTES,
  MAX_FILE_SIZE_MB,
  ACCEPTED_FILE_TYPES,
  ALL_ACCEPTED_TYPES,
  MAX_VIDEO_DURATION_SECONDS,
} from '@/lib/constants';
import { getFileExtension, getMimeCategory } from '@/lib/utils';
import type {
  ValidationResult,
  ValidationError,
  ValidationWarning,
  FileInfo,
} from '@/types/analysis';

// ============== TYPES ==============

interface UseFileValidationOptions {
  maxSizeMB?: number;
  acceptedTypes?: string[];
  validateDuration?: boolean;
}

interface UseFileValidationReturn {
  validate: (file: File) => Promise<ValidationResult>;
  isValidating: boolean;
  lastResult: ValidationResult | null;
}

// ============== MAGIC BYTES SIGNATURES ==============

const MAGIC_BYTES: Record<string, number[][]> = {
  // Images
  'image/jpeg': [[0xFF, 0xD8, 0xFF]],
  'image/png': [[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]],
  'image/webp': [[0x52, 0x49, 0x46, 0x46]], // RIFF header
  'image/gif': [[0x47, 0x49, 0x46, 0x38]], // GIF8
  
  // Videos
  'video/mp4': [
    [0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70], // ftyp
    [0x00, 0x00, 0x00, 0x1C, 0x66, 0x74, 0x79, 0x70],
    [0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70],
  ],
  'video/webm': [[0x1A, 0x45, 0xDF, 0xA3]],
  'video/quicktime': [
    [0x00, 0x00, 0x00, 0x14, 0x66, 0x74, 0x79, 0x70],
    [0x00, 0x00, 0x00, 0x08, 0x77, 0x69, 0x64, 0x65],
  ],
  
  // Audio
  'audio/mpeg': [
    [0xFF, 0xFB], // MP3 frame sync
    [0xFF, 0xFA],
    [0xFF, 0xF3],
    [0xFF, 0xF2],
    [0x49, 0x44, 0x33], // ID3 tag
  ],
  'audio/wav': [[0x52, 0x49, 0x46, 0x46]], // RIFF
  'audio/ogg': [[0x4F, 0x67, 0x67, 0x53]], // OggS
  'audio/flac': [[0x66, 0x4C, 0x61, 0x43]], // fLaC
};

// ============== HELPER FUNCTIONS ==============

/**
 * Read first N bytes of file
 */
async function readFileBytes(file: File, numBytes: number): Promise<Uint8Array> {
  const slice = file.slice(0, numBytes);
  const buffer = await slice.arrayBuffer();
  return new Uint8Array(buffer);
}

/**
 * Check if file bytes match any signature for mime type
 */
function matchesMagicBytes(bytes: Uint8Array, mimeType: string): boolean {
  const signatures = MAGIC_BYTES[mimeType];
  if (!signatures) return true; // No signature to check
  
  return signatures.some(signature => {
    if (bytes.length < signature.length) return false;
    return signature.every((byte, index) => bytes[index] === byte);
  });
}

/**
 * Get video duration using HTMLVideoElement
 */
async function getVideoDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(video.src);
      resolve(video.duration);
    };
    
    video.onerror = () => {
      URL.revokeObjectURL(video.src);
      resolve(null);
    };
    
    // Timeout after 10 seconds
    setTimeout(() => {
      URL.revokeObjectURL(video.src);
      resolve(null);
    }, 10000);
    
    video.src = URL.createObjectURL(file);
  });
}

/**
 * Get audio duration using HTMLAudioElement
 */
async function getAudioDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const audio = document.createElement('audio');
    audio.preload = 'metadata';
    
    audio.onloadedmetadata = () => {
      URL.revokeObjectURL(audio.src);
      resolve(audio.duration);
    };
    
    audio.onerror = () => {
      URL.revokeObjectURL(audio.src);
      resolve(null);
    };
    
    // Timeout after 10 seconds
    setTimeout(() => {
      URL.revokeObjectURL(audio.src);
      resolve(null);
    }, 10000);
    
    audio.src = URL.createObjectURL(file);
  });
}

/**
 * Generate preview URL for file
 */
function generatePreview(file: File): string | undefined {
  const category = getMimeCategory(file.type);
  
  // Generate preview for images and videos
  if (category === 'video' || category === 'image') {
    return URL.createObjectURL(file);
  }
  
  return undefined;
}

// ============== HOOK ==============

/**
 * Hook for client-side file validation
 * 
 * @param options - Validation options
 * @returns Validation function and state
 */
export function useFileValidation(
  options: UseFileValidationOptions = {}
): UseFileValidationReturn {
  const {
    maxSizeMB = MAX_FILE_SIZE_MB,
    acceptedTypes = ALL_ACCEPTED_TYPES,
    validateDuration = true,
  } = options;
  
  const [isValidating, setIsValidating] = useState(false);
  const [lastResult, setLastResult] = useState<ValidationResult | null>(null);
  
  const validate = useCallback(async (file: File): Promise<ValidationResult> => {
    setIsValidating(true);
    
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    
    try {
      // 1. Validate file size
      const maxSizeBytes = maxSizeMB * 1024 * 1024;
      if (file.size > maxSizeBytes) {
        errors.push({
          field: 'size',
          message: `File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds maximum of ${maxSizeMB}MB`,
        });
      }
      
      // 2. Validate file type
      const isAcceptedType = acceptedTypes.some(type => {
        if (type.endsWith('/*')) {
          const category = type.replace('/*', '');
          return file.type.startsWith(category);
        }
        return file.type === type;
      });
      
      if (!isAcceptedType) {
        errors.push({
          field: 'type',
          message: `File type "${file.type || 'unknown'}" is not supported`,
        });
      }
      
      // 3. Validate magic bytes (if type is accepted)
      if (isAcceptedType && file.size > 0) {
        const headerBytes = await readFileBytes(file, 32);
        const validMagicBytes = matchesMagicBytes(headerBytes, file.type);
        
        if (!validMagicBytes) {
          warnings.push({
            field: 'integrity',
            message: 'File header does not match declared type - file may be corrupted or renamed',
          });
        }
      }
      
      // 4. Validate duration for video/audio
      const category = getMimeCategory(file.type);
      
      if (validateDuration && errors.length === 0) {
        if (category === 'video') {
          const duration = await getVideoDuration(file);
          if (duration !== null) {
            if (duration > MAX_VIDEO_DURATION_SECONDS) {
              errors.push({
                field: 'duration',
                message: `Video duration (${Math.round(duration)}s) exceeds maximum of ${MAX_VIDEO_DURATION_SECONDS}s (5 minutes)`,
              });
            }
            if (duration < 1) {
              warnings.push({
                field: 'duration',
                message: 'Video is very short - detection accuracy may be limited',
              });
            }
          }
        } else if (category === 'audio') {
          const duration = await getAudioDuration(file);
          if (duration !== null && duration > MAX_VIDEO_DURATION_SECONDS) {
            warnings.push({
              field: 'duration',
              message: `Audio is ${Math.round(duration)}s long - processing may take longer`,
            });
          }
        }
      }
      
      // 5. Generate preview
      const preview = errors.length === 0 ? generatePreview(file) : undefined;
      
      // 6. Build file info
      const fileInfo: FileInfo = {
        name: file.name,
        size: file.size,
        type: file.type,
        extension: getFileExtension(file.name),
        preview,
      };
      
      const result: ValidationResult = {
        isValid: errors.length === 0,
        errors,
        warnings,
        fileInfo,
      };
      
      setLastResult(result);
      return result;
      
    } catch (error) {
      const result: ValidationResult = {
        isValid: false,
        errors: [{
          field: 'unknown',
          message: error instanceof Error ? error.message : 'Validation failed',
        }],
        warnings: [],
        fileInfo: null,
      };
      
      setLastResult(result);
      return result;
      
    } finally {
      setIsValidating(false);
    }
  }, [maxSizeMB, acceptedTypes, validateDuration]);
  
  return {
    validate,
    isValidating,
    lastResult,
  };
}

// ============== UTILITY EXPORTS ==============

/**
 * Quick validation without hook state
 */
export async function validateFile(
  file: File,
  options: UseFileValidationOptions = {}
): Promise<ValidationResult> {
  const hook = { validate: null as unknown as (f: File) => Promise<ValidationResult> };
  
  // Inline validation without hook
  const {
    maxSizeMB = MAX_FILE_SIZE_MB,
    acceptedTypes = ALL_ACCEPTED_TYPES,
  } = options;
  
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  
  // Size check
  const maxSizeBytes = maxSizeMB * 1024 * 1024;
  if (file.size > maxSizeBytes) {
    errors.push({
      field: 'size',
      message: `File exceeds ${maxSizeMB}MB limit`,
    });
  }
  
  // Type check
  const isAcceptedType = acceptedTypes.some(type => {
    if (type.endsWith('/*')) {
      return file.type.startsWith(type.replace('/*', ''));
    }
    return file.type === type;
  });
  
  if (!isAcceptedType) {
    errors.push({
      field: 'type',
      message: `Unsupported file type: ${file.type || 'unknown'}`,
    });
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    fileInfo: errors.length === 0 ? {
      name: file.name,
      size: file.size,
      type: file.type,
      extension: getFileExtension(file.name),
      preview: generatePreview(file),
    } : null,
  };
}

export default useFileValidation;
