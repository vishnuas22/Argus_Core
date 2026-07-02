/**
 * Argus Core - File Validation Hook
 * ==================================
 * Client-side file validation before upload.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - hooks/useFileValidation.ts
 * 
 * Role: Validate files client-side before upload. Check size, type, and generate preview.
 * Provides early feedback to users without server round-trip.
 * 
 * Integration:
 * - Used by: UploadZone.tsx
 * - Updates: uploadStore via setValidation
 * - Constants: lib/constants.ts for limits
 */

import { useMemo, useCallback, useEffect, useState } from 'react';
import { 
  MAX_FILE_SIZE_BYTES, 
  MAX_FILE_SIZE_MB,
  ALL_ACCEPTED_TYPES,
  ALL_ACCEPTED_EXTENSIONS,
  ACCEPTED_FILE_TYPES,
} from '@/lib/constants';
import type { 
  ValidationResult, 
  ValidationError, 
  ValidationWarning, 
  FileInfo 
} from '@/types/analysis';
import { getMimeCategory, getFileExtension } from '@/lib/utils';

// ============== TYPES ==============

export interface ValidationConfig {
  maxSizeMB?: number;
  acceptedTypes?: string[];
  acceptedExtensions?: string[];
  validateMagicBytes?: boolean;
}

export interface UseFileValidationReturn extends ValidationResult {
  validate: (file: File) => Promise<ValidationResult>;
  generatePreview: (file: File) => Promise<string | null>;
  revokePreview: (preview: string) => void;
}

// ============== MAGIC BYTES ==============

/**
 * File signature (magic bytes) definitions
 */
const FILE_SIGNATURES: Record<string, { bytes: number[]; offset?: number }[]> = {
  // Images
  'image/jpeg': [{ bytes: [0xFF, 0xD8, 0xFF] }],
  'image/png': [{ bytes: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] }],
  'image/webp': [{ bytes: [0x52, 0x49, 0x46, 0x46], offset: 0 }], // "RIFF"
  
  // Videos
  'video/mp4': [
    { bytes: [0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70] }, // ftyp
    { bytes: [0x00, 0x00, 0x00, 0x1C, 0x66, 0x74, 0x79, 0x70] }, // ftyp variant
    { bytes: [0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70] }, // ftyp variant
  ],
  'video/webm': [{ bytes: [0x1A, 0x45, 0xDF, 0xA3] }],
  'video/quicktime': [
    { bytes: [0x00, 0x00, 0x00, 0x14, 0x66, 0x74, 0x79, 0x70, 0x71, 0x74] },
  ],
  
  // Audio
  'audio/mpeg': [
    { bytes: [0xFF, 0xFB] }, // MP3 frame sync
    { bytes: [0xFF, 0xFA] }, // MP3 frame sync variant
    { bytes: [0x49, 0x44, 0x33] }, // ID3 tag
  ],
  'audio/wav': [{ bytes: [0x52, 0x49, 0x46, 0x46] }], // "RIFF"
  'audio/ogg': [{ bytes: [0x4F, 0x67, 0x67, 0x53] }], // "OggS"
};

// ============== VALIDATION FUNCTIONS ==============

/**
 * Validate file size
 */
function validateFileSize(
  file: File, 
  maxSizeMB: number
): { error?: ValidationError; warning?: ValidationWarning } {
  const maxBytes = maxSizeMB * 1024 * 1024;
  
  if (file.size > maxBytes) {
    return {
      error: {
        field: 'size',
        message: `File size (${formatBytes(file.size)}) exceeds ${maxSizeMB}MB limit`,
      },
    };
  }
  
  // Warning for large files
  if (file.size > maxBytes * 0.8) {
    return {
      warning: {
        field: 'size',
        message: `Large file (${formatBytes(file.size)}) may take longer to process`,
      },
    };
  }
  
  return {};
}

/**
 * Validate file type by MIME and extension
 */
function validateFileType(
  file: File,
  acceptedTypes: string[],
  acceptedExtensions: string[]
): { error?: ValidationError; warning?: ValidationWarning } {
  const extension = getFileExtension(file.name);
  const mimeType = file.type;
  
  // Check MIME type
  const isMimeValid = acceptedTypes.some(type => {
    if (type.endsWith('/*')) {
      return mimeType.startsWith(type.slice(0, -1));
    }
    return mimeType === type;
  });
  
  // Check extension
  const isExtValid = acceptedExtensions.some(
    ext => ext.toLowerCase() === `.${extension}`.toLowerCase()
  );
  
  if (!isMimeValid && !isExtValid) {
    return {
      error: {
        field: 'type',
        message: `Unsupported file type "${extension || mimeType}". Accepted: video, audio, image`,
      },
    };
  }
  
  // Warning if MIME doesn't match extension
  if (isMimeValid !== isExtValid) {
    return {
      warning: {
        field: 'type',
        message: 'File extension may not match its content type',
      },
    };
  }
  
  return {};
}

/**
 * Validate magic bytes (file signature)
 */
async function validateMagicBytes(file: File): Promise<{ 
  error?: ValidationError; 
  warning?: ValidationWarning 
}> {
  try {
    const buffer = await file.slice(0, 32).arrayBuffer();
    const bytes = new Uint8Array(buffer);
    
    const signatures = FILE_SIGNATURES[file.type];
    if (!signatures) {
      // No known signature for this type, skip validation
      return {};
    }
    
    const isValid = signatures.some(sig => {
      const offset = sig.offset ?? 0;
      return sig.bytes.every((byte, i) => bytes[offset + i] === byte);
    });
    
    if (!isValid) {
      return {
        warning: {
          field: 'content',
          message: 'File content may not match its declared type',
        },
      };
    }
    
    return {};
  } catch {
    // If we can't read the file, skip magic byte validation
    return {};
  }
}

/**
 * Format bytes to human readable string
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

// ============== HOOK ==============

/**
 * useFileValidation hook
 * 
 * Validates files and generates previews client-side
 * 
 * @param config - Optional validation configuration
 * @returns Validation result and utility functions
 */
export function useFileValidation(
  config?: ValidationConfig
): UseFileValidationReturn {
  // Default config
  const maxSizeMB = config?.maxSizeMB ?? MAX_FILE_SIZE_MB;
  const acceptedTypes = config?.acceptedTypes ?? ALL_ACCEPTED_TYPES;
  const acceptedExtensions = config?.acceptedExtensions ?? ALL_ACCEPTED_EXTENSIONS;
  const shouldValidateMagicBytes = config?.validateMagicBytes ?? true;

  // State for current validation result
  const [validationResult, setValidationResult] = useState<ValidationResult>({
    isValid: false,
    errors: [],
    warnings: [],
    fileInfo: null,
  });

  /**
   * Validate a file
   */
  const validate = useCallback(async (file: File): Promise<ValidationResult> => {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Size validation
    const sizeResult = validateFileSize(file, maxSizeMB);
    if (sizeResult.error) errors.push(sizeResult.error);
    if (sizeResult.warning) warnings.push(sizeResult.warning);

    // Type validation
    const typeResult = validateFileType(file, acceptedTypes, acceptedExtensions);
    if (typeResult.error) errors.push(typeResult.error);
    if (typeResult.warning) warnings.push(typeResult.warning);

    // Magic bytes validation (if enabled and no type errors)
    if (shouldValidateMagicBytes && !typeResult.error) {
      const magicResult = await validateMagicBytes(file);
      if (magicResult.error) errors.push(magicResult.error);
      if (magicResult.warning) warnings.push(magicResult.warning);
    }

    // Build file info
    const extension = getFileExtension(file.name);
    const fileInfo: FileInfo = {
      name: file.name,
      size: file.size,
      type: file.type,
      extension,
    };

    // Add duration warning for large videos
    if (file.type.startsWith('video/') && file.size > 100 * 1024 * 1024) {
      warnings.push({
        field: 'duration',
        message: 'Large video may take several minutes to analyze',
      });
    }

    const result: ValidationResult = {
      isValid: errors.length === 0,
      errors,
      warnings,
      fileInfo,
    };

    setValidationResult(result);
    return result;
  }, [maxSizeMB, acceptedTypes, acceptedExtensions, shouldValidateMagicBytes]);

  /**
   * Generate preview URL for file
   */
  const generatePreview = useCallback(async (file: File): Promise<string | null> => {
    const category = getMimeCategory(file.type);
    
    if (category === 'image') {
      // For images, create object URL directly
      return URL.createObjectURL(file);
    }
    
    if (category === 'video') {
      // For videos, create object URL (browser will show first frame or video player)
      return URL.createObjectURL(file);
    }
    
    if (category === 'audio') {
      // For audio, no visual preview - return null
      // UI can show audio icon instead
      return null;
    }
    
    return null;
  }, []);

  /**
   * Revoke preview URL to free memory
   */
  const revokePreview = useCallback((preview: string) => {
    if (preview && preview.startsWith('blob:')) {
      URL.revokeObjectURL(preview);
    }
  }, []);

  return {
    ...validationResult,
    validate,
    generatePreview,
    revokePreview,
  };
}

// ============== STANDALONE VALIDATION ==============

/**
 * Standalone validation function (for non-hook contexts)
 */
export async function validateFile(
  file: File,
  config?: ValidationConfig
): Promise<ValidationResult> {
  const maxSizeMB = config?.maxSizeMB ?? MAX_FILE_SIZE_MB;
  const acceptedTypes = config?.acceptedTypes ?? ALL_ACCEPTED_TYPES;
  const acceptedExtensions = config?.acceptedExtensions ?? ALL_ACCEPTED_EXTENSIONS;

  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  // Size validation
  const sizeResult = validateFileSize(file, maxSizeMB);
  if (sizeResult.error) errors.push(sizeResult.error);
  if (sizeResult.warning) warnings.push(sizeResult.warning);

  // Type validation
  const typeResult = validateFileType(file, acceptedTypes, acceptedExtensions);
  if (typeResult.error) errors.push(typeResult.error);
  if (typeResult.warning) warnings.push(typeResult.warning);

  // Build file info
  const extension = getFileExtension(file.name);
  const fileInfo: FileInfo = {
    name: file.name,
    size: file.size,
    type: file.type,
    extension,
  };

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    fileInfo,
  };
}

export default useFileValidation;
