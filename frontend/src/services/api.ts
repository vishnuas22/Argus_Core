/**
 * Argus Core - API Client Configuration
 * ======================================
 * Axios instance with interceptors for API communication.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - services/api.ts
 * 
 * Role: Configured axios instance with interceptors.
 * 
 * Integration:
 * - Used by: analysisApi.ts, systemApi.ts
 * - Connects to: Backend FastAPI server
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { API_BASE_URL, STORAGE_KEYS } from '@/lib/constants';
import type { ErrorResponse } from '@/types/analysis';

/**
 * Create configured axios instance
 */
export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Request interceptor for authentication
 * Adds JWT token to requests if available
 */
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Only access localStorage in browser environment
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem(STORAGE_KEYS.authToken);
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * Response interceptor for error handling
 * Handles common error scenarios
 */
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ErrorResponse>) => {
    // Handle specific error status codes
    if (error.response) {
      const { status, data } = error.response;
      
      switch (status) {
        case 401:
          // Unauthorized - clear token and redirect to login
          if (typeof window !== 'undefined') {
            localStorage.removeItem(STORAGE_KEYS.authToken);
            // Could dispatch auth event here
          }
          break;
          
        case 403:
          // Forbidden
          console.error('Access forbidden:', data?.message);
          break;
          
        case 404:
          // Not found
          console.error('Resource not found:', data?.message);
          break;
          
        case 429:
          // Rate limited
          console.error('Rate limit exceeded:', data?.message);
          break;
          
        case 500:
        case 502:
        case 503:
          // Server error
          console.error('Server error:', data?.message ?? 'Internal server error');
          break;
      }
    } else if (error.request) {
      // Network error - no response received
      console.error('Network error: Unable to reach server');
    }
    
    return Promise.reject(error);
  }
);

/**
 * Helper to check if error is axios error
 */
export function isAxiosError(error: unknown): error is AxiosError<ErrorResponse> {
  return axios.isAxiosError(error);
}

/**
 * Extract error message from axios error
 */
export function getErrorMessage(error: unknown): string {
  if (isAxiosError(error)) {
    return error.response?.data?.message ?? error.message ?? 'An unexpected error occurred';
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
}

/**
 * Extract error code from axios error
 */
export function getErrorCode(error: unknown): string | undefined {
  if (isAxiosError(error)) {
    return error.response?.data?.error_code;
  }
  return undefined;
}

export default api;
