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

declare module 'axios' {
  interface InternalAxiosRequestConfig {
    _retry?: boolean;
  }
}

/**
 * Create configured axios instance
 */
export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
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
let _isRefreshing = false;
let _refreshSubscribers: Array<(token: string) => void> = [];

function _onRefreshed(token: string) {
  _refreshSubscribers.forEach(cb => cb(token));
  _refreshSubscribers = [];
}

function _addRefreshSubscriber(cb: (token: string) => void) {
  _refreshSubscribers.push(cb);
}

async function _refreshToken(): Promise<string> {
  const { data } = await axios.post<{ access_token: string }>(
    `${API_BASE_URL}/api/v1/auth/anonymous`,
    { display_name: 'session' }
  );
  localStorage.setItem(STORAGE_KEYS.authToken, data.access_token);
  return data.access_token;
}

function _tryRefreshToken(): Promise<string> {
  _isRefreshing = true;
  return _refreshToken().finally(() => { _isRefreshing = false; });
}

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<ErrorResponse>) => {
    // Handle specific error status codes
    if (error.response) {
      const { status, data, config: failedConfig } = error.response;
      
      // Retry on 401 with a fresh token
      if (status === 401 && failedConfig && !failedConfig._retry) {
        if (typeof window !== 'undefined') {
          failedConfig._retry = true;
          localStorage.removeItem(STORAGE_KEYS.authToken);
          
          try {
            const newToken = _isRefreshing
              ? await new Promise<string>(resolve => _addRefreshSubscriber(resolve))
              : await _tryRefreshToken();
            
            if (failedConfig.headers) {
              failedConfig.headers.Authorization = `Bearer ${newToken}`;
            }
            return api(failedConfig);
          } catch {
            window.dispatchEvent(new CustomEvent('auth:unauthorized'));
          }
        }
        return Promise.reject(error);
      }
      
      switch (status) {
        case 400:
          // Bad request - validation error
          console.error('Bad request:', data?.message, data?.error_code);
          break;
          
        case 403:
          // Forbidden
          console.error('Access forbidden:', data?.message);
          break;
          
        case 404:
          // Not found
          console.error('Resource not found:', data?.message);
          break;

        case 413:
          // Request entity too large
          console.error('File too large:', data?.message);
          break;
          
        case 422:
          // Unprocessable entity - validation error
          console.error('Validation error:', data?.message, data?.error_code);
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
