/**
 * Argus Core - Authentication Hook
 * ==================================
 * Auto-authenticates with anonymous token on first load.
 *
 * Role: Manage JWT token lifecycle for API requests.
 *
 * Integration:
 * - Imports: services/api.ts, lib/constants.ts
 * - Used by: App layout
 * - Backend: POST /api/v1/auth/anonymous
 */

import { useEffect, useState, useCallback } from 'react';
import { STORAGE_KEYS } from '@/lib/constants';
import { api } from '@/services/api';

// ============== TYPES ==============

interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
  user_id: string;
}

// ============== HOOK ==============

/**
 * Hook to manage authentication state.
 *
 * Auto-authenticates with anonymous token on first load if no token exists.
 * Provides token state and refresh function.
 *
 * @returns Authentication state and actions
 */
export function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAuthenticating, setIsAuthenticating] = useState(true);
  const [userId, setUserId] = useState<string | null>(null);

  /**
   * Get anonymous token from backend
   */
  const getAnonymousToken = useCallback(async (): Promise<AuthToken | null> => {
    try {
      const response = await api.post<AuthToken>('/api/v1/auth/anonymous');
      return response.data;
    } catch (error) {
      console.error('Failed to get anonymous token:', error);
      return null;
    }
  }, []);

  /**
   * Refresh existing token
   */
  const refreshToken = useCallback(async (): Promise<boolean> => {
    try {
      const currentToken = localStorage.getItem(STORAGE_KEYS.authToken);
      if (!currentToken) return false;

      const response = await api.post<AuthToken>('/api/v1/auth/refresh', null, {
        headers: { Authorization: `Bearer ${currentToken}` },
      });

      const token = response.data;
      localStorage.setItem(STORAGE_KEYS.authToken, token.access_token);
      setUserId(token.user_id);
      setIsAuthenticated(true);
      return true;
    } catch (error) {
      console.error('Failed to refresh token:', error);
      // Clear invalid token
      localStorage.removeItem(STORAGE_KEYS.authToken);
      setIsAuthenticated(false);
      setUserId(null);
      return false;
    }
  }, []);

  /**
   * Initialize authentication
   */
  useEffect(() => {
    const initializeAuth = async () => {
      if (typeof window === 'undefined') {
        setIsAuthenticating(false);
        return;
      }

      const existingToken = localStorage.getItem(STORAGE_KEYS.authToken);

      if (existingToken) {
        // Validate existing token
        try {
          const parts = existingToken.split('.');
          if (parts.length === 3) {
            const payload = JSON.parse(atob(parts[1]));
            const expiry = payload.exp * 1000;
            
            if (expiry > Date.now() + 60000) {
              // Token valid and not about to expire
              setUserId(payload.sub);
              setIsAuthenticated(true);
              setIsAuthenticating(false);
              return;
            }
          }
        } catch {
          // Invalid token format, get new one
        }
      }

      // Get new anonymous token
      const token = await getAnonymousToken();
      if (token) {
        localStorage.setItem(STORAGE_KEYS.authToken, token.access_token);
        setUserId(token.user_id);
        setIsAuthenticated(true);
      }

      setIsAuthenticating(false);
    };

    initializeAuth();
  }, [getAnonymousToken]);

  return {
    isAuthenticated,
    isAuthenticating,
    userId,
    refreshToken,
  };
}

export default useAuth;
