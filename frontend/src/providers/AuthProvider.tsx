/**
 * Argus Core - Authentication Provider
 * ======================================
 * Initializes anonymous JWT authentication on app load.
 *
 * Role: Ensure all API requests include valid JWT tokens.
 *
 * Integration:
 * - Imports: hooks/useAuth.ts
 * - Used by: app/layout.tsx
 * - Backend: POST /api/v1/auth/anonymous
 */

'use client';

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { api } from '@/services/api';
import { STORAGE_KEYS } from '@/lib/constants';

// ============== TYPES ==============

interface AuthContextType {
  isAuthenticated: boolean;
  isAuthenticating: boolean;
  userId: string | null;
  error: string | null;
}

interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
  user_id: string;
}

// ============== CONTEXT ==============

const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  isAuthenticating: true,
  userId: null,
  error: null,
});

export function useAuthContext() {
  return useContext(AuthContext);
}

// ============== PROVIDER ==============

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAuthenticating, setIsAuthenticating] = useState(true);
  const [userId, setUserId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const getAnonymousToken = useCallback(async (): Promise<boolean> => {
    try {
      const response = await api.post<AuthToken>('/api/v1/auth/anonymous');
      const token = response.data;
      
      localStorage.setItem(STORAGE_KEYS.authToken, token.access_token);
      setUserId(token.user_id);
      setIsAuthenticated(true);
      setError(null);
      
      return true;
    } catch (err) {
      console.error('Failed to get anonymous token:', err);
      setError('Failed to authenticate');
      return false;
    }
  }, []);

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
              // Token is valid and not about to expire
              setUserId(payload.sub);
              setIsAuthenticated(true);
              setIsAuthenticating(false);
              return;
            }
          }
        } catch {
          // Invalid token format
        }
      }

      // Get new anonymous token
      await getAnonymousToken();
      setIsAuthenticating(false);
    };

    initializeAuth();
  }, [getAnonymousToken]);

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        isAuthenticating,
        userId,
        error,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export default AuthProvider;
