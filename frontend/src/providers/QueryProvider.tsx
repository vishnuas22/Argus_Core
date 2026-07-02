'use client';

/**
 * Argus Core - Query Provider
 * ===========================
 * TanStack Query provider for data fetching and caching.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - providers/QueryProvider.tsx
 * 
 * Role: Initialize TanStack Query client with optimized configuration.
 * 
 * Integration:
 * - Used by: app/layout.tsx
 * - Enables: useQuery, useMutation hooks throughout app
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { useState, type ReactNode } from 'react';

interface QueryProviderProps {
  children: ReactNode;
}

/**
 * QueryProvider component wrapping TanStack Query functionality
 */
export function QueryProvider({ children }: QueryProviderProps) {
  // Create QueryClient inside component to prevent sharing between requests (SSR safety)
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Stale time: 30 seconds - data considered fresh
            staleTime: 30 * 1000,
            
            // Cache time: 5 minutes - keep data in cache
            gcTime: 5 * 60 * 1000,
            
            // Retry configuration
            retry: (failureCount, error) => {
              // Don't retry on 4xx errors
              if (error && typeof error === 'object' && 'response' in error) {
                const response = (error as { response?: { status?: number } }).response;
                if (response?.status && response.status >= 400 && response.status < 500) {
                  return false;
                }
              }
              return failureCount < 3;
            },
            
            // Refetch on window focus (useful for analysis status updates)
            refetchOnWindowFocus: true,
            
            // Don't refetch on mount if data is fresh
            refetchOnMount: true,
          },
          mutations: {
            // Retry mutations once on failure
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {/* DevTools only in development */}
      {process.env.NODE_ENV === 'development' && (
        <ReactQueryDevtools initialIsOpen={false} position="bottom" />
      )}
    </QueryClientProvider>
  );
}

export default QueryProvider;
