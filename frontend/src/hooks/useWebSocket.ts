/**
 * Argus Core - WebSocket Hook
 * ===========================
 * WebSocket connection management for real-time analysis progress updates.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - hooks/useWebSocket.ts
 * 
 * Role: Manage WebSocket connection lifecycle for real-time updates.
 * Handles connection, reconnection, message parsing, and store updates.
 * 
 * Integration:
 * - Imports: store/progressStore for state updates
 * - Backend: /ws/analysis/{id} WebSocket endpoint
 * - Used by: analysis/[id]/page.tsx, ProgressIndicator.tsx
 * 
 * WebSocket Message Types:
 * - status: Current analysis status (sent on connect)
 * - progress: Progress updates during analysis
 * - completed: Final results with trust_score and verdict
 * - error: Error messages
 * - ping/pong: Keep-alive messages
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useProgressStore } from '@/store/progressStore';
import { analysisKeys } from '@/hooks/useAnalysis';
import type { AnalysisStatus, TrustScore, Verdict } from '@/types/analysis';

// ============== TYPES ==============

/**
 * WebSocket message types from backend
 */
export type WebSocketMessageType = 
  | 'status'
  | 'progress'
  | 'completed'
  | 'error'
  | 'ping'
  | 'pong'
  | 'subscribed'
  | 'unsubscribed';

/**
 * Incoming WebSocket message structure
 */
export interface WebSocketMessage {
  type: WebSocketMessageType;
  analysis_id?: string;
  status?: AnalysisStatus;
  progress_percent?: number;
  current_stage?: string;
  message?: string;
  timestamp?: string;
  trust_score?: TrustScore;
  verdict?: Verdict;
  report_url?: string;
  error_code?: string;
}

/**
 * Outgoing client message types
 */
export interface ClientMessage {
  type: 'ping' | 'subscribe' | 'unsubscribe' | 'refresh';
  analysis_id?: string;
}

/**
 * Hook return type
 */
export interface UseWebSocketReturn {
  /** Whether WebSocket is connected */
  isConnected: boolean;
  /** Last connection error */
  error: Error | null;
  /** Send a message to the server */
  send: (message: ClientMessage) => void;
  /** Manually reconnect */
  reconnect: () => void;
  /** Close connection */
  disconnect: () => void;
}

/**
 * Hook options
 */
export interface UseWebSocketOptions {
  /** Auto-connect on mount (default: true) */
  autoConnect?: boolean;
  /** Reconnect on close (default: true) */
  autoReconnect?: boolean;
  /** Max reconnect attempts (default: 5) */
  maxReconnectAttempts?: number;
  /** Reconnect delay in ms (default: 3000) */
  reconnectDelay?: number;
  /** Ping interval in ms (default: 30000) */
  pingInterval?: number;
  /** Connection timeout in ms (default: 10000) */
  connectionTimeout?: number;
}

// ============== CONSTANTS ==============

const DEFAULT_OPTIONS: Required<UseWebSocketOptions> = {
  autoConnect: true,
  autoReconnect: true,
  maxReconnectAttempts: 5,
  reconnectDelay: 3000,
  pingInterval: 30000,
  connectionTimeout: 10000,
};

/**
 * Get WebSocket URL from environment
 */
function getWebSocketUrl(analysisId: string): string {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || '';
  
  // Handle relative URL - construct from window.location
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    
    // If WS_URL is a path (starts with /), use current host
    if (wsUrl.startsWith('/')) {
      return `${protocol}//${host}${wsUrl}/ws/analysis/${analysisId}`;
    }
    
    // If WS_URL is empty, use current host with /api prefix
    if (!wsUrl) {
      return `${protocol}//${host}/api/ws/analysis/${analysisId}`;
    }
  }
  
  // Full URL provided
  return `${wsUrl}/ws/analysis/${analysisId}`;
}

// ============== HOOK ==============

/**
 * useWebSocket Hook
 * 
 * Manages WebSocket connection for real-time analysis progress updates.
 * Automatically handles connection lifecycle, reconnection, and message parsing.
 * 
 * @param analysisId - Analysis ID to subscribe to
 * @param options - Connection options
 * @returns WebSocket state and control functions
 * 
 * @example
 * ```tsx
 * const { isConnected, error } = useWebSocket(analysisId);
 * 
 * // Progress updates are automatically stored in progressStore
 * const progress = useProgressStore((s) => s.progress[analysisId]);
 * ```
 */
export function useWebSocket(
  analysisId: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  // Merge options with defaults
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const connectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Store actions
  const setProgress = useProgressStore((state) => state.setProgress);
  const setConnectionState = useProgressStore((state) => state.setConnectionState);
  const clearConnectionState = useProgressStore((state) => state.clearConnectionState);
  
  // Query client for cache invalidation
  const queryClient = useQueryClient();

  // ============== MESSAGE HANDLER ==============

  /**
   * Handle incoming WebSocket messages
   */
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data: WebSocketMessage = JSON.parse(event.data);
      
      switch (data.type) {
        case 'status':
        case 'progress':
          // Update progress store with current status
          setProgress(analysisId, {
            status: data.status,
            progressPercent: data.progress_percent ?? 0,
            currentStage: data.current_stage ?? data.status ?? 'pending',
            message: data.message,
            timestamp: data.timestamp ?? new Date().toISOString(),
          });
          break;
          
        case 'completed':
          // Update progress store with final results
          setProgress(analysisId, {
            status: 'completed',
            progressPercent: 100,
            currentStage: 'completed',
            message: data.message ?? 'Analysis complete',
            timestamp: data.timestamp ?? new Date().toISOString(),
            trustScore: data.trust_score,
            verdict: data.verdict,
            reportUrl: data.report_url,
          });
          
          // Invalidate analysis query to fetch fresh data
          queryClient.invalidateQueries({ 
            queryKey: analysisKeys.detail(analysisId) 
          });
          queryClient.invalidateQueries({ 
            queryKey: analysisKeys.lists() 
          });
          break;
          
        case 'error':
          // Update progress store with error
          setProgress(analysisId, {
            status: 'failed',
            progressPercent: 0,
            currentStage: 'failed',
            message: data.message,
            timestamp: data.timestamp ?? new Date().toISOString(),
            errorCode: data.error_code,
            errorMessage: data.message,
          });
          break;
          
        case 'pong':
          // Update last ping timestamp
          setConnectionState(analysisId, {
            lastPing: data.timestamp ?? new Date().toISOString(),
          });
          break;
          
        case 'subscribed':
        case 'unsubscribed':
          // Acknowledgment messages - no action needed
          break;
          
        default:
          console.warn(`Unknown WebSocket message type: ${data.type}`);
      }
    } catch (err) {
      console.error('Failed to parse WebSocket message:', err);
    }
  }, [analysisId, setProgress, setConnectionState, queryClient]);

  // ============== CONNECTION MANAGEMENT ==============

  /**
   * Clear all timeouts
   */
  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }
  }, []);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    // Don't connect if already connected
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    
    // Don't connect during SSR
    if (typeof window === 'undefined') {
      return;
    }
    
    try {
      const url = getWebSocketUrl(analysisId);
      
      // Set connection timeout
      connectionTimeoutRef.current = setTimeout(() => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
          wsRef.current?.close();
          setError(new Error('Connection timeout'));
        }
      }, opts.connectionTimeout);
      
      const ws = new WebSocket(url);
      wsRef.current = ws;
      
      ws.onopen = () => {
        // Clear connection timeout
        if (connectionTimeoutRef.current) {
          clearTimeout(connectionTimeoutRef.current);
          connectionTimeoutRef.current = null;
        }
        
        setIsConnected(true);
        setError(null);
        reconnectAttemptRef.current = 0;
        
        // Update connection state in store
        setConnectionState(analysisId, {
          isConnected: true,
          error: null,
          reconnectAttempts: 0,
        });
        
        // Start ping interval
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, opts.pingInterval);
      };
      
      ws.onmessage = handleMessage;
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError(new Error('WebSocket connection error'));
        setConnectionState(analysisId, {
          error: 'Connection error',
        });
      };
      
      ws.onclose = (event) => {
        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        
        setIsConnected(false);
        setConnectionState(analysisId, {
          isConnected: false,
        });
        
        // Attempt reconnection if enabled and not intentionally closed
        if (
          opts.autoReconnect &&
          event.code !== 1000 && // Normal closure
          reconnectAttemptRef.current < opts.maxReconnectAttempts
        ) {
          reconnectAttemptRef.current += 1;
          
          setConnectionState(analysisId, {
            reconnectAttempts: reconnectAttemptRef.current,
          });
          
          // Exponential backoff
          const delay = opts.reconnectDelay * Math.pow(2, reconnectAttemptRef.current - 1);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, Math.min(delay, 30000)); // Cap at 30 seconds
        } else if (reconnectAttemptRef.current >= opts.maxReconnectAttempts) {
          setError(new Error('Max reconnection attempts reached'));
          setConnectionState(analysisId, {
            error: 'Max reconnection attempts reached',
          });
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError(err instanceof Error ? err : new Error('Failed to connect'));
    }
  }, [
    analysisId,
    opts.autoReconnect,
    opts.maxReconnectAttempts,
    opts.reconnectDelay,
    opts.pingInterval,
    opts.connectionTimeout,
    handleMessage,
    setConnectionState,
  ]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    clearTimeouts();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
    clearConnectionState(analysisId);
  }, [analysisId, clearTimeouts, clearConnectionState]);

  /**
   * Manually reconnect
   */
  const reconnect = useCallback(() => {
    reconnectAttemptRef.current = 0;
    disconnect();
    connect();
  }, [disconnect, connect]);

  /**
   * Send a message to the server
   */
  const send = useCallback((message: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }, []);

  // ============== LIFECYCLE ==============

  // Connect on mount, cleanup on unmount
  useEffect(() => {
    if (opts.autoConnect && analysisId) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [analysisId, opts.autoConnect, connect, disconnect]);

  // ============== RETURN ==============

  return {
    isConnected,
    error,
    send,
    reconnect,
    disconnect,
  };
}

// ============== UTILITY HOOKS ==============

/**
 * Hook to check if analysis is still in progress
 * Returns true if WebSocket is connected and analysis is not complete
 */
export function useIsAnalysisInProgress(analysisId: string): boolean {
  const isConnected = useProgressStore((state) => 
    state.connections[analysisId]?.isConnected ?? false
  );
  const status = useProgressStore((state) => 
    state.progress[analysisId]?.status
  );
  
  return isConnected && status !== 'completed' && status !== 'failed';
}

/**
 * Hook to get current progress percentage
 */
export function useProgressPercent(analysisId: string): number {
  return useProgressStore((state) => 
    state.progress[analysisId]?.progressPercent ?? 0
  );
}

/**
 * Hook to get current stage
 */
export function useCurrentStage(analysisId: string): string {
  return useProgressStore((state) => 
    state.progress[analysisId]?.currentStage ?? 'pending'
  );
}

export default useWebSocket;
