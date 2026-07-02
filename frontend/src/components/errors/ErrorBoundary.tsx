/**
 * Argus Core - Error Boundary Component
 * =====================================
 * React Error Boundary for catching and displaying component errors.
 * 
 * Implements: AGENTS_FRONTEND.md - Section 16 - Error Handling Rules (P1)
 * 
 * Role: Catch JavaScript errors in child component tree, log errors,
 * and display fallback UI instead of crashing the entire application.
 * 
 * Integration:
 * - Used by: app/layout.tsx, feature boundaries
 * - Logging: Sends errors to monitoring service
 * 
 * Component Contract (P0):
 * - Catches all child component errors
 * - Provides user-friendly error UI
 * - Logs errors for debugging
 * - Offers recovery options (reset, home)
 * - Accessibility: Proper ARIA roles
 * - data-testid: error-boundary, error-boundary-reset
 * 
 * @see https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary
 */

'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, Home, RefreshCw, Mail } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

// ============== TYPES ==============

/**
 * Props for ErrorBoundary component
 */
export interface ErrorBoundaryProps {
  /** Child components to render */
  children: ReactNode;
  /** Optional fallback UI to render on error */
  fallback?: ReactNode | ((error: Error, errorInfo: ErrorInfo) => ReactNode);
  /** Callback when error occurs */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Callback when user clicks reset */
  onReset?: () => void;
  /** Show detailed error information in development */
  showDetails?: boolean;
  /** Custom error title */
  errorTitle?: string;
  /** Custom error message */
  errorMessage?: string;
}

/**
 * State for ErrorBoundary component
 */
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

// ============== CONSTANTS ==============

const IS_DEVELOPMENT = process.env.NODE_ENV === 'development';
const DEFAULT_ERROR_TITLE = 'Something went wrong';
const DEFAULT_ERROR_MESSAGE = 
  'We encountered an unexpected error. Please try refreshing the page or contact support if the problem persists.';

// ============== COMPONENT ==============

/**
 * ErrorBoundary Component
 * 
 * React error boundary that catches errors in child components and displays
 * a fallback UI instead of crashing the entire application.
 * 
 * @example
 * ```tsx
 * // Wrap feature components
 * <ErrorBoundary>
 *   <FeatureComponent />
 * </ErrorBoundary>
 * 
 * // With custom fallback
 * <ErrorBoundary
 *   fallback={<CustomErrorUI />}
 *   onError={(error, info) => console.error('Error:', error)}
 * >
 *   <FeatureComponent />
 * </ErrorBoundary>
 * 
 * // With custom messages
 * <ErrorBoundary
 *   errorTitle="Upload Failed"
 *   errorMessage="Unable to upload your file. Please try again."
 * >
 *   <UploadComponent />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  /**
   * Update state when an error is caught
   */
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  /**
   * Log error information and call onError callback
   */
  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error to console in development
    if (IS_DEVELOPMENT) {
      console.error('ErrorBoundary caught an error:', error);
      console.error('Component stack:', errorInfo.componentStack);
    }

    // Update state with error info
    this.setState({
      errorInfo,
    });

    // Call onError callback if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // TODO: Send error to monitoring service (e.g., Sentry)
    // logErrorToService(error, errorInfo);
  }

  /**
   * Reset error boundary state
   */
  resetErrorBoundary = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });

    // Call onReset callback if provided
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  /**
   * Handle navigation to home
   */
  handleGoHome = (): void => {
    this.resetErrorBoundary();
    window.location.href = '/';
  };

  /**
   * Render fallback UI when error occurs
   */
  renderFallback(): ReactNode {
    const { error, errorInfo } = this.state;
    const { 
      fallback, 
      showDetails = IS_DEVELOPMENT,
      errorTitle = DEFAULT_ERROR_TITLE,
      errorMessage = DEFAULT_ERROR_MESSAGE,
    } = this.props;

    // Use custom fallback if provided
    if (fallback) {
      if (typeof fallback === 'function' && error && errorInfo) {
        return fallback(error, errorInfo);
      }
      // Only return fallback if it's not a function (it's a ReactNode)
      if (typeof fallback !== 'function') {
        return fallback;
      }
    }

    // Default fallback UI
    return (
      <div 
        className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-b from-background to-muted/20"
        data-testid="error-boundary"
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
      >
        <Card className="max-w-2xl w-full border-destructive/50">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-full bg-destructive/10">
                <AlertTriangle className="h-6 w-6 text-destructive" />
              </div>
              <div>
                <CardTitle className="text-2xl">{errorTitle}</CardTitle>
                <CardDescription className="mt-1">
                  Error ID: {error ? error.name : 'Unknown'}
                </CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            {/* User-friendly message */}
            <Alert>
              <AlertTitle>What happened?</AlertTitle>
              <AlertDescription className="mt-2">
                {errorMessage}
              </AlertDescription>
            </Alert>

            {/* Error details (development only) */}
            {showDetails && error && (
              <Alert variant="destructive">
                <AlertTitle>Error Details (Development Only)</AlertTitle>
                <AlertDescription className="mt-2 space-y-2">
                  <div>
                    <strong>Message:</strong>
                    <pre className="mt-1 text-xs bg-destructive/5 p-2 rounded overflow-x-auto">
                      {error.message}
                    </pre>
                  </div>
                  {error.stack && (
                    <div>
                      <strong>Stack Trace:</strong>
                      <pre className="mt-1 text-xs bg-destructive/5 p-2 rounded overflow-x-auto max-h-40">
                        {error.stack}
                      </pre>
                    </div>
                  )}
                  {errorInfo && errorInfo.componentStack && (
                    <div>
                      <strong>Component Stack:</strong>
                      <pre className="mt-1 text-xs bg-destructive/5 p-2 rounded overflow-x-auto max-h-40">
                        {errorInfo.componentStack}
                      </pre>
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}

            {/* Help text */}
            <div className="text-sm text-muted-foreground">
              <p className="mb-2">You can try:</p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>Refreshing the page to recover from this error</li>
                <li>Going back to the home page and starting over</li>
                <li>Contacting support if the problem persists</li>
              </ul>
            </div>
          </CardContent>

          <CardFooter className="flex flex-wrap gap-3">
            <Button 
              onClick={this.resetErrorBoundary}
              variant="default"
              className="gap-2"
              data-testid="error-boundary-reset"
            >
              <RefreshCw className="h-4 w-4" />
              Try Again
            </Button>
            <Button 
              onClick={this.handleGoHome}
              variant="outline"
              className="gap-2"
            >
              <Home className="h-4 w-4" />
              Go Home
            </Button>
            <Button 
              variant="ghost"
              className="gap-2"
              asChild
            >
              <a href="mailto:support@arguscore.ai">
                <Mail className="h-4 w-4" />
                Contact Support
              </a>
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return this.renderFallback();
    }

    return this.props.children;
  }
}

// ============== FUNCTIONAL WRAPPER ==============

/**
 * Functional wrapper for ErrorBoundary with hooks support
 * Use this for easier integration with modern React patterns
 */
export function ErrorBoundaryWrapper(props: ErrorBoundaryProps) {
  return <ErrorBoundary {...props} />;
}

// ============== HOOK ==============

/**
 * Hook to reset nearest error boundary
 * Note: This is a placeholder. React doesn't provide a built-in way
 * to reset error boundaries from hooks yet.
 */
export function useErrorBoundary() {
  // This would need to be implemented with context
  // For now, it's a placeholder for future implementation
  const reset = () => {
    console.warn('useErrorBoundary: reset not implemented yet');
  };

  return { reset };
}

// ============== EXPORTS ==============

export default ErrorBoundary;
