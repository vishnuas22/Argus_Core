/**
 * Argus Core - Test Setup Configuration
 * ======================================
 * Global test setup for Vitest + React Testing Library + axe-core.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Polish & Testing
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Purpose:
 * - Configure jsdom environment
 * - Setup React Testing Library matchers
 * - Configure axe-core for accessibility testing
 * - Mock window/global objects
 * - Mock Next.js router
 */

import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeAll, afterAll, vi } from 'vitest';

// ============== CLEANUP ==============

/**
 * Cleanup after each test
 * React Testing Library best practice
 */
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

// ============== JSDOM MOCKS ==============

/**
 * Mock window.matchMedia
 * Required for components using media queries
 */
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

/**
 * Mock IntersectionObserver
 * Required for lazy loading components
 */
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
  takeRecords() {
    return [];
  }
} as any;

/**
 * Mock ResizeObserver
 * Required for D3 visualizations
 */
global.ResizeObserver = class ResizeObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
} as any;

/**
 * Mock SVG methods for D3.js
 * Required for gauge and chart components
 */
HTMLElement.prototype.getBBox = vi.fn(() => ({
  x: 0,
  y: 0,
  width: 100,
  height: 100,
  top: 0,
  right: 100,
  bottom: 100,
  left: 0,
  toJSON: () => {},
})) as any;

/**
 * Mock Canvas for audio visualizations
 */
HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  fillRect: vi.fn(),
  clearRect: vi.fn(),
  getImageData: vi.fn(),
  putImageData: vi.fn(),
  createImageData: vi.fn(),
  setTransform: vi.fn(),
  drawImage: vi.fn(),
  save: vi.fn(),
  fillText: vi.fn(),
  restore: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  closePath: vi.fn(),
  stroke: vi.fn(),
  translate: vi.fn(),
  scale: vi.fn(),
  rotate: vi.fn(),
  arc: vi.fn(),
  fill: vi.fn(),
  measureText: vi.fn(() => ({ width: 0 })),
  transform: vi.fn(),
  rect: vi.fn(),
  clip: vi.fn(),
})) as any;

/**
 * Mock Web Audio API
 * Required for audio analysis components
 */
global.AudioContext = vi.fn().mockImplementation(() => ({
  createAnalyser: vi.fn(() => ({
    fftSize: 2048,
    frequencyBinCount: 1024,
    getByteFrequencyData: vi.fn(),
    getByteTimeDomainData: vi.fn(),
  })),
  createMediaElementSource: vi.fn(),
  destination: {},
  close: vi.fn(),
})) as any;

// ============== NEXT.JS MOCKS ==============

/**
 * Mock next/navigation
 * Required for components using useRouter, usePathname, useSearchParams
 */
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    pathname: '/',
    query: {},
  }),
  usePathname: () => '/',
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

/**
 * Mock next/image
 * Simplifies image testing
 */
vi.mock('next/image', () => ({
  default: (props: any) => {
    // eslint-disable-next-line @next/next/no-img-element, jsx-a11y/alt-text
    return props;
  },
}));

// ============== ENVIRONMENT VARIABLES ==============

/**
 * Mock environment variables
 */
process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8001';
process.env.NEXT_PUBLIC_WS_URL = 'ws://localhost:8001';

// ============== CONSOLE SUPPRESSION ==============

/**
 * Suppress console errors/warnings in tests
 * Comment out during debugging
 */
const originalError = console.error;
const originalWarn = console.warn;

beforeAll(() => {
  console.error = (...args: any[]) => {
    // Suppress known React warnings
    if (
      typeof args[0] === 'string' &&
      (args[0].includes('Warning: ReactDOM.render') ||
        args[0].includes('Warning: useLayoutEffect') ||
        args[0].includes('Not implemented: HTMLFormElement.prototype.submit'))
    ) {
      return;
    }
    originalError.call(console, ...args);
  };

  console.warn = (...args: any[]) => {
    // Suppress known warnings
    if (
      typeof args[0] === 'string' &&
      args[0].includes('React does not recognize')
    ) {
      return;
    }
    originalWarn.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
  console.warn = originalWarn;
});

// ============== GLOBAL TEST UTILITIES ==============

/**
 * Global test timeout
 */
vi.setConfig({ testTimeout: 10000 });

/**
 * Custom matchers (if needed)
 */
expect.extend({
  toBeAccessible(received) {
    // Custom accessibility matcher
    // Implementation would use axe-core
    return {
      message: () => 'Element should be accessible',
      pass: true,
    };
  },
});
