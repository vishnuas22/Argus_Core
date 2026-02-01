/**
 * Argus Core - Playwright E2E Test Configuration
 * ===============================================
 * End-to-end testing configuration using Playwright.
 * 
 * Implements: AGENTS_FRONTEND.md - Section 14 - Testing Requirements (P0)
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6 - E2E Tests
 * 
 * Purpose: Configure Playwright for comprehensive E2E testing of critical
 * user journeys including file upload, analysis progress, and results display.
 * 
 * Test Requirements (P0):
 * - Test coverage for all critical user flows
 * - Cross-browser testing (Chromium, Firefox, WebKit)
 * - Mobile and desktop viewport testing
 * - Accessibility testing integration
 * - Screenshot comparison for visual regression
 * 
 * @see https://playwright.dev/docs/test-configuration
 */

import { defineConfig, devices } from '@playwright/test';
import path from 'path';

// Get environment variables
const PORT = process.env.PORT || 3000;
const BASE_URL = process.env.PLAYWRIGHT_TEST_BASE_URL || `http://localhost:${PORT}`;
const CI = !!process.env.CI;

/**
 * Playwright Test Configuration
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  // Test directory
  testDir: './tests/e2e',
  
  // Test file pattern
  testMatch: '**/*.spec.ts',
  
  // Run tests in files in parallel
  fullyParallel: !CI,
  
  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: CI,
  
  // Retry on CI only
  retries: CI ? 2 : 0,
  
  // Number of parallel workers
  workers: CI ? 1 : undefined,
  
  // Reporter configuration
  reporter: CI
    ? [
        ['html', { outputFolder: 'playwright-report', open: 'never' }],
        ['json', { outputFile: 'test-results/results.json' }],
        ['junit', { outputFile: 'test-results/junit.xml' }],
        ['github'],
      ]
    : [
        ['html', { outputFolder: 'playwright-report', open: 'on-failure' }],
        ['list'],
      ],
  
  // Shared settings for all projects
  use: {
    // Base URL for navigation
    baseURL: BASE_URL,
    
    // Collect trace on failure for debugging
    trace: CI ? 'retain-on-failure' : 'on-first-retry',
    
    // Take screenshot on failure
    screenshot: 'only-on-failure',
    
    // Record video on failure
    video: 'retain-on-failure',
    
    // Maximum time each action can take
    actionTimeout: 10000,
    
    // Navigation timeout
    navigationTimeout: 30000,
    
    // Locale and timezone
    locale: 'en-US',
    timezoneId: 'America/New_York',
    
    // Color scheme
    colorScheme: 'light',
    
    // Viewport size (will be overridden by projects)
    viewport: { width: 1280, height: 720 },
    
    // User agent (can be customized per project)
    // userAgent: 'custom-user-agent',
    
    // Extra HTTP headers
    extraHTTPHeaders: {
      'Accept-Language': 'en-US',
    },
  },
  
  // Global setup/teardown
  // globalSetup: require.resolve('./tests/e2e/global-setup.ts'),
  // globalTeardown: require.resolve('./tests/e2e/global-teardown.ts'),
  
  // Test timeout
  timeout: 60000,
  
  // Expect timeout
  expect: {
    timeout: 10000,
    
    // Timeout for toHaveScreenshot assertion
    toHaveScreenshot: {
      maxDiffPixels: 100,
    },
  },
  
  // Web Server configuration - auto-start dev server if not running
  webServer: {
    command: 'yarn dev',
    url: BASE_URL,
    timeout: 120000,
    reuseExistingServer: !CI,
    stdout: 'ignore',
    stderr: 'pipe',
  },
  
  // Project configurations for different browsers and viewports
  projects: [
    // Desktop Chrome
    {
      name: 'chromium-desktop',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
      },
    },
    
    // Desktop Firefox
    {
      name: 'firefox-desktop',
      use: {
        ...devices['Desktop Firefox'],
        viewport: { width: 1920, height: 1080 },
      },
    },
    
    // Desktop Safari
    {
      name: 'webkit-desktop',
      use: {
        ...devices['Desktop Safari'],
        viewport: { width: 1920, height: 1080 },
      },
    },
    
    // Mobile Chrome
    {
      name: 'mobile-chrome',
      use: {
        ...devices['Pixel 5'],
      },
    },
    
    // Mobile Safari
    {
      name: 'mobile-safari',
      use: {
        ...devices['iPhone 13'],
      },
    },
    
    // Tablet
    {
      name: 'tablet',
      use: {
        ...devices['iPad Pro'],
      },
    },
    
    // Accessibility testing (with axe-core)
    {
      name: 'accessibility',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
      },
      testMatch: '**/*.a11y.spec.ts',
    },
  ],
  
  // Output folder for test artifacts
  outputDir: 'test-results',
  
  // Folder for snapshots
  snapshotDir: 'tests/e2e/__snapshots__',
  
  // Snapshot path template
  snapshotPathTemplate: '{snapshotDir}/{testFileDir}/{testFileName}-snapshots/{arg}{-projectName}{-snapshotSuffix}{ext}',
});
