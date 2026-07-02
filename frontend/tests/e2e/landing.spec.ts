/**
 * Argus Core - Landing Page E2E Tests
 * ====================================
 * End-to-end tests for landing page functionality.
 * 
 * Implements: AGENTS_FRONTEND.md - Section 14 - Testing Requirements (P0)
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6 - E2E Tests
 * 
 * Test Coverage:
 * - Page loads successfully
 * - Navigation works correctly
 * - All CTA buttons are functional
 * - Responsive layout across viewports
 * - Accessibility compliance
 * - Performance metrics
 * 
 * @group e2e
 * @group landing
 */

import { test, expect } from '@playwright/test';

// ============== TEST CONSTANTS ==============

const VIEWPORT_DESKTOP = { width: 1920, height: 1080 };
const VIEWPORT_TABLET = { width: 768, height: 1024 };
const VIEWPORT_MOBILE = { width: 375, height: 667 };

// ============== LANDING PAGE TESTS ==============

test.describe('Landing Page', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to landing page before each test
    await page.goto('/');
  });

  // ============== CORE FUNCTIONALITY ==============

  test('should load successfully with correct title', async ({ page }) => {
    // Verify page loaded
    await expect(page).toHaveTitle(/Argus Core/);
    
    // Verify main heading is visible
    await expect(page.getByRole('heading', { name: /Multi-Modal.*Deepfake Detection/i })).toBeVisible();
  });

  test('should display all key sections', async ({ page }) => {
    // Verify landing page structure
    await expect(page.getByTestId('landing-page')).toBeVisible();
    await expect(page.getByTestId('hero-section')).toBeVisible();
    
    // Verify features section
    const featuresSection = page.locator('#features');
    await expect(featuresSection).toBeVisible();
    
    // Verify all feature cards are present (4 total)
    const featureCards = page.locator('[class*="grid"] > [class*="card"]').first().locator('..');
    await expect(featureCards.locator('[class*="card"]')).toHaveCount(4);
  });

  test('should have working navigation header', async ({ page }) => {
    // Verify header is sticky
    const header = page.locator('header');
    await expect(header).toBeVisible();
    await expect(header).toHaveCSS('position', 'sticky');
    
    // Verify logo/brand is visible
    await expect(page.getByText('Argus Core').first()).toBeVisible();
    
    // Verify analyze button in header
    const navAnalyzeBtn = page.getByTestId('nav-analyze-btn');
    await expect(navAnalyzeBtn).toBeVisible();
    await expect(navAnalyzeBtn).toHaveText(/Start Analysis/i);
  });

  // ============== NAVIGATION TESTS ==============

  test('should navigate to analyze page from hero CTA', async ({ page }) => {
    // Click main CTA button
    const ctaButton = page.getByTestId('cta-button');
    await expect(ctaButton).toBeVisible();
    await ctaButton.click();
    
    // Verify navigation to analyze page
    await page.waitForURL('/analyze');
    await expect(page).toHaveURL('/analyze');
    
    // Verify analyze page loaded
    await expect(page.getByTestId('analyze-page')).toBeVisible();
  });

  test('should navigate to analyze page from header button', async ({ page }) => {
    // Click header analyze button
    const navButton = page.getByTestId('nav-analyze-btn');
    await navButton.click();
    
    // Verify navigation
    await page.waitForURL('/analyze');
    await expect(page).toHaveURL('/analyze');
  });

  test('should scroll to features section when clicking Learn More', async ({ page }) => {
    // Click Learn More button
    const learnMoreBtn = page.getByRole('link', { name: /Learn More/i });
    await learnMoreBtn.click();
    
    // Verify URL has anchor
    await expect(page).toHaveURL('/#features');
    
    // Verify features section is in viewport
    const featuresSection = page.locator('#features');
    await expect(featuresSection).toBeInViewport();
  });

  // ============== CONTENT TESTS ==============

  test('should display all feature cards with correct content', async ({ page }) => {
    const expectedFeatures = [
      'Video Analysis',
      'Audio Analysis',
      'Image Analysis',
      'Trust Scoring',
    ];
    
    for (const feature of expectedFeatures) {
      await expect(page.getByRole('heading', { name: feature })).toBeVisible();
    }
  });

  test('should display benefits section with key points', async ({ page }) => {
    const benefitsList = page.getByRole('list').filter({ hasText: /Multi-modal analysis/i });
    
    // Verify benefits are listed
    await expect(benefitsList).toBeVisible();
    
    // Verify key benefits
    await expect(page.getByText(/Multi-modal analysis combining video, audio/i)).toBeVisible();
    await expect(page.getByText(/Real-time progress tracking/i)).toBeVisible();
    await expect(page.getByText(/GradCAM heatmaps/i)).toBeVisible();
  });

  test('should display statistics card', async ({ page }) => {
    // Verify stats are visible
    await expect(page.getByText('500MB')).toBeVisible();
    await expect(page.getByText('5 min')).toBeVisible();
    await expect(page.getByText('4+')).toBeVisible();
    await expect(page.getByText('Real-time')).toBeVisible();
  });

  // ============== RESPONSIVE TESTS ==============

  test('should be responsive on mobile viewport', async ({ page }) => {
    await page.setViewportSize(VIEWPORT_MOBILE);
    
    // Verify mobile layout loads
    await expect(page.getByTestId('landing-page')).toBeVisible();
    
    // Verify header is responsive
    const header = page.locator('header');
    await expect(header).toBeVisible();
    
    // Verify CTA button is visible on mobile
    await expect(page.getByTestId('cta-button')).toBeVisible();
  });

  test('should be responsive on tablet viewport', async ({ page }) => {
    await page.setViewportSize(VIEWPORT_TABLET);
    
    // Verify tablet layout loads
    await expect(page.getByTestId('landing-page')).toBeVisible();
    
    // Verify grid layout adapts
    const featuresGrid = page.locator('#features').locator('div[class*="grid"]');
    await expect(featuresGrid).toBeVisible();
  });

  // ============== ACCESSIBILITY TESTS ==============

  test('should have accessible navigation', async ({ page }) => {
    // Verify landmark regions
    await expect(page.locator('header')).toHaveRole('banner');
    await expect(page.locator('footer')).toHaveRole('contentinfo');
    
    // Verify headings hierarchy
    const h1 = page.getByRole('heading', { level: 1 });
    await expect(h1).toHaveCount(1);
    
    // Verify all interactive elements are keyboard accessible
    const analyzeBtn = page.getByTestId('nav-analyze-btn');
    await analyzeBtn.focus();
    await expect(analyzeBtn).toBeFocused();
  });

  test('should have proper focus management', async ({ page }) => {
    // Tab through interactive elements
    await page.keyboard.press('Tab');
    
    // Verify focus visible on first interactive element
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
    
    // Verify focus outline is visible
    const box = await focusedElement.boundingBox();
    expect(box).toBeTruthy();
  });

  // ============== PERFORMANCE TESTS ==============

  test('should load within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Page should load in under 3 seconds
    expect(loadTime).toBeLessThan(3000);
  });

  test('should have no console errors', async ({ page }) => {
    const errors: string[] = [];
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    await page.goto('/');
    
    // Verify no console errors
    expect(errors).toHaveLength(0);
  });

  // ============== VISUAL REGRESSION ==============

  test('should match screenshot (desktop)', async ({ page }) => {
    await page.setViewportSize(VIEWPORT_DESKTOP);
    
    // Wait for page to be fully loaded
    await page.waitForLoadState('networkidle');
    
    // Take screenshot and compare
    await expect(page).toHaveScreenshot('landing-page-desktop.png', {
      fullPage: true,
      maxDiffPixels: 100,
    });
  });

  test('should match screenshot (mobile)', async ({ page }) => {
    await page.setViewportSize(VIEWPORT_MOBILE);
    
    // Wait for page to be fully loaded
    await page.waitForLoadState('networkidle');
    
    // Take screenshot and compare
    await expect(page).toHaveScreenshot('landing-page-mobile.png', {
      fullPage: true,
      maxDiffPixels: 100,
    });
  });

  // ============== FOOTER TESTS ==============

  test('should display footer with branding', async ({ page }) => {
    const footer = page.locator('footer');
    await expect(footer).toBeVisible();
    
    // Verify footer content
    await expect(footer.getByText('Argus Core')).toBeVisible();
    await expect(footer.getByText(/Multi-Modal Deepfake Detection Platform/i)).toBeVisible();
  });
});

// ============== CROSS-BROWSER TESTS ==============

test.describe('Landing Page - Cross-Browser', () => {
  test('should work correctly in all browsers', async ({ page, browserName }) => {
    await page.goto('/');
    
    // Verify core functionality works across browsers
    await expect(page.getByTestId('landing-page')).toBeVisible();
    await expect(page.getByTestId('cta-button')).toBeVisible();
    
    // Verify navigation works
    await page.getByTestId('nav-analyze-btn').click();
    await page.waitForURL('/analyze');
    await expect(page).toHaveURL('/analyze');
    
    console.log(`✓ Landing page works in ${browserName}`);
  });
});
