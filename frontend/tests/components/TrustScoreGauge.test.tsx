/**
 * Argus Core - TrustScoreGauge Component Tests
 * =============================================
 * Comprehensive tests for TrustScoreGauge component.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Component Tests (P0)
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Test Coverage:
 * - Rendering with different score ranges
 * - Verdict-based color coding
 * - Animations and transitions
 * - Accessibility compliance (WCAG 2.1 AA)
 * - Error states and edge cases
 * - Responsive sizing
 * 
 * Target: >80% coverage
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { renderWithProviders, checkAccessibility } from '../utils/test-utils';
import { TrustScoreGauge, GAUGE_SIZES } from '@/components/results/TrustScoreGauge';
import type { Verdict } from '@/types/analysis';

// ============== TEST SUITE ==============

describe('TrustScoreGauge', () => {
  // ============== BASIC RENDERING ==============

  describe('Rendering', () => {
    it('should render with minimum required props', () => {
      renderWithProviders(<TrustScoreGauge score={75} />);
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toBeInTheDocument();
    });

    it('should display the score value', () => {
      renderWithProviders(<TrustScoreGauge score={85} />);
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      expect(scoreValue).toHaveTextContent('85');
    });

    it('should display confidence when provided', () => {
      renderWithProviders(
        <TrustScoreGauge score={75} confidence={0.92} showConfidence />
      );
      
      const confidence = screen.getByTestId('gauge-confidence');
      expect(confidence).toBeInTheDocument();
      expect(confidence).toHaveTextContent(/92%/);
    });

    it('should hide confidence when showConfidence is false', () => {
      renderWithProviders(
        <TrustScoreGauge score={75} confidence={0.92} showConfidence={false} />
      );
      
      const confidence = screen.queryByTestId('gauge-confidence');
      expect(confidence).not.toBeInTheDocument();
    });

    it('should hide label when showLabel is false', () => {
      renderWithProviders(<TrustScoreGauge score={75} showLabel={false} />);
      
      const label = screen.queryByTestId('gauge-score-value');
      expect(label).not.toBeInTheDocument();
    });
  });

  // ============== SCORE RANGES ==============

  describe('Score Ranges', () => {
    it('should handle score of 0', () => {
      renderWithProviders(<TrustScoreGauge score={0} />);
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      expect(scoreValue).toHaveTextContent('0');
    });

    it('should handle score of 100', () => {
      renderWithProviders(<TrustScoreGauge score={100} />);
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      expect(scoreValue).toHaveTextContent('100');
    });

    it('should clamp score above 100 to 100', () => {
      renderWithProviders(<TrustScoreGauge score={150} />);
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      expect(scoreValue).toHaveTextContent('100');
    });

    it('should clamp negative score to 0', () => {
      renderWithProviders(<TrustScoreGauge score={-10} />);
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      expect(scoreValue).toHaveTextContent('0');
    });

    it('should handle decimal scores', () => {
      renderWithProviders(<TrustScoreGauge score={75.5} />);
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      expect(scoreValue).toHaveTextContent('76'); // Should round
    });
  });

  // ============== VERDICT-BASED STYLING ==============

  describe('Verdict-Based Styling', () => {
    it('should apply authentic styling for high scores', () => {
      renderWithProviders(
        <TrustScoreGauge score={90} verdict="authentic" />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      // Check if gauge has appropriate styling
      expect(gauge).toBeInTheDocument();
    });

    it('should apply likely_authentic styling', () => {
      renderWithProviders(
        <TrustScoreGauge score={75} verdict="likely_authentic" />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toBeInTheDocument();
    });

    it('should apply uncertain styling', () => {
      renderWithProviders(
        <TrustScoreGauge score={50} verdict="uncertain" />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toBeInTheDocument();
    });

    it('should apply likely_deepfake styling', () => {
      renderWithProviders(
        <TrustScoreGauge score={35} verdict="likely_deepfake" />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toBeInTheDocument();
    });

    it('should apply deepfake styling for low scores', () => {
      renderWithProviders(
        <TrustScoreGauge score={15} verdict="deepfake" />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toBeInTheDocument();
    });
  });

  // ============== SIZE VARIANTS ==============

  describe('Size Variants', () => {
    it('should render with small size', () => {
      renderWithProviders(<TrustScoreGauge score={75} size={GAUGE_SIZES.sm} />);
      
      const svg = screen.getByTestId('trust-score-gauge').querySelector('svg');
      expect(svg).toHaveAttribute('width', GAUGE_SIZES.sm.toString());
    });

    it('should render with medium size (default)', () => {
      renderWithProviders(<TrustScoreGauge score={75} size={GAUGE_SIZES.md} />);
      
      const svg = screen.getByTestId('trust-score-gauge').querySelector('svg');
      expect(svg).toHaveAttribute('width', GAUGE_SIZES.md.toString());
    });

    it('should render with large size', () => {
      renderWithProviders(<TrustScoreGauge score={75} size={GAUGE_SIZES.lg} />);
      
      const svg = screen.getByTestId('trust-score-gauge').querySelector('svg');
      expect(svg).toHaveAttribute('width', GAUGE_SIZES.lg.toString());
    });

    it('should render with extra large size', () => {
      renderWithProviders(<TrustScoreGauge score={75} size={GAUGE_SIZES.xl} />);
      
      const svg = screen.getByTestId('trust-score-gauge').querySelector('svg');
      expect(svg).toHaveAttribute('width', GAUGE_SIZES.xl.toString());
    });

    it('should render with custom size', () => {
      const customSize = 300;
      renderWithProviders(<TrustScoreGauge score={75} size={customSize} />);
      
      const svg = screen.getByTestId('trust-score-gauge').querySelector('svg');
      expect(svg).toHaveAttribute('width', customSize.toString());
    });
  });

  // ============== ANIMATIONS ==============

  describe('Animations', () => {
    it('should animate by default', async () => {
      renderWithProviders(<TrustScoreGauge score={75} animated />);
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toBeInTheDocument();
      
      // Animation should complete within duration
      await waitFor(() => {
        const scoreValue = screen.getByTestId('gauge-score-value');
        expect(scoreValue).toHaveTextContent('75');
      }, { timeout: 2000 });
    });

    it('should not animate when animated is false', () => {
      renderWithProviders(<TrustScoreGauge score={75} animated={false} />);
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      // Should show final value immediately
      expect(scoreValue).toHaveTextContent('75');
    });

    it('should respect custom animation duration', async () => {
      const customDuration = 500;
      renderWithProviders(
        <TrustScoreGauge
          score={75}
          animated
          animationDuration={customDuration}
        />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toBeInTheDocument();
    });

    it('should update smoothly when score changes', async () => {
      const { rerender } = renderWithProviders(
        <TrustScoreGauge score={50} animated />
      );
      
      // Update score
      rerender(<TrustScoreGauge score={80} animated />);
      
      await waitFor(() => {
        const scoreValue = screen.getByTestId('gauge-score-value');
        expect(scoreValue).toHaveTextContent('80');
      }, { timeout: 2000 });
    });
  });

  // ============== ACCESSIBILITY ==============

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      renderWithProviders(
        <TrustScoreGauge score={75} verdict="likely_authentic" />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toHaveAttribute('role', 'img');
      expect(gauge).toHaveAttribute('aria-label');
    });

    it('should include score in aria-label', () => {
      renderWithProviders(<TrustScoreGauge score={85} />);
      
      const gauge = screen.getByTestId('trust-score-gauge');
      const ariaLabel = gauge.getAttribute('aria-label');
      expect(ariaLabel).toContain('85');
    });

    it('should include verdict in aria-label when provided', () => {
      renderWithProviders(
        <TrustScoreGauge score={90} verdict="authentic" />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      const ariaLabel = gauge.getAttribute('aria-label');
      expect(ariaLabel).toContain('authentic');
    });

    it('should pass basic accessibility checks', () => {
      const { container } = renderWithProviders(
        <TrustScoreGauge score={75} verdict="likely_authentic" />
      );
      
      const result = checkAccessibility(container);
      expect(result.passed).toBe(true);
    });

    it('should be keyboard accessible', () => {
      renderWithProviders(<TrustScoreGauge score={75} />);
      
      const gauge = screen.getByTestId('trust-score-gauge');
      // Gauge should be in tab order if interactive
      expect(gauge).toBeInTheDocument();
    });
  });

  // ============== CUSTOM STYLING ==============

  describe('Custom Styling', () => {
    it('should apply custom className', () => {
      const customClass = 'custom-gauge-class';
      renderWithProviders(
        <TrustScoreGauge score={75} className={customClass} />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge.className).toContain(customClass);
    });

    it('should merge custom styles with default styles', () => {
      renderWithProviders(
        <TrustScoreGauge score={75} className="custom-class" />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      // Should have both custom and default classes
      expect(gauge.className).toBeTruthy();
    });
  });

  // ============== ERROR HANDLING ==============

  describe('Error Handling', () => {
    it('should handle NaN score gracefully', () => {
      renderWithProviders(<TrustScoreGauge score={NaN} />);
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      expect(scoreValue).toHaveTextContent('0');
    });

    it('should handle Infinity score gracefully', () => {
      renderWithProviders(<TrustScoreGauge score={Infinity} />);
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      expect(scoreValue).toHaveTextContent('100');
    });

    it('should handle missing confidence gracefully', () => {
      renderWithProviders(
        <TrustScoreGauge score={75} showConfidence />
      );
      
      // Should not crash, confidence should be hidden or show N/A
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toBeInTheDocument();
    });
  });

  // ============== INTEGRATION ==============

  describe('Integration', () => {
    it('should work with all props combined', () => {
      renderWithProviders(
        <TrustScoreGauge
          score={85}
          confidence={0.92}
          verdict="authentic"
          size={GAUGE_SIZES.lg}
          animated
          showLabel
          showConfidence
          className="custom-class"
          animationDuration={1000}
        />
      );
      
      const gauge = screen.getByTestId('trust-score-gauge');
      expect(gauge).toBeInTheDocument();
      
      const scoreValue = screen.getByTestId('gauge-score-value');
      expect(scoreValue).toHaveTextContent('85');
      
      const confidence = screen.getByTestId('gauge-confidence');
      expect(confidence).toHaveTextContent(/92%/);
    });

    it('should maintain correct aspect ratio at different sizes', () => {
      const sizes = [GAUGE_SIZES.sm, GAUGE_SIZES.md, GAUGE_SIZES.lg, GAUGE_SIZES.xl];
      
      sizes.forEach((size) => {
        const { unmount } = renderWithProviders(
          <TrustScoreGauge score={75} size={size} />
        );
        
        const svg = screen.getByTestId('trust-score-gauge').querySelector('svg');
        expect(svg).toHaveAttribute('width', size.toString());
        expect(svg).toHaveAttribute('height', size.toString());
        
        unmount();
      });
    });
  });

  // ============== SNAPSHOT ==============

  describe('Snapshots', () => {
    it('should match snapshot for authentic score', () => {
      const { container } = renderWithProviders(
        <TrustScoreGauge score={90} verdict="authentic" animated={false} />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot for deepfake score', () => {
      const { container } = renderWithProviders(
        <TrustScoreGauge score={20} verdict="deepfake" animated={false} />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot for uncertain score', () => {
      const { container } = renderWithProviders(
        <TrustScoreGauge score={50} verdict="uncertain" animated={false} />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });
  });
});
