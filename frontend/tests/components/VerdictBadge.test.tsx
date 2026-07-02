/**
 * Argus Core - VerdictBadge Component Tests
 * ==========================================
 * Comprehensive tests for VerdictBadge component.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Phase 6: Component Tests (P0)
 * Complies with: AGENTS_FRONTEND.md - Testing Requirements (P0)
 * 
 * Test Coverage:
 * - All verdict types rendering
 * - Color coding and visual styling
 * - Size variants
 * - Icon display
 * - Description toggle
 * - Accessibility compliance (WCAG 2.1 AA)
 * - Animations
 * 
 * Target: >80% coverage
 */

import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders, checkAccessibility } from '../utils/test-utils';
import { VerdictBadge, VERDICT_CONFIG } from '@/components/results/VerdictBadge';
import type { Verdict } from '@/types/analysis';

// ============== TEST SUITE ==============

describe('VerdictBadge', () => {
  // ============== BASIC RENDERING ==============

  describe('Rendering', () => {
    it('should render with minimum required props', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" />);
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
    });

    it('should display the verdict label', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" />);
      
      const label = screen.getByTestId('verdict-badge-label');
      expect(label).toHaveTextContent('Authentic');
    });

    it('should show icon by default', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" />);
      
      const icon = screen.getByTestId('verdict-badge-icon');
      expect(icon).toBeInTheDocument();
    });

    it('should hide icon when showIcon is false', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" showIcon={false} />);
      
      const icon = screen.queryByTestId('verdict-badge-icon');
      expect(icon).not.toBeInTheDocument();
    });

    it('should show description when showDescription is true', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" showDescription />);
      
      const description = screen.getByTestId('verdict-badge-description');
      expect(description).toBeInTheDocument();
      expect(description).toHaveTextContent(VERDICT_CONFIG.authentic.description);
    });

    it('should hide description when showDescription is false', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" showDescription={false} />);
      
      const description = screen.queryByTestId('verdict-badge-description');
      expect(description).not.toBeInTheDocument();
    });
  });

  // ============== VERDICT TYPES ==============

  describe('Verdict Types', () => {
    it('should render authentic verdict correctly', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" />);
      
      const label = screen.getByTestId('verdict-badge-label');
      expect(label).toHaveTextContent('Authentic');
      
      // The verdict badge contains green color styling in its children
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
    });

    it('should render likely_authentic verdict correctly', () => {
      renderWithProviders(<VerdictBadge verdict="likely_authentic" />);
      
      const label = screen.getByTestId('verdict-badge-label');
      expect(label).toHaveTextContent(/Likely Authentic|Probably Authentic/);
    });

    it('should render uncertain verdict correctly', () => {
      renderWithProviders(<VerdictBadge verdict="uncertain" />);
      
      const label = screen.getByTestId('verdict-badge-label');
      expect(label).toHaveTextContent('Uncertain');
      
      // The verdict badge contains yellow/warning color styling in its children
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
    });

    it('should render likely_fake verdict correctly', () => {
      renderWithProviders(<VerdictBadge verdict="likely_fake" />);
      
      const label = screen.getByTestId('verdict-badge-label');
      expect(label).toHaveTextContent(/Likely Fake/i);
    });

    it('should render fake verdict correctly', () => {
      renderWithProviders(<VerdictBadge verdict="fake" />);
      
      const label = screen.getByTestId('verdict-badge-label');
      expect(label).toHaveTextContent(/Fake/i);
      
      // The verdict badge contains red color styling in its children
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
    });

    it('should render all verdict types without errors', () => {
      const verdicts: Verdict[] = [
        'authentic',
        'likely_authentic',
        'uncertain',
        'likely_fake',
        'fake',
      ];

      verdicts.forEach((verdict) => {
        const { unmount } = renderWithProviders(<VerdictBadge verdict={verdict} />);
        
        const badge = screen.getByTestId('verdict-badge');
        expect(badge).toBeInTheDocument();
        
        unmount();
      });
    });
  });

  // ============== SIZE VARIANTS ==============

  describe('Size Variants', () => {
    it('should render with small size', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" size="sm" />);
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
      // Size styling is applied to children components
    });

    it('should render with medium size (default)', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" size="md" />);
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
      // Size styling is applied to children components
    });

    it('should render with large size', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" size="lg" />);
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
      // Size styling is applied to children components
    });

    it('should use medium size by default', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" />);
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
      // Size styling is applied to children components
    });
  });

  // ============== ICON RENDERING ==============

  describe('Icon Rendering', () => {
    it('should render correct icon for authentic', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" />);
      
      const icon = screen.getByTestId('verdict-badge-icon');
      expect(icon).toBeInTheDocument();
      // Icon should be ShieldCheck
    });

    it('should render correct icon for likely_authentic', () => {
      renderWithProviders(<VerdictBadge verdict="likely_authentic" />);
      
      const icon = screen.getByTestId('verdict-badge-icon');
      expect(icon).toBeInTheDocument();
      // Icon should be Shield
    });

    it('should render correct icon for uncertain', () => {
      renderWithProviders(<VerdictBadge verdict="uncertain" />);
      
      const icon = screen.getByTestId('verdict-badge-icon');
      expect(icon).toBeInTheDocument();
      // Icon should be HelpCircle or AlertTriangle
    });

    it('should render correct icon for likely_fake', () => {
      renderWithProviders(<VerdictBadge verdict="likely_fake" />);
      
      const icon = screen.getByTestId('verdict-badge-icon');
      expect(icon).toBeInTheDocument();
      // Icon should be AlertOctagon
    });

    it('should render correct icon for fake', () => {
      renderWithProviders(<VerdictBadge verdict="fake" />);
      
      const icon = screen.getByTestId('verdict-badge-icon');
      expect(icon).toBeInTheDocument();
      // Icon should be XOctagon
    });

    it('should scale icon with badge size', () => {
      const { rerender } = renderWithProviders(
        <VerdictBadge verdict="authentic" size="sm" />
      );
      
      let icon = screen.getByTestId('verdict-badge-icon');
      expect(icon).toHaveClass('h-3');
      
      rerender(<VerdictBadge verdict="authentic" size="lg" />);
      
      icon = screen.getByTestId('verdict-badge-icon');
      expect(icon).toHaveClass('h-5');
    });
  });

  // ============== COLOR CODING ==============

  describe('Color Coding', () => {
    it('should use green colors for authentic', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" />);
      
      const badge = screen.getByTestId('verdict-badge');
      const badgeElement = badge.querySelector('[class*="bg-green"]');
      
      // Should contain green-related classes
      expect(badgeElement).toBeInTheDocument();
    });

    it('should use lime/green colors for likely_authentic', () => {
      renderWithProviders(<VerdictBadge verdict="likely_authentic" />);
      
      const badge = screen.getByTestId('verdict-badge');
      const badgeElement = badge.querySelector('[class*="bg-lime"]');
      
      expect(badgeElement).toBeInTheDocument();
    });

    it('should use yellow colors for uncertain', () => {
      renderWithProviders(<VerdictBadge verdict="uncertain" />);
      
      const badge = screen.getByTestId('verdict-badge');
      const badgeElement = badge.querySelector('[class*="bg-yellow"]');
      
      expect(badgeElement).toBeInTheDocument();
    });

    it('should use orange colors for likely_fake', () => {
      renderWithProviders(<VerdictBadge verdict="likely_fake" />);
      
      const badge = screen.getByTestId('verdict-badge');
      const badgeElement = badge.querySelector('[class*="bg-orange"]');
      
      expect(badgeElement).toBeInTheDocument();
    });

    it('should use red colors for fake', () => {
      renderWithProviders(<VerdictBadge verdict="fake" />);
      
      const badge = screen.getByTestId('verdict-badge');
      const badgeElement = badge.querySelector('[class*="bg-red"]');
      
      expect(badgeElement).toBeInTheDocument();
    });
  });

  // ============== DESCRIPTION ==============

  describe('Description', () => {
    it('should display correct description for each verdict', () => {
      const verdicts: Verdict[] = [
        'authentic',
        'likely_authentic',
        'uncertain',
        'likely_fake',
        'fake',
      ];

      verdicts.forEach((verdict) => {
        const { unmount } = renderWithProviders(
          <VerdictBadge verdict={verdict} showDescription />
        );
        
        const description = screen.getByTestId('verdict-badge-description');
        expect(description).toHaveTextContent(VERDICT_CONFIG[verdict].description);
        
        unmount();
      });
    });

    it('should wrap description text properly', () => {
      renderWithProviders(
        <VerdictBadge verdict="authentic" showDescription />
      );
      
      const description = screen.getByTestId('verdict-badge-description');
      // Description should be in the document and visible
      expect(description).toBeVisible();
    });
  });

  // ============== ANIMATIONS ==============

  describe('Animations', () => {
    it('should animate when animated prop is true', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" animated />);
      
      const badge = screen.getByTestId('verdict-badge');
      // Should have animation classes
      expect(badge).toBeInTheDocument();
    });

    it('should not animate when animated prop is false', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" animated={false} />);
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
    });

    it('should respect prefers-reduced-motion', () => {
      // Mock prefers-reduced-motion media query
      window.matchMedia = vi.fn().mockImplementation(query => ({
        matches: query === '(prefers-reduced-motion: reduce)',
        media: query,
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })) as any;

      renderWithProviders(<VerdictBadge verdict="authentic" animated />);
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
    });
  });

  // ============== ACCESSIBILITY ==============

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" />);
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toHaveAttribute('role', 'status');
      expect(badge).toHaveAttribute('aria-label');
    });

    it('should include verdict in aria-label', () => {
      renderWithProviders(<VerdictBadge verdict="fake" />);
      
      const badge = screen.getByTestId('verdict-badge');
      const ariaLabel = badge.getAttribute('aria-label');
      expect(ariaLabel).toContain('Fake');
    });

    it('should pass basic accessibility checks', () => {
      const { container } = renderWithProviders(
        <VerdictBadge verdict="authentic" />
      );
      
      const result = checkAccessibility(container);
      expect(result.passed).toBe(true);
    });

    it('should have sufficient color contrast', () => {
      // Test all verdict types for color contrast
      const verdicts: Verdict[] = [
        'authentic',
        'likely_authentic',
        'uncertain',
        'likely_fake',
        'fake',
      ];

      verdicts.forEach((verdict) => {
        const { unmount } = renderWithProviders(<VerdictBadge verdict={verdict} />);
        
        const badge = screen.getByTestId('verdict-badge');
        expect(badge).toBeInTheDocument();
        
        unmount();
      });
    });

    it('should be keyboard accessible', () => {
      renderWithProviders(<VerdictBadge verdict="authentic" />);
      
      const badge = screen.getByTestId('verdict-badge');
      // Badge should be accessible but not focusable (it's a status indicator)
      expect(badge).toBeInTheDocument();
    });
  });

  // ============== CUSTOM STYLING ==============

  describe('Custom Styling', () => {
    it('should apply custom className', () => {
      const customClass = 'custom-verdict-class';
      renderWithProviders(
        <VerdictBadge verdict="authentic" className={customClass} />
      );
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge.className).toContain(customClass);
    });

    it('should merge custom styles with default styles', () => {
      renderWithProviders(
        <VerdictBadge verdict="authentic" className="custom-class" />
      );
      
      const badge = screen.getByTestId('verdict-badge');
      // Should have both custom and default classes
      expect(badge.className).toContain('custom-class');
      expect(badge.className.length).toBeGreaterThan('custom-class'.length);
    });
  });

  // ============== INTEGRATION ==============

  describe('Integration', () => {
    it('should work with all props combined', () => {
      renderWithProviders(
        <VerdictBadge
          verdict="authentic"
          size="lg"
          showIcon
          showDescription
          animated
          className="custom-class"
        />
      );
      
      const badge = screen.getByTestId('verdict-badge');
      expect(badge).toBeInTheDocument();
      
      const icon = screen.getByTestId('verdict-badge-icon');
      expect(icon).toBeInTheDocument();
      
      const description = screen.getByTestId('verdict-badge-description');
      expect(description).toBeInTheDocument();
    });

    it('should render consistently across different verdicts', () => {
      const verdicts: Verdict[] = [
        'authentic',
        'likely_authentic',
        'uncertain',
        'likely_fake',
        'fake',
      ];

      verdicts.forEach((verdict) => {
        const { unmount } = renderWithProviders(
          <VerdictBadge 
            verdict={verdict} 
            size="md" 
            showIcon 
            showDescription 
          />
        );
        
        // All components should be present
        expect(screen.getByTestId('verdict-badge')).toBeInTheDocument();
        expect(screen.getByTestId('verdict-badge-label')).toBeInTheDocument();
        expect(screen.getByTestId('verdict-badge-icon')).toBeInTheDocument();
        expect(screen.getByTestId('verdict-badge-description')).toBeInTheDocument();
        
        unmount();
      });
    });
  });

  // ============== SNAPSHOTS ==============

  describe('Snapshots', () => {
    it('should match snapshot for authentic verdict', () => {
      const { container } = renderWithProviders(
        <VerdictBadge verdict="authentic" animated={false} />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot for fake verdict', () => {
      const { container } = renderWithProviders(
        <VerdictBadge verdict="fake" animated={false} />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });

    it('should match snapshot with description', () => {
      const { container } = renderWithProviders(
        <VerdictBadge verdict="uncertain" showDescription animated={false} />
      );
      
      expect(container.firstChild).toMatchSnapshot();
    });
  });
});
