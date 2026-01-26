# Frontend Agent Constitution (Agent-Grade Memory)

Version: 1.0

Purpose: This document defines mandatory architectural, quality, security, performance, and operational rules for autonomous or semi-autonomous frontend code generation. All agents must load, interpret, and enforce this file before writing or modifying any code.

---

## 1. Rule Severity System

Each rule is classified using the following severity levels:

* **P0 – Blocker**: Must be satisfied. If violated, the agent must stop execution and report failure.
* **P1 – Critical**: Must be fixed before merge.
* **P2 – Important**: Should be fixed if feasible.
* **P3 – Optional**: Best practice.

If any P0 rule fails → terminate the task immediately.

---

## 2. Agent Execution Protocol (Mandatory)

Agents must follow this protocol strictly:

1. Parse this document before writing code.
2. Identify applicable rules for the task.
3. Apply P0 rules first, then P1 → P3.
4. Reject any solution violating P0 rules.
5. Prefer explicit typing over inference when ambiguity exists.
6. Implement mobile layout before desktop layout.
7. Implement accessibility features before visual polish.
8. Include tests with every component or feature.
9. Document any deviation from this constitution.
10. Validate outputs using defined tooling.

---

## 3. Default Technology Stack (Binding)

Agents must assume the following production stack unless explicitly overridden.

### Frontend

* Framework: Next.js (App Router)
* Language: TypeScript (strict mode)
* Visualization: D3.js
* Realtime: WebSocket client
* Styling: Tailwind CSS or CSS Modules
* Testing: Vitest + React Testing Library + Playwright
* Linting: ESLint
* Formatting: Prettier
* Accessibility Testing: axe-core

### API Layer / Backend

* Runtime: Node.js + TypeScript
* Framework: NestJS or Fastify
* Realtime: Socket.IO

### Async Processing & Training

* Task Queue: Celery
* Message Broker / Cache: Redis

---

## 4. Enforcement Toolchain

Agents must design code compatible with the following enforcement tools:

### Code Quality

* ESLint (strict configuration)
* eslint-plugin-react
* eslint-plugin-jsx-a11y
* eslint-plugin-security
* Prettier

### Type Safety

* TypeScript strict mode
* `tsc --noEmit`

### Testing

* Vitest / Jest
* Playwright / Cypress
* axe-core

### Performance

* Lighthouse CI
* Webpack Bundle Analyzer / Vite Visualizer

### Security

* npm audit
* Snyk
* OWASP dependency-check

---

## 5. Failure Handling Policy (P0)

Agent must stop execution if any of the following occur:

* WCAG 2.1 AA violation
* Bundle size > 200KB initial load
* Test coverage < 80%
* TypeScript `any` used without explicit justification
* Detected XSS or CSRF risk
* Missing loading/error/empty states
* Missing keyboard accessibility

---

## 6. Code Quality Rules (P1)

* ESLint + Prettier enforced
* Naming conventions:

  * Components: PascalCase
  * Functions/variables: camelCase
* Comprehensive JSDoc comments
* Semantic HTML5 elements
* Zero console errors/warnings in production

---

## 7. Component Design Rules (P1)

* Single Responsibility Principle
* Atomic Design: atoms → molecules → organisms → pages
* Stateless by default
* No prop drilling beyond 2 levels
* Composition over inheritance

### Standard Component Contract (P0)

Each component must include:

* Props interface
* Loading state UI
* Error state UI
* Empty state UI
* Accessibility attributes
* Unit tests
* Storybook story
* Memoization strategy (if applicable)

---

## 8. Type Safety Rules (P0)

* TypeScript strict mode enabled
* No `any` types allowed
* Interfaces for all props and state
* Runtime type guards where applicable
* Generics for reusable logic

---

## 9. Performance Standards (P0)

* Initial bundle < 200KB
* Route-level code splitting
* Lazy loading for:

  * Images
  * Components
  * Heavy libraries
* Memoization for expensive computations
* Virtual scrolling for large lists

### Web Vitals Targets

* LCP < 2.5s
* FID < 100ms
* CLS < 0.1
* INP < 200ms

---

## 10. Accessibility Rules (P0 – Non-Negotiable)

* WCAG 2.1 AA compliance
* Keyboard navigation for all interactions
* ARIA labels where required
* Screen reader compatibility
* Color contrast ≥ 4.5:1
* Focus management with visible indicators

---

## 11. State Management Rules (P1)

* Centralized state for shared data
* Local state for UI-only concerns
* Immutable updates only
* Optimistic UI updates where applicable
* State persistence strategy defined

### Standardization

* Global: Zustand / Redux Toolkit
* Server: TanStack Query
* No direct fetch inside components
* Normalized error handling

---

## 12. API Integration Rules (P1)

* Request/response caching
* Loading, error, empty states
* Retry with exponential backoff
* Timeout configuration
* Optimistic updates
* Error boundaries around data-fetching components

---

## 13. Security Rules (P0 – Critical)

* Input sanitization (XSS prevention)
* CSRF protection
* Content Security Policy headers
* Secure cookies (HttpOnly, Secure, SameSite)
* Never store sensitive data in localStorage
* Secrets via environment variables only
* Dependency vulnerability scanning mandatory

---

## 14. Testing Requirements (P0)

* Unit coverage > 80%
* Component tests mandatory
* Integration tests for critical flows
* E2E tests for user journeys
* Automated accessibility tests
* No skipped or disabled tests

---

## 15. Responsive Design Rules (P0)

* Mobile-first approach mandatory
* Breakpoint strategy defined
* Touch targets ≥ 44x44px
* Responsive images using srcset
* Fluid typography and spacing

---

## 16. Error Handling Rules (P1)

* Error boundaries at feature boundaries
* User-friendly messages
* Error logging and monitoring
* Graceful degradation
* Fallback UI components
* Network error handling

---

## 17. SEO Optimization Rules (P1)

* Semantic HTML structure
* Meta tags (title, description, OG)
* Structured data (JSON-LD)
* Proper heading hierarchy
* Alt text for all images
* Sitemap + robots.txt

---

## 18. Browser Support Rules (P2)

* Last 2 versions of major browsers
* Graceful degradation
* Feature detection only
* Polyfills only when necessary
* Progressive enhancement

---

## 19. Code Organization Rules (P1)

* Feature-based folder structure
* One component per file
* Barrel exports
* Absolute imports with path aliases
* Colocate tests

### Standard Folder Layout

```
src/
  features/
    feature-name/
      components/
      hooks/
      services/
      store/
      tests/
      types/
  shared/
  ui/
  pages/
```

---

## 20. Asset Optimization Rules (P1)

* Images: WebP/AVIF + fallback
* Icons: SVG sprites or icon libraries
* Fonts: subset + preload
* Lazy load non-critical assets
* CDN for static assets

---

## 21. Forms & Validation Rules (P1)

* Client + server validation
* Accessible error messages
* Real-time validation feedback
* Form state persistence
* Disable submit while processing

---

## 22. Animation Rules (P2)

* Prefer CSS transitions
* Target 60fps
* Respect prefers-reduced-motion
* Animate transform and opacity only
* Avoid layout thrashing

---

## 23. Developer Experience Rules (P2)

* Hot module replacement enabled
* Pre-commit hooks (lint, format, test)
* Clear error messages
* Storybook documentation
* Git hooks as quality gates

---

## 24. Build & Deployment Rules (P1)

* Environment-specific builds
* Tree shaking enabled
* Source maps
* Asset versioning and cache busting

---

## 25. Non-Negotiable Global Rules (P0)

1. Mobile-first is mandatory
2. Accessibility is mandatory
3. Every component must support loading, error, empty states
4. Performance budgets are hard limits
5. Security vulnerabilities block deployment
6. Test coverage below 80% fails build
7. TypeScript `any` requires explicit justification

---

## 26. Agent Compliance Declaration

Agents must treat this file as authoritative. Any deviation must be explicitly documented in output with rationale and severity impact.

End of constitution.
