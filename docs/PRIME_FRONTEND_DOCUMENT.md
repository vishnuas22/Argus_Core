# PRIME FRONTEND DOCUMENT
## Argus Prime - Multi-Modal Deepfake Detection Platform
### Frontend Implementation Blueprint v1.0

**Classification:** Production-Grade Implementation Specification  
**Date:** January 2026  
**Compliance:** AGENTS_FRONTEND.md | Next.js 14+ | TypeScript Strict

---

# TABLE OF CONTENTS

1. [Section 1: The "Life of a Request" Flow - Frontend Perspective](#section-1-the-life-of-a-request-flow---frontend-perspective)
2. [Section 2: Architecture & File Manifesto](#section-2-architecture--file-manifesto)
3. [Section 3: Development Strategy](#section-3-development-strategy)
4. [Appendix A: Component Contracts](#appendix-a-component-contracts)
5. [Appendix B: API Integration Schemas](#appendix-b-api-integration-schemas)
6. [Appendix C: State Management Patterns](#appendix-c-state-management-patterns)

---

# SECTION 1: THE "LIFE OF A REQUEST" FLOW - FRONTEND PERSPECTIVE

## 1.1 User Journey: "Analyze Media File"

This traces the complete frontend journey from file selection to results display.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND REQUEST LIFECYCLE                                            │
│                        Total Time: 0ms → ~20s (user perception)                             │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: FILE SELECTION & VALIDATION (0-100ms)                                      │   │
│  │ Files: UploadZone.tsx → useFileValidation.ts → fileValidation.ts                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                      │
│      │  T+0ms:    User drags file onto UploadZone component                               │
│      │  T+10ms:   UploadZone validates drop event, extracts file                           │
│      │  T+20ms:   useFileValidation hook triggers client-side validation                   │
│      │  T+30ms:   fileValidation.ts checks: size ≤500MB, type, magic bytes                │
│      │  T+50ms:   Local file preview generated (image/video thumbnail)                     │
│      │  T+100ms:  UI updates: file card displayed with metadata                            │
│      ▼                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 2: UPLOAD INITIATION (100ms-500ms)                                            │   │
│  │ Files: AnalysisForm.tsx → useAnalysis.ts → analysisApi.ts → FormData construction   │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                      │
│      │  T+100ms:  User clicks "Analyze" button                                             │
│      │  T+110ms:  AnalysisForm.tsx collects options (heatmaps, report, defense level)     │
│      │  T+120ms:  useAnalysis mutation triggered                                           │
│      │  T+130ms:  analysisApi.ts constructs FormData with file + options                  │
│      │  T+150ms:  POST /api/v1/analyze initiated (multipart/form-data)                    │
│      │  T+200ms:  Upload progress tracked via axios onUploadProgress                       │
│      │  T+500ms:  Server responds 202 Accepted with analysis_id                           │
│      ▼                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 3: WEBSOCKET CONNECTION (500ms-600ms)                                          │   │
│  │ Files: useWebSocket.ts → WebSocketProvider.tsx → progressStore.ts                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                      │
│      │  T+500ms:  useAnalysis receives analysis_id from response                          │
│      │  T+510ms:  Router navigates to /analysis/{analysis_id}                             │
│      │  T+520ms:  AnalysisPage.tsx mounts, triggers useWebSocket connection               │
│      │  T+530ms:  WebSocket connects to ws://.../ws/analysis/{analysis_id}                │
│      │  T+550ms:  Server sends initial "status" message                                    │
│      │  T+600ms:  progressStore.ts updates with current status                            │
│      ▼                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 4: REAL-TIME PROGRESS (600ms-15s)                                              │   │
│  │ Files: ProgressIndicator.tsx → progressStore.ts → AnalysisTimeline.tsx               │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                      │
│      │  T+600ms:  ProgressIndicator shows "Pending" state                                  │
│      │  T+2s:     WebSocket receives "preprocessing" status (15%)                         │
│      │  T+2.1s:   ProgressIndicator animates to 15%, stage label updates                  │
│      │  T+4s:     WebSocket receives "analyzing" status (50%)                             │
│      │  T+4.1s:   AnalysisTimeline.tsx shows active analysis indicators                   │
│      │  T+10s:    Progress updates continue: 60%, 70%, 80%                                │
│      │  T+14s:    WebSocket receives "aggregating" status (85%)                           │
│      │  T+15s:    WebSocket receives "completed" with trust_score, verdict                │
│      ▼                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 5: RESULTS DISPLAY (15s-15.5s)                                                 │   │
│  │ Files: ResultsPanel.tsx → TrustScoreGauge.tsx → VerdictBadge.tsx → D3 visualizations │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                      │
│      │  T+15s:    progressStore updates status to "completed"                              │
│      │  T+15.1s:  TanStack Query invalidates analysis cache                               │
│      │  T+15.2s:  GET /api/v1/analyze/{id}/detail fetches full results                   │
│      │  T+15.3s:  ResultsPanel.tsx renders with animated score gauge                      │
│      │  T+15.4s:  VerdictBadge shows verdict with appropriate color                       │
│      │  T+15.5s:  D3 visualizations render: score breakdown, timeline                     │
│      ▼                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 6: DETAILED EXPLORATION (User-Driven)                                          │   │
│  │ Files: ModalityTabs.tsx → VideoAnalysisPanel.tsx → HeatmapViewer.tsx → ReportDownload│   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                      │
│      │  User clicks "Video Analysis" tab                                                   │
│      │  →  VideoAnalysisPanel.tsx renders spatial/temporal/lipsync scores                 │
│      │  →  HeatmapViewer.tsx loads GradCAM heatmap images via presigned URLs             │
│      │                                                                                      │
│      │  User clicks "Download Report"                                                      │
│      │  →  ReportDownload.tsx initiates GET /api/v1/analyze/{id}/report                   │
│      │  →  PDF downloaded via presigned URL                                               │
│      ▼                                                                                      │
│                                                                                             │
│                            ✓ USER JOURNEY COMPLETE                                          │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1.2 File Interaction Sequence Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND FILE INTERACTION SEQUENCE                              │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  USER ACTION: Drop File                                                                  │
│    │                                                                                     │
│    ▼                                                                                     │
│  UploadZone.tsx ─────────────────────────────────────────────────────────────────────┐  │
│    │                                                                                  │  │
│    │  1. onDrop event handler                                                        │  │
│    │  2. Extract File from DataTransfer                                              │  │
│    │                                                                                  │  │
│    ├──► hooks/useFileValidation.ts                                                   │  │
│    │      │                                                                           │  │
│    │      ├── validateFileSize(file, maxSize=500MB)                                  │  │
│    │      ├── validateFileType(file, allowedTypes)                                   │  │
│    │      ├── generatePreview(file) → URL.createObjectURL                           │  │
│    │      └── Returns: { isValid, errors, preview }                                  │  │
│    │                                                                                  │  │
│    ├──► store/uploadStore.ts (Zustand)                                               │  │
│    │      │                                                                           │  │
│    │      └── setFile(file, preview, metadata)                                       │  │
│    │                                                                                  │  │
│    └──► UI Update: FileCard.tsx renders                                              │  │
│                                                                                       │  │
│  ═══════════════════════════════════════════════════════════════════════════════════ │  │
│                                                                                       │  │
│  USER ACTION: Click "Analyze"                                                         │  │
│    │                                                                                  │  │
│    ▼                                                                                  │  │
│  AnalysisForm.tsx ───────────────────────────────────────────────────────────────────┤  │
│    │                                                                                  │  │
│    ├──► hooks/useAnalysis.ts (TanStack Query mutation)                               │  │
│    │      │                                                                           │  │
│    │      ├── useMutation({ mutationFn: submitAnalysis })                            │  │
│    │      │                                                                           │  │
│    │      └──► services/analysisApi.ts                                               │  │
│    │             │                                                                    │  │
│    │             ├── constructFormData(file, options)                                │  │
│    │             │      • file: File                                                 │  │
│    │             │      • generate_report: boolean                                   │  │
│    │             │      • generate_heatmaps: boolean                                 │  │
│    │             │      • defense_level: "standard" | "aggressive"                   │  │
│    │             │                                                                    │  │
│    │             ├── axios.post('/api/v1/analyze', formData, {                       │  │
│    │             │     onUploadProgress: (e) => setProgress(e.loaded/e.total)        │  │
│    │             │   })                                                              │  │
│    │             │                                                                    │  │
│    │             └── Returns: AnalysisResponse { analysis_id, status, created_at }   │  │
│    │                                                                                  │  │
│    └──► store/analysisStore.ts                                                       │  │
│           │                                                                           │  │
│           └── setCurrentAnalysis(analysis_id)                                        │  │
│                                                                                       │  │
│  ═══════════════════════════════════════════════════════════════════════════════════ │  │
│                                                                                       │  │
│  NAVIGATION: router.push(`/analysis/${analysis_id}`)                                 │  │
│    │                                                                                  │  │
│    ▼                                                                                  │  │
│  AnalysisPage.tsx ───────────────────────────────────────────────────────────────────┤  │
│    │                                                                                  │  │
│    ├──► hooks/useWebSocket.ts                                                        │  │
│    │      │                                                                           │  │
│    │      ├── Connect: new WebSocket(`ws://.../ws/analysis/${analysis_id}`)         │  │
│    │      │                                                                           │  │
│    │      ├── onmessage handler:                                                     │  │
│    │      │     • type: "status" → store/progressStore.setStatus()                   │  │
│    │      │     • type: "progress" → store/progressStore.setProgress()               │  │
│    │      │     • type: "completed" → invalidate queries, update store               │  │
│    │      │     • type: "error" → store/errorStore.setError()                        │  │
│    │      │                                                                           │  │
│    │      └── Ping/pong keep-alive every 30s                                         │  │
│    │                                                                                  │  │
│    ├──► hooks/useAnalysisDetail.ts (TanStack Query)                                  │  │
│    │      │                                                                           │  │
│    │      └── useQuery({                                                             │  │
│    │            queryKey: ['analysis', analysis_id],                                 │  │
│    │            queryFn: () => getAnalysisDetail(analysis_id),                       │  │
│    │            enabled: status === 'completed'                                      │  │
│    │          })                                                                     │  │
│    │                                                                                  │  │
│    └──► Renders:                                                                     │  │
│           ├── <ProgressIndicator /> (if status !== 'completed')                      │  │
│           ├── <ResultsPanel /> (if status === 'completed')                           │  │
│           └── <ErrorDisplay /> (if status === 'failed')                              │  │
│                                                                                       │  │
└──────────────────────────────────────────────────────────────────────────────────────┘  │
```

---

# SECTION 2: ARCHITECTURE & FILE MANIFESTO

## 2.1 Complete Directory Structure

```
/app/frontend/
├── src/
│   ├── app/                           # Next.js App Router
│   │   ├── layout.tsx                 # Root layout with providers
│   │   ├── page.tsx                   # Landing/home page
│   │   ├── globals.css                # Global styles + Tailwind
│   │   │
│   │   ├── (dashboard)/               # Dashboard route group
│   │   │   ├── layout.tsx             # Dashboard layout with sidebar
│   │   │   ├── page.tsx               # Dashboard overview
│   │   │   ├── history/page.tsx       # Analysis history
│   │   │   └── settings/page.tsx      # User settings
│   │   │
│   │   ├── analyze/                   # Analysis routes
│   │   │   ├── page.tsx               # New analysis form
│   │   │   └── text/page.tsx          # Text analysis form
│   │   │
│   │   ├── analysis/                  # Analysis results routes
│   │   │   └── [id]/                  # Dynamic analysis route
│   │   │       ├── page.tsx           # Analysis results page
│   │   │       ├── loading.tsx        # Loading skeleton
│   │   │       └── error.tsx          # Error boundary
│   │   │
│   │   └── api/                       # API Routes (BFF pattern)
│   │       └── [...path]/route.ts     # Proxy to backend API
│   │
│   ├── components/                    # React Components
│   │   ├── ui/                        # Base UI components (shadcn)
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── dialog.tsx
│   │   │   ├── progress.tsx
│   │   │   ├── tabs.tsx
│   │   │   └── ...
│   │   │
│   │   ├── layout/                    # Layout components
│   │   │   ├── Header.tsx             # Navigation header
│   │   │   ├── Sidebar.tsx            # Dashboard sidebar
│   │   │   ├── Footer.tsx             # Footer
│   │   │   └── ThemeToggle.tsx        # Dark/light mode toggle
│   │   │
│   │   ├── upload/                    # Upload components
│   │   │   ├── UploadZone.tsx         # Drag-drop upload area
│   │   │   ├── FileCard.tsx           # File preview card
│   │   │   ├── UploadProgress.tsx     # Upload progress indicator
│   │   │   └── FileTypeIcon.tsx       # File type icons
│   │   │
│   │   ├── analysis/                  # Analysis components
│   │   │   ├── AnalysisForm.tsx       # Analysis options form
│   │   │   ├── ProgressIndicator.tsx  # Real-time progress
│   │   │   ├── AnalysisTimeline.tsx   # Pipeline stage timeline
│   │   │   ├── ResultsPanel.tsx       # Results container
│   │   │   └── AnalysisCard.tsx       # Analysis list item card
│   │   │
│   │   ├── results/                   # Results display components
│   │   │   ├── TrustScoreGauge.tsx    # D3 radial gauge
│   │   │   ├── VerdictBadge.tsx       # Verdict display badge
│   │   │   ├── ScoreBreakdown.tsx     # Modality score breakdown
│   │   │   ├── ExplanationPanel.tsx   # AI explanation display
│   │   │   └── ReportDownload.tsx     # PDF report download
│   │   │
│   │   ├── modality/                  # Modality-specific components
│   │   │   ├── ModalityTabs.tsx       # Tab switcher
│   │   │   ├── VideoAnalysisPanel.tsx # Video results panel
│   │   │   ├── AudioAnalysisPanel.tsx # Audio results panel
│   │   │   ├── TextAnalysisPanel.tsx  # Text results panel
│   │   │   └── MetadataPanel.tsx      # Metadata/C2PA panel
│   │   │
│   │   ├── visualization/             # D3.js Visualizations
│   │   │   ├── HeatmapViewer.tsx      # GradCAM heatmap overlay
│   │   │   ├── SpectrogramViewer.tsx  # Audio spectrogram
│   │   │   ├── TimelineChart.tsx      # Temporal analysis chart
│   │   │   ├── RadarChart.tsx         # Multi-modality radar
│   │   │   └── FrameGallery.tsx       # Anomaly frame gallery
│   │   │
│   │   └── shared/                    # Shared components
│   │       ├── LoadingSpinner.tsx     # Loading states
│   │       ├── ErrorBoundary.tsx      # Error boundary
│   │       ├── EmptyState.tsx         # Empty state display
│   │       └── Tooltip.tsx            # Info tooltips
│   │
│   ├── hooks/                         # Custom React Hooks
│   │   ├── useAnalysis.ts             # Analysis CRUD mutations
│   │   ├── useAnalysisDetail.ts       # Analysis detail query
│   │   ├── useAnalysisList.ts         # Analysis history query
│   │   ├── useWebSocket.ts            # WebSocket connection
│   │   ├── useFileValidation.ts       # File validation logic
│   │   ├── useUpload.ts               # Upload with progress
│   │   └── useMediaPreview.ts         # Generate previews
│   │
│   ├── services/                      # API Service Layer
│   │   ├── api.ts                     # Axios instance config
│   │   ├── analysisApi.ts             # Analysis endpoints
│   │   ├── systemApi.ts               # Health, models endpoints
│   │   └── websocket.ts               # WebSocket client
│   │
│   ├── store/                         # Zustand State Stores
│   │   ├── uploadStore.ts             # Upload state
│   │   ├── analysisStore.ts           # Current analysis state
│   │   ├── progressStore.ts           # Real-time progress
│   │   ├── uiStore.ts                 # UI state (modals, theme)
│   │   └── errorStore.ts              # Global error state
│   │
│   ├── types/                         # TypeScript Types
│   │   ├── analysis.ts                # Analysis types
│   │   ├── api.ts                     # API response types
│   │   ├── websocket.ts               # WebSocket message types
│   │   └── index.ts                   # Barrel export
│   │
│   ├── lib/                           # Utility Libraries
│   │   ├── utils.ts                   # General utilities (cn, etc.)
│   │   ├── fileValidation.ts          # File validation rules
│   │   ├── formatters.ts              # Data formatters
│   │   ├── constants.ts               # App constants
│   │   └── d3/                        # D3 visualization helpers
│   │       ├── gauge.ts               # Gauge chart factory
│   │       ├── radar.ts               # Radar chart factory
│   │       └── timeline.ts            # Timeline chart factory
│   │
│   ├── providers/                     # React Context Providers
│   │   ├── QueryProvider.tsx          # TanStack Query provider
│   │   ├── ThemeProvider.tsx          # Theme provider
│   │   └── WebSocketProvider.tsx      # WebSocket context
│   │
│   └── styles/                        # Additional Styles
│       ├── animations.css             # CSS animations
│       └── d3-charts.css              # D3 chart styles
│
├── public/                            # Static Assets
│   ├── icons/                         # App icons
│   └── images/                        # Static images
│
├── tests/                             # Test Files
│   ├── components/                    # Component tests
│   ├── hooks/                         # Hook tests
│   ├── e2e/                           # Playwright E2E tests
│   └── utils/                         # Utility tests
│
├── .storybook/                        # Storybook Configuration
├── next.config.js                     # Next.js config
├── tailwind.config.js                 # Tailwind config
├── tsconfig.json                      # TypeScript config
├── vitest.config.ts                   # Vitest config
└── package.json                       # Dependencies
```

---

## 2.2 File Manifesto (Frontend)

### APP ROUTER LAYER

---

#### File: `src/app/layout.tsx`

**Role:** Root layout wrapping all pages. Initializes providers, fonts, metadata.

**Integration:**
- **Imports:** `providers/QueryProvider`, `providers/ThemeProvider`, `components/layout/Header`
- **Inputs:** `children: React.ReactNode`
- **Outputs:** Complete HTML document structure

**Schema:**
```typescript
// No data schema - layout component
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}): JSX.Element
```

**Why this approach:** App Router layouts enable persistent UI and shared state across navigations without full page reloads.

---

#### File: `src/app/page.tsx`

**Role:** Landing page with hero section, feature overview, and call-to-action to start analysis.

**Integration:**
- **Imports:** `components/ui/button`, `components/shared/AnimatedBackground`
- **Inputs:** None (static page)
- **Outputs:** Landing page JSX

**Why this approach:** Static landing page with client components for animations. No server data fetching needed.

---

#### File: `src/app/analyze/page.tsx`

**Role:** Main analysis upload page. Contains upload zone and analysis form.

**Integration:**
- **Imports:** `components/upload/UploadZone`, `components/analysis/AnalysisForm`
- **Inputs:** None (client-side only)
- **Outputs:** Analysis form page

**Schema:**
```typescript
// Page uses client components for interactivity
'use client';

export default function AnalyzePage(): JSX.Element
```

**Why this approach:** Client component for file handling and form state. No SSR needed for upload form.

---

#### File: `src/app/analysis/[id]/page.tsx`

**Role:** Dynamic analysis results page. Displays progress or results based on status.

**Integration:**
- **Imports:** `hooks/useAnalysisDetail`, `hooks/useWebSocket`, `components/analysis/*`, `components/results/*`
- **Inputs:** `params: { id: string }` (dynamic route parameter)
- **Outputs:** Analysis progress or results UI

**Schema:**
```typescript
interface AnalysisPageProps {
  params: {
    id: string;  // analysis_id from URL
  };
}

export default function AnalysisPage({ params }: AnalysisPageProps): JSX.Element
```

**Why this approach:** Dynamic route enables bookmarkable analysis URLs. WebSocket provides real-time updates.

---

### COMPONENTS - UPLOAD

---

#### File: `src/components/upload/UploadZone.tsx`

**Role:** Drag-and-drop file upload zone with validation feedback.

**Integration:**
- **Imports:** `hooks/useFileValidation`, `store/uploadStore`, `components/upload/FileCard`
- **Inputs:** `onFileSelect: (file: File) => void`, `maxSizeMB: number`, `acceptedTypes: string[]`
- **Outputs:** Drag-drop zone UI with visual feedback

**Schema:**
```typescript
interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  maxSizeMB?: number;              // Default: 500
  acceptedTypes?: string[];        // Default: video/*, audio/*, image/*
  disabled?: boolean;
  className?: string;
}

interface UploadZoneState {
  isDragging: boolean;
  error: string | null;
  preview: string | null;
}
```

**Component Contract (P0):**
- ✅ Props interface defined
- ✅ Loading state: Shows upload spinner during processing
- ✅ Error state: Displays validation errors inline
- ✅ Empty state: Shows instructions and accepted formats
- ✅ Accessibility: Keyboard navigation, screen reader support
- ✅ data-testid: `upload-zone`, `upload-zone-input`, `upload-zone-error`

**Why this approach:** Drag-drop provides intuitive UX. Client-side validation prevents unnecessary server requests.

---

#### File: `src/components/upload/FileCard.tsx`

**Role:** Display selected file with preview, metadata, and remove action.

**Integration:**
- **Imports:** `components/upload/FileTypeIcon`, `lib/formatters`
- **Inputs:** `file: File`, `preview: string | null`, `onRemove: () => void`
- **Outputs:** File card with thumbnail, name, size, type

**Schema:**
```typescript
interface FileCardProps {
  file: File;
  preview: string | null;
  onRemove: () => void;
  uploadProgress?: number;  // 0-100, if uploading
  error?: string;
}
```

**Why this approach:** Separate card component enables reuse in history and batch upload scenarios.

---

### COMPONENTS - ANALYSIS

---

#### File: `src/components/analysis/AnalysisForm.tsx`

**Role:** Analysis options form with submit action.

**Integration:**
- **Imports:** `hooks/useAnalysis`, `store/uploadStore`, `components/ui/*`
- **Inputs:** Form options
- **Outputs:** Form UI with submit button

**Schema:**
```typescript
interface AnalysisFormProps {
  file: File;
  onSubmitSuccess: (analysisId: string) => void;
  onSubmitError: (error: Error) => void;
}

interface AnalysisOptions {
  generateReport: boolean;       // Default: true
  generateHeatmaps: boolean;     // Default: true
  defenseLevel: 'none' | 'standard' | 'aggressive';  // Default: 'standard'
  modalities?: ('video' | 'audio' | 'image' | 'text')[];  // Auto-detect if empty
}
```

**Component Contract (P0):**
- ✅ Loading state: Disable form, show spinner on submit
- ✅ Error state: Display server errors
- ✅ Accessibility: Form labels, error announcements

**Why this approach:** Controlled form with React Hook Form for validation. Options map directly to API parameters.

---

#### File: `src/components/analysis/ProgressIndicator.tsx`

**Role:** Real-time analysis progress display with stage visualization.

**Integration:**
- **Imports:** `store/progressStore`, `components/ui/progress`, animation utilities
- **Inputs:** `analysisId: string`
- **Outputs:** Animated progress bar with stage labels

**Schema:**
```typescript
interface ProgressIndicatorProps {
  analysisId: string;
  className?: string;
}

interface ProgressState {
  status: AnalysisStatus;
  progressPercent: number;
  currentStage: string;
  message?: string;
}

type AnalysisStatus = 
  | 'pending'
  | 'preprocessing'
  | 'analyzing'
  | 'aggregating'
  | 'completed'
  | 'failed';
```

**Why this approach:** Zustand store synced with WebSocket enables reactive progress updates without prop drilling.

---

#### File: `src/components/analysis/AnalysisTimeline.tsx`

**Role:** Visual timeline of analysis pipeline stages.

**Integration:**
- **Imports:** `store/progressStore`, animation utilities
- **Inputs:** `analysisId: string`
- **Outputs:** Horizontal timeline with stage icons

**Schema:**
```typescript
interface TimelineStage {
  id: string;
  label: string;
  icon: React.ComponentType;
  status: 'pending' | 'active' | 'completed' | 'error';
  duration?: number;  // ms
}

const PIPELINE_STAGES: TimelineStage[] = [
  { id: 'upload', label: 'Upload', icon: UploadIcon, ... },
  { id: 'preprocess', label: 'Preprocessing', icon: ProcessIcon, ... },
  { id: 'analyze', label: 'Analysis', icon: ScanIcon, ... },
  { id: 'aggregate', label: 'Scoring', icon: CalculatorIcon, ... },
  { id: 'complete', label: 'Complete', icon: CheckIcon, ... },
];
```

**Why this approach:** Visual timeline gives users clear understanding of multi-stage process and current position.

---

### COMPONENTS - RESULTS

---

#### File: `src/components/results/TrustScoreGauge.tsx`

**Role:** D3.js radial gauge displaying Trust Score (0-100).

**Integration:**
- **Imports:** `d3`, `lib/d3/gauge`, animation utilities
- **Inputs:** `score: number`, `confidence: number`, `verdict: Verdict`
- **Outputs:** Animated radial gauge SVG

**Schema:**
```typescript
interface TrustScoreGaugeProps {
  score: number;           // 0-100
  confidence: number;      // 0-1
  verdict: Verdict;
  size?: number;           // Default: 200
  animated?: boolean;      // Default: true
  showLabel?: boolean;     // Default: true
}

type Verdict = 
  | 'authentic'
  | 'likely_authentic'
  | 'uncertain'
  | 'likely_fake'
  | 'fake';
```

**D3 Implementation:**
```typescript
// lib/d3/gauge.ts
export function createGauge(
  container: SVGElement,
  score: number,
  config: GaugeConfig
): void {
  const arc = d3.arc()
    .innerRadius(config.innerRadius)
    .outerRadius(config.outerRadius)
    .startAngle(-Math.PI / 2)
    .endAngle((score / 100) * Math.PI - Math.PI / 2);
  
  // Color scale based on verdict thresholds
  const colorScale = d3.scaleThreshold<number, string>()
    .domain([20, 40, 60, 80])
    .range(['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e']);
  
  // Animate arc from 0 to score
  container.select('.score-arc')
    .transition()
    .duration(1000)
    .attrTween('d', arcTween(score));
}
```

**Why this approach:** D3 provides smooth animations and precise control for data visualization. SVG ensures crisp rendering at any size.

---

#### File: `src/components/results/VerdictBadge.tsx`

**Role:** Verdict display badge with color coding and icon.

**Integration:**
- **Imports:** `components/ui/badge`, lucide icons
- **Inputs:** `verdict: Verdict`, `size: 'sm' | 'md' | 'lg'`
- **Outputs:** Colored badge with verdict text

**Schema:**
```typescript
interface VerdictBadgeProps {
  verdict: Verdict;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  className?: string;
}

const VERDICT_CONFIG: Record<Verdict, VerdictStyle> = {
  authentic: { 
    label: 'Authentic', 
    color: 'bg-green-500', 
    icon: ShieldCheck,
    description: 'High confidence authentic content'
  },
  likely_authentic: { 
    label: 'Likely Authentic', 
    color: 'bg-lime-500', 
    icon: Shield,
    description: 'Content appears authentic with minor concerns'
  },
  uncertain: { 
    label: 'Uncertain', 
    color: 'bg-yellow-500', 
    icon: AlertTriangle,
    description: 'Analysis inconclusive - review recommended'
  },
  likely_fake: { 
    label: 'Likely Fake', 
    color: 'bg-orange-500', 
    icon: AlertOctagon,
    description: 'Content shows manipulation indicators'
  },
  fake: { 
    label: 'Fake', 
    color: 'bg-red-500', 
    icon: XOctagon,
    description: 'High confidence manipulated content'
  },
};
```

**Why this approach:** Consistent verdict styling across app. Config object enables easy customization.

---

#### File: `src/components/results/ResultsPanel.tsx`

**Role:** Main results container organizing score, verdict, explanation, and actions.

**Integration:**
- **Imports:** All results components, `hooks/useAnalysisDetail`
- **Inputs:** `analysisId: string`
- **Outputs:** Full results layout

**Schema:**
```typescript
interface ResultsPanelProps {
  analysisId: string;
  className?: string;
}

// Internal layout structure
const ResultsPanel: React.FC<ResultsPanelProps> = ({ analysisId }) => {
  const { data, isLoading, error } = useAnalysisDetail(analysisId);
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left: Score Gauge + Verdict */}
      <div className="lg:col-span-1">
        <TrustScoreGauge ... />
        <VerdictBadge ... />
      </div>
      
      {/* Center: Explanation + Key Findings */}
      <div className="lg:col-span-2">
        <ExplanationPanel ... />
        <ScoreBreakdown ... />
      </div>
      
      {/* Actions */}
      <div className="lg:col-span-3">
        <ReportDownload ... />
      </div>
    </div>
  );
};
```

**Why this approach:** Grid layout provides responsive structure. Component composition keeps ResultsPanel focused on layout.

---

#### File: `src/components/results/ScoreBreakdown.tsx`

**Role:** Visual breakdown of scores per modality with D3 bar chart.

**Integration:**
- **Imports:** `d3`, types
- **Inputs:** `videoResult`, `audioResult`, `textResult`, `metadataResult`
- **Outputs:** Horizontal bar chart of modality scores

**Schema:**
```typescript
interface ScoreBreakdownProps {
  videoResult?: VideoResult;
  audioResult?: AudioResult;
  textResult?: TextResult;
  weights: Record<string, number>;
}

interface BreakdownItem {
  modality: string;
  score: number;
  weight: number;
  contribution: number;  // score * weight
  color: string;
}
```

**Why this approach:** Bar chart clearly shows contribution of each modality to final score. Interactive hover reveals details.

---

### COMPONENTS - MODALITY PANELS

---

#### File: `src/components/modality/ModalityTabs.tsx`

**Role:** Tab navigation for modality-specific analysis panels.

**Integration:**
- **Imports:** `components/ui/tabs`, modality panels
- **Inputs:** Analysis results, available modalities
- **Outputs:** Tabbed panel interface

**Schema:**
```typescript
interface ModalityTabsProps {
  analysisId: string;
  availableModalities: Modality[];
  videoResult?: VideoResult;
  audioResult?: AudioResult;
  textResult?: TextResult;
  metadataResult?: MetadataResult;
}
```

**Why this approach:** Tabs organize complex multi-modality results. Lazy loading panels improve initial render performance.

---

#### File: `src/components/modality/VideoAnalysisPanel.tsx`

**Role:** Detailed video analysis results with spatial/temporal/lipsync breakdowns.

**Integration:**
- **Imports:** `components/visualization/HeatmapViewer`, `components/visualization/TimelineChart`, `components/visualization/FrameGallery`
- **Inputs:** `VideoResult`
- **Outputs:** Video analysis details UI

**Schema:**
```typescript
interface VideoAnalysisPanelProps {
  result: VideoResult;
  analysisId: string;
}

interface VideoResult {
  spatial: {
    score: number;
    perFrameScores: number[];
    anomalyIndices: number[];
    heatmapUrls: string[];
  };
  temporal: {
    consistencyScore: number;
    flickeringDetected: boolean;
    anomalyTimestamps: number[];
  };
  lipSync?: {
    syncScore: number;
    manipulationProbability: number;
    detectedTechnology?: string;
  };
  aggregateScore: number;
  framesAnalyzed: number;
  faceDetected: boolean;
}
```

**Why this approach:** Dedicated panel for video provides space for complex visualizations. Sub-sections organized by analysis type.

---

#### File: `src/components/modality/AudioAnalysisPanel.tsx`

**Role:** Audio analysis results with spectrogram and vocoder artifact display.

**Integration:**
- **Imports:** `components/visualization/SpectrogramViewer`
- **Inputs:** `AudioResult`
- **Outputs:** Audio analysis details UI

**Schema:**
```typescript
interface AudioAnalysisPanelProps {
  result: AudioResult;
  analysisId: string;
}

interface AudioResult {
  syntheticProbability: number;
  vocoderArtifactsDetected: boolean;
  voiceConsistencyScore: number;
  spectrogramUrl?: string;
}
```

**Why this approach:** Spectrogram visualization helps experts understand detection rationale.

---

### COMPONENTS - VISUALIZATIONS

---

#### File: `src/components/visualization/HeatmapViewer.tsx`

**Role:** GradCAM heatmap overlay viewer with zoom and frame navigation.

**Integration:**
- **Imports:** Image component, zoom library
- **Inputs:** `heatmapUrls: string[]`, `originalFrameUrls: string[]`
- **Outputs:** Heatmap viewer with toggle and navigation

**Schema:**
```typescript
interface HeatmapViewerProps {
  heatmapUrls: string[];
  originalFrameUrls?: string[];
  selectedIndex?: number;
  onIndexChange?: (index: number) => void;
  showOverlay?: boolean;  // Toggle heatmap visibility
}
```

**Why this approach:** Heatmap visualization explains where model detected anomalies. Toggle enables comparison with original.

---

#### File: `src/components/visualization/TimelineChart.tsx`

**Role:** D3 timeline chart showing temporal analysis with anomaly markers.

**Integration:**
- **Imports:** `d3`, animation utilities
- **Inputs:** Per-frame scores, anomaly timestamps
- **Outputs:** Interactive timeline chart

**Schema:**
```typescript
interface TimelineChartProps {
  scores: number[];
  anomalyIndices: number[];
  fps?: number;
  onFrameSelect?: (index: number) => void;
  height?: number;
}
```

**D3 Implementation:**
```typescript
// lib/d3/timeline.ts
export function createTimelineChart(
  container: SVGElement,
  data: TimelineData,
  config: TimelineConfig
): void {
  const x = d3.scaleLinear()
    .domain([0, data.scores.length])
    .range([0, config.width]);
  
  const y = d3.scaleLinear()
    .domain([0, 1])
    .range([config.height, 0]);
  
  const line = d3.line<number>()
    .x((_, i) => x(i))
    .y(d => y(d))
    .curve(d3.curveMonotoneX);
  
  // Draw score line
  container.append('path')
    .datum(data.scores)
    .attr('class', 'score-line')
    .attr('d', line);
  
  // Mark anomalies
  container.selectAll('.anomaly-marker')
    .data(data.anomalyIndices)
    .enter()
    .append('circle')
    .attr('cx', d => x(d))
    .attr('cy', d => y(data.scores[d]))
    .attr('r', 5)
    .attr('class', 'anomaly-marker');
}
```

**Why this approach:** Timeline shows temporal patterns missed by aggregate scores. Click-to-seek enables exploration.

---

### HOOKS

---

#### File: `src/hooks/useAnalysis.ts`

**Role:** TanStack Query mutations for analysis CRUD operations.

**Integration:**
- **Imports:** `@tanstack/react-query`, `services/analysisApi`
- **Inputs:** Mutation options
- **Outputs:** Mutation functions and state

**Schema:**
```typescript
interface UseAnalysisReturn {
  submitAnalysis: UseMutationResult<AnalysisResponse, Error, SubmitAnalysisInput>;
  deleteAnalysis: UseMutationResult<void, Error, string>;
}

interface SubmitAnalysisInput {
  file: File;
  options: AnalysisOptions;
  onUploadProgress?: (progress: number) => void;
}

export function useAnalysis(): UseAnalysisReturn {
  const queryClient = useQueryClient();
  
  const submitAnalysis = useMutation({
    mutationFn: async ({ file, options, onUploadProgress }: SubmitAnalysisInput) => {
      return analysisApi.submitAnalysis(file, options, onUploadProgress);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['analyses'] });
    },
  });
  
  // ...
}
```

**Why this approach:** TanStack Query provides caching, loading/error states, and automatic refetching. Mutations handle optimistic updates.

---

#### File: `src/hooks/useWebSocket.ts`

**Role:** WebSocket connection management for real-time updates.

**Integration:**
- **Imports:** `store/progressStore`, `services/websocket`
- **Inputs:** `analysisId: string`
- **Outputs:** Connection state, send function

**Schema:**
```typescript
interface UseWebSocketReturn {
  isConnected: boolean;
  error: Error | null;
  send: (message: WebSocketMessage) => void;
  subscribe: (analysisId: string) => void;
  unsubscribe: (analysisId: string) => void;
}

interface WebSocketMessage {
  type: 'ping' | 'subscribe' | 'unsubscribe' | 'refresh';
  analysisId?: string;
}

export function useWebSocket(analysisId: string): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const setProgress = useProgressStore((s) => s.setProgress);
  
  useEffect(() => {
    const ws = new WebSocket(
      `${process.env.NEXT_PUBLIC_WS_URL}/ws/analysis/${analysisId}`
    );
    
    ws.onopen = () => setIsConnected(true);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'status':
        case 'progress':
          setProgress(analysisId, data);
          break;
        case 'completed':
          setProgress(analysisId, data);
          queryClient.invalidateQueries(['analysis', analysisId]);
          break;
        case 'error':
          // Handle error
          break;
      }
    };
    
    // Ping every 30s
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
    
    return () => {
      clearInterval(pingInterval);
      ws.close();
    };
  }, [analysisId]);
  
  // ...
}
```

**Why this approach:** WebSocket provides instant updates vs polling. Automatic reconnection ensures reliability.

---

#### File: `src/hooks/useFileValidation.ts`

**Role:** Client-side file validation before upload.

**Integration:**
- **Imports:** `lib/fileValidation`
- **Inputs:** `File`, validation config
- **Outputs:** Validation result

**Schema:**
```typescript
interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  fileInfo: FileInfo;
}

interface FileInfo {
  name: string;
  size: number;
  type: string;
  extension: string;
  preview?: string;
}

export function useFileValidation(
  file: File | null,
  config?: ValidationConfig
): ValidationResult {
  return useMemo(() => {
    if (!file) return { isValid: false, errors: [], warnings: [], fileInfo: null };
    
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    
    // Size validation
    if (file.size > (config?.maxSizeMB ?? 500) * 1024 * 1024) {
      errors.push({ field: 'size', message: `File exceeds ${config?.maxSizeMB}MB limit` });
    }
    
    // Type validation
    const allowedTypes = config?.acceptedTypes ?? ACCEPTED_MEDIA_TYPES;
    if (!allowedTypes.some(t => file.type.startsWith(t))) {
      errors.push({ field: 'type', message: 'Unsupported file type' });
    }
    
    // Duration warning (videos)
    if (file.type.startsWith('video/') && file.size > 100 * 1024 * 1024) {
      warnings.push({ field: 'duration', message: 'Large video may take several minutes to analyze' });
    }
    
    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      fileInfo: {
        name: file.name,
        size: file.size,
        type: file.type,
        extension: file.name.split('.').pop() ?? '',
      },
    };
  }, [file, config]);
}
```

**Why this approach:** Early validation prevents wasted uploads. Warnings inform users without blocking.

---

### SERVICES

---

#### File: `src/services/api.ts`

**Role:** Axios instance configuration with interceptors.

**Integration:**
- **Imports:** `axios`
- **Inputs:** Environment config
- **Outputs:** Configured axios instance

**Schema:**
```typescript
import axios, { AxiosInstance, AxiosError } from 'axios';

export const api: AxiosInstance = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ErrorResponse>) => {
    if (error.response?.status === 401) {
      // Handle auth error
    }
    return Promise.reject(error);
  }
);
```

**Why this approach:** Centralized axios config ensures consistent behavior. Interceptors handle cross-cutting concerns.

---

#### File: `src/services/analysisApi.ts`

**Role:** Analysis API endpoint functions.

**Integration:**
- **Imports:** `services/api`, types
- **Inputs:** API parameters
- **Outputs:** Typed API responses

**Schema:**
```typescript
export const analysisApi = {
  submitAnalysis: async (
    file: File,
    options: AnalysisOptions,
    onUploadProgress?: (progress: number) => void
  ): Promise<AnalysisResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('generate_report', String(options.generateReport));
    formData.append('generate_heatmaps', String(options.generateHeatmaps));
    formData.append('defense_level', options.defenseLevel);
    
    const response = await api.post<AnalysisResponse>('/api/v1/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (e.total && onUploadProgress) {
          onUploadProgress(Math.round((e.loaded / e.total) * 100));
        }
      },
    });
    
    return response.data;
  },
  
  getAnalysis: async (id: string): Promise<AnalysisResponse> => {
    const response = await api.get<AnalysisResponse>(`/api/v1/analyze/${id}`);
    return response.data;
  },
  
  getAnalysisDetail: async (id: string): Promise<AnalysisDetailResponse> => {
    const response = await api.get<AnalysisDetailResponse>(`/api/v1/analyze/${id}/detail`);
    return response.data;
  },
  
  listAnalyses: async (params?: ListParams): Promise<AnalysisResponse[]> => {
    const response = await api.get<AnalysisResponse[]>('/api/v1/analyze', { params });
    return response.data;
  },
  
  deleteAnalysis: async (id: string): Promise<void> => {
    await api.delete(`/api/v1/analyze/${id}`);
  },
  
  getReport: async (id: string): Promise<{ reportUrl: string }> => {
    const response = await api.get<{ report_url: string }>(`/api/v1/analyze/${id}/report`);
    return { reportUrl: response.data.report_url };
  },
  
  getHeatmaps: async (id: string): Promise<HeatmapResponse> => {
    const response = await api.get<HeatmapResponse>(`/api/v1/analyze/${id}/heatmaps`);
    return response.data;
  },
};
```

**Why this approach:** Typed API functions provide autocomplete and compile-time safety. Centralized endpoints simplify testing.

---

### STORE (ZUSTAND)

---

#### File: `src/store/progressStore.ts`

**Role:** Real-time progress state for WebSocket updates.

**Integration:**
- **Imports:** `zustand`
- **Inputs:** Progress updates from WebSocket
- **Outputs:** Progress state and actions

**Schema:**
```typescript
import { create } from 'zustand';

interface ProgressEntry {
  status: AnalysisStatus;
  progressPercent: number;
  currentStage: string;
  message?: string;
  timestamp: string;
}

interface ProgressStore {
  // State
  progress: Record<string, ProgressEntry>;
  
  // Actions
  setProgress: (analysisId: string, update: ProgressEntry) => void;
  clearProgress: (analysisId: string) => void;
  
  // Selectors
  getProgress: (analysisId: string) => ProgressEntry | undefined;
}

export const useProgressStore = create<ProgressStore>((set, get) => ({
  progress: {},
  
  setProgress: (analysisId, update) => {
    set((state) => ({
      progress: {
        ...state.progress,
        [analysisId]: update,
      },
    }));
  },
  
  clearProgress: (analysisId) => {
    set((state) => {
      const { [analysisId]: _, ...rest } = state.progress;
      return { progress: rest };
    });
  },
  
  getProgress: (analysisId) => {
    return get().progress[analysisId];
  },
}));
```

**Why this approach:** Zustand provides lightweight, TypeScript-friendly state management. Progress keyed by analysis_id supports multiple concurrent analyses.

---

#### File: `src/store/uploadStore.ts`

**Role:** Upload state management.

**Integration:**
- **Imports:** `zustand`
- **Inputs:** File selection events
- **Outputs:** Upload state

**Schema:**
```typescript
interface UploadStore {
  // State
  file: File | null;
  preview: string | null;
  uploadProgress: number;
  error: string | null;
  
  // Actions
  setFile: (file: File, preview?: string) => void;
  clearFile: () => void;
  setUploadProgress: (progress: number) => void;
  setError: (error: string | null) => void;
}

export const useUploadStore = create<UploadStore>((set) => ({
  file: null,
  preview: null,
  uploadProgress: 0,
  error: null,
  
  setFile: (file, preview) => {
    set({ file, preview, error: null });
  },
  
  clearFile: () => {
    set({ file: null, preview: null, uploadProgress: 0, error: null });
  },
  
  setUploadProgress: (progress) => {
    set({ uploadProgress: progress });
  },
  
  setError: (error) => {
    set({ error });
  },
}));
```

**Why this approach:** Centralized upload state enables progress display across components. Auto-cleanup on clear.

---

### TYPES

---

#### File: `src/types/analysis.ts`

**Role:** TypeScript type definitions for analysis data.

**Schema:**
```typescript
// ============== ENUMS ==============

export type Modality = 'video' | 'audio' | 'image' | 'text';

export type AnalysisStatus =
  | 'pending'
  | 'preprocessing'
  | 'analyzing'
  | 'aggregating'
  | 'completed'
  | 'failed';

export type Verdict =
  | 'authentic'
  | 'likely_authentic'
  | 'uncertain'
  | 'likely_fake'
  | 'fake';

// ============== CORE TYPES ==============

export interface TrustScore {
  value: number;        // 0-100
  confidence: number;   // 0-1
  calibrated: boolean;
}

export interface Explanation {
  summary: string;
  keyFindings: string[];
  manipulationRegions: ManipulationRegion[];
  confidenceRationale: string;
  methodologyUsed: string[];
}

export interface ManipulationRegion {
  regionType: string;
  location: string;
  confidence: number;
  frameIndices?: number[];
}

// ============== MODALITY RESULTS ==============

export interface SpatialResult {
  score: number;
  perFrameScores: number[];
  anomalyIndices: number[];
  heatmapUrls: string[];
}

export interface TemporalResult {
  consistencyScore: number;
  flickeringDetected: boolean;
  anomalyTimestamps: number[];
}

export interface LipSyncResult {
  syncScore: number;
  manipulationProbability: number;
  detectedTechnology?: string;
}

export interface VideoResult {
  spatial: SpatialResult;
  temporal: TemporalResult;
  lipSync?: LipSyncResult;
  aggregateScore: number;
  framesAnalyzed: number;
  faceDetected: boolean;
}

export interface AudioResult {
  syntheticProbability: number;
  vocoderArtifactsDetected: boolean;
  voiceConsistencyScore: number;
  spectrogramUrl?: string;
}

export interface TextResult {
  aiProbability: number;
  perplexityScore: number;
  burstinessScore: number;
  radarScore?: number;
}

export interface MetadataResult {
  c2pa: {
    present: boolean;
    valid?: boolean;
    issuer?: string;
    issuedAt?: string;
  };
  exifAnomalies: string[];
  fileStructureValid: boolean;
}

// ============== API RESPONSES ==============

export interface AnalysisResponse {
  analysisId: string;
  status: AnalysisStatus;
  trustScore?: TrustScore;
  verdict?: Verdict;
  explanation?: Explanation;
  reportUrl?: string;
  createdAt: string;
  completedAt?: string;
}

export interface AnalysisDetailResponse extends AnalysisResponse {
  videoResult?: VideoResult;
  audioResult?: AudioResult;
  textResult?: TextResult;
  metadataResult?: MetadataResult;
  processingTimeSeconds?: number;
}
```

**Why this approach:** Comprehensive types ensure type safety across frontend. Matches backend schema exactly.

---

# SECTION 3: DEVELOPMENT STRATEGY

## 3.1 Implementation Phases

### Phase 1: Foundation (Days 1-2)

**Goal:** Core infrastructure and routing

| Priority | File | Task |
|----------|------|------|
| P0 | `src/app/layout.tsx` | Root layout with providers |
| P0 | `src/providers/*` | QueryProvider, ThemeProvider |
| P0 | `src/services/api.ts` | Axios instance |
| P0 | `src/types/analysis.ts` | Core type definitions |
| P1 | `src/lib/utils.ts` | Utility functions (cn, formatters) |
| P1 | `src/store/uiStore.ts` | Basic UI state |

**Validation:** App renders without errors, theme toggle works.

---

### Phase 2: Upload Flow (Days 3-4)

**Goal:** Complete file upload experience

| Priority | File | Task |
|----------|------|------|
| P0 | `src/components/upload/UploadZone.tsx` | Drag-drop with validation |
| P0 | `src/hooks/useFileValidation.ts` | Client-side validation |
| P0 | `src/store/uploadStore.ts` | Upload state |
| P1 | `src/components/upload/FileCard.tsx` | File preview card |
| P1 | `src/components/upload/UploadProgress.tsx` | Progress bar |
| P0 | `src/services/analysisApi.ts` | submitAnalysis function |
| P0 | `src/hooks/useAnalysis.ts` | Submit mutation |
| P1 | `src/components/analysis/AnalysisForm.tsx` | Options form |
| P0 | `src/app/analyze/page.tsx` | Upload page |

**Validation:** Can upload file, receive analysis_id, navigate to results.

---

### Phase 3: Real-Time Progress (Days 5-6)

**Goal:** WebSocket integration and progress display

| Priority | File | Task |
|----------|------|------|
| P0 | `src/hooks/useWebSocket.ts` | WebSocket connection |
| P0 | `src/store/progressStore.ts` | Progress state |
| P0 | `src/components/analysis/ProgressIndicator.tsx` | Progress bar |
| P1 | `src/components/analysis/AnalysisTimeline.tsx` | Stage timeline |
| P0 | `src/app/analysis/[id]/page.tsx` | Results page |
| P1 | `src/app/analysis/[id]/loading.tsx` | Loading skeleton |

**Validation:** Progress updates in real-time via WebSocket.

---

### Phase 4: Results Display (Days 7-9)

**Goal:** Complete results visualization

| Priority | File | Task |
|----------|------|------|
| P0 | `src/components/results/TrustScoreGauge.tsx` | D3 gauge |
| P0 | `src/lib/d3/gauge.ts` | Gauge chart factory |
| P0 | `src/components/results/VerdictBadge.tsx` | Verdict display |
| P0 | `src/components/results/ResultsPanel.tsx` | Results layout |
| P1 | `src/components/results/ScoreBreakdown.tsx` | Modality breakdown |
| P1 | `src/components/results/ExplanationPanel.tsx` | AI explanation |
| P1 | `src/hooks/useAnalysisDetail.ts` | Detail query |

**Validation:** Full results display for completed analysis.

---

### Phase 5: Modality Panels (Days 10-12)

**Goal:** Deep-dive modality analysis views

| Priority | File | Task |
|----------|------|------|
| P0 | `src/components/modality/ModalityTabs.tsx` | Tab navigation |
| P1 | `src/components/modality/VideoAnalysisPanel.tsx` | Video panel |
| P1 | `src/components/modality/AudioAnalysisPanel.tsx` | Audio panel |
| P1 | `src/components/visualization/HeatmapViewer.tsx` | Heatmap viewer |
| P2 | `src/components/visualization/TimelineChart.tsx` | Timeline chart |
| P2 | `src/components/visualization/SpectrogramViewer.tsx` | Spectrogram |

**Validation:** All modality panels render correctly with data.

---

### Phase 6: Polish & Testing (Days 13-15)

**Goal:** Production readiness

| Priority | File | Task |
|----------|------|------|
| P0 | `tests/components/*.test.tsx` | Component tests (80% coverage) |
| P0 | `tests/e2e/*.spec.ts` | Playwright E2E tests |
| P1 | All components | Accessibility audit (axe-core) |
| P1 | All components | Loading/error/empty states |
| P2 | `.storybook/*` | Storybook documentation |
| P2 | Performance optimization | Bundle analysis, code splitting |

**Validation:** All tests pass, Lighthouse score > 90.

---

## 3.2 API Integration Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND ↔ BACKEND API MAP                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐         ┌─────────────────────────────────────┐   │
│  │     FRONTEND        │         │           BACKEND                   │   │
│  └─────────────────────┘         └─────────────────────────────────────┘   │
│                                                                             │
│  analysisApi.submitAnalysis()    POST /api/v1/analyze                      │
│  ───────────────────────────────────────────────────────────────────────►  │
│  Request: FormData { file, options }                                        │
│  Response: { analysis_id, status, created_at }                              │
│                                                                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                             │
│  useWebSocket()                  WS /ws/analysis/{id}                       │
│  ◄──────────────────────────────────────────────────────────────────────   │
│  Messages: { type: "progress", status, progress_percent, ... }              │
│                                                                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                             │
│  analysisApi.getAnalysis()       GET /api/v1/analyze/{id}                   │
│  ───────────────────────────────────────────────────────────────────────►  │
│  Response: { analysis_id, status, trust_score, verdict, ... }               │
│                                                                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                             │
│  analysisApi.getAnalysisDetail() GET /api/v1/analyze/{id}/detail            │
│  ───────────────────────────────────────────────────────────────────────►  │
│  Response: { ..., video_result, audio_result, text_result, ... }            │
│                                                                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                             │
│  analysisApi.getReport()         GET /api/v1/analyze/{id}/report            │
│  ───────────────────────────────────────────────────────────────────────►  │
│  Response: { report_url } → Presigned URL to MinIO PDF                      │
│                                                                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                             │
│  analysisApi.getHeatmaps()       GET /api/v1/analyze/{id}/heatmaps          │
│  ───────────────────────────────────────────────────────────────────────►  │
│  Response: { heatmaps: [{ key, url }], count }                              │
│                                                                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                             │
│  analysisApi.listAnalyses()      GET /api/v1/analyze                        │
│  ───────────────────────────────────────────────────────────────────────►  │
│  Query: ?status=completed&limit=20&offset=0                                 │
│  Response: AnalysisResponse[]                                               │
│                                                                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                             │
│  analysisApi.deleteAnalysis()    DELETE /api/v1/analyze/{id}                │
│  ───────────────────────────────────────────────────────────────────────►  │
│  Response: 204 No Content                                                   │
│                                                                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                             │
│  systemApi.getHealth()           GET /api/v1/health                         │
│  ───────────────────────────────────────────────────────────────────────►  │
│  Response: { status, timestamp, version, components }                       │
│                                                                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                             │
│  systemApi.getModels()           GET /api/v1/models                         │
│  ───────────────────────────────────────────────────────────────────────►  │
│  Response: { models: [...], count }                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# APPENDIX A: COMPONENT CONTRACTS

## Standard Component Contract (P0 Compliance)

Every interactive component MUST implement:

```typescript
interface ComponentContract {
  // 1. Props Interface
  props: {
    data: T;                    // Typed data input
    onAction?: () => void;      // Optional callbacks
    className?: string;         // Style override
    'data-testid'?: string;     // Test identifier
  };
  
  // 2. States
  states: {
    loading: JSX.Element;       // Loading skeleton/spinner
    error: JSX.Element;         // Error display
    empty: JSX.Element;         // Empty state
  };
  
  // 3. Accessibility
  accessibility: {
    ariaLabels: boolean;        // All interactive elements labeled
    keyboardNav: boolean;       // Tab navigation works
    focusVisible: boolean;      // Focus indicators
  };
  
  // 4. Testing
  testing: {
    dataTestIds: string[];      // All interactive elements have IDs
    unitTest: boolean;          // Unit test exists
  };
}
```

---

# APPENDIX B: API INTEGRATION SCHEMAS

## Request/Response Transformations

```typescript
// ============== REQUEST TRANSFORMERS ==============

// Transform camelCase frontend options to snake_case backend
export function transformAnalysisOptions(options: AnalysisOptions): Record<string, any> {
  return {
    generate_report: options.generateReport,
    generate_heatmaps: options.generateHeatmaps,
    defense_level: options.defenseLevel,
    modalities: options.modalities?.join(','),
  };
}

// ============== RESPONSE TRANSFORMERS ==============

// Transform snake_case backend response to camelCase frontend
export function transformAnalysisResponse(raw: any): AnalysisResponse {
  return {
    analysisId: raw.analysis_id,
    status: raw.status,
    trustScore: raw.trust_score ? {
      value: raw.trust_score.value,
      confidence: raw.trust_score.confidence,
      calibrated: raw.trust_score.calibrated,
    } : undefined,
    verdict: raw.verdict,
    explanation: raw.explanation ? {
      summary: raw.explanation.summary,
      keyFindings: raw.explanation.key_findings,
      manipulationRegions: raw.explanation.manipulation_regions?.map(transformRegion),
      confidenceRationale: raw.explanation.confidence_rationale,
      methodologyUsed: raw.explanation.methodology_used,
    } : undefined,
    reportUrl: raw.report_url,
    createdAt: raw.created_at,
    completedAt: raw.completed_at,
  };
}

export function transformVideoResult(raw: any): VideoResult {
  return {
    spatial: {
      score: raw.spatial.score,
      perFrameScores: raw.spatial.per_frame_scores,
      anomalyIndices: raw.spatial.anomaly_indices,
      heatmapUrls: raw.spatial.heatmap_urls,
    },
    temporal: {
      consistencyScore: raw.temporal.consistency_score,
      flickeringDetected: raw.temporal.flickering_detected,
      anomalyTimestamps: raw.temporal.anomaly_timestamps,
    },
    lipSync: raw.lip_sync ? {
      syncScore: raw.lip_sync.sync_score,
      manipulationProbability: raw.lip_sync.manipulation_probability,
      detectedTechnology: raw.lip_sync.detected_technology,
    } : undefined,
    aggregateScore: raw.aggregate_score,
    framesAnalyzed: raw.frames_analyzed,
    faceDetected: raw.face_detected,
  };
}
```

---

# APPENDIX C: STATE MANAGEMENT PATTERNS

## Zustand Store Pattern

```typescript
// Pattern for all Zustand stores
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface StoreState {
  // State
  data: T;
  isLoading: boolean;
  error: Error | null;
}

interface StoreActions {
  // Actions
  setData: (data: T) => void;
  reset: () => void;
  
  // Async actions
  fetchData: () => Promise<void>;
}

type Store = StoreState & StoreActions;

const initialState: StoreState = {
  data: null,
  isLoading: false,
  error: null,
};

export const useStore = create<Store>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,
        
        setData: (data) => set({ data }),
        
        reset: () => set(initialState),
        
        fetchData: async () => {
          set({ isLoading: true, error: null });
          try {
            const data = await api.getData();
            set({ data, isLoading: false });
          } catch (error) {
            set({ error: error as Error, isLoading: false });
          }
        },
      }),
      { name: 'store-name' }
    ),
    { name: 'StoreName' }
  )
);
```

---

## TanStack Query Pattern

```typescript
// Pattern for all queries
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// Query hook
export function useAnalysisDetail(analysisId: string) {
  return useQuery({
    queryKey: ['analysis', analysisId, 'detail'],
    queryFn: () => analysisApi.getAnalysisDetail(analysisId),
    enabled: !!analysisId,
    staleTime: 30_000,      // 30 seconds
    refetchInterval: (data) => 
      data?.status === 'completed' ? false : 5_000,  // Poll if not complete
  });
}

// Mutation hook
export function useDeleteAnalysis() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => analysisApi.deleteAnalysis(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analyses'] });
    },
    onError: (error) => {
      toast.error(`Failed to delete: ${error.message}`);
    },
  });
}
```

---

**END OF PRIME FRONTEND DOCUMENT**

Document Version: 1.0.0  
Last Updated: January 2026  
Compliance: AGENTS_FRONTEND.md v1.0
