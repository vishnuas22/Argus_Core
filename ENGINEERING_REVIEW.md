# Argus Core Deepfake Detection Platform — Production-Grade Engineering Review

**Date:** June 28, 2026
**Reviewer:** Engineering Review Panel
**Version:** 1.0.0

---

## 1. Executive Summary

Argus Core is a multi-modal deepfake detection platform supporting image, video, and audio analysis with explainable AI (XAI), court-admissible PDF forensic reports, and a premium dark-theme Next.js frontend. The system is containerized via Docker Compose with 6 services: frontend (Next.js), backend (FastAPI), celery-worker (Celery), MongoDB, Redis, and MinIO.

**Overall Engineering Score: 6.0 / 10**

**Strengths:**
- Comprehensive modality coverage (image, video, audio) with ensemble fusion
- Well-structured async architecture with Celery distributed task processing
- Robust ONNX Runtime inference engine with VRAM-aware batching
- Extensive XAI pipeline (GradCAM++, DCT analysis, spectrogram analysis, occlusion sensitivity)
- Production-grade PDF forensic report generation with chain of custody
- Premium dark-theme UI with thorough state management
- Strong TypeScript typing across the frontend

**Critical Risks:**
1. `cffi==2.0.0` — does not exist on PyPI; build will fail
2. `cryptography==46.0.3` — far ahead of current stable; likely unresolvable
3. MongoDB has no authentication configured
4. CORS set to `*` in all configurations
5. JWT secret variable naming mismatch (`JWT_SECRET` vs `SECRET_KEY`)
6. Model download versioning is non-deterministic (latest tags, no pinning)
7. No TLS on any connection

---

## 2. System Overview

| Property | Value |
|----------|-------|
| **Architecture Style** | Layered microservices with event-driven async processing |
| **Backend** | Python 3.11 + FastAPI + Celery |
| **Frontend** | Next.js 14 (React 18) + TypeScript + Tailwind CSS |
| **Database** | MongoDB 7 (primary), Redis 7 (cache + broker) |
| **Object Storage** | MinIO |
| **ML Runtime** | ONNX Runtime (primary), PyTorch (secondary) |
| **Deployment** | Docker Compose, 6 containers |
| **Total Backend LOC** | ~15,000+ Python |
| **Total Frontend LOC** | ~8,000+ TypeScript/TSX |

---

## 3. Architecture Breakdown

### 3.1 Service Topology

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│   Backend    │────▶│  Celery      │
│  :3000       │     │  :8000       │     │  Worker      │
│  Next.js 14  │◀────│  FastAPI     │────▶│  4 prefork   │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                            │                     │
                    ┌───────┴───────┐     ┌───────┴───────┐
                    │    MongoDB    │     │     Redis     │
                    │    :27017     │     │    :6379      │
                    └───────────────┘     └───────────────┘
                            │
                    ┌───────┴───────┐
                    │    MinIO      │
                    │    :9000      │
                    └───────────────┘
```

### 3.2 Module Hierarchy

```
backend/
├── server.py              # FastAPI entry point + lifespan
├── config.py              # pydantic-settings (env-based)
├── api/                   # REST + WebSocket endpoints
│   ├── router.py          # All API routes (878 lines)
│   ├── middleware.py      # Logging, auth, rate limit, CORS (679 lines)
│   ├── deps.py            # Dependency injection + ServiceManager (1130 lines)
│   └── websocket.py       # WebSocket progress updates
├── core/                  # Business logic
│   ├── orchestrator.py    # Celery task definitions (1541 lines)
│   ├── engine.py          # ONNX Runtime inference engine (761 lines)
│   ├── fusion.py          # Multi-modal fusion (574 lines)
│   ├── scorer.py          # Trust score + Platt calibration (606 lines)
│   ├── xai.py             # XAI evidence generation (1281 lines)
│   └── explain.py         # Template-based explanations (707 lines)
├── analyzers/             # Modality-specific analysis
│   ├── base.py            # BaseAnalyzer, metrics (635 lines)
│   ├── image.py           # Image deepfake detection (1125 lines)
│   ├── video_analyzer.py  # Video + spatial + temporal + lipsync
│   ├── audio.py           # Audio/synthetic voice detection
│   └── metadata.py        # C2PA + EXIF analysis
├── models/                # ML model management
│   ├── manager.py         # LRU-cached model lifecycle (889 lines)
│   ├── registry.py        # Model metadata registry (576 lines)
│   ├── bootstrap.py       # Startup model download
│   └── model_downloader.py# HuggingFace downloader
├── processing/            # Preprocessing pipeline
│   ├── preprocess.py      # Media extraction + content type detection (399 lines)
│   ├── sanitize.py        # Input validation + sanitization
│   └── extract.py         # Frame/audio extraction
├── forensics/             # Report generation
│   └── report.py          # PDF + text report (1055 lines)
├── storage/               # Persistence layer
│   ├── db.py              # Async MongoDB client (472 lines)
│   └── storage.py         # MinIO + local fallback (847 lines)
├── schemas/               # Pydantic v2 models
│   └── schemas.py         # All data models (679 lines)
├── interfaces/            # Abstract base classes
│   ├── storage.py         # IStorage interface
│   └── model.py           # ModelInfo dataclass
├── utils/                 # Utilities
│   ├── logging.py         # structlog configuration
│   ├── errors.py          # Custom exception hierarchy
│   ├── metrics.py         # Prometheus metrics
│   └── hardware.py        # Hardware detection
├── encoders/              # Modality feature encoders
├── fusion_layers/         # Cross-attention + self-attention layers
└── training/              # Training pipeline

frontend/
├── src/
│   ├── app/               # Next.js App Router
│   │   ├── layout.tsx     # Root layout + providers
│   │   ├── page.tsx       # Landing page (324 lines)
│   │   ├── analyze/page.tsx    # Upload page (148 lines)
│   │   └── analysis/[id]/page.tsx  # Results page (660 lines)
│   ├── components/        # React components
│   │   ├── ui/            # shadcn/ui primitives
│   │   ├── upload/        # Upload zone + form
│   │   ├── analysis/      # Progress, timeline, form
│   │   ├── results/       # Results panel + verdict
│   │   ├── modality/      # Modality-specific panels
│   │   └── xai/           # XAI explanation panel + gallery
│   ├── hooks/             # Custom hooks
│   │   ├── useXAI.ts      # XAI data + derived queries (644 lines)
│   │   └── useWebSocket.ts # WebSocket connection (514 lines)
│   ├── store/             # Zustand stores
│   │   ├── progressStore.ts  # Analysis progress (362 lines)
│   │   ├── xaiStore.ts       # XAI explanation data (506 lines)
│   │   ├── uploadStore.ts    # File upload state (308 lines)
│   │   └── uiStore.ts        # Global UI state (382 lines)
│   ├── services/          # API client layer
│   │   └── analysisApi.ts # Axios-based API service (262 lines)
│   └── types/             # TypeScript type definitions
│       └── analysis.ts    # All domain types (663 lines)
```

### 3.3 Startup Sequence

```
docker-compose up
  │
  ├─ Redis (health: redis-cli ping)
  ├─ MongoDB (health: mongosh ping)
  ├─ MinIO (health: curl /minio/health/live)
  │
  └─ Backend (waits for Redis + MongoDB + MinIO)
      ├─ docker-entrypoint.sh
      │   ├─ Create directories
      │   ├─ Hardware detection
      │   ├─ Download models (if enabled)
      │   ├─ Validate ONNX models
      │   └─ Warmup models
      └─ uvicorn server:app
          ├─ lifespan startup
          │   ├─ startup_dependencies()
          │   │   ├─ ServiceManager.start_all_services()
          │   │   ├─ wait_for_redis()
          │   │   ├─ wait_for_minio()
          │   │   ├─ StorageClient.ensure_default_buckets()
          │   │   ├─ DatabaseClient.connect()
          │   │   ├─ InferenceEngine.warmup()
          │   │   └─ ModelManager.warmup()
          │   └─ WebSocket startup
          └─ Accept requests
              └─ Celery worker (started separately by compose)
```

---

## 4. Data Flow Analysis

### 4.1 Primary Analysis Flow

```
User Upload ──▶ Frontend (/analyze)
    │
    ├─ File selected → uploadStore.setFile()
    ├─ Options set → AnalysisForm
    └─ Submit → analysisApi.submitAnalysis()
                │
    POST /api/v1/analyze ──▶ Backend router.analyze_media()
                │
                ├─ Validate file (InputSanitizer)
                ├─ Create AnalysisDocument (MongoDB)
                ├─ Upload to MinIO (argus-uploads)
                └─ Enqueue Celery task
                    │
                    ┌─── run_analysis_pipeline ───▶ Celery Worker
                    │       │
                    │       ├─ Phase 1: Preprocessing
                    │       │   ├─ Download from MinIO
                    │       │   ├─ Content type detection
                    │       │   ├─ Frame extraction (video)
                    │       │   ├─ Audio extraction (video)
                    │       │   ├─ Face detection
                    │       │   └─ Upload preprocessed data to MinIO
                    │       │
                    │       ├─ Phase 2: Parallel Analysis
                    │       │   ├─ ImageAnalyzer → InferenceEngine → ONNX
                    │       │   ├─ VideoAnalyzer → Spatial + Temporal + LipSync
                    │       │   └─ AudioAnalyzer → ONNX (AASIST/Purdue-M2)
                    │       │
                    │       ├─ Phase 3: Aggregation
                    │       │   ├─ MultiModalFusion.aggregate()
                    │       │   ├─ TrustScorer.compute()
                    │       │   └─ XAI generation
                    │       │       ├─ GradCAM++ heatmaps
                    │       │       ├─ DCT frequency analysis
                    │       │       ├─ Feature importance
                    │       │       └─ Evidence package
                    │       │
                    │       ├─ Phase 4: Persist to MongoDB
                    │       └─ Phase 5: Async PDF report generation
                    │           └─ Upload PDF to MinIO (argus-results)
                    │
                    └─── Frontend polls + WebSocket
                        ├─ GET /api/v1/analyze/{id} → status
                        ├─ GET /api/v1/analyze/{id}/detail → full results
                        └─ WebSocket /ws/analysis/{id} → real-time progress
```

### 4.2 Upload Data Flow

```
Request (multipart/form-data)
  │
  ├─ File: bytes (up to 500MB)
  │
  ▼
router.analyze_media()
  │
  ├─ 1. Read file: await file.read()           ← Entire file in memory
  ├─ 2. Validate: InputSanitizer.sanitize()     ← Extension, magic bytes, scan
  ├─ 3. Create AnalysisDocument:
  │     analysis_id = uuid4()
  │     status = PENDING
  │     input = FileInput(file_id, type, hash, size)
  │
  ├─ 4. Upload to MinIO:
  │     storage.upload_file(file, bucket_uploads, key)
  │
  └─ 5. Enqueue Celery:
        run_analysis_pipeline.delay(analysis_id, modalities, options)
        │
        ▼
      Response: { analysis_id, status: "pending", created_at }
```

### 4.3 Trust Score Computation

```
Modality Scores (0-1, higher = more fake)
  │
  ├─ Video: aggregate_score (0.30 weight)
  │   ├─ Spatial score (frame-level CNN)
  │   ├─ Temporal score (X-CLIP)
  │   └─ LipSync score (LipInc)
  │
  ├─ Audio: synthetic_probability (0.20 weight)
  │   ├─ AASIST anti-spoofing
  │   └─ Voice consistency check
  │
  ├─ Image: ai_generated_probability (primary weight)
  │   ├─ EfficientNet-B3
  │   ├─ SigLIP
  │   └─ DCT frequency analysis
  │
  └─ Metadata: confidence adjustment (±0.1)
      ├─ C2PA validity
      └─ EXIF anomalies
          │
          ▼
MultiModalFusion.aggregate()
  ├─ Confidence-weighted ensemble average
  ├─ Disagreement penalization (pull toward 0.5)
  └─ Metadata adjustment
          │
          ▼
TrustScorer.compute()
  ├─ Invert: authenticity = 1.0 - raw_score
  ├─ Platt calibration (if enabled)
  ├─ Power transform (if score_power != 1.0)
  └─ Scale to 0-100
          │
          ▼
Verdict determination:
  80-100 → authentic
  60-79  → likely_authentic
  40-59  → uncertain
  20-39  → likely_fake
  0-19   → fake
```

---

## 5. Business Logic Analysis

### 5.1 Core Business Rules

| Rule | Location | Logic |
|------|----------|-------|
| **Verdict thresholds** | `config.py:137-156` | 80+ authentic, 60-79 likely_authentic, 40-59 uncertain, 20-39 likely_fake, <20 fake |
| **Scoring weights** | `config.py:61-66` | video_spatial 0.30, video_temporal 0.25, audio 0.20, metadata 0.15 |
| **Disagreement penalization** | `fusion.py:494-498` | When modality scores diverge, pull ensemble toward 0.5 proportionally to std dev |
| **Metadata adjustment** | `fusion.py:537-556` | Valid C2PA → -0.1 (more authentic); invalid structure → +0.1 |
| **Confidence-dependent regression** | `fusion.py:208-225` | Low-confidence single modality → pull toward 0.5 |
| **Platt calibration** | `scorer.py:214-218` | Logistic regression on authenticity probability; SKIPPED for IMAGE_ONLY |
| **Uncertainty penalty** | `scorer.py:249-253` | If uncertainty > max threshold, verdict = UNCERTAIN regardless of score |
| **Model LRU eviction** | `manager.py:597-650` | When VRAM is insufficient, evict least-recently-used models |
| **Minimum model size** | `manager.py:391` | Reject models < 10KB (placeholder guard) |

### 5.2 State Machine

```
PENDING ──▶ PREPROCESSING ──▶ ANALYZING ──▶ AGGREGATING ──▶ COMPLETED
                │                 │              │
                ▼                 ▼              ▼
              FAILED             FAILED        FAILED
```

### 5.3 Hidden Assumptions

1. **ONNX models are always available** — If download fails or `purdue_m2.onnx` is corrupted, pipeline continues with neutral scores
2. **GPU memory is 4GB** — Hardcoded VRAM limits for RTX 3050 class hardware
3. **MongoDB requires no authentication** — Default Docker config has no credentials
4. **MinIO credentials are hardcoded fallbacks** — `minioadmin`/`minioadmin` in `.env`
5. **CORS `*` is safe** — Relies on reverse proxy but none is configured
6. **Import errors during XAI generation are acceptable** — Silently skipped with `except Exception`
7. **Text modality scores contributed 10%** — Now removed post-refactor
8. **Gemini API key was optional** — Now removed post-refactor

---

## 6. Dependency Analysis

### 6.1 Internal Dependency Graph

```
server.py
  ├── api/router.py → api/deps.py → core/orchestrator.py
  │   ├── processing/preprocess.py → processing/sanitize.py, processing/extract.py
  │   ├── analyzers/image.py, video_analyzer.py, audio.py, metadata.py
  │   │   └── analyzers/base.py
  │   ├── core/engine.py → models/manager.py → models/registry.py
  │   ├── core/fusion.py → fusion_layers/ (optional)
  │   ├── core/scorer.py
  │   ├── core/xai.py → core/explain.py
  │   └── storage/db.py → MongoDB (motor)
  ├── storage/storage.py → MinIO SDK (+ LocalStorageClient fallback)
  └── utils/logging.py ← (all modules)
```

### 6.2 External Dependencies (Critical)

| Package | Version | Purpose | Risk |
|---------|---------|---------|------|
| `cffi` | 2.0.0 **NONEXISTENT** | C Foreign Function Interface | **BUILD BLOCKER** |
| `cryptography` | 46.0.3 **UNRELEASED** | Cryptographic operations | **HIGH** — likely won't install |
| `onnxruntime` | 1.24.1 | ONNX model inference | Low (stable) |
| `torch` | 2.2.0 | PyTorch (limited use) | Medium (outdated) |
| `fastapi` | 0.110.1 | Web framework | Low |
| `celery` | 5.6.2 | Async task queue | Low |
| `motor` | 3.3.1 | Async MongoDB driver | Low |
| `stripe` | 14.1.0 | Payment processing | **UNUSED** — no payment endpoint found |

---

## 7. Code Quality Review

### 7.1 Critical Issues

| # | Issue | Location | Evidence |
|---|-------|----------|----------|
| CQ-1 | **`import threading` missing** | `analyzers/image.py:4` | `threading.Lock()` used without import — caused `NameError` |
| CQ-2 | **Dead backup files** | `core/xai.py.bak`, `.bak2` | 2,352 lines of stale code left in production image (REMOVED) |
| CQ-3 | **Unused forensic analyzer** | `analyzers/forensic_analyzer.py` | 755-line multi-pass engine, zero callers in pipeline (REMOVED) |
| CQ-4 | **Double sigmoid calibration** | `analyzers/image.py:924` + `core/scorer.py:214` | Sigmoid(logit_diff) then Platt sigmoid — scores compressed toward 0.5 (FIXED) |

### 7.2 Medium Issues

| # | Issue | Location | Evidence |
|---|-------|----------|----------|
| MQ-1 | **God orchestrator** | `core/orchestrator.py` | 1,541 lines, 30+ functions, 3 Celery task definitions, DB access, Redis pub/sub |
| MQ-2 | **Silent exception swallowing** | 10+ locations | `except Exception: logger.warning(...)` — hides failures |
| MQ-3 | **Global mutable state** | `core/engine.py:748` | Module-level `_engine` not thread-safe on write |
| MQ-4 | **Private attribute access** | `api/router.py:466-468` | `storage._run_sync`, `storage._client` accessed directly |
| MQ-5 | **Cross-reference globals** | `core/xai.py:402` | Accesses `_primary_onnx_session` from `analyzers.image` |
| MQ-6 | **99-line health check** | `api/router.py:443-541` | Monolithic, synchronous Redis ping in async handler, accesses private members |
| MQ-7 | **Hardcoded model paths** | `core/xai.py:395-397` | `/models/deepfake_detector_v3.onnx` hardcoded in occlusion heatmap |
| MQ-8 | **Unused neural fusion** | `core/fusion.py:260-297` | Cross-attention engine lazy imports module that may not exist; `_neural_fusion` fallback always runs instead |
| MQ-9 | **Indentation bug** | `core/explain.py:441` | `return ManipulationType.UNKNOWN` inside for loop — only checks last modality |

### 7.3 Minor Issues

| # | Issue | Location |
|---|-------|----------|
| m-1 | `os.environ.get()` bypasses pydantic-settings | `config.py:24` |
| m-2 | `allow_pickle=True` | `analyzers/forensic_analyzer.py:295` (REMOVED) |
| m-3 | Hardcoded audio constants (10ms frames, 8kHz) | `core/xai.py:649` |
| m-4 | Timestamp in reproducibility hash | `core/xai.py:1209` |
| m-5 | `_get_compatible_model_name()` is a no-op | `models/manager.py:227` |
| m-6 | Per-request sanitizer ignored for deps sanitizer | `api/router.py:123` vs `api/deps.py:866` |
| m-7 | `ensure_default_buckets()` called on every upload | `api/router.py:149` |
| m-8 | `ModalityXAI` interface duplicated in store and types | `frontend/` |

---

## 8. Performance Review

### 8.1 Bottlenecks

| # | Issue | Impact | Location |
|---|-------|--------|----------|
| P-1 | **Sync ONNX inference in async handlers** | Blocks event loop | `core/engine.py:241` — `session.run()` is synchronous |
| P-2 | **No DB connection pooling in Celery** | Per-task connection overhead | Each Celery task creates fresh MongoDB connection |
| P-3 | **Occlusion heatmap O(n²) ONNX calls** | ~300 inferences/image | `core/xai.py:362-445` — nested loop over patches |
| P-4 | **Redundant `ensure_default_buckets()`** | S3 API call per upload | `api/router.py:149` |
| P-5 | **Whole-file buffering** | 500MB file → 500MB RAM | `api/router.py:120` — `await file.read()` |
| P-6 | **Synchronous Redis in health check** | Blocks event loop | `api/router.py:477` — `redis.Redis().ping()` |
| P-7 | **Synchronous Celery inspect in health check** | Blocks event loop | `api/router.py:488` — `inspect().stats()` |
| P-8 | **nvidia-smi on every VRAM check** | ~50ms subprocess call | `models/manager.py:725` — no caching of VRAM reads |

### 8.2 Caching Opportunities

| Opportunity | Current State | Expected Gain |
|-------------|--------------|---------------|
| Redis response caching | Not used for API | ~10ms → ~1ms for repeated queries |
| VRAM usage caching | Fresh `nvidia-smi` every call | ~50ms → ~0ms |
| Model metadata caching | LRU already implemented | Already good |
| Preprocessed frame caching | Not implemented | High for repeated analysis of same media |

---

## 9. Database Review

### 9.1 MongoDB Schema

**Collection: `analyses`**
```
{
  _id: ObjectId,
  analysis_id: string (UUID),
  status: string (enum),
  input: { file_id, file_type, original_filename, file_hash, file_size },
  options: { generate_report, generate_heatmaps, defense_level, modalities },
  trust_score: { value, confidence, calibrated },
  verdict: string,
  video_result: { ... },
  audio_result: { ... },
  image_result: { ... },
  metadata_result: { ... },
  explanation: { ... },
  evidence_package: { ... },
  feature_importance: [...],
  scientific_references: [...],
  report_url: string | null,
  processing_time_seconds: number | null,
  created_at: datetime,
  completed_at: datetime | null,
  error_message: string | null
}
```

### 9.2 Indexes

| Collection | Index | Type |
|------------|-------|------|
| `analyses` | `analysis_id` | Unique (created) |
| `analyses` | `status` | Single field (created) |
| `analyses` | `created_at` | Single field (created) |
| `jobs` | `job_id` | Unique (created) |
| `audit_log` | `timestamp` | Single field (NOT TTL despite intent) |

### 9.3 Issues

| Issue | Evidence | Impact |
|-------|----------|--------|
| **No TTL on audit_log** | Comment says "TTL index" but creates DESC index | Logs grow unbounded |
| **No composite indexes** | No `{status, created_at}` index for listing queries | Full collection scan for filtered + sorted queries |
| **No text indexes** | No full-text search support | Cannot search analysis metadata |
| **No sharding strategy** | Single replica | Scale limited by single node |
| **No authentication** | MongoDB container has no credentials | Anyone on network can read/write |
| **Connection pool hardcoded** | `maxPoolSize=50, minPoolSize=10` | Not configurable without code change |

---

## 10. API Review

### 10.1 Endpoints

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| POST | `/api/v1/analyze` | Upload + analyze media | Optional |
| GET | `/api/v1/analyze/{id}` | Get analysis status | No |
| GET | `/api/v1/analyze/{id}/detail` | Get full results | No |
| DELETE | `/api/v1/analyze/{id}` | Delete analysis | Required |
| GET | `/api/v1/analyze` | List analyses | No |
| GET | `/api/v1/analyze/{id}/report` | Get report URL | No |
| GET | `/api/v1/analyze/{id}/heatmaps` | Get heatmap URLs | No |
| GET | `/api/v1/analyze/{id}/xai` | Get XAI explanations | No |
| GET | `/api/v1/analyze/{id}/xai/heatmaps` | Get XAI heatmap overlays | No |
| GET | `/api/v1/health` | Health check | No |
| GET | `/api/v1/models` | List models | No |
| GET | `/api/v1/stats` | Aggregate stats | No |

### 10.2 Issues

| Issue | Evidence |
|-------|----------|
| **No response pagination metadata** | `list_analyses` has `limit`/`offset` but no `total`/`next` |
| **Inconsistent error shapes** | Mix of `HTTPException` (plain dict) and `ErrorResponse` (structured) |
| **Health check is 99 lines** | Monolithic, accesses private members, synchronous Redis calls |
| **Delete has no ownership check** | Anyone can delete any analysis (if authenticated) |
| **No caching headers** | Every request hits MongoDB |
| **No rate limit headers** | Rate limiting exists but no `X-RateLimit-*` headers returned |
| **Placeholder report endpoint** | `get_report()` returns presigned URL without generating PDF |

---

## 11. Security Review (OWASP Top 10)

### 11.1 Findings

| OWASP Category | Issue | Severity | Location |
|----------------|-------|----------|----------|
| **A01: Broken Access Control** | No ownership check on DELETE | High | `api/router.py:348` |
| **A02: Cryptographic Failures** | No TLS; secrets in `.env` | High | Infrastructure |
| **A03: Injection** | Path traversal in LocalStorage | Low | `storage/storage.py:66-73` (mitigated) |
| **A04: Insecure Design** | CORS `*`; MongoDB no auth | Critical | `config.py:79`, `docker-compose.yml` |
| **A05: Security Misconfiguration** | JWT secret mismatch; default creds | High | `.env` vs `docker-compose.yml` |
| **A06: Vulerable Components** | `cffi==2.0.0` (nonexistent) | Critical | `requirements.txt` |
| **A07: Auth Failures** | JWT optional on most endpoints | Medium | `api/router.py` |
| **A08: Data Integrity Failures** | No CSRF; no integrity checks on upload | Medium | Frontend |
| **A09: Logging Failures** | Audit trail not linked (previous_hash=None) | Medium | `storage/db.py` |
| **A10: SSRF** | Model download from HuggingFace | Low | `docker-entrypoint.sh` |

### 11.2 Specific Issues

1. **CORS `*`** — `config.py:79`, `.env:51`. Any website can make authenticated requests.
2. **MongoDB no auth** — No `MONGO_INITDB_ROOT_USERNAME` set in compose.
3. **JWT secret mismatch** — `.env` uses `JWT_SECRET`, compose uses `SECRET_KEY`.
4. **Hardcoded secrets** — `"change-this-in-production"` for SECRET_KEY and API_KEY_SALT.
5. **MinIO weak credentials** — `minioadmin`/`minioadmin` (`.env`) or `argusadmin`/`argussecret123` (compose).
6. **All ports exposed** — No reverse proxy; Redis, MongoDB, MinIO ports exposed to host.
7. **No TLS** — All HTTP, WS, MinIO S3 API traffic in plaintext.
8. **`allow_pickle=True`** — In removed file, no longer a concern.

---

## 12. Reliability Review

### 12.1 Mechanisms

| Mechanism | Status | Assessment |
|-----------|--------|------------|
| **Celery task retries** | Implemented | Exponential backoff `30 * (retries + 1)`, max 3 retries |
| **Storage retry** | Implemented | `_retry_operation` with exponential backoff (1s base, 10s max, 3 retries) |
| **Storage local fallback** | Implemented | After 3 MinIO failures, permanently switches to local filesystem |
| **MinIO reconnection** | Implemented | `_reconnect()` recreates client on errors |
| **MongoDB retries** | None | Single attempt, raises on failure |
| **Circuit breaker** | None | No pattern used downstream |
| **WebSocket reconnection** | Implemented | Exponential backoff (3s base, 30s max, 5 attempts) |
| **Graceful degradation** | Partial | XAI failures → continue without; model missing → neutral scores |
| **Health checks** | Docker-level | Container healthchecks for all 6 services |

### 12.2 Issues

| Issue | Impact | Location |
|-------|--------|----------|
| **Permanent local fallback after 3 MinIO failures** | No recovery to MinIO | `storage/storage.py:339` |
| **No circuit breaker for MongoDB** | Cascading failures | `storage/db.py` |
| **No timeout on model loading** | Infinite hang possible | `models/manager.py:358` |
| **`download_stream` has no local fallback** | Inconsistent with `download_file` | `storage/storage.py:597-631` |
| **No request timeout on API endpoints** | Long-running requests hold connections | `api/router.py` |
| **`Except Exception` hides real failures** | Difficult to diagnose | Multiple locations |

---

## 13. Scalability Review

### 13.1 Scaling Characteristics

| Dimension | Current State | Limit |
|-----------|--------------|-------|
| **Horizontal (backend)** | Stateless FastAPI; replicable | Unlimited (behind reverse proxy) |
| **Horizontal (workers)** | 4 prefork workers configured | Per-node CPU cores |
| **Database** | Single MongoDB | Write throughput limited |
| **Storage** | Single MinIO | Capacity + throughput limited |
| **GPU inference** | Single GPU, 4GB VRAM | Cannot run larger models |
| **Concurrent uploads** | No explicit limit | Memory-bound (500MB/file) |

### 13.2 Issues

| Issue | Evidence |
|-------|----------|
| **No reverse proxy for load balancing** | All services expose raw ports |
| **Single MongoDB has no replicas** | No read scaling, no failover |
| **MinIO has no clustering** | Default standalone mode |
| **Frontend is single instance** | No horizontal scaling for Next.js |
| **Celery concurrency hardcoded to 4** | Not configurable via env |
| **No response caching** | Every request hits DB |
| **No read replicas** | All reads go to primary |

---

## 14. Testability Review

### 14.1 Test Infrastructure

| Tool | Purpose | Coverage |
|------|---------|----------|
| **pytest** | Python testing | ~10 test files remaining |
| **pytest-asyncio** | Async test support | auto mode |
| **vitest** (v4) | Frontend testing | Configured but no test files found |
| **Playwright** | E2E testing | Configured but no test files found |
| **Coverage** | Threshold 60% | Set in `pyproject.toml` |

### 14.2 Issues

| Issue | Evidence |
|-------|----------|
| **No unit tests for core modules** | `core/engine.py`, `core/fusion.py`, `core/scorer.py` — no test files |
| **No frontend tests** | `vitest` configured but `tests/` dir is empty |
| **No E2E tests** | Playwright configured but no spec files |
| **60% coverage threshold** | Low for security-sensitive application |
| **Tight coupling hinders mocking** | Global singletons, `from config import config` |
| **`pyproject.toml` includes `.` as test path** | May collect non-test files |

---

## 15. Documentation Review

| Artifact | Quality | Issues |
|----------|---------|--------|
| **Inline docstrings** | Excellent | Comprehensive module-level and function-level docstrings |
| **API documentation** | Good | FastAPI auto-generates OpenAPI docs at `/docs` and `/redoc` |
| **README** | Not reviewed | Not in scope |
| **Architecture docs** | Partial | Docstrings reference `PRIME_ARGUS_DOCUMENT.md` which was not found |
| **Code comments** | Good | Generally well-commented with rationale |
| **Type hints** | Strong | Python type hints throughout; TypeScript types comprehensive |
| **Naming consistency** | Good | Snake_case in Python, camelCase in TypeScript (intentional) |

---

## 16. Prioritized Issues Table

| Priority | ID | Category | Issue | Effort | Confidence |
|----------|----|----------|-------|--------|------------|
| 🔴 Critical | INF-1 | Build | `cffi==2.0.0` does not exist on PyPI | 5 min | High |
| 🔴 Critical | INF-2 | Build | `cryptography==46.0.3` far ahead of stable | 5 min | High |
| 🔴 Critical | INF-3 | Build | `click-plugins==1.1.1.2` may not resolve | 5 min | High |
| 🔴 Critical | SEC-1 | Security | MongoDB has no authentication | 30 min | High |
| 🔴 Critical | SEC-2 | Security | CORS `*` in production | 10 min | High |
| 🔴 Critical | SEC-3 | Security | JWT secret variable name mismatch | 15 min | High |
| 🟡 High | CQ-1 | Code Quality | God orchestrator (1,541 lines) | 4 hr | High |
| 🟡 High | CQ-2 | Code Quality | Silent exception swallowing | 2 hr | Medium |
| 🟡 High | PERF-1 | Performance | Sync ONNX blocks async event loop | 4 hr | High |
| 🟡 High | PERF-2 | Performance | No DB connection pooling in Celery | 4 hr | Medium |
| 🟡 High | SEC-4 | Security | Hardcoded fallback secrets in compose | 10 min | High |
| 🟡 High | SEC-5 | Security | All ports exposed, no reverse proxy | 2 hr | High |
| 🟡 High | REL-1 | Reliability | Permanent local storage fallback | 1 hr | Medium |
| 🔵 Medium | CQ-3 | Code Quality | Global mutable state in engine | 1 hr | Medium |
| 🔵 Medium | CQ-4 | Code Quality | Private attribute access in router | 30 min | High |
| 🔵 Medium | DB-1 | Database | No TTL index on audit_log | 10 min | High |
| 🔵 Medium | DB-2 | Database | No composite indexes for listing | 30 min | Medium |
| 🔵 Medium | API-1 | API | Health check is 99-line monolith | 1 hr | High |
| 🔵 Medium | INF-4 | Infrastructure | GPU disabled in compose (`GPU_ENABLED: false`) | 5 min | High |
| 🔵 Medium | INF-5 | Infrastructure | No resource limits on containers | 30 min | High |
| ⚪ Low | m-1 | Code Quality | Duplicate `ModalityXAI` interface | 10 min | High |
| ⚪ Low | m-2 | Code Quality | Placeholder report endpoint | 30 min | Medium |
| ⚪ Low | m-3 | Code Quality | Redundant `ensure_default_buckets()` per upload | 15 min | High |
| ⚪ Low | m-4 | Code Quality | Timestamp in reproducibility hash | 15 min | Medium |
| ⚪ Low | TEST-1 | Testing | No frontend tests | 2 hr | High |

---

## 17. Technical Debt Assessment

### 17.1 Debt by Category

| Category | Estimated Debt | Severity |
|----------|---------------|----------|
| **Build blockers** | 1-2 days | Critical — system cannot build |
| **Security** | 2-3 days | High — data at risk |
| **Code quality** | 3-5 days | Medium — maintainability |
| **Performance** | 2-3 days | Medium — scalability |
| **Testing** | 3-5 days | Medium — confidence |
| **Infrastructure** | 1-2 days | Low — hardening |
| **Documentation** | 1 day | Low |

**Total estimated technical debt: 2-3 weeks**

### 17.2 Hotspots

- **`core/orchestrator.py`** (1,541 lines) — 17% of all backend business logic in one file
- **`core/xai.py`** (1,281 lines) — Hardcoded paths, global state access, no timeouts on expensive ops
- **`api/deps.py`** (1,130 lines) — Service auto-starter + dependency injection
- **`forensics/report.py`** (1,055 lines) — 259-line PDF generation function
- **`api/router.py`** (878 lines) — 99-line health check function, private member access

---

## 18. Refactoring Roadmap

### Phase 1 — Build Fixes (Day 1)
- Fix `cffi==2.0.0` → `cffi>=1.17.0`
- Fix `cryptography==46.0.3` → `cryptography>=44.0.0`
- Fix `click-plugins==1.1.1.2` → `click-plugins>=1.1.1`
- Fix `click-didyoumean==0.3.1` → `click-didyoumean>=0.3.0`

### Phase 2 — Security (Day 1-2)
- Configure MongoDB authentication in docker-compose
- Restrict CORS to frontend origin (not `*`)
- Fix JWT secret variable name consistency (`SECRET_KEY` everywhere)
- Move secrets from .env and compose to environment variables/secrets manager
- Add reverse proxy (nginx/traefik) with TLS termination
- Remove `GPU_ENABLED: false` override if GPU should be used

### Phase 3 — Infrastructure Hardening (Day 2-3)
- Add resource limits (mem_limit, cpus) to all compose services
- Add log rotation configuration
- Add restart policy to frontend service
- Set MinIO to specific version (not `latest`)
- Add `--no-cache-dir` consistently in Dockerfile

### Phase 4 — Code Quality (Week 1-2)
- Split `core/orchestrator.py` into modules (orchestrator + analysis + results)
- Add explicit DB connection pooling for Celery tasks
- Replace silent `except Exception` blocks with specific exception handling
- Move health check to dedicated module
- Remove global state patterns (singletons with reset)
- Fix `core/explain.py` indentation bug
- Move hardcoded constants to config

### Phase 5 — Performance (Week 2)
- Move synchronous ONNX inference to thread pool executor
- Add Redis response caching for GET endpoints
- Add MongoDB connection pooling for Celery
- Add VRAM usage caching with configurable TTL
- Add streaming upload support for large files

### Phase 6 — Testing (Week 2-3)
- Add unit tests for core modules (engine, fusion, scorer)
- Add integration tests for Celery pipeline
- Write frontend component tests
- Add E2E tests with Playwright
- Raise coverage threshold to 80%

---

## 19. Production-Grade Refactored Code

The following critical fixes have already been applied:

1. **`analyzers/image.py:4`** — Added `import threading` (Critical: NameError crash)
2. **`core/scorer.py:214`** — Skip Platt calibration for `IMAGE_ONLY` (High: double sigmoid bug)
3. **Removed `core/xai.py.bak`, `.bak2`** — 2,352 lines dead code
4. **Removed `analyzers/forensic_analyzer.py`** — 755 lines unused code
5. **Removed Gemini integration** — 15 files, ~5,400 lines (config, analyzers, reasoning, chat, schemas, API, frontend)
6. **Removed text modality** — All text analysis routing, scoring, UI removed
7. **Fixed `docker-compose.yml:228-229`** — `NEXT_PUBLIC_API_URL` uses Docker service name
8. **Fixed `backend/Dockerfile:52`** — `libssl3` → `libssl3t64` for Debian Bookworm

### Docker Networking Verification

```bash
# Verify all 6 containers are healthy
docker compose ps

# Expected output:
# NAME                STATUS
# argus-redis         healthy
# argus-mongodb       healthy  
# argus-minio         healthy
# argus-backend       healthy
# argus-celery-worker healthy
# argus-frontend      healthy
```

### Rebuild Commands

```bash
# Rebuild all images with no cache for clean build
docker compose build --no-cache

# Start all services
docker compose up -d

# Verify health
docker compose ps

# Check backend logs for successful startup
docker compose logs backend | tail -20

# Check Celery logs for worker registration
docker compose logs celery-worker | tail -20
```

---

## 20. Risks Remaining

### Unresolved Blockers

| Risk | Impact | Workaround |
|------|--------|------------|
| **`purdue_m2.onnx` is corrupted** | Audio analyzer scores always neutral | Disable audio or replace model file |
| **PyTorch models blocked from engine** | Video analyzers bypass engine | Already working via `manager.get_model()` |
| **Unused `stripe` package** | Unnecessary dependency | Remove if no payment integration planned |

### Architectural Risks

1. **Permanent local storage fallback** — A transient MinIO outage permanently degrades storage
2. **No recovery from local fallback** — Once switched, never retries MinIO
3. **Cross-attention fusion is unused** — The neural fusion path imports a module that may not exist; the ensemble fallback always runs instead
4. **Audio model is a single point of failure** — If `purdue_m2.onnx` is corrupted, audio analysis silently returns neutral scores

---

## 21. Future Recommendations

### Short Term (Next 2 Weeks)

1. **Fix build blockers** — Resolve nonexistent package versions in `requirements.txt`
2. **Secure MongoDB** — Add authentication to the Docker configuration
3. **Restrict CORS** — Change from `*` to frontend origin
4. **Add reverse proxy** — nginx with TLS in front of all services
5. **Fix audio model** — Download working `purdue_m2.onnx` or replace with alternative

### Medium Term (1-2 Months)

1. **Add comprehensive test suite** — Unit + integration + E2E
2. **Split orchestrator** — Break 1,541-line god file into focused modules
3. **Add request timeouts** — Prevent long-running requests from holding connections
4. **Add OpenTelemetry tracing** — End-to-end request observability
5. **Move to async ONNX inference** — Thread pool to avoid blocking event loop

### Long Term (3+ Months)

1. **Kubernetes deployment** — Replace Docker Compose with K8s for scaling
2. **Database sharding** — Horizontal scaling for MongoDB
3. **Model versioning** — Pin model versions with integrity hashes
4. **Streaming file upload** — Avoid buffering 500MB files in memory
5. **GPU support** — Enable TensorRT and GPU inference in production

---

## 22. Overall Engineering Score

**Score: 6.0 / 10**

### Breakdown

| Category | Score | Reasoning |
|----------|-------|-----------|
| **Architecture** | 7/10 | Well-layered, event-driven, but god objects exist |
| **Code Quality** | 6/10 | Strong typing, comprehensive, but dead code + silent errors |
| **Performance** | 5/10 | Sync ONNX in async, no connection pooling, whole-file buffering |
| **Security** | 4/10 | CORS `*`, MongoDB no auth, no TLS, hardcoded secrets |
| **Reliability** | 6/10 | Retries + fallbacks, but permanent degradation + no circuit breakers |
| **Scalability** | 4/10 | Single MongoDB, single MinIO, no caching, no reverse proxy |
| **Testability** | 3/10 | No unit tests, no frontend tests, 60% coverage threshold |
| **Documentation** | 7/10 | Excellent inline docs, comprehensive types, missing README review |
| **Infrastructure** | 5/10 | Dockerized but no resource limits, no TLS, no log rotation |
| **Maintainability** | 6/10 | Well-organized modules, but god classes + dead code reduce agility |

### Strengths
- Comprehensive modality coverage with proper ensemble fusion
- Production-grade async architecture with Celery
- Strong XAI pipeline with scientific references
- Premium frontend UX with thorough state management
- Extensive type safety (Python + TypeScript)

### Weaknesses
- Build blockers prevent clean installation
- Security posture needs hardening for production
- Limited test coverage
- Performance bottlenecks in critical paths
- Infrastructure lacks production hardening

---

*Report generated June 28, 2026. Findings based on comprehensive codebase analysis of ~23,000 lines across 60+ files.*
