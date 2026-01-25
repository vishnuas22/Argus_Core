# PRIME ARGUS DOCUMENT
## Multi-Modal Deepfake Detection & Forensic Analysis Platform
### Implementation Blueprint v1.0

**Classification:** Production-Grade Implementation Specification  
**Date:** January 2026  
**Compliance:** AGENTS.md (Backend) | AGENTS_FRONTEND.md (Frontend)

---

# TABLE OF CONTENTS

1. [Section 1: The "Life of a Request" Flow](#section-1-the-life-of-a-request-flow)
2. [Section 2: Architecture & File Manifesto](#section-2-architecture--file-manifesto)
3. [Section 3: Development Strategy](#section-3-development-strategy)
4. [Appendix A: Shared Schemas](#appendix-a-shared-schemas)
5. [Appendix B: Configuration Reference](#appendix-b-configuration-reference)

---

# SECTION 1: THE "LIFE OF A REQUEST" FLOW

## 1.1 Complete Request Lifecycle

This traces a single `POST /api/analyze` request from upload to response.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE REQUEST LIFECYCLE                                     │
│                           Total Time: ~15-20s (30s video)                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: INGESTION (0-500ms)                                                     │   │
│  │ Files: router.py → sanitize.py → storage.py → schemas.py                        │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                  │
│      │  T+0ms:    router.py receives multipart/form-data                               │
│      │  T+5ms:    sanitize.py validates magic bytes, size, content-type                │
│      │  T+50ms:   sanitize.py runs adversarial input sanitization                      │
│      │  T+100ms:  storage.py streams file to MinIO (async chunked upload)             │
│      │  T+400ms:  schemas.py creates AnalysisRequest, generates analysis_id           │
│      │  T+500ms:  MongoDB insert (analyses collection, status="pending")              │
│      ▼                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 2: PREPROCESSING (500ms-2s)                                                │   │
│  │ Files: orchestrator.py → preprocess.py → extract.py                              │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                  │
│      │  T+500ms:  orchestrator.py enqueues "preprocess" job to Redis/Celery           │
│      │  T+600ms:  Celery worker picks up job, calls preprocess.py                     │
│      │  T+700ms:  preprocess.py detects media type, routes to appropriate extractor   │
│      │  T+800ms:  extract.py (video): ffmpeg extracts frames + audio track            │
│      │  T+1.5s:   extract.py: frame sampling (every 5th frame for <30s video)         │
│      │  T+1.8s:   extract.py: face detection (RetinaFace) on sampled frames           │
│      │  T+2s:     Preprocessed data stored in MinIO /preprocessed/{analysis_id}/      │
│      ▼                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 3: MULTI-MODEL INFERENCE (2s-14s)                                          │   │
│  │ Files: orchestrator.py → engine.py → video.py → audio.py → text.py → image.py   │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                  │
│      │  T+2s:     orchestrator.py dispatches parallel analysis jobs                   │
│      │  T+2.1s:   engine.py loads required models via ModelManager (LRU cache)        │
│      │                                                                                  │
│      │  ┌─── PARALLEL EXECUTION ───────────────────────────────────────────────┐      │
│      │  │                                                                       │      │
│      │  │  video.py (T+2.2s → T+10s):                                          │      │
│      │  │    • spatial.py: EfficientNet-B3 per-frame analysis                  │      │
│      │  │    • temporal.py: X-CLIP temporal consistency                        │      │
│      │  │    • lipsync.py: LIPINC-V2 lip-sync verification                    │      │
│      │  │    • explain.py: GradCAM heatmap generation                         │      │
│      │  │                                                                       │      │
│      │  │  audio.py (T+2.2s → T+6s):                                           │      │
│      │  │    • spectrogram extraction                                          │      │
│      │  │    • Purdue-M2 vocoder artifact detection                           │      │
│      │  │    • voice consistency analysis                                      │      │
│      │  │                                                                       │      │
│      │  │  metadata.py (T+2.2s → T+3s):                                        │      │
│      │  │    • C2PA manifest extraction                                        │      │
│      │  │    • EXIF analysis                                                   │      │
│      │  │    • Hash verification                                               │      │
│      │  │                                                                       │      │
│      │  └───────────────────────────────────────────────────────────────────────┘      │
│      │                                                                                  │
│      │  T+14s:    All analysis results returned to orchestrator.py                    │
│      ▼                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 4: AGGREGATION & SCORING (14s-15s)                                         │   │
│  │ Files: fusion.py → scorer.py → explain.py                                        │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                  │
│      │  T+14s:    fusion.py receives all modality results                             │
│      │  T+14.2s:  scorer.py computes weighted Trust Score (dynamic weights)           │
│      │  T+14.5s:  explain.py generates human-readable explanation via LLM             │
│      │  T+14.8s:  scorer.py determines verdict (authentic/uncertain/fake)             │
│      │  T+15s:    Results written to MongoDB (status="completed")                     │
│      ▼                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 5: RESPONSE (15s-15.5s)                                                    │   │
│  │ Files: router.py → schemas.py                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                  │
│      │  T+15s:    router.py fetches completed analysis from MongoDB                   │
│      │  T+15.2s:  schemas.py serializes AnalysisResponse                             │
│      │  T+15.3s:  JSON response returned to client                                    │
│      │  T+15.5s:  WebSocket notification sent (if subscribed)                        │
│      ▼                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 6: ASYNC REPORT GENERATION (Background, 15s-30s)                           │   │
│  │ Files: orchestrator.py → report.py → forensics.py                                │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│      │                                                                                  │
│      │  T+15s:    orchestrator.py queues "report" job (non-blocking)                  │
│      │  T+16s:    report.py generates PDF with heatmaps, findings                     │
│      │  T+25s:    forensics.py adds C2PA manifest (if requested)                      │
│      │  T+30s:    PDF uploaded to MinIO, report_url updated in MongoDB               │
│      ▼                                                                                  │
│                                                                                         │
│                            ✓ REQUEST COMPLETE                                          │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1.2 File Interaction Sequence

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              FILE INTERACTION SEQUENCE                                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  CLIENT                                                                              │
│    │                                                                                 │
│    │ POST /api/analyze (multipart/form-data)                                        │
│    ▼                                                                                 │
│  router.py ──────────────────────────────────────────────────────────────────────┐  │
│    │                                                                              │  │
│    │ Dependency Injection:                                                        │  │
│    │   • sanitize.py (InputSanitizer)                                            │  │
│    │   • storage.py (StorageClient)                                              │  │
│    │   • db.py (DatabaseClient)                                                  │  │
│    │                                                                              │  │
│    ├──► sanitize.py.validate_file(file: UploadFile) → SanitizedFile             │  │
│    │      │                                                                       │  │
│    │      ├── Magic byte verification                                            │  │
│    │      ├── Content-type validation                                            │  │
│    │      ├── Size limit check (from config.py)                                  │  │
│    │      └── Adversarial pattern detection                                      │  │
│    │                                                                              │  │
│    ├──► storage.py.upload_file(file, bucket) → ObjectKey                        │  │
│    │      │                                                                       │  │
│    │      └── MinIO async chunked upload                                         │  │
│    │                                                                              │  │
│    ├──► schemas.py.AnalysisRequest.create() → AnalysisRequest                   │  │
│    │                                                                              │  │
│    ├──► db.py.insert_analysis(request) → analysis_id                            │  │
│    │                                                                              │  │
│    └──► orchestrator.py.enqueue_analysis(analysis_id) → job_id                  │  │
│           │                                                                       │  │
│           └──► Redis LPUSH "argus:jobs:preprocess"                               │  │
│                                                                                   │  │
│  ═══════════════════════════════════════════════════════════════════════════════ │  │
│                                                                                   │  │
│  CELERY WORKER (Background Process)                                               │  │
│    │                                                                              │  │
│    ▼                                                                              │  │
│  orchestrator.py ────────────────────────────────────────────────────────────────┤  │
│    │                                                                              │  │
│    ├──► preprocess.py.process(analysis_id)                                       │  │
│    │      │                                                                       │  │
│    │      ├──► extract.py.extract_video_data(file) → VideoData                   │  │
│    │      │      • Frame extraction (ffmpeg-python)                              │  │
│    │      │      • Audio track extraction                                        │  │
│    │      │      • Face detection (RetinaFace)                                   │  │
│    │      │                                                                       │  │
│    │      └──► storage.py.upload_preprocessed(data)                              │  │
│    │                                                                              │  │
│    ├──► engine.py.analyze(analysis_id, modalities)                               │  │
│    │      │                                                                       │  │
│    │      ├── ModelManager.load_models(required_models)                          │  │
│    │      │                                                                       │  │
│    │      ├──► video.py.analyze(frames, faces) → VideoResult                     │  │
│    │      │      ├── spatial.py.analyze_frames() → SpatialScore                  │  │
│    │      │      ├── temporal.py.analyze_consistency() → TemporalScore           │  │
│    │      │      ├── lipsync.py.verify_sync(frames, audio) → LipSyncScore        │  │
│    │      │      └── explain.py.generate_heatmaps() → List[HeatmapURL]           │  │
│    │      │                                                                       │  │
│    │      ├──► audio.py.analyze(audio_track) → AudioResult                       │  │
│    │      │      ├── Mel-spectrogram generation                                  │  │
│    │      │      └── Purdue-M2 vocoder detection                                 │  │
│    │      │                                                                       │  │
│    │      └──► metadata.py.analyze(file) → MetadataResult                        │  │
│    │             ├── C2PA extraction                                              │  │
│    │             └── EXIF analysis                                                │  │
│    │                                                                              │  │
│    ├──► fusion.py.aggregate(video_result, audio_result, metadata_result)         │  │
│    │      │                                                                       │  │
│    │      └──► scorer.py.compute_trust_score(aggregated) → TrustScore           │  │
│    │                                                                              │  │
│    ├──► explain.py.generate_explanation(results) → Explanation                   │  │
│    │                                                                              │  │
│    └──► db.py.update_analysis(analysis_id, results)                              │  │
│                                                                                   │  │
└──────────────────────────────────────────────────────────────────────────────────┘  │
```

---

# SECTION 2: ARCHITECTURE & FILE MANIFESTO

## 2.1 Complete Directory Structure

```
/app/backend/
├── server.py                 # FastAPI application entry point
├── config.py                 # Configuration loader (env vars, YAML)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (DO NOT COMMIT)
│
├── api/                      # API Layer (HTTP Interface)
│   ├── __init__.py
│   ├── router.py             # Main API router, endpoint definitions
│   ├── deps.py               # Dependency injection providers
│   ├── middleware.py         # CORS, rate limiting, auth middleware
│   └── websocket.py          # WebSocket handlers for real-time updates
│
├── core/                     # Core Business Logic
│   ├── __init__.py
│   ├── orchestrator.py       # Job orchestration, Celery task definitions
│   ├── engine.py             # Model loading, inference engine
│   ├── fusion.py             # Multi-modal result aggregation
│   ├── scorer.py             # Trust Score computation
│   └── explain.py            # Explainability (GradCAM, textual)
│
├── analyzers/                # Modality-Specific Analyzers
│   ├── __init__.py
│   ├── base.py               # Abstract base class for analyzers
│   ├── video.py              # Video deepfake detection
│   ├── audio.py              # Audio deepfake detection
│   ├── image.py              # Image analysis
│   ├── text.py               # AI-generated text detection
│   └── metadata.py           # C2PA, EXIF analysis
│
├── analyzers/video/          # Video Sub-Analyzers
│   ├── __init__.py
│   ├── spatial.py            # Per-frame spatial analysis
│   ├── temporal.py           # Temporal consistency
│   └── lipsync.py            # Lip-sync verification
│
├── processing/               # Data Processing Pipeline
│   ├── __init__.py
│   ├── preprocess.py         # Media preprocessing orchestration
│   ├── extract.py            # Frame/audio extraction
│   ├── sanitize.py           # Input validation, adversarial defense
│   └── transform.py          # Data transformations for inference
│
├── storage/                  # Storage Layer
│   ├── __init__.py
│   ├── storage.py            # MinIO client wrapper
│   └── db.py                 # MongoDB client wrapper
│
├── models/                   # ML Model Management
│   ├── __init__.py
│   ├── manager.py            # Model loading, caching, VRAM management
│   ├── registry.py           # Model registry, version tracking
│   └── optimize.py           # ONNX/TensorRT optimization utilities
│
├── forensics/                # Forensic & Reporting
│   ├── __init__.py
│   ├── forensics.py          # C2PA integration
│   ├── report.py             # PDF report generation
│   └── audit.py              # Audit trail logging
│
├── schemas/                  # Data Schemas
│   ├── __init__.py
│   ├── schemas.py            # Pydantic models for all data structures
│   ├── requests.py           # API request schemas
│   ├── responses.py          # API response schemas
│   └── internal.py           # Internal data transfer objects
│
├── interfaces/               # Abstract Interfaces
│   ├── __init__.py
│   ├── analyzer.py           # IAnalyzer abstract base
│   ├── storage.py            # IStorage abstract base
│   └── model.py              # IModel abstract base
│
└── utils/                    # Utilities
    ├── __init__.py
    ├── logging.py            # Structured logging setup
    ├── metrics.py            # Prometheus metrics
    └── errors.py             # Custom exception classes
```

---

## 2.2 File Manifesto (Backend)

### API LAYER

---

#### File: `server.py`

**Role:** FastAPI application entry point. Initializes app, includes routers, configures middleware, manages lifecycle events.

**SOTA Algorithm:** None (infrastructure only)

**Integration:**
- **Imports:** `config.py`, `api/router.py`, `api/middleware.py`, `storage/db.py`
- **Inputs:** None (entry point)
- **Outputs:** FastAPI application instance

**Schema:**
```python
# No data schema - infrastructure file
```

**Why this approach:** Single entry point follows 12-factor app principles. Lifecycle hooks ensure clean startup/shutdown of database connections and model caches.

---

#### File: `api/router.py`

**Role:** Define all HTTP endpoints. Route requests to appropriate handlers. Handle request validation and response serialization.

**SOTA Algorithm:** None (routing only)

**Integration:**
- **Imports:** `api/deps.py`, `schemas/requests.py`, `schemas/responses.py`, `core/orchestrator.py`
- **Inputs:** `AnalyzeRequest`, `UploadFile`
- **Outputs:** `AnalysisResponse`, `AnalysisStatusResponse`

**Schema:**
```python
# Endpoint: POST /api/analyze
async def analyze(
    file: UploadFile,
    options: AnalyzeOptions = Depends(),
    sanitizer: InputSanitizer = Depends(get_sanitizer),
    storage: StorageClient = Depends(get_storage),
    orchestrator: Orchestrator = Depends(get_orchestrator)
) -> AnalysisResponse

# Endpoint: GET /api/analyze/{analysis_id}
async def get_analysis(
    analysis_id: str,
    db: DatabaseClient = Depends(get_db)
) -> AnalysisStatusResponse
```

**Why this approach:** Dependency injection enables testing and modularity. Async handlers prevent blocking during I/O operations.

---

#### File: `api/deps.py`

**Role:** Dependency injection providers. Create and cache service instances per request or application lifetime.

**SOTA Algorithm:** None (DI patterns)

**Integration:**
- **Imports:** `storage/storage.py`, `storage/db.py`, `processing/sanitize.py`, `core/orchestrator.py`
- **Inputs:** None
- **Outputs:** Service instances (StorageClient, DatabaseClient, etc.)

**Schema:**
```python
def get_storage() -> StorageClient:
    """Singleton MinIO client"""
    
def get_db() -> DatabaseClient:
    """Connection-pooled MongoDB client"""
    
def get_sanitizer() -> InputSanitizer:
    """Per-request sanitizer instance"""
    
def get_orchestrator() -> Orchestrator:
    """Celery task orchestrator"""
```

**Why this approach:** Centralized dependency management enables easy mocking for tests and consistent resource handling.

---

#### File: `api/middleware.py`

**Role:** Request/response middleware for cross-cutting concerns: CORS, rate limiting, authentication, request logging.

**SOTA Algorithm:** Token bucket rate limiting (probabilistic), JWT validation

**Integration:**
- **Imports:** `config.py`, `utils/logging.py`, `utils/metrics.py`
- **Inputs:** Raw HTTP requests
- **Outputs:** Processed requests or 4xx/5xx responses

**Schema:**
```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter with Redis backend"""
    
class AuthMiddleware(BaseHTTPMiddleware):
    """JWT validation, role extraction"""
    
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured logging with correlation IDs"""
```

**Why this approach:** Middleware pattern separates concerns cleanly. Redis-backed rate limiting scales horizontally.

---

#### File: `api/websocket.py`

**Role:** WebSocket handlers for real-time analysis progress updates.

**SOTA Algorithm:** Pub/Sub pattern via Redis

**Integration:**
- **Imports:** `storage/db.py`, `config.py`
- **Inputs:** WebSocket connections, analysis_id subscriptions
- **Outputs:** Real-time progress events (JSON)

**Schema:**
```python
@router.websocket("/ws/analysis/{analysis_id}")
async def analysis_progress(
    websocket: WebSocket,
    analysis_id: str
) -> None:
    """Stream analysis progress updates"""
```

**Why this approach:** WebSockets provide low-latency updates without polling. Redis Pub/Sub enables multi-worker broadcasting.

---

### CORE LAYER

---

#### File: `core/orchestrator.py`

**Role:** Celery task definitions. Job queuing, status tracking, retry logic. Coordinates the entire analysis pipeline.

**SOTA Algorithm:** Directed Acyclic Graph (DAG) task scheduling with dependency resolution

**Integration:**
- **Imports:** `processing/preprocess.py`, `core/engine.py`, `core/fusion.py`, `forensics/report.py`, `storage/db.py`
- **Inputs:** `analysis_id: str`, `modalities: List[Modality]`
- **Outputs:** Job status updates to MongoDB

**Schema:**
```python
@celery_app.task(bind=True, max_retries=3)
def run_analysis_pipeline(
    self,
    analysis_id: str,
    options: dict
) -> AnalysisResult:
    """
    Main analysis pipeline task.
    
    Pipeline:
    1. preprocess_task (extract frames/audio)
    2. analyze_task (parallel modality analysis)
    3. aggregate_task (fusion + scoring)
    4. report_task (async PDF generation)
    """

@celery_app.task
def preprocess_task(analysis_id: str) -> PreprocessedData

@celery_app.task
def analyze_modality_task(
    analysis_id: str,
    modality: Modality,
    data: PreprocessedData
) -> ModalityResult

@celery_app.task
def aggregate_results_task(
    analysis_id: str,
    results: List[ModalityResult]
) -> AggregatedResult
```

**Why this approach:** Celery provides robust distributed task execution with automatic retries. Task chaining enables complex workflows with clean error handling.

---

#### File: `core/engine.py`

**Role:** Model inference engine. Manages model loading, VRAM allocation, batch inference execution.

**SOTA Algorithm:** 
- **Model Loading:** LRU cache with VRAM pressure monitoring
- **Inference:** ONNX Runtime with TensorRT/OpenVINO execution providers
- **Batching:** Dynamic batching based on available memory

**Integration:**
- **Imports:** `models/manager.py`, `config.py`, `schemas/internal.py`
- **Inputs:** `preprocessed_data: PreprocessedData`, `model_name: str`
- **Outputs:** `InferenceResult` with confidence scores

**Schema:**
```python
class InferenceEngine:
    """
    Manages model inference with hardware optimization.
    """
    
    async def infer(
        self,
        model_name: str,
        inputs: Union[np.ndarray, List[np.ndarray]],
        batch_size: Optional[int] = None
    ) -> InferenceResult:
        """
        Run inference with automatic batching.
        
        Args:
            model_name: Registry key for model
            inputs: Input tensor(s)
            batch_size: Override auto-batch sizing
            
        Returns:
            InferenceResult with predictions and confidence
        """
    
    def get_optimal_batch_size(
        self,
        model_name: str,
        input_shape: Tuple[int, ...]
    ) -> int:
        """Calculate optimal batch size given VRAM constraints"""
```

**Why this approach:** Centralized inference engine enables consistent optimization across all models. VRAM management prevents OOM errors on RTX 3050.

---

#### File: `core/fusion.py`

**Role:** Multi-modal result aggregation. Combines outputs from all analyzers using attention-weighted fusion.

**SOTA Algorithm:** 
- **Attention-Based Fusion:** Learned attention weights based on modality confidence
- **Uncertainty Quantification:** Monte Carlo dropout for confidence calibration

**Integration:**
- **Imports:** `schemas/internal.py`, `config.py`
- **Inputs:** `List[ModalityResult]`
- **Outputs:** `AggregatedResult`

**Schema:**
```python
class MultiModalFusion:
    """
    Attention-weighted fusion of modality results.
    
    Algorithm:
    1. Extract confidence scores from each modality
    2. Compute attention weights: softmax(confidence * learned_bias)
    3. Weighted aggregation: Σ(weight_i × score_i)
    4. Uncertainty estimation via ensemble disagreement
    """
    
    def aggregate(
        self,
        results: List[ModalityResult],
        content_type: ContentType
    ) -> AggregatedResult:
        """
        Fuse multi-modal results.
        
        Args:
            results: Results from each analyzer
            content_type: Affects weight distribution
            
        Returns:
            AggregatedResult with fused score and uncertainty
        """
```

**Why this approach:** Attention-based fusion adapts to modality reliability per-sample, unlike fixed weights. Uncertainty quantification enables "uncertain" verdicts when models disagree.

---

#### File: `core/scorer.py`

**Role:** Trust Score computation. Converts aggregated results into final 0-100 score and verdict.

**SOTA Algorithm:**
- **Bayesian Score Calibration:** Platt scaling for well-calibrated probabilities
- **Dynamic Thresholding:** Content-type aware thresholds loaded from config

**Integration:**
- **Imports:** `schemas/internal.py`, `config.py`
- **Inputs:** `AggregatedResult`
- **Outputs:** `TrustScore`, `Verdict`

**Schema:**
```python
class TrustScorer:
    """
    Computes calibrated Trust Score and verdict.
    
    Score Ranges (configurable):
    - 80-100: Authentic
    - 60-79: Likely Authentic
    - 40-59: Uncertain (flag for human review)
    - 20-39: Likely Fake
    - 0-19: Fake
    """
    
    def compute(
        self,
        aggregated: AggregatedResult
    ) -> Tuple[TrustScore, Verdict]:
        """
        Compute final score with Platt calibration.
        
        Returns:
            Tuple of (TrustScore 0-100, Verdict enum)
        """
    
    def calibrate_probability(
        self,
        raw_score: float,
        content_type: ContentType
    ) -> float:
        """Apply Platt scaling for probability calibration"""
```

**Why this approach:** Platt scaling ensures scores represent true probabilities (e.g., score 70 means 70% confidence). Dynamic thresholds account for different base rates across content types.

---

#### File: `core/explain.py`

**Role:** Explainability module. Generates GradCAM heatmaps and textual explanations.

**SOTA Algorithm:**
- **Visual:** GradCAM++ (improved gradient weighting)
- **Textual:** Template-based generation with dynamic slot filling (no external LLM required)

**Integration:**
- **Imports:** `schemas/internal.py`, `models/manager.py`
- **Inputs:** `model_activations`, `ModalityResult`
- **Outputs:** `Explanation` (heatmaps + text)

**Schema:**
```python
class ExplainabilityEngine:
    """
    Generate human-interpretable explanations.
    """
    
    def generate_gradcam(
        self,
        model: ONNXModel,
        input_tensor: np.ndarray,
        target_class: int = 1  # "fake" class
    ) -> np.ndarray:
        """
        Generate GradCAM++ heatmap.
        
        Returns:
            Normalized heatmap array (H, W) in [0, 1]
        """
    
    def generate_textual_explanation(
        self,
        results: AggregatedResult,
        heatmap_regions: List[Region]
    ) -> str:
        """
        Generate natural language explanation.
        
        Template-based approach with dynamic slots:
        "Analysis detected {manipulation_type} in the {region} 
         with {confidence}% confidence. Key indicators include 
         {indicators}."
        """
    
    def localize_manipulation(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5
    ) -> List[Region]:
        """Extract manipulation regions from heatmap"""
```

**Why this approach:** GradCAM++ provides better localization than original GradCAM. Template-based text avoids external LLM dependencies while remaining interpretable.

---

### ANALYZERS LAYER

---

#### File: `analyzers/base.py`

**Role:** Abstract base class defining the analyzer interface. All modality analyzers inherit from this.

**SOTA Algorithm:** None (interface definition)

**Integration:**
- **Imports:** `interfaces/analyzer.py`, `schemas/internal.py`
- **Inputs:** N/A (abstract)
- **Outputs:** N/A (abstract)

**Schema:**
```python
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.
    
    All analyzers must implement:
    - analyze(): Main analysis method
    - get_required_models(): List models needed
    - supports_modality(): Check if modality is supported
    """
    
    @abstractmethod
    async def analyze(
        self,
        data: PreprocessedData,
        engine: InferenceEngine
    ) -> ModalityResult:
        """Run analysis on preprocessed data"""
        pass
    
    @abstractmethod
    def get_required_models(self) -> List[str]:
        """Return list of model registry keys needed"""
        pass
    
    @abstractmethod
    def supports_modality(self, modality: Modality) -> bool:
        """Check if this analyzer handles the given modality"""
        pass
    
    def validate_input(self, data: PreprocessedData) -> None:
        """Validate input data before analysis"""
        pass
```

**Why this approach:** Abstract base ensures consistent interface across analyzers. Enables polymorphic processing in orchestrator.

---

#### File: `analyzers/video.py`

**Role:** Video deepfake detection orchestrator. Coordinates spatial, temporal, and lip-sync sub-analyzers.

**SOTA Algorithm:**
- **Ensemble:** Weighted voting across 3 sub-analyzers
- **Anomaly Detection:** Z-score based frame anomaly flagging

**Integration:**
- **Imports:** `analyzers/video/spatial.py`, `analyzers/video/temporal.py`, `analyzers/video/lipsync.py`, `core/engine.py`
- **Inputs:** `VideoData` (frames, faces, audio)
- **Outputs:** `VideoResult`

**Schema:**
```python
class VideoAnalyzer(BaseAnalyzer):
    """
    Multi-stage video deepfake detection.
    
    Pipeline:
    1. Spatial analysis (per-frame artifacts)
    2. Temporal analysis (cross-frame consistency)
    3. Lip-sync verification (if audio present)
    4. Ensemble aggregation with anomaly flagging
    """
    
    async def analyze(
        self,
        data: VideoData,
        engine: InferenceEngine
    ) -> VideoResult:
        """
        Run complete video analysis pipeline.
        
        Returns:
            VideoResult with spatial, temporal, lip-sync scores,
            anomaly frames, and heatmap URLs
        """
    
    def get_required_models(self) -> List[str]:
        return [
            "efficientnet_b3_spatial",
            "xclip_temporal",
            "lipinc_v2"
        ]
```

**Why this approach:** Modular sub-analyzers enable targeted optimization. Ensemble voting improves robustness against adversarial attacks.

---

#### File: `analyzers/video/spatial.py`

**Role:** Per-frame spatial artifact detection using EfficientNet-B3 backbone with CLIP guidance.

**SOTA Algorithm:**
- **Model:** EfficientNet-B3 (from DeepfakeBench) fine-tuned on FaceForensics++
- **Enhancement:** CLIP visual encoder for generalization to unseen forgery types
- **Inference:** ONNX INT8 quantized for RTX 3050

**Integration:**
- **Imports:** `core/engine.py`, `core/explain.py`
- **Inputs:** `List[np.ndarray]` (face crops)
- **Outputs:** `SpatialResult` (per-frame scores, heatmaps)

**Schema:**
```python
class SpatialAnalyzer:
    """
    Per-frame spatial artifact detection.
    
    Models:
    - efficientnet_b3: Binary classification (real/fake)
    - clip_vit_b16: Feature extraction for generalization
    
    Features Detected:
    - Blending boundaries
    - Texture inconsistencies
    - Frequency domain artifacts
    """
    
    async def analyze_frames(
        self,
        face_crops: List[np.ndarray],
        engine: InferenceEngine
    ) -> SpatialResult:
        """
        Analyze face crops for spatial artifacts.
        
        Args:
            face_crops: List of (224, 224, 3) face images
            engine: Inference engine
            
        Returns:
            SpatialResult with per-frame scores and aggregate
        """
    
    def detect_frequency_artifacts(
        self,
        image: np.ndarray
    ) -> FrequencyFeatures:
        """DCT analysis for GAN fingerprints"""
```

**Why this approach:** EfficientNet-B3 provides optimal accuracy/speed tradeoff for RTX 3050. CLIP guidance enables zero-shot generalization to novel deepfake types.

---

#### File: `analyzers/video/temporal.py`

**Role:** Cross-frame temporal consistency analysis using X-CLIP transformer.

**SOTA Algorithm:**
- **Model:** X-CLIP with Multiframe Integration Transformer (KDD 2025)
- **Analysis:** Optical flow consistency, facial landmark tracking
- **Anomaly:** Frame-to-frame coherence scoring

**Integration:**
- **Imports:** `core/engine.py`
- **Inputs:** `List[np.ndarray]` (sequence of frames)
- **Outputs:** `TemporalResult`

**Schema:**
```python
class TemporalAnalyzer:
    """
    Temporal consistency analysis.
    
    Detects:
    - Flickering artifacts
    - Unnatural motion patterns
    - Landmark jitter
    - Inter-frame color inconsistency
    """
    
    async def analyze_consistency(
        self,
        frame_sequence: List[np.ndarray],
        engine: InferenceEngine
    ) -> TemporalResult:
        """
        Analyze temporal consistency across frames.
        
        Args:
            frame_sequence: Ordered list of frames
            engine: Inference engine
            
        Returns:
            TemporalResult with consistency score and anomaly indices
        """
    
    def compute_optical_flow_consistency(
        self,
        frames: List[np.ndarray]
    ) -> float:
        """OpenCV optical flow analysis"""
```

**Why this approach:** X-CLIP captures long-range temporal dependencies that CNNs miss. Optical flow provides physics-grounded consistency checks.

---

#### File: `analyzers/video/lipsync.py`

**Role:** Lip-sync deepfake detection using LIPINC-V2 architecture.

**SOTA Algorithm:**
- **Model:** LIPINC-V2 (Vision Temporal Transformer with multihead cross-attention)
- **Detection:** Audio-visual synchronization scoring
- **Targets:** Wav2Lip, Diff2Lip, Video_Retalking, IP_LAP artifacts

**Integration:**
- **Imports:** `core/engine.py`
- **Inputs:** `frames: List[np.ndarray]`, `audio: np.ndarray`
- **Outputs:** `LipSyncResult`

**Schema:**
```python
class LipSyncAnalyzer:
    """
    Lip-sync deepfake detection.
    
    Specialized for detecting:
    - Wav2Lip manipulations
    - Diff2Lip artifacts
    - Video_Retalking inconsistencies
    - Audio-visual desynchronization
    """
    
    async def verify_sync(
        self,
        mouth_crops: List[np.ndarray],
        audio_features: np.ndarray,
        engine: InferenceEngine
    ) -> LipSyncResult:
        """
        Verify audio-visual lip synchronization.
        
        Args:
            mouth_crops: Cropped mouth regions (frames)
            audio_features: MFCC/mel-spectrogram features
            engine: Inference engine
            
        Returns:
            LipSyncResult with sync score and manipulation probability
        """
```

**Why this approach:** LIPINC-V2 is SOTA for lip-sync specific deepfakes. Multihead cross-attention captures fine-grained audio-visual correlations.

---

#### File: `analyzers/audio.py`

**Role:** Audio deepfake detection using Purdue-M2 architecture.

**SOTA Algorithm:**
- **Model:** Purdue-M2 AI-Synthesized Voice Generalization (AAAI 2025)
- **Features:** Mel-spectrogram, MFCC, raw waveform
- **Detection:** Vocoder artifacts, spectral inconsistencies

**Integration:**
- **Imports:** `core/engine.py`, `processing/transform.py`
- **Inputs:** `audio_data: np.ndarray` (waveform)
- **Outputs:** `AudioResult`

**Schema:**
```python
class AudioAnalyzer(BaseAnalyzer):
    """
    Synthetic voice detection.
    
    Features Extracted:
    - Mel-spectrogram (80 mel bands)
    - MFCC (13 coefficients + deltas)
    - Raw waveform (for RawNet-style)
    
    Artifacts Detected:
    - Vocoder artifacts (phase discontinuities)
    - Unnatural harmonics
    - Bandwidth limitations
    - Background noise inconsistencies
    """
    
    async def analyze(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        engine: InferenceEngine
    ) -> AudioResult:
        """
        Analyze audio for synthetic generation artifacts.
        
        Args:
            audio_data: Raw waveform (1D array)
            sample_rate: Audio sample rate (target 16kHz)
            engine: Inference engine
            
        Returns:
            AudioResult with synthetic probability and feature scores
        """
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> np.ndarray:
        """Extract mel-spectrogram features"""
    
    def detect_vocoder_artifacts(
        self,
        spectrogram: np.ndarray
    ) -> VocoderArtifactScore:
        """Analyze spectral patterns for vocoder signatures"""
```

**Why this approach:** Purdue-M2 generalizes across unseen TTS systems. Multi-feature approach catches diverse vocoder types.

---

#### File: `analyzers/text.py`

**Role:** AI-generated text detection using perplexity/burstiness analysis and RADAR model.

**SOTA Algorithm:**
- **Primary:** RADAR (IBM NeurIPS) - adversarially robust detector
- **Secondary:** GPT-2 perplexity scoring, burstiness analysis
- **Ensemble:** Weighted combination for robustness

**Integration:**
- **Imports:** `core/engine.py`
- **Inputs:** `text: str`
- **Outputs:** `TextResult`

**Schema:**
```python
class TextAnalyzer(BaseAnalyzer):
    """
    AI-generated text detection.
    
    Metrics:
    - Perplexity: Low = likely AI (too predictable)
    - Burstiness: Low variance = likely AI
    - Vocabulary diversity: Low = likely AI
    - RADAR score: Adversarially trained classifier
    """
    
    async def analyze(
        self,
        text: str,
        engine: InferenceEngine
    ) -> TextResult:
        """
        Analyze text for AI generation patterns.
        
        Args:
            text: Input text (minimum 50 characters)
            engine: Inference engine
            
        Returns:
            TextResult with AI probability and metric breakdown
        """
    
    def compute_perplexity(self, text: str) -> float:
        """GPT-2 based perplexity scoring"""
    
    def compute_burstiness(self, text: str) -> float:
        """Sentence length variance analysis"""
```

**Why this approach:** RADAR provides robustness against paraphrasing attacks. Multiple metrics reduce false positives from naturally low-perplexity human text.

---

#### File: `analyzers/image.py`

**Role:** Single-image deepfake and AI-generated image detection.

**SOTA Algorithm:**
- **Model:** SigLIP-based classifier (HuggingFace deepfake-detector-model-v1)
- **Analysis:** Frequency domain (DCT), CLIP embeddings
- **Explainability:** GradCAM overlay generation

**Integration:**
- **Imports:** `core/engine.py`, `core/explain.py`
- **Inputs:** `image: np.ndarray`
- **Outputs:** `ImageResult`

**Schema:**
```python
class ImageAnalyzer(BaseAnalyzer):
    """
    Image deepfake and AI-generation detection.
    
    Detection Targets:
    - Face swaps (DeepFaceLab, etc.)
    - AI-generated faces (StyleGAN, Midjourney)
    - Edited/manipulated images
    - Stable Diffusion outputs
    """
    
    async def analyze(
        self,
        image: np.ndarray,
        engine: InferenceEngine
    ) -> ImageResult:
        """
        Analyze single image for manipulation.
        
        Args:
            image: Input image (RGB, any size)
            engine: Inference engine
            
        Returns:
            ImageResult with fake probability, heatmap URL
        """
    
    def analyze_dct(self, image: np.ndarray) -> DCTFeatures:
        """Discrete Cosine Transform for GAN fingerprints"""
```

**Why this approach:** SigLIP provides excellent generalization across image generators. DCT analysis catches frequency-domain GAN signatures.

---

#### File: `analyzers/metadata.py`

**Role:** Media metadata analysis including C2PA Content Credentials and EXIF data.

**SOTA Algorithm:**
- **C2PA:** Content Credentials verification per C2PA v2.3 specification
- **EXIF:** Anomaly detection via metadata consistency analysis

**Integration:**
- **Imports:** `forensics/forensics.py`
- **Inputs:** `file_bytes: bytes`
- **Outputs:** `MetadataResult`

**Schema:**
```python
class MetadataAnalyzer(BaseAnalyzer):
    """
    Media metadata and provenance analysis.
    
    Analyzes:
    - C2PA Content Credentials (if present)
    - EXIF data consistency
    - File structure anomalies
    - Hash verification
    """
    
    async def analyze(
        self,
        file_bytes: bytes,
        original_filename: str
    ) -> MetadataResult:
        """
        Analyze file metadata for authenticity signals.
        
        Args:
            file_bytes: Raw file content
            original_filename: Original filename for extension check
            
        Returns:
            MetadataResult with C2PA status, EXIF anomalies
        """
    
    def extract_c2pa_manifest(
        self,
        file_bytes: bytes
    ) -> Optional[C2PAManifest]:
        """Extract and validate C2PA Content Credentials"""
    
    def analyze_exif_consistency(
        self,
        exif_data: dict
    ) -> List[ExifAnomaly]:
        """Check for suspicious EXIF patterns"""
```

**Why this approach:** C2PA is the emerging standard for content authenticity. EXIF analysis catches amateur manipulations that modify metadata.

---

### PROCESSING LAYER

---

#### File: `processing/preprocess.py`

**Role:** Media preprocessing orchestration. Routes files to appropriate extractors based on type.

**SOTA Algorithm:** None (orchestration)

**Integration:**
- **Imports:** `processing/extract.py`, `processing/sanitize.py`, `storage/storage.py`
- **Inputs:** `analysis_id: str`, `file_key: str`
- **Outputs:** `PreprocessedData`

**Schema:**
```python
class Preprocessor:
    """
    Orchestrates media preprocessing.
    
    Steps:
    1. Download file from MinIO
    2. Detect media type
    3. Route to appropriate extractor
    4. Upload preprocessed data
    5. Update job status
    """
    
    async def process(
        self,
        analysis_id: str,
        file_key: str,
        storage: StorageClient
    ) -> PreprocessedData:
        """
        Preprocess media file for analysis.
        
        Returns:
            PreprocessedData with extracted features
        """
```

**Why this approach:** Single entry point for preprocessing simplifies orchestration. Type-based routing enables modality-specific optimization.

---

#### File: `processing/extract.py`

**Role:** Media extraction utilities. Frame extraction from video, audio track separation, face detection.

**SOTA Algorithm:**
- **Face Detection:** RetinaFace (SOTA accuracy)
- **Frame Extraction:** FFmpeg with smart keyframe detection
- **Audio:** FFmpeg audio stream extraction

**Integration:**
- **Imports:** `config.py`
- **Inputs:** Video/audio bytes
- **Outputs:** `VideoData`, `AudioData`

**Schema:**
```python
class MediaExtractor:
    """
    Extract analyzable data from media files.
    """
    
    async def extract_video_data(
        self,
        video_bytes: bytes,
        frame_sample_rate: int = 5
    ) -> VideoData:
        """
        Extract frames, faces, and audio from video.
        
        Args:
            video_bytes: Raw video file
            frame_sample_rate: Sample every Nth frame
            
        Returns:
            VideoData with frames, face crops, audio track
        """
    
    async def detect_faces(
        self,
        frames: List[np.ndarray]
    ) -> List[FaceDetection]:
        """
        Run RetinaFace on frames.
        
        Returns:
            List of detections with bounding boxes, landmarks
        """
    
    async def extract_audio_track(
        self,
        video_bytes: bytes
    ) -> AudioData:
        """Extract audio as 16kHz mono WAV"""
```

**Why this approach:** RetinaFace provides robust face detection across poses. Smart frame sampling balances accuracy and speed for RTX 3050.

---

#### File: `processing/sanitize.py`

**Role:** Input validation and adversarial defense. Prevents malicious uploads and adversarial attacks.

**SOTA Algorithm:**
- **Validation:** Magic byte verification, content-type matching
- **Adversarial Defense:** JPEG compression + Gaussian noise injection

**Integration:**
- **Imports:** `config.py`, `utils/errors.py`
- **Inputs:** `UploadFile`
- **Outputs:** `SanitizedFile`

**Schema:**
```python
class InputSanitizer:
    """
    Validate and sanitize uploaded files.
    
    Security Checks:
    1. Magic byte verification (not extension-based)
    2. Content-type matching
    3. Size limits
    4. Malware pattern detection
    
    Adversarial Defense:
    1. JPEG recompression (removes perturbations)
    2. Gaussian noise injection
    3. Multi-scale analysis
    """
    
    async def validate(
        self,
        file: UploadFile,
        max_size_mb: int = 500
    ) -> SanitizedFile:
        """
        Validate and sanitize input file.
        
        Raises:
            InvalidFileError: If validation fails
            
        Returns:
            SanitizedFile with verified metadata
        """
    
    def apply_adversarial_defense(
        self,
        image: np.ndarray,
        defense_level: str = "standard"
    ) -> np.ndarray:
        """
        Apply preprocessing to defeat adversarial attacks.
        
        Levels:
        - "none": No defense (faster)
        - "standard": JPEG compression Q=85
        - "aggressive": Compression + noise + blur
        """
```

**Why this approach:** Magic byte validation prevents extension spoofing. Adversarial preprocessing defeats 90%+ of perturbation attacks with minimal accuracy loss.

---

#### File: `processing/transform.py`

**Role:** Data transformations for model inference. Normalization, resizing, feature extraction.

**SOTA Algorithm:** Standard ImageNet normalization, mel-spectrogram extraction

**Integration:**
- **Imports:** `config.py`
- **Inputs:** Raw media data
- **Outputs:** Model-ready tensors

**Schema:**
```python
class DataTransformer:
    """
    Transform raw data for model inference.
    """
    
    def transform_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ) -> np.ndarray:
        """
        Prepare image for model input.
        
        Applies:
        - Resize to target_size
        - RGB normalization (ImageNet mean/std)
        - Channel ordering (CHW for PyTorch)
        """
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        n_mels: int = 80
    ) -> np.ndarray:
        """
        Extract mel-spectrogram features.
        
        Args:
            audio: Raw waveform
            sample_rate: Target sample rate
            n_mels: Number of mel bands
        """
```

**Why this approach:** Standardized transforms ensure consistency across all inference paths. Centralized implementation reduces duplication.

---

### STORAGE LAYER

---

#### File: `storage/storage.py`

**Role:** MinIO client wrapper. Handles file upload/download, presigned URLs, bucket management.

**SOTA Algorithm:** None (storage client)

**Integration:**
- **Imports:** `config.py`
- **Inputs:** File bytes, object keys
- **Outputs:** Object URLs, file streams

**Schema:**
```python
class StorageClient:
    """
    MinIO object storage client.
    
    Buckets:
    - argus-uploads: Raw uploaded files
    - argus-preprocessed: Extracted frames, audio
    - argus-results: Heatmaps, reports
    """
    
    async def upload_file(
        self,
        file: Union[bytes, BinaryIO],
        bucket: str,
        object_key: str,
        content_type: str
    ) -> str:
        """
        Upload file to MinIO.
        
        Returns:
            Object key for retrieval
        """
    
    async def download_file(
        self,
        bucket: str,
        object_key: str
    ) -> bytes:
        """Download file from MinIO"""
    
    async def get_presigned_url(
        self,
        bucket: str,
        object_key: str,
        expires_seconds: int = 3600
    ) -> str:
        """Generate presigned download URL"""
```

**Why this approach:** MinIO provides S3-compatible storage that scales horizontally. Presigned URLs enable secure client-side access without proxy overhead.

---

#### File: `storage/db.py`

**Role:** MongoDB client wrapper. Handles analysis CRUD, job status updates, connection pooling.

**SOTA Algorithm:** None (database client)

**Integration:**
- **Imports:** `config.py`, `schemas/schemas.py`
- **Inputs:** Analysis data, queries
- **Outputs:** Analysis documents

**Schema:**
```python
class DatabaseClient:
    """
    MongoDB async client with connection pooling.
    
    Collections:
    - analyses: Main analysis records
    - jobs: Celery job tracking
    - audit_log: Immutable audit trail
    """
    
    async def insert_analysis(
        self,
        analysis: AnalysisDocument
    ) -> str:
        """
        Insert new analysis record.
        
        Returns:
            Inserted analysis_id
        """
    
    async def update_analysis(
        self,
        analysis_id: str,
        updates: dict
    ) -> None:
        """Update analysis with results"""
    
    async def get_analysis(
        self,
        analysis_id: str
    ) -> Optional[AnalysisDocument]:
        """Retrieve analysis by ID"""
```

**Why this approach:** MongoDB's flexible schema accommodates evolving analysis results. Connection pooling prevents connection exhaustion under load.

---

### MODELS LAYER

---

#### File: `models/manager.py`

**Role:** Model loading, caching, and VRAM management. Implements LRU eviction for 4GB VRAM constraint.

**SOTA Algorithm:**
- **Caching:** LRU with VRAM-aware eviction
- **Loading:** Lazy loading with warmup option
- **Memory:** Real-time VRAM monitoring

**Integration:**
- **Imports:** `models/registry.py`, `config.py`
- **Inputs:** Model names
- **Outputs:** Loaded ONNX sessions

**Schema:**
```python
class ModelManager:
    """
    Intelligent model loading for constrained VRAM.
    
    Features:
    - LRU eviction when VRAM pressure detected
    - Lazy loading (models loaded on first use)
    - Warmup mode (preload critical models)
    - Real-time VRAM monitoring via nvidia-smi
    """
    
    def __init__(
        self,
        max_vram_mb: int = 3500,
        model_cache_dir: str = "/models"
    ):
        """Initialize with VRAM budget"""
    
    async def get_model(
        self,
        model_name: str
    ) -> ort.InferenceSession:
        """
        Get model session, loading if necessary.
        
        Evicts LRU models if VRAM insufficient.
        
        Returns:
            ONNX Runtime InferenceSession
        """
    
    def get_vram_usage(self) -> int:
        """Get current VRAM usage in MB"""
    
    def evict_lru(self, required_mb: int) -> None:
        """Evict least recently used models to free space"""
```

**Why this approach:** LRU eviction ensures most-used models stay loaded. VRAM monitoring prevents OOM on RTX 3050.

---

#### File: `models/registry.py`

**Role:** Model registry. Maps model names to paths, metadata, and version info.

**SOTA Algorithm:** None (registry)

**Integration:**
- **Imports:** `config.py`
- **Inputs:** Model names
- **Outputs:** Model paths, metadata

**Schema:**
```python
class ModelRegistry:
    """
    Central registry of available models.
    
    Registry Structure:
    {
        "efficientnet_b3_spatial": {
            "path": "/models/efficientnet_b3_int8.onnx",
            "input_shape": [1, 3, 224, 224],
            "vram_mb": 300,
            "version": "1.0.0",
            "provider": ["TensorRTExecutionProvider", "CUDAExecutionProvider"]
        }
    }
    """
    
    def get_model_info(
        self,
        model_name: str
    ) -> ModelInfo:
        """Get model metadata from registry"""
    
    def list_models(self) -> List[str]:
        """List all registered models"""
```

**Why this approach:** Registry enables version tracking and dynamic model updates without code changes.

---

#### File: `models/optimize.py`

**Role:** Model optimization utilities. ONNX export, INT8 quantization, TensorRT conversion.

**SOTA Algorithm:**
- **Quantization:** Static INT8 with calibration
- **Optimization:** TensorRT graph optimization

**Integration:**
- **Imports:** `config.py`
- **Inputs:** PyTorch models
- **Outputs:** Optimized ONNX files

**Schema:**
```python
class ModelOptimizer:
    """
    Optimize models for efficient inference.
    
    Pipeline:
    1. Export PyTorch → ONNX
    2. Apply ONNX graph optimizations
    3. Quantize to INT8 (with calibration data)
    4. Build TensorRT engine (optional)
    """
    
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str
    ) -> None:
        """Export PyTorch model to ONNX"""
    
    def quantize_int8(
        self,
        onnx_path: str,
        calibration_data: np.ndarray,
        output_path: str
    ) -> None:
        """Apply static INT8 quantization"""
```

**Why this approach:** INT8 quantization provides 4x speedup with <2% accuracy loss. TensorRT maximizes RTX 3050 performance.

---

### FORENSICS LAYER

---

#### File: `forensics/forensics.py`

**Role:** C2PA Content Credentials integration. Extract, validate, and create provenance manifests.

**SOTA Algorithm:** C2PA v2.3 specification compliance

**Integration:**
- **Imports:** c2pa-python library
- **Inputs:** Media files
- **Outputs:** C2PA manifests

**Schema:**
```python
class ForensicsEngine:
    """
    C2PA Content Credentials integration.
    
    Capabilities:
    - Extract existing C2PA manifests
    - Validate cryptographic signatures
    - Verify trust list membership
    - Create new manifests for analysis results
    """
    
    def extract_manifest(
        self,
        file_bytes: bytes
    ) -> Optional[C2PAManifest]:
        """
        Extract C2PA manifest from media file.
        
        Returns:
            C2PAManifest if present, None otherwise
        """
    
    def validate_manifest(
        self,
        manifest: C2PAManifest
    ) -> ValidationResult:
        """
        Validate manifest integrity and trust.
        
        Checks:
        - Cryptographic signature validity
        - Certificate chain verification
        - Trust list membership
        - Tampering detection
        """
    
    def create_analysis_manifest(
        self,
        analysis_result: AnalysisResult,
        original_file: bytes
    ) -> C2PAManifest:
        """Create C2PA manifest documenting analysis"""
```

**Why this approach:** C2PA is the emerging global standard for content authenticity. Integration provides legal-grade provenance.

---

#### File: `forensics/report.py`

**Role:** PDF forensic report generation with embedded evidence.

**SOTA Algorithm:** None (report generation)

**Integration:**
- **Imports:** `schemas/schemas.py`, `storage/storage.py`
- **Inputs:** `AnalysisResult`
- **Outputs:** PDF bytes

**Schema:**
```python
class ReportGenerator:
    """
    Generate forensic PDF reports.
    
    Report Sections:
    1. Executive Summary (score, verdict)
    2. Methodology (models used, versions)
    3. Findings by Modality
    4. Evidence (heatmaps, spectrograms)
    5. Technical Appendix
    6. Chain of Custody
    """
    
    async def generate(
        self,
        analysis: AnalysisResult,
        storage: StorageClient
    ) -> bytes:
        """
        Generate comprehensive PDF report.
        
        Returns:
            PDF file bytes
        """
```

**Why this approach:** PDF reports provide legal-admissible documentation. Embedded evidence enables offline verification.

---

#### File: `forensics/audit.py`

**Role:** Immutable audit trail logging for chain of custody.

**SOTA Algorithm:** Append-only log with cryptographic chaining

**Integration:**
- **Imports:** `storage/db.py`
- **Inputs:** Audit events
- **Outputs:** None (side effect: log entry)

**Schema:**
```python
class AuditLogger:
    """
    Immutable audit trail for forensic chain of custody.
    
    Events Logged:
    - File upload
    - Analysis started
    - Analysis completed
    - Report generated
    - File accessed
    - File deleted
    
    Each entry includes:
    - Timestamp (UTC)
    - Event type
    - Actor (user/system)
    - Resource ID
    - Cryptographic hash of previous entry (chain)
    """
    
    async def log_event(
        self,
        event_type: AuditEventType,
        resource_id: str,
        actor: str,
        metadata: dict
    ) -> None:
        """Log immutable audit event"""
```

**Why this approach:** Cryptographic chaining provides tamper-evidence for legal proceedings.

---

### SCHEMAS LAYER

---

#### File: `schemas/schemas.py`

**Role:** Core Pydantic models for all data structures. Single source of truth for types.

**SOTA Algorithm:** None (data modeling)

**Integration:**
- **Imports:** pydantic
- **Inputs:** N/A (definitions)
- **Outputs:** N/A (definitions)

**Schema:**
```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class Modality(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"

class Verdict(str, Enum):
    AUTHENTIC = "authentic"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    LIKELY_FAKE = "likely_fake"
    FAKE = "fake"

class TrustScore(BaseModel):
    value: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    calibrated: bool = True

class SpatialResult(BaseModel):
    score: float
    per_frame_scores: List[float]
    anomaly_indices: List[int]
    heatmap_urls: List[str]

class TemporalResult(BaseModel):
    consistency_score: float
    flickering_detected: bool
    anomaly_timestamps: List[float]

class LipSyncResult(BaseModel):
    sync_score: float
    manipulation_probability: float
    detected_technology: Optional[str]

class VideoResult(BaseModel):
    spatial: SpatialResult
    temporal: TemporalResult
    lip_sync: Optional[LipSyncResult]
    aggregate_score: float

class AudioResult(BaseModel):
    synthetic_probability: float
    vocoder_artifacts_detected: bool
    voice_consistency_score: float
    spectrogram_url: Optional[str]

class TextResult(BaseModel):
    ai_probability: float
    perplexity_score: float
    burstiness_score: float
    radar_score: Optional[float]

class MetadataResult(BaseModel):
    c2pa_present: bool
    c2pa_valid: Optional[bool]
    provenance_chain: List[dict]
    exif_anomalies: List[str]

class Explanation(BaseModel):
    summary: str
    key_findings: List[str]
    manipulation_regions: List[dict]
    confidence_rationale: str

class AnalysisResult(BaseModel):
    analysis_id: str
    status: str
    trust_score: TrustScore
    verdict: Verdict
    video_result: Optional[VideoResult]
    audio_result: Optional[AudioResult]
    text_result: Optional[TextResult]
    metadata_result: MetadataResult
    explanation: Explanation
    report_url: Optional[str]
    processing_time_seconds: float
    created_at: datetime
    completed_at: Optional[datetime]
```

**Why this approach:** Pydantic provides runtime validation and automatic JSON serialization. Centralized schemas prevent drift between components.

---

### UTILITIES LAYER

---

#### File: `utils/logging.py`

**Role:** Structured logging configuration with correlation IDs.

**SOTA Algorithm:** None (logging)

**Integration:**
- **Imports:** structlog
- **Inputs:** N/A
- **Outputs:** Configured loggers

**Schema:**
```python
def setup_logging(
    level: str = "INFO",
    json_format: bool = True
) -> None:
    """Configure structured logging"""

def get_logger(name: str) -> structlog.BoundLogger:
    """Get logger instance with context"""
```

**Why this approach:** Structured logging enables machine parsing for monitoring. Correlation IDs trace requests across services.

---

#### File: `utils/metrics.py`

**Role:** Prometheus metrics collection for observability.

**SOTA Algorithm:** None (metrics)

**Integration:**
- **Imports:** prometheus_client
- **Inputs:** N/A
- **Outputs:** Metrics

**Schema:**
```python
# Counters
analysis_requests_total = Counter(
    "argus_analysis_requests_total",
    "Total analysis requests",
    ["status", "modality"]
)

# Histograms
analysis_duration_seconds = Histogram(
    "argus_analysis_duration_seconds",
    "Analysis processing time",
    ["modality"]
)

# Gauges
model_vram_usage_bytes = Gauge(
    "argus_model_vram_usage_bytes",
    "Current VRAM usage",
    ["model"]
)
```

**Why this approach:** Prometheus metrics enable alerting and dashboards. Standard histogram buckets match latency SLOs.

---

#### File: `utils/errors.py`

**Role:** Custom exception classes for structured error handling.

**SOTA Algorithm:** None (error handling)

**Integration:**
- **Imports:** None
- **Inputs:** N/A
- **Outputs:** Exception classes

**Schema:**
```python
class ArgusError(Exception):
    """Base exception for all Argus errors"""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

class InvalidFileError(ArgusError):
    status_code = 400
    error_code = "INVALID_FILE"

class AnalysisNotFoundError(ArgusError):
    status_code = 404
    error_code = "ANALYSIS_NOT_FOUND"

class ModelLoadError(ArgusError):
    status_code = 500
    error_code = "MODEL_LOAD_FAILED"

class InferenceError(ArgusError):
    status_code = 500
    error_code = "INFERENCE_FAILED"
```

**Why this approach:** Custom exceptions enable consistent error responses. Status codes map to HTTP semantics.

---

# SECTION 3: DEVELOPMENT STRATEGY

## 3.1 Development Order (Contract-First)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           DEVELOPMENT ORDER (DAG)                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  LAYER 0: Foundation (Day 1)                                                           │
│  ─────────────────────────────                                                         │
│  │                                                                                      │
│  ├── schemas/schemas.py          ← ALL data types defined FIRST                        │
│  ├── interfaces/analyzer.py      ← Abstract contracts                                  │
│  ├── interfaces/storage.py       ← Storage interface                                   │
│  ├── interfaces/model.py         ← Model interface                                     │
│  ├── config.py                   ← Configuration loader                                │
│  └── utils/errors.py             ← Exception classes                                   │
│      │                                                                                  │
│      ▼                                                                                  │
│  LAYER 1: Storage & Utilities (Day 2)                                                  │
│  ─────────────────────────────────────                                                 │
│  │                                                                                      │
│  ├── storage/storage.py          ← MinIO client                                        │
│  ├── storage/db.py               ← MongoDB client                                      │
│  ├── utils/logging.py            ← Logging setup                                       │
│  └── utils/metrics.py            ← Prometheus metrics                                  │
│      │                                                                                  │
│      ▼                                                                                  │
│  LAYER 2: Processing Pipeline (Day 3)                                                  │
│  ─────────────────────────────────────                                                 │
│  │                                                                                      │
│  ├── processing/sanitize.py      ← Input validation                                    │
│  ├── processing/extract.py       ← Media extraction                                    │
│  ├── processing/transform.py     ← Data transforms                                     │
│  └── processing/preprocess.py    ← Orchestrates above                                  │
│      │                                                                                  │
│      ▼                                                                                  │
│  LAYER 3: Model Infrastructure (Day 4)                                                 │
│  ───────────────────────────────────────                                               │
│  │                                                                                      │
│  ├── models/registry.py          ← Model metadata                                      │
│  ├── models/manager.py           ← VRAM management                                     │
│  └── models/optimize.py          ← Quantization utils                                  │
│      │                                                                                  │
│      ▼                                                                                  │
│  LAYER 4: Core Engine (Day 5-6)                                                        │
│  ───────────────────────────────                                                       │
│  │                                                                                      │
│  ├── core/engine.py              ← Inference engine                                    │
│  ├── core/explain.py             ← GradCAM, explanations                               │
│  ├── core/fusion.py              ← Multi-modal aggregation                             │
│  └── core/scorer.py              ← Trust Score calculation                             │
│      │                                                                                  │
│      ▼                                                                                  │
│  LAYER 5: Analyzers (Day 7-10)                                                         │
│  ─────────────────────────────                                                         │
│  │                                                                                      │
│  ├── analyzers/base.py           ← Abstract base                                       │
│  ├── analyzers/image.py          ← Start simple (single image)                         │
│  ├── analyzers/video/spatial.py  ← Per-frame analysis                                  │
│  ├── analyzers/video/temporal.py ← Temporal consistency                                │
│  ├── analyzers/video/lipsync.py  ← Lip-sync detection                                  │
│  ├── analyzers/video.py          ← Video orchestrator                                  │
│  ├── analyzers/audio.py          ← Audio analysis                                      │
│  ├── analyzers/text.py           ← Text detection                                      │
│  └── analyzers/metadata.py       ← C2PA, EXIF                                          │
│      │                                                                                  │
│      ▼                                                                                  │
│  LAYER 6: API & Orchestration (Day 11-12)                                              │
│  ─────────────────────────────────────────                                             │
│  │                                                                                      │
│  ├── api/deps.py                 ← Dependency providers                                │
│  ├── api/middleware.py           ← Auth, rate limiting                                 │
│  ├── api/router.py               ← HTTP endpoints                                      │
│  ├── api/websocket.py            ← Real-time updates                                   │
│  ├── core/orchestrator.py        ← Celery tasks                                        │
│  └── server.py                   ← FastAPI app                                         │
│      │                                                                                  │
│      ▼                                                                                  │
│  LAYER 7: Forensics (Day 13-14)                                                        │
│  ───────────────────────────────                                                       │
│  │                                                                                      │
│  ├── forensics/forensics.py      ← C2PA integration                                    │
│  ├── forensics/report.py         ← PDF generation                                      │
│  └── forensics/audit.py          ← Audit logging                                       │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## 3.2 Contract-First Rules

### Rule 1: Schemas Before Logic
```python
# CORRECT: Define schema first
class VideoResult(BaseModel):
    spatial: SpatialResult
    temporal: TemporalResult
    # ... full definition

# Then implement
async def analyze_video(...) -> VideoResult:
    pass
```

### Rule 2: Interface Before Implementation
```python
# CORRECT: Abstract interface first
class IAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, data: PreprocessedData) -> ModalityResult:
        pass

# Then concrete implementation
class VideoAnalyzer(IAnalyzer):
    async def analyze(self, data: PreprocessedData) -> ModalityResult:
        # Implementation
```

### Rule 3: No Hardcoded Values
```python
# WRONG
threshold = 0.5

# CORRECT
threshold = config.get("detection.threshold", default=0.5)
```

### Rule 4: Type Everything
```python
# WRONG
def analyze(data):
    pass

# CORRECT
async def analyze(
    data: PreprocessedData,
    engine: InferenceEngine
) -> AnalysisResult:
    pass
```

## 3.3 Testing Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                TESTING PYRAMID                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│                              ┌───────────────┐                                          │
│                              │  E2E Tests    │  ← 10% (Full pipeline)                   │
│                              │  (Playwright) │                                          │
│                              └───────┬───────┘                                          │
│                                      │                                                  │
│                         ┌────────────┴────────────┐                                     │
│                         │  Integration Tests      │  ← 30% (API + DB + Storage)         │
│                         │  (pytest + testcontainers)                                    │
│                         └────────────┬────────────┘                                     │
│                                      │                                                  │
│              ┌───────────────────────┴───────────────────────┐                          │
│              │              Unit Tests                        │  ← 60% (Pure functions)  │
│              │              (pytest + mocks)                  │                          │
│              └────────────────────────────────────────────────┘                          │
│                                                                                         │
│  Test Files:                                                                            │
│  ├── tests/unit/                                                                        │
│  │   ├── test_schemas.py       ← Pydantic validation                                   │
│  │   ├── test_scorer.py        ← Trust Score calculation                               │
│  │   ├── test_fusion.py        ← Aggregation logic                                     │
│  │   └── test_sanitize.py      ← Input validation                                      │
│  │                                                                                      │
│  ├── tests/integration/                                                                 │
│  │   ├── test_api.py           ← Endpoint tests                                        │
│  │   ├── test_storage.py       ← MinIO operations                                      │
│  │   └── test_db.py            ← MongoDB operations                                    │
│  │                                                                                      │
│  └── tests/e2e/                                                                         │
│      └── test_full_pipeline.py ← Complete analysis flow                                │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# APPENDIX A: SHARED SCHEMAS

## Complete Schema Reference

```python
# /app/backend/schemas/schemas.py

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import uuid

# ============== ENUMS ==============

class Modality(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    ANALYZING = "analyzing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"

class Verdict(str, Enum):
    AUTHENTIC = "authentic"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    LIKELY_FAKE = "likely_fake"
    FAKE = "fake"

class ContentType(str, Enum):
    VIDEO_WITH_SPEECH = "video_with_speech"
    VIDEO_NO_SPEECH = "video_no_speech"
    AUDIO_ONLY = "audio_only"
    IMAGE_ONLY = "image_only"
    TEXT_ONLY = "text_only"

# ============== BASE MODELS ==============

class BaseSchema(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True
    )

# ============== INPUT SCHEMAS ==============

class FileInput(BaseSchema):
    file_id: str
    file_type: str
    original_filename: str
    file_hash: str
    file_size: int
    duration_seconds: Optional[float] = None

class AnalyzeOptions(BaseSchema):
    modalities: Optional[List[Modality]] = None  # None = auto-detect
    generate_report: bool = True
    generate_heatmaps: bool = True
    defense_level: str = "standard"  # none, standard, aggressive

# ============== RESULT SCHEMAS ==============

class TrustScore(BaseSchema):
    value: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    calibrated: bool = True

class SpatialResult(BaseSchema):
    score: float = Field(..., ge=0, le=1)
    per_frame_scores: List[float]
    anomaly_indices: List[int]
    heatmap_urls: List[str]

class TemporalResult(BaseSchema):
    consistency_score: float = Field(..., ge=0, le=1)
    flickering_detected: bool
    anomaly_timestamps: List[float]

class LipSyncResult(BaseSchema):
    sync_score: float = Field(..., ge=0, le=1)
    manipulation_probability: float = Field(..., ge=0, le=1)
    detected_technology: Optional[str] = None

class VideoResult(BaseSchema):
    spatial: SpatialResult
    temporal: TemporalResult
    lip_sync: Optional[LipSyncResult] = None
    aggregate_score: float = Field(..., ge=0, le=1)
    frames_analyzed: int
    face_detected: bool

class AudioResult(BaseSchema):
    synthetic_probability: float = Field(..., ge=0, le=1)
    vocoder_artifacts_detected: bool
    voice_consistency_score: float = Field(..., ge=0, le=1)
    spectrogram_url: Optional[str] = None

class TextResult(BaseSchema):
    ai_probability: float = Field(..., ge=0, le=1)
    perplexity_score: float
    burstiness_score: float
    radar_score: Optional[float] = None

class C2PAManifest(BaseSchema):
    present: bool
    valid: Optional[bool] = None
    issuer: Optional[str] = None
    issued_at: Optional[datetime] = None
    assertions: List[Dict[str, Any]] = []

class MetadataResult(BaseSchema):
    c2pa: C2PAManifest
    exif_anomalies: List[str]
    file_structure_valid: bool

class ManipulationRegion(BaseSchema):
    region_type: str  # "face", "mouth", "background"
    location: str  # Description or coordinates
    confidence: float = Field(..., ge=0, le=1)
    frame_indices: Optional[List[int]] = None

class Explanation(BaseSchema):
    summary: str
    key_findings: List[str]
    manipulation_regions: List[ManipulationRegion]
    confidence_rationale: str
    methodology_used: List[str]

# ============== ANALYSIS DOCUMENT ==============

class AnalysisDocument(BaseSchema):
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: AnalysisStatus = AnalysisStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Input
    input: FileInput
    options: AnalyzeOptions
    
    # Results (populated after analysis)
    trust_score: Optional[TrustScore] = None
    verdict: Optional[Verdict] = None
    video_result: Optional[VideoResult] = None
    audio_result: Optional[AudioResult] = None
    text_result: Optional[TextResult] = None
    metadata_result: Optional[MetadataResult] = None
    explanation: Optional[Explanation] = None
    
    # Outputs
    report_url: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None

# ============== API SCHEMAS ==============

class AnalysisRequest(BaseSchema):
    options: AnalyzeOptions = AnalyzeOptions()

class AnalysisResponse(BaseSchema):
    analysis_id: str
    status: AnalysisStatus
    trust_score: Optional[TrustScore] = None
    verdict: Optional[Verdict] = None
    explanation: Optional[Explanation] = None
    report_url: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class AnalysisDetailResponse(AnalysisResponse):
    video_result: Optional[VideoResult] = None
    audio_result: Optional[AudioResult] = None
    text_result: Optional[TextResult] = None
    metadata_result: Optional[MetadataResult] = None
    processing_time_seconds: Optional[float] = None

# ============== INTERNAL SCHEMAS ==============

class PreprocessedData(BaseSchema):
    analysis_id: str
    content_type: ContentType
    frames: Optional[List[str]] = None  # MinIO keys
    face_crops: Optional[List[str]] = None  # MinIO keys
    audio_key: Optional[str] = None  # MinIO key
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = {}

class ModalityResult(BaseSchema):
    modality: Modality
    score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    details: Dict[str, Any]

class AggregatedResult(BaseSchema):
    modality_results: List[ModalityResult]
    fused_score: float = Field(..., ge=0, le=1)
    uncertainty: float = Field(..., ge=0, le=1)
    weights_used: Dict[str, float]
```

---

# APPENDIX B: CONFIGURATION REFERENCE

## Environment Variables

```bash
# /app/backend/.env

# ============== DATABASE ==============
MONGO_URL=mongodb://localhost:27017
DB_NAME=argus_prime

# ============== STORAGE ==============
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_UPLOADS=argus-uploads
MINIO_BUCKET_PREPROCESSED=argus-preprocessed
MINIO_BUCKET_RESULTS=argus-results

# ============== REDIS ==============
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# ============== ML CONFIGURATION ==============
MODEL_CACHE_DIR=/models
USE_GPU=true
GPU_MEMORY_LIMIT_MB=3500
ENABLE_TENSORRT=true
FALLBACK_TO_CPU=true

# ============== PROCESSING ==============
MAX_VIDEO_DURATION_SECONDS=300
MAX_FILE_SIZE_MB=500
FRAME_SAMPLE_RATE_SHORT=5
FRAME_SAMPLE_RATE_MEDIUM=10
FRAME_SAMPLE_RATE_LONG=15

# ============== SCORING ==============
SCORE_WEIGHT_VIDEO_SPATIAL=0.30
SCORE_WEIGHT_VIDEO_TEMPORAL=0.25
SCORE_WEIGHT_AUDIO=0.20
SCORE_WEIGHT_METADATA=0.15
SCORE_WEIGHT_TEXT=0.10

VERDICT_THRESHOLD_AUTHENTIC=80
VERDICT_THRESHOLD_LIKELY_AUTHENTIC=60
VERDICT_THRESHOLD_UNCERTAIN=40
VERDICT_THRESHOLD_LIKELY_FAKE=20

# ============== SECURITY ==============
JWT_SECRET=your-secure-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
API_RATE_LIMIT_PER_MINUTE=100

# ============== CORS ==============
CORS_ORIGINS=*

# ============== LOGGING ==============
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Configuration Loader

```python
# /app/backend/config.py

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # Database
    mongo_url: str
    db_name: str = "argus_prime"
    
    # Storage
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket_uploads: str = "argus-uploads"
    minio_bucket_preprocessed: str = "argus-preprocessed"
    minio_bucket_results: str = "argus-results"
    
    # Redis
    redis_url: str
    celery_broker_url: str
    celery_result_backend: str
    
    # ML
    model_cache_dir: str = "/models"
    use_gpu: bool = True
    gpu_memory_limit_mb: int = 3500
    enable_tensorrt: bool = True
    fallback_to_cpu: bool = True
    
    # Processing
    max_video_duration_seconds: int = 300
    max_file_size_mb: int = 500
    frame_sample_rate_short: int = 5
    frame_sample_rate_medium: int = 10
    frame_sample_rate_long: int = 15
    
    # Scoring
    score_weight_video_spatial: float = 0.30
    score_weight_video_temporal: float = 0.25
    score_weight_audio: float = 0.20
    score_weight_metadata: float = 0.15
    score_weight_text: float = 0.10
    
    verdict_threshold_authentic: int = 80
    verdict_threshold_likely_authentic: int = 60
    verdict_threshold_uncertain: int = 40
    verdict_threshold_likely_fake: int = 20
    
    # Security
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_rate_limit_per_minute: int = 100
    
    # CORS
    cors_origins: str = "*"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

config = get_settings()
```

---

**Document Version:** 1.0  
**Classification:** Implementation Blueprint  
**Status:** READY FOR IMPLEMENTATION  
**Compliance:** AGENTS.md (Backend) ✓ | AGENTS_FRONTEND.md (Frontend) ✓

---

*This document provides the complete implementation blueprint for Argus Prime. All files, schemas, and contracts are defined. Development should proceed in the order specified in Section 3.*
