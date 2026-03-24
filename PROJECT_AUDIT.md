# Argus_Core Technical Audit Report
## VC/Acquisition Level Due Diligence

**Audit Date:** 2026-02-19  
**Auditor:** Elite Senior Technical Auditor & MLOps Architect  
**Methodology:** Ruthless, Skeptical, Data-Driven Analysis of Executable Code Only

---

## THE ONE-LINER REALITY

> **"A well-architected FastAPI inference engine with production-grade infrastructure, undermined by fragile type coercion, 173 blanket exception handlers, and a dangerous PlaceholderSession fallback that silently degrades prediction quality."**

---

## PRODUCTION READINESS SCORE: **62/100**

| Category | Score | Weight | Weighted |
|----------|-------|--------|----------|
| Architecture | 4/5 | 25% | 20 |
| Error Handling | 2/5 | 20% | 8 |
| Inference Integrity | 3/5 | 25% | 15 |
| Code Hygiene | 3/5 | 15% | 9 |
| Schema Compliance | 3/5 | 15% | 10 |
| **Total** | | | **62/100** |

---

## PHASE 1: THE IDENTITY CHECK

### Architecture Classification: **Microservices**

**Evidence:**
- [`docker-compose.yml`](docker-compose.yml) defines 6 services: backend, celery-worker, frontend, redis, mongodb, minio
- Service separation with dedicated queues: preprocessing, analysis, aggregation, reports
- Celery distributed task queue with Redis broker

**Architecture Strengths:**
- Clean service boundaries with health checks
- Dedicated worker pools for memory-intensive ML tasks
- Object storage (MinIO) for media files
- MongoDB for document storage
- Redis for caching and message brokering

**Architecture Weaknesses:**
- No API gateway pattern - direct backend exposure
- No circuit breaker implementation for service failures
- Celery worker shares identical codebase with API server (not truly decoupled)

### Dependency Risk Analysis

**Critical Findings from [`requirements.txt`](backend/requirements.txt):**

| Package | Version | Risk Level | Issue |
|---------|---------|------------|-------|
| `passlib` | 1.7.4 | **HIGH** | Unmaintained since 2020, known bcrypt timing vulnerabilities |
| `python-jose` | 3.5.0 | **MEDIUM** | Cryptographic implementation concerns |
| `numpy` | unpinned | **HIGH** | No version constraint - breaking changes possible |
| `torch` | unpinned | **HIGH** | No version constraint - major API changes between versions |
| `transformers` | unpinned | **HIGH** | No version constraint - model compatibility issues |
| `fastapi` | 0.110.1 | **LOW** | Minor version behind latest (0.115+) |
| `celery` | 5.6.2 | **LOW** | Current |

**Recommendation:** Pin all ML dependencies with `==` constraints and run `pip-audit` in CI/CD.

---

## PHASE 2: PRODUCTION READINESS AUDIT

### 2.1 Error Handling: **Score 2/5 (FAIL)**

**Finding: 173 `except Exception` blocks found across codebase**

This is the single most critical production risk. The codebase uses broad exception catching that:
1. Masks root causes
2. Makes debugging impossible in production
3. Silently degrades service quality

**Critical Examples:**

```python
# backend/models/manager.py:462-465
except Exception as e:
    logger.error(f"Failed to load ONNX model {metadata.name}: {e}")
    logger.warning("Falling back to placeholder session")
    return self._create_placeholder_session(metadata)  # SILENT DEGRADATION
```

**Impact:** When model loading fails, the system returns a `PlaceholderSession` that generates heuristic-based outputs instead of real model inference. Users receive predictions without knowing they're not from trained models.

```python
# backend/core/orchestrator.py:748-750
except Exception as e:
    logger.error(f"Modality analysis failed: {modality.value}, {e}")
    raise InferenceError(modality.value, str(e))
```

**Better but still problematic** - catches all exceptions instead of specific ones.

**Files with Most Exception Handlers:**
1. `backend/api/deps.py` - 18 occurrences
2. `backend/core/orchestrator.py` - 14 occurrences
3. `backend/models/downloader.py` - 14 occurrences
4. `backend/analyzers/audio.py` - 10 occurrences

### 2.2 Inference Integrity: **Score 3/5**

#### Model Loading Analysis

**Positive Findings:**
- Real ONNX models are loaded when available ([`manager.py:423-456`](backend/models/manager.py:423))
- File size validation detects placeholder files (< 10KB threshold)
- LRU cache with VRAM pressure monitoring

**Critical Issue: PlaceholderSession Fallback**

The [`PlaceholderSession`](backend/models/manager.py:741) class is a **production risk**:

```python
class PlaceholderSession:
    """
    Placeholder ONNX session for development/testing.
    
    Returns heuristic-based outputs derived from input statistics.
    """
    
    def run(self, output_names, input_feed, run_options=None):
        # Generates fake predictions based on:
        # - Image variance
        # - Edge density  
        # - High-frequency content
        # NOT from trained neural network weights
```

**Why This Is Dangerous:**
1. No external indication that placeholder is active
2. Health endpoint shows "healthy" even with placeholders
3. Predictions appear legitimate but are heuristic-based
4. Court admissibility claims are invalid with placeholder outputs

**GradCAM/Heatmap Analysis:**

The [`xai.py`](backend/core/xai.py) implementation has two modes:

1. **Real GradCAM++** (lines 236-285): Uses actual model features when available
2. **Synthetic Heatmap** (lines 287-341): Generated from image statistics when features unavailable

```python
def _generate_synthetic_image_heatmap(self, image: np.ndarray) -> np.ndarray:
    """
    Generate synthetic attention heatmap based on image analysis.
    Uses multiple heuristics for deepfake detection:
    - Edge density analysis
    - Texture variance
    - Color coherence
    """
```

**Verdict:** Heatmaps are mathematically grounded but may not reflect actual model attention if features are unavailable.

#### Audio Pipeline Analysis

**Recent Fix in [`orchestrator.py:894-911`](backend/core/orchestrator.py:894):**

```python
def _build_audio_result(result: ModalityResult) -> AudioResult:
    details = result.details or {}
    
    # Handle vocoder_artifacts which might be a dict or bool
    vocoder_artifacts = details.get("vocoder_artifacts", False)
    if isinstance(vocoder_artifacts, dict):
        vocoder_detected = vocoder_artifacts.get("artifact_score", 0) > 0.5
    elif isinstance(vocoder_artifacts, bool):
        vocoder_detected = vocoder_artifacts
    else:
        vocoder_detected = False
```

**Assessment:** This is a **symptom patch**, not a root cause fix. The type inconsistency should be resolved at the analyzer level, not coerced in the orchestrator.

### 2.3 Code Hygiene: **Score 3/5**

#### DRY Violations Found

**Result Building Duplication:**

The `_build_*_result` functions in [`orchestrator.py`](backend/core/orchestrator.py:894) share identical patterns:

```python
def _build_audio_result(result: ModalityResult) -> AudioResult:
    details = result.details or {}
    return AudioResult(
        synthetic_probability=details.get("synthetic_probability", 1 - result.score),
        ...
    )

def _build_text_result(result: ModalityResult) -> TextResult:
    details = result.details  # No "or {}" - inconsistent!
    return TextResult(
        ai_probability=details.get("ai_probability", 1 - result.score),
        ...
    )

def _build_image_result(result: ModalityResult) -> ImageResult:
    details = result.details  # No "or {}" - inconsistent!
    return ImageResult(
        ai_generated_probability=details.get("ai_generated_probability", ...),
        ...
    )
```

**Issue:** Inconsistent null handling (`details = result.details` vs `details = result.details or {}`)

**Recommendation:** Implement Factory Pattern with base class:

```python
class ResultBuilder(ABC):
    @abstractmethod
    def build(self, result: ModalityResult) -> BaseResult:
        details = result.details or {}  # Consistent null handling
        return self._build_impl(result, details)
```

### 2.4 Schema-Driven Compliance: **Score 3/5**

**Pydantic Models in [`schemas.py`](backend/schemas/schemas.py):**

The schemas are well-defined with proper validation:

```python
class AudioResult(BaseSchema):
    synthetic_probability: float = Field(..., ge=0, le=1)
    vocoder_artifacts_detected: bool = Field(default=False)
    voice_consistency_score: float = Field(..., ge=0, le=1)
```

**However:** The orchestrator bypasses schema validation by using dictionary access:

```python
# orchestrator.py:908
synthetic_probability=details.get("synthetic_probability", 1 - result.score),
```

**Issue:** If `synthetic_probability` is outside [0, 1] range in the dict, it passes through until Pydantic validation at the response level - potentially causing 500 errors instead of handled validation errors.

**Recommendation:** Validate at the boundary using Pydantic models, not dictionaries.

---

## PHASE 3: THE TECH DEBT LIST

### Critical (Will Break in Production)

| ID | File | Line | Issue | Impact |
|----|------|------|-------|--------|
| C1 | `models/manager.py` | 462-465 | PlaceholderSession fallback | Silent prediction degradation |
| C2 | `requirements.txt` | 76-77 | Unpinned numpy/torch | Deployment failures |
| C3 | `core/orchestrator.py` | 748-750 | Broad exception catch | Debugging impossible |
| C4 | `analyzers/audio.py` | 1190-1192 | Audio loading failure returns None | Cascading failures |

### High (Will Cause Issues Under Load)

| ID | File | Line | Issue | Impact |
|----|------|------|-------|--------|
| H1 | `api/deps.py` | 82-84 | Exception returns False | Health checks unreliable |
| H2 | `core/engine.py` | 203-204 | Batch size exception defaults silently | Performance degradation |
| H3 | `models/manager.py` | 736-738 | VRAM check exception passes | Memory exhaustion |

### Medium (Technical Debt)

| ID | File | Line | Issue | Impact |
|----|------|------|-------|--------|
| M1 | `orchestrator.py` | 894-941 | DRY violation in result builders | Maintenance burden |
| M2 | `analyzers/audio.py` | 489-492 | Librosa cache workaround | Suboptimal performance |
| M3 | `core/xai.py` | 287-341 | Synthetic heatmap fallback | Misleading explanations |

---

## MARKET VIABILITY ANALYSIS

### Competitive Positioning (2025-2026 SOTA)

| Feature | Argus_Core | SOTA 2025 | Gap |
|---------|------------|-----------|-----|
| Text Detection | ModernBERT, GPT-2 perplexity | RADAR-v2, DetectGPT | Minor |
| Image Detection | EfficientNet-B3, SigLIP | DINOv2, CLIP-based | Minor |
| Audio Detection | Purdue-M2, AASIST | AASIST-L, RawNet3 | Minor |
| Video Detection | X-CLIP, EfficientNet | VideoMAE, ViViT | Moderate |
| Explainability | GradCAM++, DCT | SHAP, LIME integrated | Minor |
| Real-time Processing | Celery queues | Ray Serve, Triton | Moderate |

### Market Viability: **YES, with conditions**

**Strengths:**
- Multi-modal coverage (text, image, audio, video)
- Court-admissible evidence generation
- ONNX runtime for deployment flexibility
- Proper infrastructure architecture

**Gaps to Address:**
1. **Model versioning** - No A/B testing capability for model updates
2. **Adversarial robustness** - Defense levels exist but not benchmarked
3. **Real-time latency** - Celery adds queue latency; consider Ray Serve for <100ms SLA
4. **Model monitoring** - No drift detection or performance tracking

---

## IMPROVEMENT ROADMAP

### Immediate (P0 - Security/Correctness)

1. **Remove PlaceholderSession or add explicit health indicator**
   - Add `is_placeholder: bool` to model health response
   - Return 503 Service Unavailable if all models are placeholders

2. **Pin all dependencies**
   ```txt
   numpy==1.26.4
   torch==2.2.0
   transformers==4.38.0
   ```

3. **Replace broad exception handlers**
   ```python
   # Instead of:
   except Exception as e:
   
   # Use:
   except (ONNXRuntimeError, ModelLoadError) as e:
   ```

### Short-term (P1 - Reliability)

4. **Implement Factory Pattern for result building**
   - Create `ResultBuilderFactory` with modality-specific builders
   - Centralize null handling and type coercion

5. **Add schema validation at analyzer boundary**
   - Validate `ModalityResult.details` with Pydantic before passing to orchestrator

6. **Implement circuit breaker for model loading**
   - Track consecutive failures
   - Open circuit after threshold
   - Allow recovery with exponential backoff

### Medium-term (P2 - Architecture)

7. **Separate inference service**
   - Move model inference to dedicated service
   - Use gRPC or Ray Serve for communication
   - Enable horizontal scaling of inference workers

8. **Add model monitoring**
   - Track prediction distribution drift
   - Monitor inference latency percentiles
   - Alert on confidence degradation

9. **Implement proper type coercion layer**
   - Create `TypeCoercer` class for vocoder_artifacts handling
   - Move type handling out of orchestrator

---

## APPENDIX: Audit Methodology

### Files Analyzed

| Category | Files | Lines of Code |
|----------|-------|---------------|
| Core Engine | 6 | ~15,000 |
| Analyzers | 8 | ~25,000 |
| API Layer | 4 | ~10,000 |
| Models | 6 | ~15,000 |
| Infrastructure | 3 | ~500 |
| **Total** | **27** | **~65,500** |

### Tools Used

- Static analysis: Regex search for exception patterns
- Import analysis: Module dependency tracing
- Schema validation: Pydantic model inspection
- Architecture analysis: Docker Compose topology

---

## CONCLUSION

Argus_Core is a **well-architected system with serious production risks**. The infrastructure is solid, the multi-modal approach is market-viable, and the explainability features are differentiated. However, the **silent fallback to placeholder models** and **173 broad exception handlers** represent existential risks for a production deployment.

**Investment Recommendation:** Fund with milestone-based conditions:
1. Remove PlaceholderSession fallback or add explicit health indicators
2. Reduce broad exception handlers by 80%
3. Pin all ML dependencies
4. Add model monitoring and drift detection

**Estimated Remediation Effort:** 2-3 engineering sprints for P0 items, 2-3 months for full production hardening.
