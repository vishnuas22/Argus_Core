# Argus Core — Architecture

**Version:** 1.9.0 | **Updated:** 2026-07-02

This is a 2-page overview. For the full 900-line engineering review, see [ENGINEERING_REVIEW.md](ENGINEERING_REVIEW.md).

---

## System Topology

```
                         ┌──────────────────────────────────────────┐
                         │              Nginx (TLS)                 │
                         │   :80 → 301 :443                         │
                         │   :443 → routes to internal services     │
                         └──────────────┬───────────────────────────┘
                                        │
            ┌───────────────┬───────────┼───────────────┬───────────┐
            │               │           │               │           │
     ┌──────▼─────┐  ┌──────▼─────┐ ┌───▼────┐  ┌──────▼─────┐ ┌───▼────┐
     │  Frontend  │  │  Backend   │ │  WS    │  │  Grafana   │ │ MinIO  │
     │  Next.js   │  │  FastAPI   │ │  /ws/  │  │  /grafana/ │ │ console│
     │  :3000     │  │  :8000     │ │        │  │  :3000     │ │ :9001  │
     └────────────┘  └──────┬─────┘ └────────┘  └────────────┘ └────────┘
                            │
                  ┌─────────┼──────────┐
                  │         │          │
           ┌──────▼────┐ ┌──▼───┐ ┌────▼─────┐
           │  Celery   │ │Redis │ │ MongoDB  │
           │  Worker   │ │:6379 │ │ :27017   │
           │  + Beat   │ │      │ │          │
           └──────┬────┘ └──────┘ └──────────┘
                  │
           ┌──────▼────┐
           │   MinIO   │  Object storage
           │   :9000   │  (uploads, preprocessed, results)
           └───────────┘
```

**Internal services are NOT exposed to the host** — only Nginx on 80/443.

---

## Detection Pipeline

```
1. UPLOAD (POST /api/v1/analyze)
   │
   ├─ File validation (magic bytes, size, sanitize)
   ├─ Per-user rate limit check
   ├─ Create AnalysisDocument in MongoDB
   ├─ Upload to MinIO (argus-uploads bucket)
   └─ Enqueue Celery task
       │
2. PREPROCESSING (Celery worker)
   │
   ├─ Download from MinIO
   ├─ Content type detection
   ├─ Frame extraction (video → frames)
   ├─ Audio extraction (video → waveform)
   ├─ Face detection
   └─ Upload preprocessed to MinIO (argus-preprocessed)
       │
3. ANALYSIS (parallel, per modality)
   │
   ├─ Image: CLIP+LoRA, DINOv2, SigLIP, SBI, EfficientNet-B3, DCT
   ├─ Video: Spatial (CNN), Temporal (X-CLIP/VideoMAE), LipSync
   └─ Audio: Wav2Vec2-XLS-R, AASIST3, ECAPA-TDNN, vocoder, voice consistency
       │
4. FUSION (evidential Dirichlet)
   │
   ├─ Per-modality evidence → alpha_fake, alpha_real
   ├─ Fused score = alpha_fake / (alpha_fake + alpha_real)
   ├─ Uncertainty = K / sum(alpha)
   └─ Metadata adjustment (C2PA, EXIF anomalies)
       │
5. SCORING (Platt calibration)
   │
   ├─ authenticity_prob = 1 - fused_score
   ├─ Platt calibration (if enabled, not IMAGE_ONLY)
   ├─ Trust score = authenticity_prob × 100
   └─ Verdict: authentic ≥80 | likely_authentic ≥60 | uncertain ≥40 | likely_fake ≥20 | fake
       │
6. XAI + REPORT
   │
   ├─ GradCAM++ heatmaps
   ├─ DCT frequency analysis
   ├─ Feature importance ranking
   ├─ Evidence package with reproducibility hash
   ├─ Scientific references
   └─ PDF forensic report (async)
       │
7. PERSIST
   │
   └─ Update MongoDB analysis document
       │
8. NOTIFY (WebSocket)
   │
   └─ Push progress + completion to frontend via Redis pub/sub
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Celery for analysis** | ML inference is CPU/GPU-bound; async event loop must not block. Celery prefork isolates each analysis. |
| **Evidential Dirichlet fusion** | Principled uncertainty quantification (Sensoy et al. NeurIPS 2018). Better than weighted average when modalities disagree. |
| **Platt calibration skipped for IMAGE_ONLY** | Image analyzer already applies sigmoid internally; double-calibration compresses scores toward 0.5. |
| **Per-model ONNX session locks** | `session.run()` is not thread-safe; per-model asyncio.Lock serializes access without blocking the event loop. |
| **Provenance tracking** | Every prediction tagged as `model_inference` / `heuristic_only` / `placeholder`. Placeholder predictions cannot be the sole basis for a verdict. |
| **Stuck-task reaper** | Celery worker crash = analysis stuck in STARTED forever. Reaper marks FAILED after 2× expected duration. |
| **Graceful drain on shutdown** | 25-second drain window lets in-flight requests complete before container exits. New requests get 503 + Retry-After. |
| **Daily MongoDB backup to MinIO** | mongodump → tar.gz → MinIO with 30-day lifecycle. Restore tested quarterly. |

---

## Data Flow

| Data | Where it lives | Retention |
|---|---|---|
| Analysis documents | MongoDB `analyses` collection | 90 days (TTL index) |
| Audit log | MongoDB `audit_log` collection | 90 days (TTL index) |
| Uploaded files | MinIO `argus-uploads` bucket | 30 days |
| Preprocessed data | MinIO `argus-preprocessed` bucket | 7 days |
| PDF reports | MinIO `argus-results` bucket | 90 days |
| Celery results | Redis DB 1 | 24 hours |
| ML models | Docker volume `models_data` | Permanent (versioned) |
| Prometheus metrics | Prometheus volume | 30 days |

---

## Reliability Mechanisms

| Mechanism | What it protects against |
|---|---|
| Celery task retries (3×, exponential backoff) | Transient failures |
| Storage retry + local fallback | MinIO outage |
| Stuck-task reaper (every 5 min) | Worker crash |
| Graceful drain (25s) | Lost requests on deploy |
| DrainingMiddleware (503) | New traffic during shutdown |
| Daily MongoDB backup | Data loss |
| Per-user rate limit | Single-user DoS |
| Per-IP rate limit | Anonymous abuse |
| Circuit breaker (planned) | Cascading DB failures |
| Health check alerts | Silent failures |

---

## Observability

| Signal | Tool | Where |
|---|---|---|
| Metrics | Prometheus → Grafana | `https://domain/grafana/` |
| Alerts | Alertmanager → Slack | `#argus-alerts` channel |
| Logs | structlog (JSON) → stdout | `docker logs argus-backend` |
| Error tracking | Sentry (optional) | Configure `SENTRY_DSN` |
| Distributed tracing | OpenTelemetry (planned) | — |

---

## Scaling

| Dimension | Strategy |
|---|---|
| Backend | Stateless — scale horizontally behind nginx |
| Celery workers | Scale `--concurrency` and/or add worker containers |
| MongoDB | Single primary (replica set for failover — post-MVP) |
| MinIO | Single node (distributed mode post-MVP) |
| Models | Shared volume — all workers see same weights |

**Throughput ceiling** (single node, M1 Max, MPS):
- Image: ~4 analyses/min (parallel)
- Video: ~1 analysis/2min (CPU-bound)
- Audio: ~6 analyses/min

---

## Security

| Layer | Mechanism |
|---|---|
| Network | TLS 1.2/1.3 via Nginx, internal services not exposed |
| Auth | JWT (HS256), optional on most endpoints, required for delete |
| Authz | Role-based (user, admin, analyst) — post-MVP |
| Rate limit | Per-IP (100/min) + per-user (10/hour free, 100/hour pro) |
| Upload | Magic-byte validation, 500MB cap, chunked read, ClamAV (planned) |
| Secrets | `.env` file (Docker secrets / Vault post-MVP) |
| Headers | HSTS, CSP, X-Frame-Options, X-Content-Type-Options |
| CORS | Explicit origins only in production (no wildcard) |

---

For the full engineering review (known issues, technical debt, refactoring roadmap), see [ENGINEERING_REVIEW.md](ENGINEERING_REVIEW.md).
