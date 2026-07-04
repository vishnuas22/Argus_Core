# Argus Core — Multi-Modal Deepfake Detection Platform

A production-grade platform for detecting AI-generated and manipulated content across **image**, **video**, and **audio** modalities, with explainable AI (XAI), court-admissible forensic reports, and continuous learning.

## Quick Start

### Prerequisites

- **Docker** and Docker Compose v2
- **Python 3.11+** (for native Mac development with MPS)
- **Node.js 18+** and Yarn (for frontend)
- **8 GB RAM minimum** (16 GB recommended for video analysis)

### Option 1: Docker (Linux/x86 with GPU)

```bash
# 1. Clone and configure
git clone <repo-url> argus
cd argus
cp .env.example .env
# Edit .env — set JWT_SECRET, MONGO_PASSWORD, REDIS_PASSWORD, MINIO_SECRET_KEY

# 2. Generate strong secrets
openssl rand -hex 32  # for JWT_SECRET
openssl rand -hex 32  # for MONGO_PASSWORD
openssl rand -hex 32  # for REDIS_PASSWORD
openssl rand -hex 32  # for MINIO_SECRET_KEY

# 3. Start the stack
docker compose -f docker-compose.prod.yml up -d

# 4. Verify health
curl https://localhost/api/v1/health  # If TLS configured
# OR
docker compose -f docker-compose.prod.yml exec backend curl localhost:8000/api/v1/health

# 5. Open the frontend
# https://localhost (or http://localhost:3000 in dev)
```

### Option 2: Native Mac Development (M1/M2/M3/M4 — uses MPS GPU)

```bash
# 1. Start stateful services in Docker (MongoDB, Redis, MinIO)
docker compose -f docker-compose.mac-dev.yml up -d

# 2. Run the setup script (creates venv, installs MPS PyTorch, runs tests)
chmod +x scripts/setup_mac_dev.sh
./scripts/setup_mac_dev.sh

# 3. Verify MPS is available
python scripts/verify_mps.py

# 4. Start the backend (native, with MPS acceleration)
source backend/.venv/bin/activate
cd backend
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# 5. (Another terminal) Start Celery worker
cd backend
celery -A core.orchestrator.celery_app worker --loglevel=info --pool=solo

# 6. (Another terminal) Start the frontend
cd frontend
yarn install && yarn dev
```

## What It Does

Upload an image, video, or audio file. Argus Core:

1. **Preprocesses** the media (extract frames, audio, faces, metadata)
2. **Runs multiple SOTA detectors** in parallel per modality:
   - **Image**: CLIP+LoRA, DINOv2, SigLIP, SBI, EfficientNet-B3, DCT analysis
   - **Video**: Spatial (frame-level CNN), Temporal (X-CLIP/VideoMAE/TimeSformer), LipSync
   - **Audio**: Wav2Vec2-XLS-R antispoofing, AASIST3, ECAPA-TDNN, CDP-Mamba, vocoder artifacts
3. **Fuses** results with evidential Dirichlet fusion (Sensoy et al. NeurIPS 2018)
4. **Calibrates** the trust score with Platt scaling
5. **Generates XAI evidence**: GradCAM++ heatmaps, DCT analysis, feature importance, provenance records
6. **Produces a court-admissible PDF report** with chain of custody

## Architecture

```
                    ┌──────────────┐
                    │   Frontend   │  Next.js 14 + TypeScript
                    │  :3000       │  Tailwind + shadcn/ui
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Backend    │  FastAPI + Pydantic v2
                    │  :8000       │  JWT auth, rate limiting, WS
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼─────┐ ┌───▼────┐ ┌────▼─────┐
       │  Celery    │ │ MongoDB│ │  MinIO   │
       │  Worker    │ │  :27017│ │  :9000   │
       │  (analysis)│ │        │ │          │
       └──────┬─────┘ └────────┘ └──────────┘
              │
       ┌──────▼─────┐
       │   Redis    │  Task broker + cache
       │   :6379    │
       └────────────┘
```

**Detection pipeline:**

```
Upload → Preprocess → [Image | Video | Audio analyzers] → Ensemble fusion
  → Platt-calibrated trust score → Verdict → XAI artifacts
  → Court-admissible PDF report
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

## Key Features

- **Multi-modal**: image, video, and audio deepfake detection
- **SOTA detector ensemble**: 23 models from HuggingFace, pinned with SHA-256 verification
- **Evidential Dirichlet fusion**: principled uncertainty quantification
- **Platt calibration**: honest probability estimates
- **Explainable AI**: GradCAM++ heatmaps, DCT analysis, feature importance, provenance records
- **Court-admissible reports**: PDF with chain of custody, reproducibility hash, confidence intervals
- **Adversarial defenses**: randomized preprocessing, certified robustness, adversarial gate
- **Continuous learning**: A/B testing, LoRA retraining, drift detection
- **Production infrastructure**: Docker, Celery, MongoDB, MinIO, Prometheus, Grafana, Alertmanager
- **Security**: TLS, JWT auth, per-IP + per-user rate limiting, file upload validation, security headers
- **Reliability**: graceful shutdown, stuck-task reaper, circuit breakers, daily backups

## Documentation

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 2-page system overview with diagram |
| [PRODUCTION_READINESS_AUDIT.md](PRODUCTION_READINESS_AUDIT.md) | Gap analysis vs production-ready |
| [RUNBOOK.md](RUNBOOK.md) | Ops runbook — what to do when things break |
| [AGENTS.md](AGENTS.md) | Developer guide — architecture, known issues, recent fixes |
| [ENGINEERING_REVIEW.md](ENGINEERING_REVIEW.md) | Deep engineering review (900 lines) |
| [CHANGELOG.md](CHANGELOG.md) | All notable changes |
| [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) | Maximum-performance optimization pass details |

## API

Once running, API docs are at:
- **Swagger UI**: `https://localhost/api/v1/docs` (or `http://localhost:8000/docs` in dev)
- **ReDoc**: `https://localhost/api/v1/redoc`

Key endpoints:

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/analyze` | Upload and analyze media |
| GET | `/api/v1/analyze/{id}` | Get analysis status |
| GET | `/api/v1/analyze/{id}/detail` | Get full results |
| GET | `/api/v1/analyze/{id}/report` | Get PDF report URL |
| GET | `/api/v1/analyze/{id}/heatmaps` | Get heatmap URLs |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/models` | List loaded models |
| WS | `/ws/analysis/{id}` | Real-time progress updates |

## Testing

```bash
# CPU-runnable unit tests (no torch/onnx/MongoDB required)
cd backend
python -m pytest tests/ \
  --ignore=tests/test_lip_sync_module.py \
  --ignore=tests/test_training_pipeline.py \
  -q
```

Expected: 268 passed, 34 skipped (MongoDB-dependent).

## Deployment

See [RUNBOOK.md](RUNBOOK.md) § "Initial Deployment" for the full production deployment checklist.

Quick version:

1. Provision a Linux server (8 vCPU, 32 GB RAM, 500 GB SSD)
2. Install Docker + Docker Compose
3. Obtain TLS certificates (Let's Encrypt or commercial)
4. `cp .env.example .env` and fill in real secrets
5. Configure nginx with your domain
6. `docker compose -f docker-compose.prod.yml up -d`
7. Verify health: `curl https://your-domain/api/v1/health`
8. Configure Slack/PagerDuty alerts in `alertmanager/alertmanager.yml`

## License

See [LICENSE](LICENSE).

## Contributing

1. Read [AGENTS.md](AGENTS.md) for architecture and conventions
2. Run `python -m pytest tests/` before submitting PRs
3. Follow the existing code style (black, isort, line length 100)
4. Update [CHANGELOG.md](CHANGELOG.md) for all changes
