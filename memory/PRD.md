# ARGUS CORE - Product Requirements Document

## Original Problem Statement
Build "Argus Core - Multi-Modal Deepfake Detection & Forensic Analysis Platform" to address a $40B projected fraud threat. Current industry standard for human detection is only 24.5%. Need automated, multi-modal AI analysis (Text, Audio, Video, Image).

## Phase 1 Status: Research & Architecture
**Status:** COMPLETED ✅  
**Date:** January 2026

## Phase 2 Status: Backend Implementation
**Status:** COMPLETED ✅
**All Layers (0-7):** FULLY IMPLEMENTED
**Date:** January 2026

## Phase 3 Status: Frontend Implementation
**Status:** PENDING 🔜
**Current Focus:** React Frontend Development
**Date:** January 2026

## Core Requirements
1. Multi-modal detection (Video, Audio, Text, Image)
2. Trust Score Engine (0-100 weighted scoring)
3. Explainable AI reports with GradCAM heatmaps
4. C2PA forensic integration
5. Hardware: RTX 3050 (4GB VRAM), 16GB RAM, 500GB SSD

## User Personas
- **Security Analysts**: Need detailed forensic reports
- **Journalists**: Quick verification of media authenticity
- **Legal Teams**: Legal-admissible evidence generation
- **Content Moderators**: Batch processing capability

## Implementation Progress by Layer

### Layer 0: Foundation ✅ COMPLETED
- [x] schemas/schemas.py - All Pydantic data models
- [x] interfaces/analyzer.py - Abstract analyzer interface
- [x] interfaces/storage.py - Storage interface
- [x] interfaces/model.py - Model interface
- [x] config.py - Configuration loader
- [x] utils/errors.py - Exception classes

### Layer 1: Storage & Utilities ✅ COMPLETED
- [x] storage/storage.py - MinIO client wrapper
- [x] storage/db.py - MongoDB client wrapper
- [x] utils/logging.py - Structured logging
- [x] utils/metrics.py - Prometheus metrics

### Layer 2: Processing Pipeline ✅ COMPLETED
- [x] processing/sanitize.py - Input validation
- [x] processing/extract.py - Media extraction
- [x] processing/transform.py - Data transforms
- [x] processing/preprocess.py - Preprocessing orchestration

### Layer 3: Model Infrastructure ✅ COMPLETED
- [x] models/registry.py - Model metadata registry
- [x] models/manager.py - VRAM management
- [x] models/optimize.py - INT8 quantization utilities

### Layer 4: Core Engine ✅ COMPLETED
- [x] core/engine.py - Inference engine
- [x] core/explain.py - GradCAM/explanations
- [x] core/fusion.py - Multi-modal aggregation
- [x] core/scorer.py - Trust Score calculation

### Layer 5: Analyzers ✅ COMPLETED
- [x] analyzers/base.py - Abstract base class
- [x] analyzers/image.py - Image deepfake detection
- [x] analyzers/video/spatial.py - Per-frame spatial analysis
- [x] analyzers/video/temporal.py - Temporal consistency
- [x] analyzers/video/lipsync.py - Lip-sync detection
- [x] analyzers/video.py - Video orchestrator
- [x] analyzers/audio.py - Audio deepfake detection
- [x] analyzers/text.py - AI text detection
- [x] analyzers/metadata.py - C2PA, EXIF analysis

### Layer 6: API & Orchestration ✅ COMPLETED
- [x] api/deps.py - Dependency providers with JWT auth
- [x] api/middleware.py - CORS, rate limiting, auth middleware
- [x] api/router.py - Complete HTTP endpoints (analyze, status, list, delete, text)
- [x] api/websocket.py - Real-time progress updates with Redis Pub/Sub
- [x] core/orchestrator.py - Celery task definitions with DAG scheduling
- [x] server.py - FastAPI application entry point with lifecycle management

### Layer 7: Forensics ✅ COMPLETED
- [x] forensics/forensics.py - C2PA v2.3 manifest extraction & validation
- [x] forensics/report.py - PDF forensic report generation with ReportLab
- [x] forensics/audit.py - Cryptographic chain audit logging

## Key Technical Decisions
1. DeepfakeBench as unified detection framework
2. CLIP/VLM approach for generalization (not CNN-only)
3. rPPG as secondary signal only (compromised by modern deepfakes)
4. Batch processing over real-time (hardware constraint)
5. ONNX INT8 quantization for optimization

## Recent Changes
- **2026-01-26**: Backend Layer 0-7 COMPLETE - All backend files implemented
  - Full API layer with FastAPI endpoints
  - WebSocket support for real-time progress
  - Celery orchestrator for distributed processing
  - C2PA forensics integration
  - PDF report generation
  - Cryptographic audit trail

## Prioritized Backlog


# Argus Core - Project Documentation

## Overview
Multi-Modal Deepfake Detection & Forensic Analysis Platform

## Repository
- **Source**: https://github.com/vishnuas22/Argus_Core.git
- **Branch**: main

## Architecture
- **Backend**: FastAPI (Python)
- **Database**: MongoDB
- **Storage**: MinIO (optional - disabled for current env)
- **Cache**: Redis (optional - disabled for current env)

## What's Been Implemented
- [2026-01-26] Full repo replacement from GitHub
- [2026-01-26] Dependencies installed (100+ Python packages)
- [2026-01-26] Backend service running on port 8001
- [2026-01-26] Modified startup to make MinIO/Redis optional

## Available Endpoints
- `GET /` - API info
- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc documentation
- `GET /metrics` - Prometheus metrics

## Services Running
| Service | Port | Status |
|---------|------|--------|
| Backend (FastAPI) | 8001 | ✅ Running |
| MongoDB | 27017 | ✅ Running |
| Redis | 6379 | ✅ Running |
| MinIO | 9000 (API), 9001 (Console) | ✅ Running |
| Redis | 6379 | ✅ Running (Auto-configured) |
| MinIO | 9000 (API), 9001 (Console) | ✅ Running (Auto-configured) |

## Infrastructure Auto-Setup (2026-01-27)
- Redis installed and configured via supervisor (auto-start enabled)
- MinIO installed and configured via supervisor (auto-start enabled)
- All services automatically start with backend on boot
- Health checks passing for all components
- Storage integration fully operational

### Auto-Start Configuration
Services are managed via `/etc/supervisor/conf.d/`:
- `redis.conf` - Redis server on port 6379
- `minio.conf` - MinIO server on port 9000 (API) and 9001 (Console)

## MinIO Buckets Created
- `argus-uploads` - File uploads
- `argus-preprocessed` - Preprocessed media
- `argus-results` - Analysis results

## Redis
- Pub/Sub for WebSocket cross-worker messaging
- Celery broker/backend ready



### P0 (MVP - Critical)
- [x] Backend architecture scaffolding
- [ ] **React frontend with file upload** ← NEXT
- [x] MinIO integration for storage (storage/storage.py)
- [x] Basic image detection (analyzers/image.py)
- [ ] **Trust Score display in frontend** ← NEXT

### P1 (Core Features)
- [x] Video frame extraction pipeline (processing/extract.py)
- [x] Audio extraction and analysis (analyzers/audio.py)
- [x] Purdue-M2 voice detection (analyzers/audio.py)
- [x] GradCAM heatmap generation (core/explain.py)

### P2 (Advanced)
- [x] LIPINC-V2 lip-sync detection (analyzers/video/lipsync.py)
- [x] Temporal consistency analysis (analyzers/video/temporal.py)
- [x] Text/LLM detection (analyzers/text.py)
- [x] C2PA forensic integration (forensics/forensics.py)
- [x] PDF report generation (forensics/report.py)

### P3 (Production)
- [ ] INT8 quantization runtime
- [ ] TensorRT optimization
- [ ] OpenVINO CPU fallback
- [ ] Load testing
- [ ] Security hardening


| Service | Port | Status |
|---------|------|--------|
| Backend API | 8001 | ✅ Running |
| MongoDB | 27017 | ✅ Running |
| MinIO API | 9000 | ✅ Running |
| MinIO Console | 9001 | ✅ Running |
| Redis | 6379 | ✅ Running |

### Storage Buckets (MinIO)
- `argus-uploads` - Raw uploaded media files
- `argus-preprocessed` - Extracted frames, audio tracks
- `argus-results` - Heatmaps, analysis reports

---

## Implementation Progress

### Phase 1: Infrastructure Setup ✅
- [x] Repository cloned from GitHub
- [x] Python dependencies installed
- [x] MongoDB configured and running
- [x] MinIO installed and configured (supervisor)
- [x] Redis installed and configured (supervisor)
- [x] All default buckets created
- [x] Backend health checks passing

### Phase 2: Core API (From PRIME_ARGUS_DOCUMENT.md)
- [x] server.py - FastAPI entry point
- [x] config.py - Configuration loader
- [x] api/router.py - API endpoints
- [x] api/deps.py - Dependency injection
- [x] api/websocket.py - Real-time updates
- [x] api/middleware.py - CORS, rate limiting
- [x] storage/db.py - MongoDB client
- [x] storage/storage.py - MinIO client

### Phase 3: Analysis Pipeline (Pending)
- [ ] core/orchestrator.py - Celery task management
- [ ] core/engine.py - Inference engine
- [ ] core/fusion.py - Multi-modal fusion
- [ ] core/scorer.py - Trust scoring
- [ ] analyzers/* - Modality analyzers

---

## API Endpoints

### Health & System
- `GET /` - Root info
- `GET /health` - Basic health
- `GET /api/v1/health` - Detailed health with component status
- `GET /metrics` - Prometheus metrics

### Analysis (From router.py)
- `POST /api/v1/analyze` - Submit media for analysis
- `GET /api/v1/analyze/{analysis_id}` - Get analysis status/results

### WebSocket
- `WS /ws/analysis/{analysis_id}` - Real-time progress updates
- `WS /ws/updates` - Global system updates

---

## Configuration (.env)

```
# Database
MONGO_URL=mongodb://localhost:27017
DB_NAME=argus_core

# Storage (MinIO)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

---

## Backlog

### P0 - Critical
- [ ] React frontend with file upload
- [ ] Trust Score display in frontend
- [ ] WebSocket real-time progress

### P1 - High Priority
- [ ] ML model loading via ModelManager
- [ ] End-to-end media analysis flow

### P2 - Medium Priority
- [ ] PDF report generation
- [ ] C2PA forensics integration
---

## Notes
- All services managed via Supervisor for auto-restart
- Backend has hot-reload enabled for development
- MinIO credentials are default (change for production)



## Next Steps
1. **Create React Frontend** (PRIORITY)
   - File upload component with drag & drop
   - Real-time analysis progress via WebSocket
   - Trust Score visualization (gauge/meter)
   - Verdict display with color coding
   - Heatmap image viewer
   - PDF report download

2. Integration Testing
   - End-to-end media upload flow
   - WebSocket connection testing
   - Error handling verification

3. Production Hardening
   - Model optimization
   - Security audit
   - Performance benchmarking
