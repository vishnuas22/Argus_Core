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
