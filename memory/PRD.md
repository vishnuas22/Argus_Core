# ARGUS CORE - Product Requirements Document

## Original Problem Statement
Build "Argus Core - Multi-Modal Deepfake Detection & Forensic Analysis Platform" to address a $40B projected fraud threat. Current industry standard for human detection is only 24.5%. Need automated, multi-modal AI analysis (Text, Audio, Video, Image).

## Phase 1 Status: Research & Architecture
**Status:** COMPLETED ✅  
**Date:** January 2026

## Phase 2 Status: Backend Implementation
**Status:** IN PROGRESS 🔄
**Current Layer:** Layer 5 - Analyzers
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
- [x] analyzers/metadata.py - C2PA, EXIF analysis ✅ JUST ADDED

### Layer 6: API & Orchestration 🔜 NEXT
- [ ] api/deps.py - Dependency providers
- [ ] api/middleware.py - Auth, rate limiting
- [ ] api/router.py - HTTP endpoints
- [ ] api/websocket.py - Real-time updates
- [ ] core/orchestrator.py - Celery tasks
- [ ] server.py - FastAPI app

### Layer 7: Forensics 🔜 PENDING
- [ ] forensics/forensics.py - C2PA integration
- [ ] forensics/report.py - PDF generation
- [ ] forensics/audit.py - Audit logging

## Key Technical Decisions
1. DeepfakeBench as unified detection framework
2. CLIP/VLM approach for generalization (not CNN-only)
3. rPPG as secondary signal only (compromised by modern deepfakes)
4. Batch processing over real-time (hardware constraint)
5. ONNX INT8 quantization for optimization

## Recent Changes
- **2026-01-26**: Added `analyzers/metadata.py` - Complete C2PA and EXIF metadata analyzer
  - Magic bytes file format detection
  - EXIF metadata extraction and consistency analysis
  - C2PA Content Credentials extraction and validation
  - File structure anomaly detection
  - Tampering indicator detection
  - Authenticity scoring based on metadata signals

## Prioritized Backlog

### P0 (MVP - Critical)
- [x] Backend architecture scaffolding
- [ ] React frontend with file upload
- [ ] MinIO integration for storage
- [ ] Basic image detection (EfficientNet-B3)
- [ ] Trust Score display

### P1 (Core Features)
- [x] Video frame extraction pipeline (processing/extract.py)
- [ ] M2F2-Det integration
- [x] Audio extraction and analysis (analyzers/audio.py)
- [x] Purdue-M2 voice detection (analyzers/audio.py)
- [x] GradCAM heatmap generation (core/explain.py)

### P2 (Advanced)
- [x] LIPINC-V2 lip-sync detection (analyzers/video/lipsync.py)
- [x] Temporal consistency analysis (analyzers/video/temporal.py)
- [x] Text/LLM detection (analyzers/text.py)
- [x] C2PA forensic integration (analyzers/metadata.py)
- [ ] PDF report generation

### P3 (Production)
- [ ] INT8 quantization
- [ ] TensorRT optimization
- [ ] OpenVINO CPU fallback
- [ ] Load testing
- [ ] Security hardening

## Next Steps
1. Implement Layer 6: API & Orchestration
   - Start with api/deps.py (dependency injection)
   - Then api/router.py (HTTP endpoints)
   - Then core/orchestrator.py (Celery tasks)
   - Finally server.py (FastAPI app entry point)

2. Implement Layer 7: Forensics
   - forensics/forensics.py for C2PA signing
   - forensics/report.py for PDF reports
   - forensics/audit.py for audit trail
