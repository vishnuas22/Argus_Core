# ARGUS CORE - Product Requirements Document

## Original Problem Statement
Build "Argus Core - Multi-Modal Deepfake Detection & Forensic Analysis Platform" to address a $40B projected fraud threat. Current industry standard for human detection is only 24.5%. Need automated, multi-modal AI analysis (Text, Audio, Video, Image).

## Phase 1 Status: Research & Architecture
**Status:** COMPLETED ✅  
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

## What's Been Implemented
- [x] Deep web research on SOTA 2025-2026 detection methods
- [x] Master Architecture Document created
- [x] Executive Summary with top 5 tools identified
- [x] Hardware feasibility assessment (RTX 3050 viable with INT8)
- [x] Implementation roadmap (6-week plan)

## Key Technical Decisions
1. DeepfakeBench as unified detection framework
2. CLIP/VLM approach for generalization (not CNN-only)
3. rPPG as secondary signal only (compromised by modern deepfakes)
4. Batch processing over real-time (hardware constraint)
5. ONNX INT8 quantization for optimization

## Prioritized Backlog

### P0 (MVP - Critical)
- [ ] FastAPI backend scaffolding
- [ ] React frontend with file upload
- [ ] MinIO integration for storage
- [ ] Basic image detection (EfficientNet-B3)
- [ ] Trust Score display

### P1 (Core Features)
- [ ] Video frame extraction pipeline
- [ ] M2F2-Det integration
- [ ] Audio extraction and analysis
- [ ] Purdue-M2 voice detection
- [ ] GradCAM heatmap generation

### P2 (Advanced)
- [ ] LIPINC-V2 lip-sync detection
- [ ] Temporal consistency analysis
- [ ] Text/LLM detection
- [ ] C2PA forensic integration
- [ ] PDF report generation

### P3 (Production)
- [ ] INT8 quantization
- [ ] TensorRT optimization
- [ ] OpenVINO CPU fallback
- [ ] Load testing
- [ ] Security hardening

## Next Steps
Awaiting user approval to proceed to Phase 2: Implementation
