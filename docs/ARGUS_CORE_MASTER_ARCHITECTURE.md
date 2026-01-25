# ARGUS CORE: Multi-Modal Deepfake Detection & Forensic Analysis Platform

## Master Architecture Document v1.0
**Date:** January 2026  
**Classification:** Technical Architecture & Implementation Roadmap  
**Hardware Target:** RTX 3050 (4GB VRAM) | 16GB RAM | 500GB SSD  
**Deployment:** Vercel (MVP) → On-Premise Production

---

# PART 1: EXECUTIVE SUMMARY

## 1.1 State-of-the-Art in Deepfake Detection (2025-2026)

### Current Landscape
The deepfake detection field has undergone significant evolution. Key findings from our research:

| Challenge | 2023 Reality | 2025-2026 SOTA |
|-----------|--------------|----------------|
| **Human Detection Accuracy** | 24.5% | Still ~25-30% |
| **AI Detection on Old Fakes** | 95%+ | Maintained |
| **AI Detection on Diffusion Fakes** | 50-60% | 85-95% with new models |
| **Cross-Dataset Generalization** | Poor (75-80% drop) | Improved via CLIP/VLMs |
| **rPPG/Biological Signals** | Effective | **Compromised** - modern deepfakes preserve heartbeats |

### Critical Insight: rPPG is No Longer Reliable
2025 research from Fraunhofer HHI confirms that high-quality deepfakes (DeepFaceLive, dual-decoder autoencoders) **retain realistic heart rate signals** from source videos with correlation coefficients of 0.57-0.82. This fundamentally undermines FakeCatcher-style biological detection. Our architecture will use rPPG as a **secondary signal only**, not primary.

### Winning Detection Paradigm
The current SOTA approach combines:
1. **Vision-Language Models (CLIP/SigLIP)** for generalization
2. **Temporal Transformers** for video consistency analysis
3. **Multi-modal fusion** (audio + visual + text + metadata)
4. **Explainable AI (GradCAM/SHAP)** for forensic evidence

---

## 1.2 Top 5 Recommended Open-Source Libraries/Models

### Tier 1: MUST INTEGRATE (Production-Ready)

| # | Model/Framework | Use Case | GitHub/Source | Why This? |
|---|-----------------|----------|---------------|-----------|
| **1** | **DeepfakeBench** | Unified Detection Framework | [github.com/SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | 15 SOTA detectors, 9 datasets, standardized pipeline |
| **2** | **M2F2-Det** | Video Face Forgery (Explainable) | [github.com/CHELSEA234/M2F2_Det](https://github.com/CHELSEA234/M2F2_Det) | CVPR 2025 Oral, CLIP + LLM explanations |
| **3** | **Purdue-M2 Voice Generalization** | Audio Deepfake Detection | [github.com/Purdue-M2/AI-Synthesized-Voice-Generalization](https://github.com/Purdue-M2/AI-Synthesized-Voice-Generalization) | AAAI 2025, vocoder artifact detection |
| **4** | **LIPINC-V2** | Lip-Sync Deepfake Detection | [github.com/skrantidatta/LIPINC-V2](https://github.com/skrantidatta/LIPINC-V2) | Specialized for Wav2Lip, Diff2Lip detection |
| **5** | **c2pa-rs / c2pa-python** | Content Authenticity | [github.com/contentauth/c2pa-rs](https://github.com/contentauth/c2pa-rs) | C2PA v2.3 standard for forensic provenance |

### Tier 2: SUPPLEMENTARY (For Specific Use Cases)

| Model | Use Case | Source |
|-------|----------|--------|
| **DFD-FCG** | Facial Component Guided Detection | CVPR 2025 |
| **GPTZero API** | AI Text Detection | gptzero.me (API) |
| **OpenVINO** | CPU Inference Optimization | Intel |
| **RADAR** | Robust AI Text Detection | IBM/NeurIPS |
| **deepfake-detector-model-v1** | Quick Image Classification | HuggingFace |

---

## 1.3 Hardware Feasibility Assessment

### RTX 3050 (4GB VRAM) Constraints & Solutions

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MEMORY BUDGET ANALYSIS                           │
├─────────────────────────────────────────────────────────────────────┤
│ Available VRAM: 4GB                                                 │
│ OS/Display Overhead: ~0.3GB                                         │
│ Usable for Inference: ~3.7GB                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Model Requirements (FP32):                                          │
│   • EfficientNet-B3: ~1.2GB                                        │
│   • ResNet50: ~0.8GB                                               │
│   • CLIP ViT-B/16: ~0.6GB                                          │
│   • Audio Model (RawNet3): ~0.3GB                                  │
│   • Total FP32: ~2.9GB ✅ FITS                                      │
├─────────────────────────────────────────────────────────────────────┤
│ With INT8 Quantization:                                             │
│   • All models: ~0.8GB                                             │
│   • Frame buffer (5 frames): ~0.5GB                                │
│   • Total Optimized: ~1.3GB ✅ COMFORTABLE                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Optimization Strategy

1. **ONNX + INT8 Quantization**: 4x model size reduction, 3-4x speedup
2. **TensorRT EP**: Maximize RTX 3050's Ampere architecture
3. **OpenVINO CPU Fallback**: For parallel processing when GPU busy
4. **Batch Processing**: Queue-based with Celery/Redis
5. **Model Caching**: Keep frequently used models in VRAM

### Expected Performance (RTX 3050)

| Analysis Type | Time per Item | Throughput |
|---------------|---------------|------------|
| Image (single frame) | ~50ms | 20 fps |
| Video (30s, sampled) | ~8-12s | 5 videos/min |
| Audio (60s clip) | ~3-5s | 12 clips/min |
| Text (500 words) | ~200ms | 300 docs/min |
| Full Multi-modal | ~15-20s | 3-4 analyses/min |

---

# PART 2: TECHNICAL ARCHITECTURE

## 2.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ARGUS CORE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   WEB UI     │    │  REST API    │    │  WEBHOOK     │    │  BATCH CLI   │  │
│  │  (React)     │    │  (FastAPI)   │    │  (MinIO)     │    │  (Python)    │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │                   │          │
│         └───────────────────┼───────────────────┼───────────────────┘          │
│                             ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         API GATEWAY / ROUTER                             │   │
│  │                    (Authentication, Rate Limiting)                       │   │
│  └───────────────────────────────────┬─────────────────────────────────────┘   │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         JOB ORCHESTRATOR                                 │   │
│  │                    (Celery + Redis Queue)                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ Ingest Job  │  │ Preprocess  │  │ Analysis    │  │ Report Gen  │     │   │
│  │  │             │  │ Job         │  │ Job         │  │ Job         │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └───────────────────────────────────┬─────────────────────────────────────┘   │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    MULTI-MODAL DETECTION ENGINE                          │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌─────────────┐  │   │
│  │  │ VIDEO ANALYZER│ │ AUDIO ANALYZER│ │ TEXT ANALYZER │ │IMAGE ANALYZER│  │   │
│  │  │               │ │               │ │               │ │             │  │   │
│  │  │ • FaceExtract │ │ • Spectrogram │ │ • Perplexity  │ │• CLIP/SigLIP│  │   │
│  │  │ • Temporal    │ │ • Vocoder Det │ │ • Burstiness  │ │• Artifact   │  │   │
│  │  │ • Lip-Sync    │ │ • Voice Bio   │ │ • RADAR       │ │• GradCAM    │  │   │
│  │  │ • M2F2-Det    │ │ • Purdue-M2   │ │               │ │             │  │   │
│  │  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └─────┬───────┘  │   │
│  │          └─────────────────┴─────────────────┴───────────────┘          │   │
│  │                                    ▼                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │              TRUST SCORE ENGINE (Weighted Fusion)                │    │   │
│  │  │    Score = w₁×Visual + w₂×Audio + w₃×Text + w₄×Metadata        │    │   │
│  │  │    (Weights adaptive based on available modalities)             │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └───────────────────────────────────┬─────────────────────────────────────┘   │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      EXPLAINABLE AI MODULE                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │   │
│  │  │ GradCAM      │  │ SHAP         │  │ Textual      │                   │   │
│  │  │ Heatmaps     │  │ Importance   │  │ Explanations │                   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │   │
│  └───────────────────────────────────┬─────────────────────────────────────┘   │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      FORENSIC REPORT GENERATOR                           │   │
│  │  • PDF Report with Evidence                                              │   │
│  │  • C2PA Content Credentials (if available)                               │   │
│  │  • Chain of Custody Metadata                                             │   │
│  │  • Heatmap Visualizations                                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐             │
│  │   MinIO          │  │   MongoDB        │  │   Redis          │             │
│  │   (File Storage) │  │   (Analysis DB)  │  │   (Job Queue)    │             │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 Detection Module Deep Dive

### A. Video Analysis Pipeline

```python
# Conceptual Pipeline Flow
class VideoAnalyzer:
    """
    Multi-stage video deepfake detection
    """
    
    stages = [
        # Stage 1: Face Extraction & Tracking
        "RetinaFace/MTCNN → Face crops (every N frames)",
        
        # Stage 2: Spatial Analysis (per-frame)
        "M2F2-Det with CLIP encoder → Artifact detection",
        "EfficientNet-B3 → Binary classification",
        
        # Stage 3: Temporal Analysis
        "X-CLIP Transformer → Frame consistency",
        "LIPINC-V2 → Lip-sync verification (if speech)",
        
        # Stage 4: Biological Signals (Secondary)
        "rPPG extraction → Heart rate anomaly (low weight)",
        
        # Stage 5: Ensemble & Explanation
        "Weighted voting → Final score",
        "GradCAM → Manipulation heatmap"
    ]
```

**Frame Sampling Strategy for RTX 3050:**
- Short videos (<30s): Every 5th frame
- Medium videos (30s-2min): Every 10th frame
- Long videos (>2min): Key frame extraction + every 15th frame
- Smart sampling: Extra frames around detected anomalies

### B. Audio Analysis Pipeline

```python
class AudioAnalyzer:
    """
    Synthetic voice and audio manipulation detection
    """
    
    stages = [
        # Stage 1: Audio Preprocessing
        "Extract audio track → 16kHz mono WAV",
        "Voice Activity Detection → Speech segments",
        
        # Stage 2: Feature Extraction
        "Mel-spectrogram + MFCC features",
        "Raw waveform (for RawNet3-style)",
        
        # Stage 3: Vocoder Artifact Detection
        "Purdue-M2 model → Synthesis artifacts",
        "Spectral analysis → Unnatural harmonics",
        
        # Stage 4: Cross-Modal Verification
        "Audio-Visual sync check (if video)",
        "Lip movement correlation"
    ]
```

### C. Text Analysis Pipeline

```python
class TextAnalyzer:
    """
    AI-generated text detection
    """
    
    metrics = {
        "perplexity": "Low perplexity indicates LLM generation",
        "burstiness": "Uniform sentence length = AI pattern",
        "vocabulary_diversity": "AI uses more common words",
        "repetition_patterns": "Subtle phrase repetitions",
        "coherence_score": "Too perfect = suspicious"
    }
    
    models = [
        "GPTZero API (if available)",
        "RADAR (IBM) - adversarial robust",
        "Custom BERT classifier (fallback)"
    ]
```

### D. Image Analysis Pipeline

```python
class ImageAnalyzer:
    """
    Single-frame deepfake and AI-generated image detection
    """
    
    detectors = [
        "SigLIP-based classifier (HuggingFace)",
        "EfficientNet-B3 (DeepfakeBench)",
        "Frequency domain analysis (DCT artifacts)",
        "EXIF/metadata analysis"
    ]
    
    explainability = [
        "GradCAM heatmap generation",
        "SHAP feature importance",
        "Artifact localization"
    ]
```

---

## 2.3 Trust Score Engine

### Weighted Fusion Algorithm

```
TRUST_SCORE = Σ(wᵢ × sᵢ × cᵢ) / Σ(wᵢ × cᵢ)

Where:
- wᵢ = Base weight for modality i
- sᵢ = Raw score from detector (0-100)
- cᵢ = Confidence of detection (0-1)
```

### Default Weight Configuration

| Modality | Base Weight | Rationale |
|----------|-------------|-----------|
| **Video (Spatial)** | 0.30 | Primary indicator for face swaps |
| **Video (Temporal)** | 0.25 | Catches flickering, inconsistencies |
| **Audio** | 0.20 | Critical for voice cloning |
| **Metadata/C2PA** | 0.15 | Provenance is strong evidence |
| **Text** | 0.10 | Context-dependent |

### Adaptive Weighting

```python
def compute_adaptive_weights(available_modalities, content_type):
    """
    Adjust weights based on what's available and content type
    """
    if content_type == "video_with_speech":
        # Lip-sync becomes very important
        weights["lip_sync"] = 0.20
        weights["audio"] = 0.25
    
    elif content_type == "image_only":
        # All weight to spatial analysis
        weights["spatial"] = 0.70
        weights["metadata"] = 0.30
    
    # Normalize to sum to 1.0
    return normalize(weights)
```

---

## 2.4 Explainable AI (XAI) Implementation

### GradCAM Heatmap Generation

```python
def generate_gradcam_heatmap(model, input_image, target_class="fake"):
    """
    Generate Class Activation Map showing manipulation regions
    
    Output: Overlay image with red regions indicating high manipulation probability
    """
    # Forward pass
    features = model.feature_extractor(input_image)
    predictions = model.classifier(features)
    
    # Backward pass for target class
    gradients = torch.autograd.grad(
        predictions[target_class],
        features,
        retain_graph=True
    )
    
    # Generate heatmap
    weights = torch.mean(gradients, dim=[2, 3])
    cam = torch.sum(weights * features, dim=1)
    cam = F.relu(cam)  # Only positive contributions
    
    # Normalize and resize to input dimensions
    heatmap = normalize_and_resize(cam, input_image.shape)
    
    return overlay_heatmap(input_image, heatmap)
```

### Textual Explanation Generation

Using M2F2-Det's LLM explanation capability:

```
Input: Detected deepfake frame
Output: "This frame shows signs of manipulation in the mouth region 
        (confidence: 87%). The lip movements appear inconsistent with 
        the audio track, suggesting Wav2Lip or similar lip-sync 
        technology was used. Additional artifacts detected around 
        the jaw line indicate face boundary blending."
```

---

## 2.5 C2PA Forensic Integration

### Content Credentials Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    C2PA INTEGRATION FLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │ Input Media  │                                               │
│  └──────┬───────┘                                               │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────┐              │
│  │ C2PA Manifest Extraction                      │              │
│  │ • Check for existing Content Credentials      │              │
│  │ • Validate cryptographic signatures           │              │
│  │ • Extract provenance chain                    │              │
│  └──────┬───────────────────────────────────────┘              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────┐              │
│  │ Provenance Analysis                           │              │
│  │ • Original creation tool/device              │              │
│  │ • Edit history (if any)                      │              │
│  │ • AI generation indicators                   │              │
│  │ • Trust list verification                    │              │
│  └──────┬───────────────────────────────────────┘              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────┐              │
│  │ Report Generation                             │              │
│  │ • Include C2PA status in Trust Score         │              │
│  │ • Flag "No Provenance" as risk factor        │              │
│  │ • Display verified creation info             │              │
│  └──────────────────────────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# PART 3: IMPLEMENTATION ROADMAP

## 3.1 Development Phases

### Phase 1: Foundation (Weeks 1-2) - MVP
**Goal:** Basic upload → analysis → result flow

```
Week 1:
├── Day 1-2: Project scaffolding
│   ├── FastAPI backend structure
│   ├── React frontend with file upload
│   ├── MinIO setup for file storage
│   └── MongoDB schemas
│
├── Day 3-4: Single-modality detection
│   ├── Image analysis with EfficientNet-B3
│   ├── Basic ONNX runtime integration
│   └── Simple confidence score output
│
└── Day 5-7: Basic UI/UX
    ├── Upload interface
    ├── Progress indicators
    └── Results display

Week 2:
├── Day 1-3: Video analysis (basic)
│   ├── Frame extraction pipeline
│   ├── Per-frame analysis
│   └── Aggregate scoring
│
├── Day 4-5: Audio analysis (basic)
│   ├── Audio extraction from video
│   ├── Spectrogram generation
│   └── Basic synthetic voice detection
│
└── Day 6-7: Integration & Testing
    ├── Multi-modal result aggregation
    └── Basic Trust Score calculation
```

### Phase 2: Advanced Detection (Weeks 3-4)
**Goal:** SOTA models + explainability

```
Week 3:
├── M2F2-Det integration (video)
├── Purdue-M2 integration (audio)
├── Text analysis module
└── GradCAM heatmap generation

Week 4:
├── Lip-sync detection (LIPINC-V2)
├── Temporal consistency analysis
├── SHAP feature importance
└── Trust Score fine-tuning
```

### Phase 3: Forensics & Production (Weeks 5-6)
**Goal:** Legal-grade reports + optimization

```
Week 5:
├── C2PA integration
├── PDF report generation
├── Chain of custody metadata
└── Export functionality

Week 6:
├── INT8 quantization
├── TensorRT optimization
├── OpenVINO CPU fallback
├── Load testing
└── Security hardening
```

---

## 3.2 Database Schema Design

### MongoDB Collections

```javascript
// analyses collection
{
  "_id": ObjectId,
  "analysis_id": "uuid",
  "created_at": ISODate,
  "status": "pending|processing|completed|failed",
  
  "input": {
    "file_id": "minio_object_key",
    "file_type": "video|audio|image|text",
    "original_filename": "string",
    "file_hash": "sha256",
    "file_size": Number,
    "duration_seconds": Number  // for video/audio
  },
  
  "results": {
    "trust_score": 0-100,
    "verdict": "authentic|likely_authentic|uncertain|likely_fake|fake",
    
    "video_analysis": {
      "spatial_score": Number,
      "temporal_score": Number,
      "lip_sync_score": Number,
      "face_detected": Boolean,
      "frames_analyzed": Number,
      "anomaly_frames": [Number],
      "heatmap_urls": [String]
    },
    
    "audio_analysis": {
      "synthetic_probability": Number,
      "vocoder_artifacts_detected": Boolean,
      "voice_consistency_score": Number,
      "spectrogram_url": String
    },
    
    "text_analysis": {
      "ai_probability": Number,
      "perplexity_score": Number,
      "burstiness_score": Number
    },
    
    "metadata_analysis": {
      "c2pa_present": Boolean,
      "c2pa_valid": Boolean,
      "provenance_chain": [Object],
      "exif_anomalies": [String]
    }
  },
  
  "explanation": {
    "summary": "Human-readable explanation",
    "key_findings": [String],
    "manipulation_regions": [
      {
        "type": "face|audio|text",
        "location": "description or coordinates",
        "confidence": Number
      }
    ]
  },
  
  "report_url": "string",  // Generated PDF
  "processing_time_seconds": Number
}

// jobs collection (for queue management)
{
  "_id": ObjectId,
  "job_id": "uuid",
  "analysis_id": "uuid",
  "job_type": "ingest|preprocess|analyze|report",
  "status": "queued|running|completed|failed",
  "created_at": ISODate,
  "started_at": ISODate,
  "completed_at": ISODate,
  "error_message": String,
  "retry_count": Number
}
```

---

## 3.3 API Endpoint Design

```yaml
# Core Analysis Endpoints
POST   /api/analyze              # Submit file for analysis
GET    /api/analyze/{id}         # Get analysis status/results
GET    /api/analyze/{id}/report  # Download PDF report
DELETE /api/analyze/{id}         # Delete analysis

# Batch Operations
POST   /api/analyze/batch        # Submit multiple files
GET    /api/analyze/batch/{id}   # Get batch status

# Quick Analysis (No storage)
POST   /api/quick/image          # Quick image check
POST   /api/quick/text           # Quick text check

# Explainability
GET    /api/analyze/{id}/heatmap        # Get GradCAM overlay
GET    /api/analyze/{id}/explanation    # Get detailed explanation

# C2PA
GET    /api/analyze/{id}/provenance     # Get C2PA chain
POST   /api/verify/c2pa                 # Verify C2PA credentials

# Health & Metrics
GET    /api/health               # Service health
GET    /api/metrics              # Processing metrics
```

---

## 3.4 Frontend Component Structure

```
src/
├── components/
│   ├── upload/
│   │   ├── DropZone.jsx         # Drag & drop upload
│   │   ├── UploadProgress.jsx   # Upload progress bar
│   │   └── FilePreview.jsx      # Preview uploaded media
│   │
│   ├── analysis/
│   │   ├── TrustScoreGauge.jsx  # Visual score display
│   │   ├── VerdictBadge.jsx     # Authentic/Fake badge
│   │   ├── ModalityBreakdown.jsx # Score per modality
│   │   └── Timeline.jsx         # Video timeline with flags
│   │
│   ├── explainability/
│   │   ├── HeatmapViewer.jsx    # GradCAM overlay viewer
│   │   ├── FindingsList.jsx     # Key findings display
│   │   └── ExplanationCard.jsx  # Human-readable explanation
│   │
│   ├── forensics/
│   │   ├── ProvenanceChain.jsx  # C2PA visualization
│   │   ├── MetadataTable.jsx    # EXIF/metadata display
│   │   └── ReportDownload.jsx   # PDF download button
│   │
│   └── common/
│       ├── LoadingSpinner.jsx
│       ├── ErrorBoundary.jsx
│       └── Toast.jsx
│
├── pages/
│   ├── Dashboard.jsx            # Main analysis dashboard
│   ├── AnalysisDetail.jsx       # Single analysis view
│   ├── BatchResults.jsx         # Batch analysis view
│   └── Settings.jsx             # Configuration
│
└── hooks/
    ├── useAnalysis.js           # Analysis state management
    ├── useUpload.js             # File upload logic
    └── useWebSocket.js          # Real-time updates
```

---

# PART 4: HARDWARE OPTIMIZATION GUIDE

## 4.1 Model Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                MODEL OPTIMIZATION WORKFLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original Model (PyTorch FP32)                                  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────┐                           │
│  │ Export to ONNX                   │                           │
│  │ torch.onnx.export(model, ...)    │                           │
│  └─────────────────────────────────┘                           │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────┐                           │
│  │ Static INT8 Quantization         │                           │
│  │ • Calibration dataset (100-500   │                           │
│  │   representative samples)        │                           │
│  │ • Per-channel quantization       │                           │
│  │ • ~4x size reduction            │                           │
│  └─────────────────────────────────┘                           │
│         │                                                       │
│         ├────────────────┬──────────────────┐                  │
│         ▼                ▼                  ▼                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ TensorRT     │ │ OpenVINO     │ │ ONNX Runtime │           │
│  │ (GPU Path)   │ │ (CPU Path)   │ │ (Fallback)   │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4.2 Memory Management Strategy

```python
class ModelManager:
    """
    Intelligent model loading for 4GB VRAM constraint
    """
    
    def __init__(self, max_vram_gb=3.5):
        self.max_vram = max_vram_gb * 1024  # MB
        self.loaded_models = {}
        self.model_sizes = {
            "efficientnet_b3_int8": 300,    # MB
            "clip_vit_b16_int8": 200,       # MB
            "audio_model_int8": 100,        # MB
            "lip_sync_model_int8": 150,     # MB
        }
    
    def load_for_task(self, task_type):
        """
        Load only models needed for current task
        Unload others if memory pressure
        """
        required = self.get_required_models(task_type)
        
        # Calculate memory needed
        needed_mb = sum(self.model_sizes[m] for m in required)
        current_mb = sum(self.model_sizes[m] for m in self.loaded_models)
        
        # Unload if necessary
        if current_mb + needed_mb > self.max_vram:
            self.unload_least_recently_used()
        
        # Load required models
        for model_name in required:
            if model_name not in self.loaded_models:
                self.load_model(model_name)
```

## 4.3 Batch Processing Configuration

```yaml
# Recommended settings for RTX 3050
batch_processing:
  # Video settings
  video:
    max_concurrent_jobs: 2
    frame_batch_size: 4          # Frames processed together
    max_frames_in_memory: 20
    frame_skip_short: 5          # Every 5th frame for <30s
    frame_skip_long: 15          # Every 15th frame for >2min
    
  # Audio settings
  audio:
    max_concurrent_jobs: 4
    chunk_duration_seconds: 10   # Process 10s chunks
    
  # Image settings  
  image:
    max_concurrent_jobs: 8
    batch_size: 4                # 4 images at once
    
  # Queue settings
  queue:
    max_queue_size: 100
    job_timeout_seconds: 300
    retry_max: 3
```

---

# PART 5: DATASETS & BENCHMARKING

## 5.1 Recommended Datasets

| Dataset | Size | Content | Use Case |
|---------|------|---------|----------|
| **FaceForensics++** | 3,600 videos | 4 manipulation types | Training (primary) |
| **DFDC** | 104,500 videos | Diverse, large-scale | Testing |
| **Celeb-DF-v2** | 5,639 videos | High-quality celeb swaps | Cross-dataset eval |
| **LipSyncTimit** | 9,090 videos | Wav2Lip, Diff2Lip | Lip-sync detection |
| **DeepFakeVox-HQ** | 1.3M samples | Audio deepfakes | Voice detection |
| **OpenFake** | Modern | GPT-Image, SD | 2025 generators |

## 5.2 Benchmark Targets

```
┌─────────────────────────────────────────────────────────────┐
│                    PERFORMANCE TARGETS                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Detection Accuracy (AUROC):                                │
│  ├── FF++ (in-domain):     > 95%                           │
│  ├── DFDC (cross-dataset): > 85%                           │
│  ├── Celeb-DF-v2:          > 88%                           │
│  └── OpenFake (modern):    > 80%                           │
│                                                             │
│  False Positive Rate:      < 5%                             │
│  False Negative Rate:      < 10%                            │
│                                                             │
│  Processing Speed (RTX 3050):                               │
│  ├── Image:    < 100ms                                     │
│  ├── Video (30s): < 15s                                    │
│  └── Audio (60s): < 5s                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# PART 6: SECURITY & COMPLIANCE

## 6.1 Security Considerations

```
┌─────────────────────────────────────────────────────────────┐
│                   SECURITY ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Validation:                                          │
│  ├── File type verification (magic bytes, not extension)   │
│  ├── Size limits (configurable per plan)                   │
│  ├── Malware scanning (ClamAV integration)                 │
│  └── Rate limiting per user/IP                             │
│                                                             │
│  Data Handling:                                             │
│  ├── Files encrypted at rest (MinIO server-side)          │
│  ├── Automatic deletion after configurable period          │
│  ├── No PII stored without consent                         │
│  └── GDPR-compliant data export/deletion                   │
│                                                             │
│  Access Control:                                            │
│  ├── JWT authentication                                    │
│  ├── Role-based access (admin, analyst, viewer)           │
│  └── API key management for integrations                   │
│                                                             │
│  Audit Trail:                                               │
│  ├── All analysis requests logged                          │
│  ├── Chain of custody for forensic reports                 │
│  └── Immutable audit log                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 6.2 Adversarial Defense

### Known Attack Vectors

1. **Adversarial Perturbations**: Noise added to fool detectors
2. **Compression Attacks**: Heavy compression to remove artifacts
3. **Anti-Forensic Techniques**: Targeted artifact removal

### Defense Strategy

```python
class AdversarialDefense:
    """
    Multi-layer defense against adversarial attacks
    """
    
    strategies = [
        # 1. Ensemble voting (harder to fool multiple models)
        "Use 3+ independent detectors, require majority agreement",
        
        # 2. Multi-scale analysis
        "Analyze at multiple resolutions (catches perturbations)",
        
        # 3. Frequency domain analysis
        "DCT/FFT analysis less susceptible to pixel perturbations",
        
        # 4. Compression robustness training
        "Train on heavily compressed samples",
        
        # 5. Confidence calibration
        "Flag low-confidence results for human review"
    ]
```

---

# APPENDIX A: QUICK REFERENCE

## A.1 GitHub Repositories

```
Core Detection:
├── https://github.com/SCLBD/DeepfakeBench
├── https://github.com/CHELSEA234/M2F2_Det
├── https://github.com/Purdue-M2/AI-Synthesized-Voice-Generalization
├── https://github.com/skrantidatta/LIPINC-V2
└── https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection

Forensics & Standards:
├── https://github.com/contentauth/c2pa-rs
└── https://github.com/contentauth/c2pa-python

Optimization:
├── https://github.com/openvinotoolkit/openvino
└── https://github.com/microsoft/onnxruntime

Platforms:
└── https://github.com/AHU-VLab/Deepfake-o-Meter (Reference)
```

## A.2 Key Python Dependencies

```
# Core ML
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
timm>=0.9.0

# ONNX & Optimization
onnx>=1.14.0
onnxruntime-gpu>=1.15.0
openvino>=2023.0

# Video/Audio Processing
opencv-python>=4.8.0
ffmpeg-python>=0.2.0
librosa>=0.10.0
moviepy>=1.0.3

# Explainability
captum>=0.6.0  # GradCAM
shap>=0.42.0

# Forensics
c2pa-python>=0.4.0
python-magic>=0.4.27
pillow>=10.0.0
```

## A.3 Environment Variables

```bash
# Database
MONGO_URL=mongodb://localhost:27017
DB_NAME=argus_core

# Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=argus-files

# ML Configuration
MODEL_CACHE_DIR=/models
USE_GPU=true
GPU_MEMORY_LIMIT=3500  # MB
ENABLE_TENSORRT=true
FALLBACK_TO_CPU=true

# Processing
MAX_VIDEO_DURATION=300  # seconds
MAX_FILE_SIZE_MB=500
FRAME_SAMPLE_RATE=5
BATCH_SIZE=4

# Security
JWT_SECRET=your-secret-key
API_RATE_LIMIT=100  # requests per minute
```

---

# APPENDIX B: DECISION LOG

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| **DeepfakeBench as base** | Unified framework, 15 models, standardized evaluation | Building from scratch (too slow) |
| **CLIP/VLM approach** | Best 2025 generalization to unseen fakes | CNN-only (poor on diffusion fakes) |
| **rPPG as secondary only** | 2025 research shows deepfakes preserve heartbeats | Primary rPPG (compromised) |
| **Batch over real-time** | RTX 3050 constraint, better accuracy | Real-time (hardware limitation) |
| **MongoDB** | Flexible schema for evolving analysis results | PostgreSQL (rigid schema) |
| **MinIO** | S3-compatible, self-hosted, webhook support | Local filesystem (no scaling) |
| **INT8 quantization** | 4x speedup with <2% accuracy loss | FP16 (still too large) |

---

**Document Version:** 1.0  
**Status:** AWAITING APPROVAL  
**Next Step:** Upon approval, proceed to Phase 2 (Implementation)

---

*This document was compiled based on comprehensive research of 2025-2026 SOTA deepfake detection methods, optimized for the specified hardware constraints (RTX 3050 4GB, 16GB RAM).*
