# Argus Core — Model Audit & Architecture Recommendation

**Date:** 2026-07-02
**Audience:** Engineering team
**Purpose:** Curate the model registry to keep only valid, useful, best-in-class models per modality, and ensure lazy loading.

---

## Executive Summary

The current registry declares **22 models**. After audit:

| Category | Count | Action |
|---|---|---|
| ✅ Keep — valid, useful, best-in-class | 13 | Lazy-load |
| ❌ Remove — dead (never referenced in code) | 3 | Delete from registry |
| ❌ Remove — no working HF source | 2 | Delete from registry |
| ⚠️ Consolidate — duplicates | 1 | Merge into one entry |
| ⚠️ License-restricted | 1 | Gate behind config flag (already done) |
| ⚠️ Questionable source | 1 | Replace with verified alternative |
| 🔧 Fix — non-commercial license issue | 1 | Already gated; document clearly |

**Net result: 22 → 14 models.** Startup time drops from "load 22 models" to "load 0 models at startup, load on first use." First analysis takes a one-time hit (~3-10s for the first modality); subsequent analyses are fast.

---

## Detailed Findings

### Models to REMOVE (dead code — never referenced in any analyzer/detector)

#### 1. `xclip_temporal` — DEAD
- **Declared:** `models/registry.py:178`
- **Referenced in code:** grep finds zero `engine.infer("xclip_temporal")` calls
- **HF source:** `microsoft/xclip-base-patch16` — exists but not wired up
- **Action:** Remove from registry. If temporal X-CLIP analysis is wanted later, re-add with proper detector wiring.

#### 2. `clip_vit_l14` — DEAD
- **Declared:** `models/registry.py:264`
- **Referenced in code:** zero calls. `clip_vit_b16` is used instead.
- **Action:** Remove. The L/14 variant is 3× the VRAM (1200MB vs 400MB) with no proven accuracy benefit in this pipeline.

#### 3. `dinov2_vit_b14` — DEAD (duplicate)
- **Declared:** `models/registry.py:283`
- **Referenced in code:** zero calls. `dinov2_image_detector` is used instead (same backbone, with detection head).
- **Action:** Remove. The detector variant supersedes the bare feature extractor.

### Models to REMOVE (no working source)

#### 4. `cdp_mamba_audio_detector` — NO SOURCE
- **Declared:** `models/registry.py:540`
- **HF source:** `microsoft/CDP-Mamba-audio-deepfake` — **placeholder, "actual repo TBD"** per registry comment
- **download_url:** empty string
- **Action:** Remove from registry and from `detectors/__init__.py`. The CDP-Mamba paper (ICASSP 2025) is real but no public weights exist. Re-add when weights are published.

#### 5. `altfree_video_detector` — NO CANONICAL SOURCE
- **Declared:** `models/registry.py:588`
- **HF source:** `facebook/altfree-video-base` — **does not exist on HuggingFace** (per manifest note: "No canonical HF port exists")
- **Manifest says:** "Detector falls back to EfficientNet-B0 + transformer stub when the HF pull fails"
- **Action:** Remove. A stub that falls back to a different architecture is worse than no detector — it produces misleading scores. Use VideoMAE as the primary video temporal detector.

### Models to CONSOLIDATE (duplicates)

#### 6. `videomae_temporal` vs `videomae_video_detector`
- **Both point to:** `MCG-NJU/videomae-base` (same HF repo, same weights)
- **`videomae_temporal`:** used by `analyzers/video/temporal.py` for frame consistency
- **`videomae_video_detector`:** used by `detectors/videomae_detector.py` for deepfake classification
- **Action:** Keep ONE entry (`videomae_base`) used by both. The temporal analyzer and the detector share the same backbone — no need for two registry entries.

### Models to KEEP (with caveats)

#### 7. `aasist3_audio_detector` — QUESTIONABLE SOURCE
- **HF source:** `facebook/aasist3-base` — manifest says `verified_public: false`
- **Manifest alternatives:** `dima806/audio_deepfake_detection`, `MelodyMachine/Deepfake-audio-detection-V2`
- **Action:** Keep in registry but switch the default source to `MelodyMachine/Deepfake-audio-detection-V2` (most-downloaded, Apache-2.0, verified labels). Document the label polarity ({0:fake, 1:real}).

#### 8. `timesformer_video_detector` — NON-COMMERCIAL LICENSE
- **License:** CC-BY-NC-4.0
- **Status:** Already gated behind `ENABLE_TIMESFORMER` config flag (default true; should be false for commercial)
- **Action:** Keep but change default to `false`. Document prominently. For commercial deployment, this model is NOT usable.

### Models to KEEP (best-in-class, valid, useful)

#### Image modality (5 models — diverse ensemble)
| Model | Role | Why it's the best |
|---|---|---|
| `dinov2_image_detector` | Primary | DINOv2 backbone — 92% robustness under transformations vs CLIP's 42% (per Argus_Master research) |
| `clip_image_detector` | Secondary | CLIP + LoRA — different failure modes from DINOv2, ensemble diversity |
| `siglip_image_detector` | Tertiary | SigLIP's sigmoid loss produces uncorrelated features — 3rd diversity axis |
| `sbi_image_detector` | Boundary specialist | Self-Blended Images — detects face-swap boundary artifacts others miss |
| `ucf_cross_forgery_detector` | Cross-generator | UCF generalizes to unseen forgery families (AAAI 2024) |
| `deepfake_detector_v3` | Legacy primary | Used by video/spatial too — keep for backward compat, will deprecate |
| `retinaface` | Face detection | Required for face crop preprocessing |

#### Audio modality (3 models — diverse ensemble)
| Model | Role | Why it's the best |
|---|---|---|
| `wav2vec2_antispoof` | Primary ONNX | Wav2Vec2 XLSR, 4.01% EER on ASVspoof2019, INT8 quantized |
| `wav2vec2_xls_r_audio_detector` | SOTA PyTorch | Wav2Vec2-XLS-R-300M + MoE-LoRA — heaviest but most accurate |
| `aasist3_audio_detector` | Spectro-temporal | AASIST3 — different architecture (graph attention) for diversity |
| `ecapa_audio_detector` | Embedding-distance | ECAPA-TDNN — MIT license, commercially safe, requires reference centroid |
| `wav2vec2_base` | Feature extractor | Required by lipsync + voice consistency |

#### Video modality (3 models)
| Model | Role | Why it's the best |
|---|---|---|
| `videomae_base` (consolidated) | Temporal | VideoMAE — NeurIPS 2022, tube-masking, best open video backbone |
| `lipinc_v2` | Lip-sync | LIPINC-V2 — detects Wav2Lip/Diff2Lip with cross-attention |
| `deepfake_detector_v3` | Spatial (frame-level) | Reused from image — frame-level CNN |

---

## Lazy Loading Architecture

### Current behavior (problematic)
1. `server.py` lifespan calls `ensure_primary_models()` at startup
2. `bootstrap.py` downloads `deepfake_detector_v3` and `clip_vit_b16` if missing
3. `ModelManager.warmup()` preloads `deepfake_detector_v3` + `retinaface` into memory
4. Each analyzer calls `ensure_models_for_analyzer()` which checks availability of ALL required models before analysis

**Problem:** Steps 1-3 block startup for 30-60 seconds even if the user only uploads audio. Steps 4 blocks every analysis on model availability checks for models that may not be needed (e.g., image analyzer checks `retinaface` even for non-face images).

### New behavior (lazy)
1. `server.py` lifespan does NOT call `ensure_primary_models()` or `warmup()`
2. Models are downloaded on FIRST inference call, not at startup
3. `ModelManager.get_model()` already loads lazily — just remove the eager warmup
4. Analyzers no longer call `ensure_models_for_analyzer()` before analysis; the engine's `get_model()` handles it
5. Optional: background pre-load of the most-likely-needed models (image: dinov2 + retinaface) AFTER startup completes, so first request is fast without blocking startup

### Implementation
- Remove `warmup()` call from `server.py` lifespan (or make it opt-in via `WARMUP_ON_STARTUP=false`)
- Remove `ensure_primary_models()` call from lifespan
- Remove `ensure_models_for_analyzer()` calls from analyzers (the engine handles loading)
- Add `LazyModelLoader` wrapper that tracks load state and exposes `is_loaded`, `load()`, `ensure_loaded()`
- Add background pre-load task that runs AFTER `/api/v1/health` returns OK for the first time

### Expected startup time improvement
- **Before:** 30-60 seconds (download + load 2 models)
- **After:** 2-3 seconds (just import modules, no model loading)
- **First image analysis:** +8 seconds (one-time DINOv2 load)
- **First audio analysis:** +5 seconds (one-time wav2vec2 load)
- **First video analysis:** +12 seconds (one-time VideoMAE load)
- **Subsequent analyses:** 0 seconds extra (models cached)

---

## Final Curated Registry (14 models)

```python
# IMAGE (7)
"deepfake_detector_v3"      # Legacy ViT — used by video/spatial too
"retinaface"                # Face detection
"dinov2_image_detector"     # Primary: DINOv2 + MAC head
"clip_image_detector"       # Secondary: CLIP + LoRA
"siglip_image_detector"     # Tertiary: SigLIP (diversity)
"sbi_image_detector"        # Boundary artifacts
"ucf_cross_forgery_detector" # Cross-generator

# AUDIO (5)
"wav2vec2_antispoof"         # Primary ONNX (INT8, 4.01% EER)
"wav2vec2_xls_r_audio_detector" # SOTA PyTorch (MoE-LoRA)
"aasist3_audio_detector"     # Spectro-temporal (graph attention)
"ecapa_audio_detector"       # Embedding-distance (MIT license)
"wav2vec2_base"              # Feature extractor (lipsync + voice consistency)

# VIDEO (2)
"videomae_base"              # Temporal (consolidated from videomae_temporal + videomae_video_detector)
"lipinc_v2"                  # Lip-sync detection

# FEATURE (shared, 1)
"clip_vit_b16"               # CLIP vision encoder (used by video/spatial for generalization)
```

**Removed:** `xclip_temporal`, `clip_vit_l14`, `dinov2_vit_b14`, `cdp_mamba_audio_detector`, `altfree_video_detector`, `videomae_temporal` (merged into `videomae_base`), `timesformer_video_detector` (license-restricted, gated off by default — kept in registry but disabled)

**Total: 14 active models + 1 disabled (timesformer) = 15 entries.**
