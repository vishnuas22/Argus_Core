# SOTA Model Audit for Argus Multimodal Deepfake Platform

**Audit Date:** 2026-03-05  
**Status:** Decision Complete (Doc-first before code changes)

## Scope
This audit validates whether each model key in the current build maps to the correct task category and a credible state-of-the-art (SOTA) source.

## Method
1. Inspected runtime wiring in current code (`registry.py`, analyzers, downloader, manager, docker entrypoint).
2. Verified model provenance against primary references:
- DeepfakeBench: https://github.com/SCLBD/DeepfakeBench
- X-CLIP paper: https://arxiv.org/abs/2207.07285
- Wav2Vec2 paper: https://arxiv.org/abs/2006.11477
- AASIST paper: https://arxiv.org/abs/2110.01200
- HuggingFace model cards for deployed checkpoints.

## Reality Check vs Vision
The platform vision is multimodal deepfake detection with non-random, category-correct models. Current build has strong components, but several critical mapping/provenance issues were found.

### Category Alignment Matrix
| Category | Runtime Key | Current State | Decision |
|---|---|---|---|
| Image AI/Real | `ai_real_detector` | Key category is correct, but source/provenance was inconsistent and SDXL-specialized | Standardize to `capcheck/ai-human-generated-image-detection` with dynamic label parsing for robust AI-vs-human mapping |
| Video Spatial | `ai_real_detector` + `clip_vit_b16` | Category usage is acceptable as pragmatic detector + feature model | Keep |
| Video Temporal | `xclip_temporal` | X-CLIP is a general temporal/video-language model, not a deepfake-specific classifier | Keep as temporal feature model, not as sole deepfake authority |
| Video Landmark Dependency | `retinaface_detector` (in code) | Invalid key (not in registry; registry key is `retinaface`) | Fix key to `retinaface` |
| Lip-sync Primary | `lipinc_v2` | Analyzer key is valid, downloader mapping incorrectly points to `wav2vec2` checkpoint | Keep key, remove incorrect implied provenance by fixing dependency wiring |
| Lip-sync Audio Features | `wav2vec2_features` (in code) | Invalid key (registry key is `wav2vec2_base`) | Fix key to `wav2vec2_base` |
| Audio Deepfake | `aasist_antispoof` + `purdue_m2` + `wav2vec2_base` | Category is broadly valid; naming/provenance in comments can be cleaner | Keep |
| Text AI detection | `roberta_ai_detector` | Key points to ModernBERT path by design alias; naming is confusing but functional | Keep key for compatibility; clarify as alias in metadata/comments |

## Mandatory Corrections (Implementation)
1. Fix invalid analyzer dependency keys:
- `retinaface_detector` -> `retinaface`
- `wav2vec2_features` -> `wav2vec2_base`

2. Remove non-operational image model source references:
- Replace `Nahrawy/AI-Generated-Detector-Versatile` with a production source and keep label mapping dynamic (`capcheck/ai-human-generated-image-detection`).

3. Startup warmup correctness:
- Replace stale warmup keys (`efficientnet_b3_spatial`, `siglip_deepfake`) with currently registered/used models.

## Out of Scope for This Patch
- Full model-architecture swap to benchmark winners requiring retraining (e.g., FTCN/AV-HuBERT/SpecRNet integrations).
- Large schema/API changes.

## Expected Outcome After Patch
- Model-category mappings become internally consistent.
- Startup/download path stops using inaccessible image detector source.
- Analyzer required-model checks resolve correctly without registry-key failures.
- Build is better aligned with project vision: multimodal, category-correct, non-random model usage.
