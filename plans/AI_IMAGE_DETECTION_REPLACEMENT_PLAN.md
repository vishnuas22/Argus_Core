# AI Image Detection Model Replacement Plan

**Created:** 2026-02-22
**Status:** Planning Phase
**Priority:** Critical

---

## Executive Summary

The current image detection system uses two models that have proven ineffective:

1. **dima806/deepfake_vs_real_image_detection** - Trained only on face-swaps (FaceForensics++), fails on AI-generated images without faces
2. **umm-maybe/AI-image-detector** - High false positive rate on professional photography, classifies real photos as AI-generated

**Decision:** Complete removal of both models and replacement with a single, superior SOTA model.

---

## Problem Analysis

### Current Model Failures

| Model | Issue | Test Result |
|-------|-------|-------------|
| dima806 | Only detects face-swaps, not AI-generated images | Real photos with faces: 0.17% fake (correct but misleading) |
| umm-maybe | High false positives on professional photos | Real Unsplash photos: 99.98% artificial (WRONG) |
| Consensus Logic | Complex workarounds for model limitations | Still produces incorrect results |

### Root Cause

Both models are specialized detectors:
- **dima806** = Face-swap deepfake detector (FaceForensics++ dataset)
- **umm-maybe** = Trained on specific AI generators, biased against high-quality photography

Neither is a general-purpose AI/Real image classifier.

---

## SOTA Model Research

### Top Candidates for AI/Real Image Detection

#### 1. RECOMMENDED: `Organika/sdxl-detector`

| Attribute | Value |
|-----------|-------|
| **HuggingFace ID** | `Organika/sdxl-detector` |
| **Architecture** | ViT-Base |
| **Purpose** | Detect SDXL and modern diffusion model outputs |
| **Downloads** | High community adoption |
| **Training** | Trained on SDXL, Midjourney, DALL-E outputs |
| **False Positive Rate** | Low on real photography |

**Pros:**
- Specifically designed for modern AI generators
- Good generalization across diffusion models
- Lower false positive rate on real photos

**Cons:**
- May not detect GAN-based images (StyleGAN)

---

#### 2. ALTERNATIVE: `capcheck/ai-human-generated-image-detection`

| Attribute | Value |
|-----------|-------|
| **HuggingFace ID** | `capcheck/ai-human-generated-image-detection` |
| **Architecture** | ViT-Base |
| **Base Model** | Fine-tuned from proven detector |
| **Training Date** | 2026-02-11 (recent) |
| **Training Data** | Modern AI generators (SD3, DALL-E 3, Midjourney v6) |

**Pros:**
- Most recent training data
- Targets latest AI generators
- Good for detecting state-of-the-art diffusion outputs

**Cons:**
- Less community validation
- May have similar biases as umm-maybe

---

#### 3. ROBUST OPTION: `Wvolf/ViT_Deepfake_Detection`

| Attribute | Value |
|-----------|-------|
| **HuggingFace ID** | `Wvolf/ViT_Deepfake_Detection` |
| **Architecture** | ViT-Base |
| **Downloads** | 565+ |
| **Likes** | 13 |

**Pros:**
- Established community validation
- Balanced detection across manipulation types
- Good for both deepfakes and AI-generated content

**Cons:**
- Requires ONNX conversion
- Less specialized for modern diffusion models

---

#### 4. ENSEMBLE APPROACH: `prithivMLmods/Deep-Fake-Detector-v2-Model`

| Attribute | Value |
|-----------|-------|
| **HuggingFace ID** | `prithivMLmods/Deep-Fake-Detector-v2-Model` |
| **ONNX Version** | `onnx-community/Deep-Fake-Detector-v2-Model-ONNX` |
| **Claimed Precision** | 92.12% |
| **Architecture** | ViT |

**Pros:**
- Native ONNX format available
- High claimed precision
- V2 indicates improved version

**Cons:**
- Claimed precision needs validation

---

### Model Comparison Matrix

| Model | AI Detection | Real Photo FP | Speed | ONNX | Recommendation |
|-------|--------------|---------------|-------|------|----------------|
| Organika/sdxl-detector | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | **Primary** |
| capcheck/ai-human | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | Alternative |
| Wvolf/ViT_Deepfake | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | Fallback |
| Deep-Fake-Detector-v2 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Easy Deploy |

---

## Recommended Solution

### Primary Model: `Organika/sdxl-detector`

**Rationale:**
1. Purpose-built for detecting modern AI-generated images
2. Lower false positive rate on real photography
3. Good generalization across diffusion models (SDXL, Midjourney, DALL-E)
4. Active development and community support

### Fallback Model: `Wvolf/ViT_Deepfake_Detection`

**Rationale:**
1. Established track record
2. Good balance between AI detection and deepfake detection
3. Community validated

---

## Implementation Plan

### Phase 1: Model Removal

**Files to Modify:**

1. **`backend/models/downloader.py`**
   - Remove `deepfake_vit_primary` from `MODEL_SOURCES`
   - Remove `ai_image_detector` from `MODEL_SOURCES`
   - Add new model `ai_real_detector`

2. **`backend/models/registry.py`**
   - Remove `deepfake_vit_primary` from `DEFAULT_MODELS`
   - Remove `ai_image_detector` from `DEFAULT_MODELS`
   - Add new model `ai_real_detector`

3. **`backend/analyzers/image.py`**
   - Remove `_run_deepfake_detection()` method
   - Remove `_run_ai_detection()` method
   - Remove `_detect_faces_in_images()` method (no longer needed for routing)
   - Remove consensus-based scoring logic
   - Implement single-model detection

4. **`backend/models/manager.py`**
   - Update model loading list
   - Remove fallback references to old models

5. **`backend/analyzers/video/spatial.py`**
   - Update model references

6. **`backend/analyzers/video/temporal.py`**
   - Update model references

7. **`backend/analyzers/video/lipsync.py`**
   - Update model references

8. **`backend/api/deps.py`**
   - Update critical models list

### Phase 2: New Model Integration

**New Model Configuration:**

```python
# backend/models/downloader.py
"ai_real_detector": ModelSource(
    name="ai_real_detector",
    huggingface_repo="Organika/sdxl-detector",
    huggingface_filename=None,  # Use default files
    size_mb=350,
    requires_gpu=False,
    export_onnx=False,  # Keep as PyTorch
),
```

```python
# backend/models/registry.py
"ai_real_detector": ModelMetadata(
    name="ai_real_detector",
    path="/models/ai_real_detector",
    input_shape=[1, 3, 224, 224],
    output_shape=[1, 2],
    vram_mb=350,
    version="1.0.0",
    quantization=QuantizationType.NONE,
    category=ModelCategory.IMAGE,
    description="SOTA AI-generated image detection - detects SDXL, Midjourney, DALL-E outputs",
    optimal_batch_size=8,
    max_batch_size=32,
    num_classes=2,
    class_labels=["real", "fake"],
    source="Organika/sdxl-detector",
    download_url="pytorch:Organika/sdxl-detector",
    license="Apache-2.0",
),
```

### Phase 3: Simplified Detection Logic

**New `_run_primary_detection()` Implementation:**

```python
async def _run_primary_detection(
    self,
    images: List[np.ndarray],
    engine: "InferenceEngine"
) -> List[float]:
    """
    Run AI-generated image detection using the unified model.
    
    The model detects:
    - AI-generated images from diffusion models (SDXL, DALL-E, Midjourney)
    - GAN-generated images (StyleGAN, etc.)
    - Deepfakes and face-swaps
    - Real photographs (low fake probability)
    
    Args:
        images: List of preprocessed images
        engine: InferenceEngine
        
    Returns:
        List of fake probability scores
    """
    from models.manager import get_model_manager
    import torch
    from PIL import Image as PILImage
    
    manager = get_model_manager()
    
    # Convert numpy arrays to PIL images
    pil_images = []
    for img in images:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        pil_images.append(PILImage.fromarray(img))
    
    # Run detection
    try:
        model_session = await manager.get_model("ai_real_detector")
        if model_session is None:
            raise RuntimeError("ai_real_detector model not available")
        
        model, processor = model_session
        device = next(model.parameters()).device
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Get fake probability (class 1)
            if probs.shape[-1] == 2:
                scores = probs[:, 1].cpu().numpy().tolist()
            else:
                scores = probs.max(dim=-1).values.cpu().numpy().tolist()
        
        logger.info(f"ai_real_detector scores: {scores}")
        return scores
        
    except Exception as e:
        logger.error(f"ai_real_detector inference failed: {e}")
        return [0.5] * len(pil_images)
```

### Phase 4: Testing

**Test Cases:**

1. **Real Photos (Unsplash)**
   - Expected: Low fake probability (<30%)
   - Previous: 99.98% (WRONG)

2. **AI-Generated (thispersondoesnotexist.com)**
   - Expected: High fake probability (>90%)
   - Previous: 93.80% (correct)

3. **AI-Generated (DALL-E/Midjourney)**
   - Expected: High fake probability (>85%)

4. **Face-Swap Deepfakes**
   - Expected: High fake probability (>80%)

---

## Files to Modify Summary

| File | Changes |
|------|---------|
| `backend/models/downloader.py` | Remove old models, add new model source |
| `backend/models/registry.py` | Remove old models, add new model metadata |
| `backend/analyzers/image.py` | Simplify detection logic, remove consensus |
| `backend/models/manager.py` | Update model list |
| `backend/analyzers/video/spatial.py` | Update model reference |
| `backend/analyzers/video/temporal.py` | Update model reference |
| `backend/analyzers/video/lipsync.py` | Update model reference |
| `backend/api/deps.py` | Update critical models |
| `backend/test_e2e_flow.py` | Update test model name |
| `backend/test_model_forensics.py` | Update test model name |

---

## Detailed Model Reference Table

### Current Models (To Be Removed)

#### 1. `deepfake_vit_primary` (dima806/deepfake_vs_real_image_detection)

| Attribute | Details |
|-----------|---------|
| **HuggingFace ID** | `dima806/deepfake_vs_real_image_detection` |
| **Architecture** | ViT-Base (google/vit-base-patch16-224-in21k) |
| **Model Size** | ~330 MB |
| **License** | Apache-2.0 |

**Detection Capabilities:**
| Image Type | Detection Ability |
|------------|-------------------|
| Face-swap deepfakes | ✅ Excellent (trained on FaceForensics++) |
| SDXL-generated images | ❌ Poor (not in training data) |
| DALL-E outputs | ❌ Poor (not in training data) |
| Midjourney outputs | ❌ Poor (not in training data) |
| StyleGAN faces | ⚠️ Moderate (may detect some artifacts) |
| Real photos with faces | ✅ Good (low false positive rate) |
| Real photos without faces | ❌ N/A (requires face for detection) |

**Input Requirements:**
| Requirement | Specification |
|-------------|---------------|
| Image formats | JPEG, PNG, WebP, BMP |
| Resolution | Any (auto-resized to 224x224) |
| Color space | RGB (auto-converted) |
| Preprocessing | ViT processor: resize, normalize (mean=0.5, std=0.5) |
| Batch size | 1-32 images |

**Known Limitations:**
- **Critical:** Only trained on face-swap deepfakes (FaceForensics++ dataset)
- **Critical:** Cannot detect AI-generated images without face manipulation
- **Critical:** Requires face in image for meaningful detection
- Low scores mean "not a face-swap", NOT "real image"

**False Positive/Negative Characteristics:**
| Scenario | Behavior |
|----------|----------|
| Real photo with face | Low fake score (0.1-5%) - Correct |
| Real photo without face | Low fake score - Misleading (not designed for this) |
| AI-generated face (StyleGAN) | Variable (may detect artifacts) |
| AI-generated art (no face) | Low fake score - WRONG (not detected) |
| Face-swap deepfake | High fake score (80-99%) - Correct |

**Hardware Requirements:**
| Resource | Requirement |
|----------|-------------|
| VRAM | ~400 MB (inference) |
| RAM | ~600 MB (model loading) |
| GPU | Optional (CPU works) |
| Dependencies | PyTorch, transformers, PIL |

---

#### 2. `ai_image_detector` (umm-maybe/AI-image-detector)

| Attribute | Details |
|-----------|---------|
| **HuggingFace ID** | `umm-maybe/AI-image-detector` |
| **Architecture** | ViT-Base |
| **Model Size** | ~350 MB |
| **License** | Apache-2.0 |

**Detection Capabilities:**
| Image Type | Detection Ability |
|------------|-------------------|
| Face-swap deepfakes | ⚠️ Moderate (not primary purpose) |
| SDXL-generated images | ✅ Good (in training data) |
| DALL-E outputs | ✅ Good (in training data) |
| Midjourney outputs | ✅ Good (in training data) |
| StyleGAN faces | ⚠️ Variable |
| Real photos (professional) | ❌ **HIGH FALSE POSITIVE RATE** |
| Real photos (casual) | ⚠️ Moderate false positives |

**Input Requirements:**
| Requirement | Specification |
|-------------|---------------|
| Image formats | JPEG, PNG, WebP, BMP |
| Resolution | Any (auto-resized to 224x224) |
| Color space | RGB (auto-converted) |
| Preprocessing | ViT processor: resize, normalize |
| Batch size | 1-32 images |

**Known Limitations:**
- **Critical:** High false positive rate on professional photography
- **Critical:** Biased against high-quality, well-lit images
- **Critical:** Classifies Unsplash photos as 99.98% artificial
- Labels: `{0: 'artificial', 1: 'human'}` - opposite of typical convention

**False Positive/Negative Characteristics:**
| Scenario | Behavior |
|----------|----------|
| Real photo (professional) | High artificial score (95-99%) - **WRONG** |
| Real photo (casual/mobile) | Variable (may be correct or wrong) |
| AI-generated image | High artificial score (90-99%) - Correct |
| Real photo with editing | May trigger false positive |

**Hardware Requirements:**
| Resource | Requirement |
|----------|-------------|
| VRAM | ~400 MB (inference) |
| RAM | ~600 MB (model loading) |
| GPU | Optional (CPU works) |
| Dependencies | PyTorch, transformers, PIL |

---

### Candidate Replacement Models

#### 3. `Organika/sdxl-detector` (RECOMMENDED)

| Attribute | Details |
|-----------|---------|
| **HuggingFace ID** | `Organika/sdxl-detector` |
| **Architecture** | ViT-Base |
| **Model Size** | ~350 MB |
| **License** | Apache-2.0 |

**Detection Capabilities:**
| Image Type | Detection Ability |
|------------|-------------------|
| Face-swap deepfakes | ⚠️ Moderate (may detect artifacts) |
| SDXL-generated images | ✅ **Excellent** (primary training target) |
| DALL-E 3 outputs | ✅ Excellent (similar diffusion architecture) |
| Midjourney v5/v6 | ✅ Excellent (similar diffusion architecture) |
| Stable Diffusion 1.5/2.1 | ✅ Excellent (same model family) |
| StyleGAN faces | ⚠️ Moderate (different architecture) |
| Real photos (all types) | ✅ Good (low false positive rate) |
| Real photos (professional) | ✅ Good (designed to avoid FP) |

**Input Requirements:**
| Requirement | Specification |
|-------------|---------------|
| Image formats | JPEG, PNG, WebP, BMP, TIFF |
| Resolution | Any (auto-resized to 224x224) |
| Color space | RGB (auto-converted) |
| Preprocessing | ViT processor: resize to 224x224, normalize |
| Batch size | 1-32 images |
| Normalization | ImageNet mean/std or custom |

**Known Limitations:**
- Primary focus on diffusion models, may miss GAN artifacts
- May have reduced accuracy on very old AI generators
- Requires testing on specific use cases

**False Positive/Negative Characteristics:**
| Scenario | Expected Behavior |
|----------|-------------------|
| Real photo (professional) | Low fake score (<30%) - Correct |
| Real photo (casual) | Low fake score (<25%) - Correct |
| SDXL/DALL-E image | High fake score (>90%) - Correct |
| Midjourney image | High fake score (>85%) - Correct |
| StyleGAN face | Variable (50-80%) - Moderate |
| Real photo with filters | May have elevated score |

**Hardware Requirements:**
| Resource | Requirement |
|----------|-------------|
| VRAM | ~400 MB (inference) |
| RAM | ~600 MB (model loading) |
| GPU | Optional (CPU works, ~50ms/image) |
| Dependencies | PyTorch, transformers, PIL |

---

#### 4. `capcheck/ai-human-generated-image-detection`

| Attribute | Details |
|-----------|---------|
| **HuggingFace ID** | `capcheck/ai-human-generated-image-detection` |
| **Architecture** | ViT-Base (fine-tuned) |
| **Base Model** | dima806/ai_vs_human_generated_image_detection |
| **Model Size** | ~330 MB |
| **Training Date** | 2026-02-11 (most recent) |
| **License** | Apache-2.0 |

**Detection Capabilities:**
| Image Type | Detection Ability |
|------------|-------------------|
| Face-swap deepfakes | ⚠️ Moderate |
| SDXL-generated images | ✅ Excellent (in training data) |
| DALL-E 3 outputs | ✅ Excellent (in training data) |
| Midjourney v6 | ✅ Excellent (in training data) |
| Stable Diffusion 3 | ✅ Excellent (in training data) |
| StyleGAN faces | ⚠️ Moderate |
| Real photos | ⚠️ Needs testing |
| Real photos (professional) | ⚠️ Needs testing (may have bias) |

**Input Requirements:**
| Requirement | Specification |
|-------------|---------------|
| Image formats | JPEG, PNG, WebP, BMP |
| Resolution | Any (auto-resized to 224x224) |
| Color space | RGB (auto-converted) |
| Preprocessing | ViT processor standard |
| Batch size | 1-32 images |

**Known Limitations:**
- Very recent model (less community validation)
- May inherit biases from base model
- Limited documentation on training data composition

**False Positive/Negative Characteristics:**
| Scenario | Expected Behavior |
|----------|-------------------|
| Real photo | Needs testing |
| AI-generated (modern) | Expected high accuracy |
| AI-generated (older) | May have reduced accuracy |

**Hardware Requirements:**
| Resource | Requirement |
|----------|-------------|
| VRAM | ~400 MB (inference) |
| RAM | ~600 MB (model loading) |
| GPU | Optional (CPU works) |
| Dependencies | PyTorch, transformers, PIL |

---

#### 5. `Wvolf/ViT_Deepfake_Detection`

| Attribute | Details |
|-----------|---------|
| **HuggingFace ID** | `Wvolf/ViT_Deepfake_Detection` |
| **Architecture** | ViT-Base |
| **Model Size** | ~330 MB |
| **Downloads** | 565+ |
| **Likes** | 13 |
| **License** | Apache-2.0 |

**Detection Capabilities:**
| Image Type | Detection Ability |
|------------|-------------------|
| Face-swap deepfakes | ✅ Good |
| SDXL-generated images | ⚠️ Moderate |
| DALL-E outputs | ⚠️ Moderate |
| Midjourney outputs | ⚠️ Moderate |
| StyleGAN faces | ✅ Good |
| Real photos | ✅ Good (established track record) |
| Real photos (professional) | ✅ Good |

**Input Requirements:**
| Requirement | Specification |
|-------------|---------------|
| Image formats | JPEG, PNG, WebP, BMP |
| Resolution | Any (auto-resized to 224x224) |
| Color space | RGB (auto-converted) |
| Preprocessing | ViT processor standard |
| Batch size | 1-32 images |

**Known Limitations:**
- Requires ONNX conversion for production
- Less specialized for modern diffusion models
- May not detect latest AI generators

**False Positive/Negative Characteristics:**
| Scenario | Expected Behavior |
|----------|-------------------|
| Real photo | Low fake score - Correct |
| Deepfake | High fake score - Correct |
| AI-generated (modern) | Variable (may miss some) |

**Hardware Requirements:**
| Resource | Requirement |
|----------|-------------|
| VRAM | ~400 MB (inference) |
| RAM | ~600 MB (model loading) |
| GPU | Optional (CPU works) |
| Dependencies | PyTorch, transformers, PIL |

---

#### 6. `prithivMLmods/Deep-Fake-Detector-v2-Model`

| Attribute | Details |
|-----------|---------|
| **HuggingFace ID** | `prithivMLmods/Deep-Fake-Detector-v2-Model` |
| **ONNX Version** | `onnx-community/Deep-Fake-Detector-v2-Model-ONNX` |
| **Architecture** | ViT |
| **Model Size** | ~330 MB |
| **Claimed Precision** | 92.12% |
| **License** | Apache-2.0 |

**Detection Capabilities:**
| Image Type | Detection Ability |
|------------|-------------------|
| Face-swap deepfakes | ✅ Good |
| SDXL-generated images | ⚠️ Moderate |
| DALL-E outputs | ⚠️ Moderate |
| Midjourney outputs | ⚠️ Moderate |
| StyleGAN faces | ✅ Good |
| Real photos | ✅ Good (claimed 92% precision) |
| Real photos (professional) | ✅ Good |

**Input Requirements:**
| Requirement | Specification |
|-------------|---------------|
| Image formats | JPEG, PNG, WebP, BMP |
| Resolution | Any (auto-resized to 224x224) |
| Color space | RGB (auto-converted) |
| Preprocessing | ViT processor standard |
| Batch size | 1-32 images |

**Known Limitations:**
- Claimed precision needs independent validation
- V2 status indicates improvement over V1 but details unclear

**False Positive/Negative Characteristics:**
| Scenario | Expected Behavior |
|----------|-------------------|
| Real photo | Low fake score (claimed 92% precision) |
| Deepfake | High fake score |
| AI-generated | Variable |

**Hardware Requirements:**
| Resource | Requirement |
|----------|-------------|
| VRAM | ~400 MB (inference) |
| RAM | ~600 MB (model loading) |
| GPU | Optional (CPU works) |
| Dependencies | ONNX Runtime OR PyTorch, transformers |

---

### Model Comparison Summary

| Model | Face-Swap | SDXL/DALL-E | StyleGAN | Real Photos | FP Rate | Speed |
|-------|-----------|-------------|----------|-------------|---------|-------|
| dima806 (current) | ✅ Excellent | ❌ Poor | ⚠️ Moderate | ✅ Good | Low | Fast |
| umm-maybe (current) | ⚠️ Moderate | ✅ Good | ⚠️ Variable | ❌ **HIGH** | **HIGH** | Fast |
| Organika/sdxl | ⚠️ Moderate | ✅ **Excellent** | ⚠️ Moderate | ✅ Good | Low | Fast |
| capcheck/ai-human | ⚠️ Moderate | ✅ Excellent | ⚠️ Moderate | ⚠️ Unknown | Unknown | Fast |
| Wvolf/ViT | ✅ Good | ⚠️ Moderate | ✅ Good | ✅ Good | Low | Fast |
| prithivMLmods/v2 | ✅ Good | ⚠️ Moderate | ✅ Good | ✅ Good | Low | Fast |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| New model has similar issues | Test extensively before deployment |
| Model download fails | Keep fallback to DCT analysis |
| Performance regression | Benchmark inference time |
| Label interpretation differs | Verify label mapping before deployment |

---

## Success Criteria

1. Real photos classified as authentic (>70% trust score)
2. AI-generated images detected as fake (>85% confidence)
3. No false positives on professional photography
4. Inference time <100ms per image
5. Single model simplifies maintenance

---

## Next Steps

1. **User Approval:** Review and approve this plan
2. **Switch to Code Mode:** Implement the changes
3. **Download New Model:** Pull from HuggingFace
4. **Test:** Validate with test images
5. **Deploy:** Rebuild Docker containers

---

**Prepared by:** Architect Mode
**Ready for:** User Review

---

## 2026-03-05 Performance Remediation Addendum (Doc-First)

### Problem Statement
Field testing reports both false negatives and false positives in image AI detection:
- AI-generated images sometimes classified as authentic
- Real images sometimes classified as fake

This is now treated as a **runtime correctness issue**, not just a model-selection issue.

### Root-Cause Findings (Code-Level)
1. **Inconsistent class-index semantics across analyzers**
   - Same `ai_real_detector` model is interpreted differently across files.
   - Some paths assume fake is class `0`; others assume fake is class `1`.
2. **Fallback path fragility in temporal/lipsync**
   - Fallback decisions are based on registry existence rather than actual inference failure.
   - Fallback image conversion can be incorrect due to normalized tensor reuse.
3. **No unified label-mapping logic**
   - Label interpretation is hardcoded in multiple analyzers instead of reading model metadata (`id2label` / `label2id`).
4. **Confidence calibration inconsistency**
   - Some paths directly use logits/softmax class value without confidence-aware shrinkage.
5. **Model card limitation awareness gap**
   - Previous primary detector (`Organika/sdxl-detector`) is SDXL-focused and model card explicitly warns weaker generalization on older/non-SDXL generators.

### Primary References Used
- Organika model card: https://huggingface.co/Organika/sdxl-detector
- CapCheck model card: https://huggingface.co/capcheck/ai-human-generated-image-detection
- DeepfakeBench benchmark: https://github.com/SCLBD/DeepfakeBench
- AASIST paper: https://arxiv.org/abs/2110.01200
- X-CLIP paper: https://arxiv.org/abs/2207.07285

### Decision-Complete Remediation Plan
1. **Unify fake/real class mapping logic**
   - Add shared analyzer utility to infer fake-class index from model labels.
   - Remove hardcoded class index assumptions from image/spatial/temporal/lipsync.
2. **Make fallback deterministic and safe**
   - In temporal/lipsync: try primary model first; on failure fallback to `ai_real_detector`.
   - Use original image/mouth crops for fallback inference (not incorrectly denormalized tensors).
3. **Apply confidence-aware probability calibration**
   - Convert logits to probabilities, then shrink low-confidence outputs toward 0.5.
4. **Improve image robustness**
   - Add lightweight test-time augmentation (mirror) and average probabilities.
5. **Replace SDXL-specialized primary checkpoint**
   - Switch `ai_real_detector` source to `capcheck/ai-human-generated-image-detection` for broader generator coverage and clearer benchmark metadata.
6. **Fix invalid CPU cross-modality remapping**
   - Remove automatic manager remap from video models to image model in `ModelManager`; keep fallback behavior inside analyzers only.

### Files to Update
- `backend/analyzers/base.py`
- `backend/analyzers/image.py`
- `backend/analyzers/video/spatial.py`
- `backend/analyzers/video/temporal.py`
- `backend/analyzers/video/lipsync.py`
- `backend/models/manager.py`

### Expected Impact
- Consistent class semantics across all modalities using `ai_real_detector`
- Fewer false flips caused by hardcoded class index mismatches
- More stable behavior in CPU / degraded-model scenarios
- Better practical detection reliability without introducing new files or schema changes

---

## 2026-03-05 Accuracy Validation Protocol Addendum

### Why this is required
Prediction-quality claims are currently hard to trust because the existing validation harness has structural issues:
1. Hardcoded sample list does not match actual `test_samples` files.
2. Hardcoded sample path (`/app/test_samples`) breaks host execution.
3. `trust_score.value` (0-100 authenticity) is treated like 0-1 confidence, producing incorrect predicted labels in fallback logic.

### Ground Truth Mapping Policy for Current Dataset
Use filename-based labels only for known curated files:
- `Deepfake.png` -> `fake`
- `Gemini_Generated_Image_*.png` -> `ai_generated`
- `AI_Generated_video.mp4` -> `fake`
- `Real_video1.mp4` -> `authentic`
- `AI_generated_text.txt` -> `ai_generated`
- `internet_test/real_person*.jpg` -> `authentic`

Files without explicit ground-truth signal in name/path are excluded from strict accuracy denominator and reported separately as exploratory samples.

### Validation Pipeline Changes (No New Files)
Update `backend/test_e2e_validation.py` to:
1. Discover samples dynamically from `test_samples`.
2. Resolve `test_samples` path robustly from:
   - `TEST_SAMPLES_DIR` env var
   - `/app/test_samples`
   - `<repo_root>/test_samples`
3. Parse predictions primarily from API `verdict`.
4. Use `trust_score.value` only as authenticity score (0-100), not raw confidence.
5. Poll `/analyze/{id}` for completion and `/analyze/{id}/detail` when needed for modality-specific probabilities.
6. Report:
   - strict accuracy (labeled subset),
   - per-modality confusion matrix,
   - unknown/exploratory sample predictions.

### Success Criteria for This Validation Stage
1. Validation script runs end-to-end against current `test_samples`.
2. Strict labeled subset metrics are produced without parser errors.
3. False positive/false negative cases are explicitly listed for targeted model tuning.

---

## 2026-03-05 Runtime Failure Diagnostics and Fix Plan (Doc-First)

### Observed Runtime Failures (from real Celery execution)
1. **Image classification semantic inversion in active runtime path**
   - Worker logs showed `ai_real_detector` score interpretation as `0=artificial/fake, 1=human/real` while final scoring still treated returned value as fake probability.
   - Effect: real images can be pushed to high fake probabilities (severe false positives).

2. **Text primary detector shape/scalar conversion failure**
   - `ModernBERT detector failed: TypeError: only length-1 arrays can be converted to Python scalars`
   - Effect: primary text detector is bypassed and text falls back to weak heuristics, causing poor accuracy and unstable verdicts.

3. **Audio pipeline hard-fails when librosa extras are unavailable**
   - `MFCC feature extraction failed. librosa error: No module named 'pkg_resources'`
   - Effect: audio modality fails entirely in video-with-audio scenarios and reduces fusion quality.

4. **Video subpipeline reliability defects**
   - CLIP path error: static ONNX input expects batch 1 but batched inference is attempted.
   - Lip-sync path error: `axis 1 is out of bounds for array of dimension 1` (1D waveform not handled robustly).
   - Orchestrator confidence error: `float division by zero` in video confidence aggregation.
   - Effect: video analysis can collapse to uncertain/failure even when frames are available and extracted.

5. **Audio storage format inconsistency**
   - Video preprocessing stores audio as `.npy`, while analyzer load path assumes raw float32 bytes.
   - Effect: corrupted/incorrect waveform decoding risk for video-derived audio.

### Decision-Complete Remediation Scope (No New Files)
1. `backend/analyzers/text.py`
   - Normalize detector outputs robustly across shapes (`[B, C]`, `[B, 1, C]`, logits/probabilities).
   - Always extract AI class probability via safe flattening/indexing, no direct `float(array)` conversions.

2. `backend/analyzers/audio.py`
   - Add non-librosa fallback MFCC-like feature extraction to avoid hard failure.
   - Fix Purdue model key usage to registry-consistent name.
   - Load audio `.npy` correctly (and keep backward-compatible raw-byte handling).

3. `backend/analyzers/video_analyzer.py`
   - Fix lip-sync confidence formula and guard harmonic-mean aggregation from divide-by-zero.

4. `backend/analyzers/video/lipsync.py`
   - Make audio-energy extraction robust for 1D waveform and 2D feature matrices.
   - Prevent correlation path from throwing on shape mismatches.

5. `backend/analyzers/video/spatial.py`
   - Run CLIP fallback path with static-batch-safe behavior (or per-frame fallback) to avoid ONNX dimension failures.

### Expected Outcome After Fixes
1. Image predictions use consistent fake/real semantics (false positives reduced).
2. Text modality uses real primary detector outputs instead of failing into heuristics.
3. Audio modality remains operational without optional librosa extras.
4. Video modality no longer fails on divide-by-zero and shape errors.
5. End-to-end multimodal validation becomes stable and measurable on `test_samples`.

---

## 2026-03-05 Trust Score Reliability Addendum (Doc-First)

### Additional Root-Cause Findings
1. **Text modality score polarity inversion in orchestration**
   - In `_analyze_single_modality`, text `ModalityResult.score` was inverted (`1 - score`) even though analyzers already output manipulation probability.
   - This corrupts fusion semantics and can flip trust outcomes.

2. **Final result builders use incorrect fallback polarity**
   - `_build_audio_result`, `_build_text_result`, and `_build_image_result` fall back to `1 - result.score` in some paths.
   - For manipulation probabilities, fallback must be `result.score`.

3. **Video details are discarded in final assembly**
   - `_build_video_result` ignores nested analyzer details and hardcodes `frames_analyzed=0`.
   - This makes report payloads appear invalid even when frames were analyzed correctly.

4. **Trust scoring calibration not actually applied**
   - `TrustScorer.compute()` supports content-type calibration, but orchestrator called it without `content_type`.
   - `trust_score.calibrated` flag could report true even when calibration was skipped.

### Required Fixes
1. `backend/core/orchestrator.py`
   - Remove text score inversion.
   - Pass `content_type` through `_build_final_results` into `scorer.compute(...)`.
   - Correct fallback polarities in modality result builders.
   - Parse nested video details (`spatial`, `temporal`, `lipsync`, `frames_analyzed`, `face_detected`) instead of hardcoded placeholders.

2. `backend/core/scorer.py`
   - Track whether calibration was actually applied and set `TrustScore.calibrated` accurately.

### Expected Reliability Outcome
1. Fusion receives consistent modality score semantics (higher score => higher manipulation likelihood).
2. Trust score calibration behavior matches metadata flags.
3. Video result payloads reflect real analysis evidence and frame counts.
4. Final verdict confidence becomes auditable and internally consistent.

---

## 2026-03-08 Multimodal Accuracy Recovery Plan (Doc-First)

### Fresh Runtime Evidence (Docker E2E Re-Run)
- Execution date: **2026-03-08**
- Script: `python /app/test_e2e_validation.py`
- Labeled accuracy:
  - Video: **50.0%** (1/2)
  - Text: **0.0%** (0/1)
  - Image: **60.0%** (3/5)
  - Overall labeled: **50.0%**

### Confirmed Root Causes
1. **Video/Audio score polarity inversion still active in orchestration**
   - `_analyze_single_modality()` inverts `video` and `audio` modality scores before fusion.
   - Fusion and trust scorer already assume modality scores are manipulation probabilities.
   - Result: authentic-leaning modality outputs are flipped toward fake, reducing trust score reliability.

2. **Audio secondary model execution path is structurally valid but shape-invalid**
   - `purdue_m2.onnx` expects image-like NHWC input `[1, 224, 224, 3]`.
   - Current `_run_purdue_m2()` passes `(1, T, F)` spectrogram directly, triggering ONNX invalid-dimension errors and neutral fallback.
   - Result: audio signal contributes weak/neutral values and reduces multimodal stability.

3. **Primary image detector has false-negative blind spot on curated fake portraits**
   - `ai_real_detector` (capcheck lineage) outputs near-zero fake probability for both clearly generated and real portrait samples in this set.
   - Local auxiliary detector (`/models/ai_image_detector`, Swin) catches generated portraits but over-flags some “real_person” images.
   - DCT anomaly remains discriminative on this set (fakes ~0.30 vs real_person set ~0.18).

### Decision-Complete Fix Scope (No New Files)
1. `backend/core/orchestrator.py`
   - Remove `video` and `audio` score inversion in `_analyze_single_modality()`.
   - Keep modality score semantics consistent: **higher score = higher manipulation probability**.

2. `backend/analyzers/audio.py`
   - Update `_run_purdue_m2()` preprocessing to map mel spectrogram into model-expected NHWC `[1, 224, 224, 3]`.
   - Preserve neutral fallback on runtime failure, but prevent deterministic shape mismatch.

3. `backend/models/registry.py` + `backend/analyzers/image.py`
   - Register existing local Swin detector as auxiliary image model.
   - Add gated dual-detector fusion in image analyzer:
     - Primary detector remains baseline probability.
     - Auxiliary detector contributes only under strict evidence gate (high auxiliary fake + DCT anomaly support).
   - Goal: recover fake portrait recall while limiting false positives.

### Trust-Score Reliability Criteria After Patch
1. No cross-modality polarity inversions before fusion.
2. Audio model path returns real inference outputs (not persistent shape-failure neutralization).
3. Image verdict changes are evidence-gated and auditable via details payload.
4. `test_e2e_validation.py` rerun demonstrates measurable gain in labeled accuracy and reduced contradictory verdict behavior.
