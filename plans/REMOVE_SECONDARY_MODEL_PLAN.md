# Plan: Remove Secondary Ensemble Model (deepfake_vit_v2)

## Objective
Remove the secondary ensemble model `onnx-community/Deep-Fake-Detector-v2-Model-ONNX` and use only the primary model `dima806/deepfake_vs_real_image_detection` for accurate deepfake detection.

## Rationale
- The secondary model averages predictions, which can reduce accuracy
- Single model (primary) provides cleaner, more direct predictions
- Primary model (dima806) is SOTA with 50K+ downloads and proven accuracy
- Reduces complexity and potential for conflicting signals

## Files to Modify

### 1. backend/analyzers/image.py
**Current State:**
- Uses ensemble detection with both primary and secondary models
- `_run_ensemble_detection()` combines scores from both models
- `get_required_models()` returns both models

**Changes Required:**
- Remove `_run_ensemble_detection()` method
- Replace with `_run_primary_detection()` using only deepfake_vit_primary
- Update `get_required_models()` to return only primary model
- Simplify `_run_analysis_pipeline()` to use primary model directly
- Remove ensemble metadata tracking

### 2. backend/analyzers/video/spatial.py
**Current State:**
- `get_required_models()` returns `["deepfake_vit_v2", "clip_vit_b16"]`

**Changes Required:**
- Update to use `deepfake_vit_primary` instead of `deepfake_vit_v2`
- Modify `_run_efficientnet()` to use PyTorch model directly

### 3. backend/models/registry.py
**Current State:**
- `DEFAULT_MODELS` contains both `deepfake_vit_primary` and `deepfake_vit_v2`

**Changes Required:**
- Remove `deepfake_vit_v2` from `DEFAULT_MODELS` dictionary

### 4. backend/api/deps.py
**Current State:**
- `critical_models` list includes `deepfake_vit_v2`

**Changes Required:**
- Remove `deepfake_vit_v2` from `critical_models` list

### 5. backend/models/manager.py
**Current State:**
- `warmup()` method loads both models in GPU mode

**Changes Required:**
- Remove `deepfake_vit_v2` from warmup models list

## Implementation Order
1. Update models/registry.py - Remove model definition
2. Update models/manager.py - Update warmup list
3. Update api/deps.py - Update critical models
4. Update analyzers/image.py - Major refactor
5. Update analyzers/video/spatial.py - Update model reference

## Testing
- Verify application starts without errors
- Test image analysis with single model
- Verify video spatial analysis works
- Check health endpoint shows correct loaded models
