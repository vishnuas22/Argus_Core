# Argus Development Guide

## Quick Start
```bash
# Start all services
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# View logs
docker compose logs -f backend celery-worker

# Rebuild backend (after requirements.txt changes)
docker compose build backend
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d backend celery-worker

# Restart celery worker (after code changes without dev volume mount)
docker exec argus-celery-worker sh -c 'kill -9 1'

# Deploy code changes (if not using dev volume mounts)
docker cp backend/analyzers/audio.py argus-celery-worker:/app/analyzers/audio.py
docker exec argus-celery-worker sh -c 'find / -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null'
```

## Architecture
- 8 Docker services: backend, celery-worker, frontend, redis, mongodb, minio, prometheus, grafana
- CPU-only on Apple Silicon (arm64), no NVIDIA GPU
- Analysis runs in celery-worker via `--pool=solo`
- Models downloaded on first use into `/models/` volume

## Known Issues
1. **purdue_m2.onnx is corrupt** — InvalidProtobuf, cannot load. Needs re-download or replacement.
2. **aasist_antispoof unavailable** — HF model ID `dima806/audio_deepfake_detection` is invalid (was changed from `clovaai/aasist-l`). No working AASIST model in registry.
3. **Uncertain verdict** — Single-modality analysis or low confidence (<0.4) routes to `uncertain`. TrustScore ~73 for audio sine wave, ~60 for image analysis.
4. **Video analysis slow on CPU** — DINOv2 model takes ~17s for 6 frames. CLIP/X-CLIP also very slow without GPU.
5. **Backend build uses python:3.11-slim** — Set `BACKEND_BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` for GPU builds.

## Recent Fixes
- `wav2vec2_antispoof.onnx` inference fixed: engine's `_infer_multi_input` had a pre-existing bug where `_get_model_lock()` was not awaited (`core/engine.py:520`), and the synchronous `with model_lock:` caused `TypeError`. Also, the dtype conversion at line 514-515 converted float32 → int64, corrupting inputs. Fixed: proper `await` + `async with` + `run_in_executor`, and `np.issubdtype` guard to leave float types intact.
- `Wav2Vec2AudioDetector` fixed: `SequenceClassifierOutput` doesn't have `.last_hidden_state` in newer transformers. Added `output_hidden_states=True` parameter and fallback to `outputs.logits` / `outputs.hidden_states[-1]`.
- Audio neural pipeline restored: `AudioAnalyzer._run_wav2vec2_antispoof` now passes `{"input_values": batch, "attention_mask": mask}` as dict (int32 dtype) instead of a single array, so the engine routes to the fixed multi-input path. Neural scores are real again — `any_neural_available=True` removes the 80% dampening.
- `frequency_anomaly_score` now properly computed and persisted (was 0.0 due to missing field in `AudioAnalysisDetails` and `_build_audio_result`)
- Audio false positive dampening: `0.5 + (aggregate - 0.5) * 0.2` when no neural models available (now only triggered if ALL neural models fail — no longer the case since wav2vec2 works)
- Frontend 400 Bad Request: removed explicit `Content-Type: multipart/form-data` header breaking FastAPI multipart parsing
- Frontend TypeScript types aligned with backend schemas
- Docker dev workflow: `docker-compose.dev.yml` mounts backend source code for hot-reload

## Useful Commands
```bash
# E2E test
python3 scripts/test_end_to_end.py

# Verify system
python3 scripts/validate_system.py

# Test audio-only with sine wave
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Authorization: Bearer $(curl -s -X POST http://localhost:8000/api/v1/auth/anonymous -H 'Content-Type: application/json' -d '{"display_name":"test"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"access_token\"])')" \
  -F "file=@/tmp/test_sine.wav" -F "generate_report=false"

# Check celery logs
docker logs argus-celery-worker --tail 50

# Clear old analyses from MongoDB
docker exec argus-mongodb mongosh -u argusadmin -p arguspass123 --authenticationDatabase admin argus_core --eval 'db.analyses.deleteMany({})'
```
