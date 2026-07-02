# Argus Core — Complete Docker Deployment & Fine-Tuning Runbook

**Version:** 1.8.4 | **Date:** 2026-06-29 | **Scope:** Production deployment guide

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start (5 Minutes)](#2-quick-start-5-minutes)
3. [Environment Configuration](#3-environment-configuration)
4. [GPU Setup & Verification](#4-gpu-setup--verification)
5. [Building & Starting the Stack](#5-building--starting-the-stack)
6. [Verifying Everything Works](#6-verifying-everything-works)
7. [Fine-Tuning LoRA Adapter Weights](#7-fine-tuning-lora-adapter-weights)
8. [Wiring Fine-Tuned Weights into the Platform](#8-wiring-fine-tuned-weights-into-the-platform)
9. [Using Pre-Trained Public Heads (No Training)](#9-using-pre-trained-public-heads-no-training)
10. [3-Mode Execution System](#10-3-mode-execution-system)
11. [Calibration & Conformal Prediction Setup](#11-calibration--conformal-prediction-setup)
12. [Continuous Learning & A/B Testing](#12-continuous-learning--ab-testing)
13. [Observability (Prometheus + Grafana)](#13-observability-prometheus--grafana)
14. [System Validation](#14-system-validation)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Prerequisites

### Hardware

| Mode | CPU | RAM | GPU | Disk |
|------|-----|-----|-----|------|
| Lite | 4 cores | 4 GB | None | 10 GB |
| Balanced | 8 cores | 8 GB | T4 16GB or A10 24GB | 50 GB |
| Research | 16 cores | 16 GB | A100 40GB+ | 100 GB |

### Software

```bash
# Docker Engine 24+
docker --version

# Docker Compose v2
docker compose version

# NVIDIA Container Toolkit (for GPU mode)
# Ubuntu/Debian:
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

### HuggingFace Token (optional but recommended)

Some models require HF authentication. Create a token at
https://huggingface.co/settings/tokens:

```bash
export HUGGINGFACE_TOKEN="hf_your_token_here"
```

---

## 2. Quick Start (5 Minutes)

```bash
# 1. Unzip the project
unzip Argus_Core_Iteration9.8.zip
cd Argus_Core-main

# 2. Create .env from template
cp .env.example .env

# 3. Edit .env — set ALL mandatory values (see Section 3)
nano .env

# 4. Build and start
docker compose up -d --build

# 5. Wait for models to download (~4 minutes first time)
docker compose logs -f backend | grep -m 1 "ARGUS CORE - Ready"

# 6. Verify
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed | python3 -m json.tool
```

Access points:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3030 (admin / your GRAFANA_ADMIN_PASSWORD)
- **MinIO Console:** http://localhost:9000 (your MINIO_ROOT_USER / MINIO_ROOT_PASSWORD)

---

## 3. Environment Configuration

Edit `.env` — **every variable marked `must be set` is mandatory.**
The stack will refuse to start without them.

### Mandatory Settings

```bash
# ===== Environment =====
ENVIRONMENT=production
LOG_LEVEL=INFO

# ===== MongoDB (auth mandatory) =====
MONGO_USER=argusadmin
MONGO_PASSWORD=<generate-a-32-char-random-string>

# ===== Redis (password mandatory) =====
REDIS_PASSWORD=<generate-a-32-char-random-string>

# ===== MinIO =====
MINIO_ROOT_USER=argusadmin
MINIO_ROOT_PASSWORD=<generate-a-32-char-random-string>

# ===== Security =====
# Use: openssl rand -hex 32
JWT_SECRET=<64-hex-char-string>
API_KEY_SALT=<32-char-random-string>

# ===== CORS (MUST be explicit in production) =====
CORS_ORIGINS=https://your-domain.com,http://localhost:3000

# ===== Grafana =====
GRAFANA_ADMIN_PASSWORD=<strong-password>
```

### GPU Settings

```bash
# For GPU mode (default — auto-detects):
EXECUTION_MODE=balanced
GPU_PROFILE=t4          # cpu | rtx3050 | t4 | a10 | a100
GPU_MEMORY_LIMIT_MB=14000
GPU_COUNT=1

# For CPU-only mode:
EXECUTION_MODE=lite
BACKEND_BASE_IMAGE=python:3.11-slim
USE_GPU=false
```

### SOTA Detector Settings

```bash
# Enable SOTA detectors (CLIP, DINOv2, SigLIP, AASIST3, Wav2Vec2-XLS-R, VideoMAE, etc.)
ENABLE_SOTA_DETECTORS=true

# Optional: use pre-trained public heads for instant benchmark numbers
ARGUS_CLIP_FINE_TUNED_HEAD=dima806/deepfake_detection_model_image
ARGUS_WAV2VEC2_FINE_TUNED_HEAD=MelodyMachine/Deepfake-audio-detection-V2

# Optional: HF token for private/gated models
HUGGINGFACE_TOKEN=hf_your_token_here
```

### Full .env Template

See `.env.example` in the project root for all 40+ configuration options.

---

## 4. GPU Setup & Verification

### Step 1: Verify Host GPU

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2    |
| GPU  Name        Persistence-M| Bus-Id          Display.A  Volatile.Uncorr.|
|   0  Tesla T4    On            | 00000000:00:1E.0  Off     0               |
| 30%  45C    P8    16W / 70W   |  1582MiB / 15360MiB  10%   Default      |
+-----------------------------------------------------------------------------+
```

### Step 2: Verify Docker GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

If this fails, reinstall NVIDIA Container Toolkit (see Section 1).

### Step 3: Verify Argus Backend Sees the GPU

```bash
# Start the stack
docker compose up -d

# Check backend logs for GPU detection
docker compose logs backend | grep -i "GPU\|CUDA\|device"
```

Expected:
```
GPU detected: Tesla T4 (15360MB)
ModeManager initialized: mode=balanced, device=cuda:0, precision=fp16
```

### Step 4: Check GPU Utilization During Inference

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Submit an analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test.jpg" \
  -F "modalities=image"
```

GPU memory should spike during inference.

### Step 5: Multi-GPU (Optional)

If you have multiple GPUs:

```bash
# In .env:
GPU_COUNT=2
ARGUS_PREFER_MULTI_GPU=true

# Docker Compose will allocate 2 GPUs
```

Verify sharding:
```bash
curl http://localhost:8000/health/detailed | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(json.dumps(d.get('subsystems', {}).get('models', {}), indent=2))
"
```

---

## 5. Building & Starting the Stack

### Step 1: Build Images

```bash
# Build all services (backend, celery-worker, frontend)
docker compose build

# For CPU-only builds (no CUDA):
BACKEND_BASE_IMAGE=python:3.11-slim docker compose build

# Build with no cache (if you changed Dockerfile or requirements)
docker compose build --no-cache
```

Build time: ~10-15 minutes first time, ~2-3 minutes with cache.

### Step 2: Start Services

```bash
# Start all services
docker compose up -d

# Watch backend startup (models download on first start)
docker compose logs -f backend
```

First start sequence:
1. MongoDB + Redis + MinIO start (~5 seconds)
2. Backend starts, downloads models from HuggingFace (~3-4 minutes)
3. Celery worker starts
4. Frontend starts
5. Prometheus + Grafana start

### Step 3: Verify All Services Are Healthy

```bash
docker compose ps
```

Expected:
```
NAME                 STATUS                   PORTS
argus-backend        Up (healthy)             0.0.0.0:8000->8000/tcp
argus-celery-worker  Up (healthy)
argus-frontend       Up (healthy)             0.0.0.0:3000->3000/tcp
argus-grafana        Up                       0.0.0.0:3030->3000/tcp
argus-minio          Up (healthy)             0.0.0.0:9000->9000/tcp
argus-mongodb        Up (healthy)
argus-prometheus     Up                       0.0.0.0:9090->9090/tcp
argus-redis          Up (healthy)
```

### Step 4: Start Celery Beat (for automatic retraining + drift checks)

```bash
# In a separate terminal:
docker compose exec celery-worker celery -A core.orchestrator.celery_app beat --loglevel=info &
```

This enables:
- Daily LoRA retraining at 02:00/03:00/04:00 UTC
- Drift checks every 6 hours
- A/B test evaluation every hour

---

## 6. Verifying Everything Works

### Step 1: Basic Health Check

```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### Step 2: Detailed Health Check

```bash
curl http://localhost:8000/health/detailed | python3 -m json.tool
```

Verify:
- `subsystems.models` shows all 9 SOTA detectors with `path_exists: true`
- `subsystems.defenses` shows `rps_enabled: true`
- `subsystems.continuous_learning` shows `enabled: true`
- `subsystems.calibration` shows calibration status

### Step 3: Prometheus Metrics

```bash
curl http://localhost:8000/metrics | grep argus_
```

Verify these metrics exist:
- `argus_inference_total`
- `argus_drift_score`
- `argus_calibration_ece`
- `argus_adversarial_flagged_total`
- `argus_feedback_buffer_size`

### Step 4: Submit a Test Analysis

```bash
# Get auth token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/anonymous | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

# Upload a file
FILE_ID=$(curl -s -X POST http://localhost:8000/api/v1/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test.jpg" | python3 -c "import json,sys; print(json.load(sys.stdin).get('file_id',''))")

# Start analysis
ANALYSIS_ID=$(curl -s -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"file_id\":\"$FILE_ID\",\"modalities\":[\"image\"]}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('analysis_id',''))")

# Poll for results
curl http://localhost:8000/api/v1/analyze/$ANALYSIS_ID \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

### Step 5: Run Full System Validation

```bash
# Run all validation suites
docker compose exec backend python /app/../scripts/validate_system.py \
  --suite all --api-url http://localhost:8000

# View the report
cat /tmp/argus_validation_report.md
```

### Step 6: Check Grafana Dashboard

Open http://localhost:3030 → login with your GRAFANA_ADMIN_PASSWORD →
the "Argus Core — Platform Observability" dashboard should be
auto-provisioned with live data.

### Step 7: CPU-Only Verification

```bash
# Verify the platform runs without GPU
EXECUTION_MODE=lite docker compose exec backend \
  python ../scripts/verify_cpu_only.py
```

### Step 8: Reproducibility Verification

```bash
docker compose exec backend \
  python ../scripts/verify_reproducibility.py --runs 5 --tolerance 1e-4
```

---

## 7. Fine-Tuning LoRA Adapter Weights

### Overview

The platform ships with **frozen SOTA backbones** (CLIP, DINOv2, SigLIP,
Wav2Vec2-XLS-R, VideoMAE, etc.) loaded from HuggingFace. To achieve real
benchmark accuracy, you train **LoRA adapters** on your deepfake datasets.

### Step 1: Prepare Datasets

#### Image (FaceForensics++ or Celeb-DF v2)

```bash
# Download Celeb-DF v2 (free for research)
python scripts/dataset_download.py --dataset celebdf_v2 --output /data

# Organize as:
# /data/celebdf/real/train/*.jpg
# /data/celebdf/fake/train/*.jpg
# /data/celebdf/real/val/*.jpg
# /data/celebdf/fake/val/*.jpg
```

#### Audio (ASVspoof 2019 LA)

```bash
# Download from https://www.asvspoof.org/index2019.html (EULA required)
python scripts/dataset_download.py --dataset asvspoof2019 --output /data

# Organize as:
# /data/asvspoof2019/LA/train/flac/*.flac
# /data/asvspoof2019/LA/train/train.txt
# /data/asvspoof2019/LA/eval/flac/*.flac
# /data/asvspoof2019/LA/eval/ASVspoof2019.LA.evalcm.txt
```

#### Video (FaceForensics++)

```bash
python scripts/dataset_download.py --dataset faceforensics --output /data

# Extract 16 frames per video at 1 fps:
for f in /data/faceforensics/train/real/*.mp4; do
  out_dir="/data/faceforensics/train/real_frames/$(basename ${f%.mp4})"
  mkdir -p "$out_dir"
  ffmpeg -i "$f" -vf fps=1 "$out_dir/frame_%03d.png"
done
```

### Step 2: Train Image LoRA Adapter (CLIP + LoRA)

```bash
cd backend

python ../scripts/train_lora_adapters.py \
    --modality image \
    --backbone clip \
    --dataset faceforensics \
    --dataset-root /data/celebdf \
    --output-dir /models/clip_lora_image_adapter \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --lora-r 16 \
    --lora-alpha 32
```

**Expected:** ~3 hours on T4, ~1 hour on A10.
**Output:** `/models/clip_lora_image_adapter/` containing:
- `adapter_config.json`
- `adapter_model.safetensors`
- `classifier.pt`
- `classifier_best.pt`
- `training_metrics.json`

### Step 3: Train Image LoRA Adapter (DINOv2 + MAC head)

```bash
python ../scripts/train_lora_adapters.py \
    --modality image \
    --backbone dinov2 \
    --dataset faceforensics \
    --dataset-root /data/celebdf \
    --output-dir /models/dinov2_image_adapter \
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-4
```

Rename the classifier for the DINOv2 detector:
```bash
cp /models/dinov2_image_adapter/classifier_best.pt /models/dinov2_image_adapter/mac_head.pt
```

### Step 4: Train Audio LoRA Adapter (Wav2Vec2-XLS-R + MoE-LoRA)

```bash
python ../scripts/train_lora_adapters.py \
    --modality audio \
    --backbone wav2vec2_xls_r \
    --dataset asvspoof2019 \
    --dataset-root /data/asvspoof2019 \
    --output-dir /models/wav2vec2_xls_r_moe_lora \
    --epochs 20 \
    --batch-size 8 \
    --lr 5e-5 \
    --lora-r 16
```

**Expected:** ~8 hours on A10.

### Step 5: Train Video LoRA Adapter (VideoMAE)

```bash
python ../scripts/train_lora_adapters.py \
    --modality video \
    --backbone videomae \
    --dataset faceforensics \
    --dataset-root /data/faceforensics \
    --output-dir /models/videomae_finetune \
    --epochs 15 \
    --batch-size 4 \
    --lr 1e-4
```

**Expected:** ~12 hours on A10.

### Step 6: Build ECAPA-TDNN Reference Centroid

The ECAPA-TDNN audio detector needs a reference centroid of real-audio
embeddings:

```bash
python3 -c "
import asyncio, sys, numpy as np
sys.path.insert(0, 'backend')
from detectors import ECAPATDNNAudioDetector

async def main():
    det = ECAPATDNNAudioDetector()
    embeddings = []
    # Feed ~100 real audio files
    for wav_path in real_audio_paths:
        import librosa
        wav, _ = librosa.load(wav_path, sr=16000)
        emb = await det.embed(wav, sample_rate=16000)
        embeddings.append(emb)
    det.build_reference_centroid(np.array(embeddings))

asyncio.run(main())
"
```

### Step 7: Watermark the Trained Adapters (IP Protection)

```bash
python3 -c "
import sys; sys.path.insert(0, 'backend')
from security import get_default_watermarker

# Watermark each adapter
for adapter_dir in [
    '/models/clip_lora_image_adapter',
    '/models/dinov2_image_adapter',
    '/models/wav2vec2_xls_r_moe_lora',
    '/models/videomae_finetune',
]:
    result = get_default_watermarker().embed_in_lora_adapter(adapter_dir)
    print(f'{adapter_dir}: {result.message}')

# Verify later:
result = get_default_watermarker().verify_lora_adapter('/models/clip_lora_image_adapter')
print(f'Verification: BER={result.ber:.4f}')
"
```

---

## 8. Wiring Fine-Tuned Weights into the Platform

### Step 1: Place Adapter Files

The Docker Compose mounts `backend_models:/models` — place your trained
adapters there:

```bash
# Copy trained adapters into the Docker volume
docker compose cp /local/models/clip_lora_image_adapter backend:/models/
docker compose cp /local/models/dinov2_image_adapter backend:/models/
docker compose cp /local/models/wav2vec2_xls_r_moe_lora backend:/models/
docker compose cp /local/models/videomae_finetune backend:/models/
docker compose cp /local/models/ecapa_reference_centroid.npy backend:/models/
```

### Step 2: Restart Backend

```bash
docker compose restart backend celery-worker
```

### Step 3: Verify Adapters Are Loaded

```bash
docker compose logs backend | grep -i "adapter loaded"
```

Expected:
```
CLIP LoRA adapter loaded from /models/clip_lora_image_adapter
DINOv2 MAC head loaded from /models/dinov2_image_adapter/mac_head.pt
Wav2Vec2-XLS-R MoE-LoRA adapter loaded from /models/wav2vec2_xls_r_moe_lora
VideoMAE classifier head loaded from /models/videomae_finetune/classifier.pt
ECAPA reference centroid loaded from /models/ecapa_reference_centroid.npy
```

### Step 4: Run Benchmark to Verify Accuracy

```bash
# Image benchmark on Celeb-DF v2 test set
docker compose exec backend python ../scripts/benchmark_sota.py \
    --modality image \
    --test-set celebdf_v2 \
    --test-root /data/Celeb-DF_v2/Test \
    --output /tmp/bench_image.json

cat /tmp/bench_image.json | python3 -m json.tool
```

Expected with trained LoRA:
```json
{
  "per_detector": {
    "clip_lora": {"auc": 0.96, "eer": 0.08},
    "dinov2": {"auc": 0.93, "eer": 0.12}
  },
  "ensemble": {
    "diversity_ensemble": {"auc": 0.97, "eer": 0.06}
  }
}
```

---

## 9. Using Pre-Trained Public Heads (No Training)

If you don't have GPU time for training, use verified public models
as drop-in heads:

### Image

```bash
# In .env:
ARGUS_CLIP_FINE_TUNED_HEAD=dima806/deepfake_detection_model_image
# Or:
ARGUS_SIGLIP_FINE_TUNED_HEAD=dima806/ai_vs_real_image_detection
```

### Audio

```bash
# In .env:
ARGUS_WAV2VEC2_FINE_TUNED_HEAD=MelodyMachine/Deepfake-audio-detection-V2
```

### Apply

```bash
docker compose down
docker compose up -d
```

The detectors will load the public fine-tuned head as their primary
classifier. No training needed — you get ~0.95 AUC immediately.

---

## 10. 3-Mode Execution System

Switch modes with a single env var — **no code changes**:

### Lite Mode (CPU-only, any laptop)

```bash
# In .env:
EXECUTION_MODE=lite
BACKEND_BASE_IMAGE=python:3.11-slim
USE_GPU=false

docker compose down
docker compose up -d --build
```

Characteristics:
- CPU only, INT8 quantized
- SOTA detectors disabled (legacy ONNX only)
- Eigen-CAM XAI (no AttnLRP)
- Target: <2s per image

### Balanced Mode (GPU if available)

```bash
EXECUTION_MODE=balanced
docker compose up -d
```

Characteristics:
- GPU with FP16, CPU fallback with FP32
- All SOTA detectors enabled
- RPS defense enabled
- Target: <500ms per image on T4

### Research Mode (GPU required, maximum accuracy)

```bash
EXECUTION_MODE=research
docker compose up -d
```

Characteristics:
- GPU required (degrades to Balanced CPU if unavailable)
- FP16 mixed precision, batch=16
- All defenses: RPS + Adversarial Gate + RS-lite + Certified Robustness
- Full XAI: AttnLRP + Eigen-CAM + audio STFT + temporal occlusion
- Target: maximum accuracy

### Verify Current Mode

```bash
curl http://localhost:8000/health/detailed | python3 -c "
import json, sys
d = json.load(sys.stdin)
cl = d.get('subsystems', {}).get('continuous_learning', {})
print(f'Continuous learning: {cl}')
"
```

---

## 11. Calibration & Conformal Prediction Setup

### Step 1: Prepare Calibration Set

Create a JSON file with ~2000 held-out samples:

```json
{
  "logits": [[2.1, -1.3], [1.5, 0.5], ...],
  "labels": [0, 0, 1, 1, ...],
  "embeddings": [[...], ...]
}
```

### Step 2: Fit Temperature Scaling + Conformal RAPS

```bash
docker compose exec backend python ../scripts/fit_calibration.py \
    --modality image \
    --calibration-json /data/calibration_image.json \
    --output-dir /models/calibration
```

Output:
- `/models/calibration/temperature_scaler.json` (T value)
- `/models/calibration/conformal_raps.json` (q_hat threshold)
- `/models/calibration/audit_report.json` (ECE before/after)
- `/models/calibration/drift_reference.{json,npz}` (for drift detection)

### Step 3: Verify Calibration

```bash
cat /models/calibration/audit_report.json | python3 -m json.tool
```

Expected:
```json
{
  "before_ts": {"ece_15": 0.12, "brier_score": 0.18},
  "after_ts": {"ece_15": 0.025, "brier_score": 0.08},
  "temperature": 1.35
}
```

ECE should drop from ~12% to <3% after temperature scaling.

---

## 12. Continuous Learning & A/B Testing

### Step 1: Submit Feedback

After each analysis, submit the ground-truth label:

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "modality": "image",
    "input_hash": "sha256-of-input-bytes",
    "label": 1,
    "predicted_score": 0.85,
    "confidence": 0.92,
    "model_version": "argus-1.8.4"
  }'
```

### Step 2: Check Feedback Buffer

```bash
curl http://localhost:8000/api/v1/feedback/stats \
  -H "Authorization: Bearer $TOKEN"
```

### Step 3: Trigger Retrain (when buffer reaches 50+ samples)

```bash
# Requires admin role
curl -X POST http://localhost:8000/api/v1/retrain/image \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Step 4: Monitor A/B Test

```bash
curl http://localhost:8000/api/v1/ab_test/image \
  -H "Authorization: Bearer $TOKEN"
```

The candidate adapter gets 10% of traffic. If accuracy > 0.85 and
AUC > 0.9, it's auto-promoted. If accuracy < 0.7, it's rolled back.

---

## 13. Observability (Prometheus + Grafana)

### Access Grafana

```
http://localhost:3030
Username: admin
Password: your GRAFANA_ADMIN_PASSWORD
```

The dashboard "Argus Core — Platform Observability" is auto-provisioned
with 11 panels:
1. Inference Rate (per modality)
2. Inference Latency p50/p95/p99
3. Drift Score (PSI + MMD)
4. Drift Severity
5. Calibration ECE
6. Retrain Cycles (24h)
7. A/B Test Accuracy
8. Feedback Buffer Size
9. Conformal Route-to-Human (1h)
10. Adversarial Flags (1h)
11. Certified Robustness Radius (p50)

### Prometheus Raw Metrics

```
http://localhost:9090
```

Query example:
```promql
rate(argus_inference_total[5m])
```

---

## 14. System Validation

### Run Full Validation Suite

```bash
docker compose exec backend python ../scripts/validate_system.py \
  --suite all \
  --api-url http://localhost:8000 \
  --output-json /tmp/validation.json \
  --output-md /tmp/validation.md

cat /tmp/validation.md
```

### Individual Suites

```bash
# End-to-end user flow (28 stages)
python scripts/validate_system.py --suite e2e

# Endpoint validation (12 endpoints × 10 scenarios)
python scripts/validate_system.py --suite endpoints

# Failure simulation (12 chaos scenarios)
python scripts/validate_system.py --suite failures

# Regression (11 metrics)
python scripts/validate_system.py --suite regression

# Unit tests
python scripts/validate_system.py --suite unit
```

### CPU-Only Verification

```bash
EXECUTION_MODE=lite CUDA_VISIBLE_DEVICES="" \
  python scripts/verify_cpu_only.py
```

### Reproducibility Verification

```bash
python scripts/verify_reproducibility.py --runs 5 --tolerance 1e-4
```

### Adversarial Robustness Benchmark

```bash
python scripts/benchmark_adversarial.py \
    --test-set celebdf_v2 \
    --test-root /data/Celeb-DF_v2/Test \
    --output /tmp/bench_adv.json \
    --epsilon 0.031 \
    --pgd-steps 20
```

---

## 15. Troubleshooting

### Backend won't start

```bash
# Check logs
docker compose logs backend

# Common issues:
# 1. Missing .env variables → "JWT_SECRET must be set"
# 2. MongoDB not ready → wait 30 seconds
# 3. Model download failed → check HUGGINGFACE_TOKEN
```

### GPU not detected

```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi

# If fails:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# In .env, verify:
EXECUTION_MODE=balanced  # not "lite"
USE_GPU=true
```

### Model download fails

```bash
# Check HF token
docker compose exec backend python -c "from huggingface_hub import whoami; print(whoami())"

# Manual download
docker compose exec backend python -c "
from models.downloader import pull_sota_snapshot
import asyncio
asyncio.run(pull_sota_snapshot('clip_image_detector'))
"
```

### Out of Memory (OOM)

```bash
# Switch to lite mode
EXECUTION_MODE=lite docker compose up -d

# Or reduce batch size in config
# In .env:
GPU_MEMORY_LIMIT_MB=8000
```

### Celery worker not processing tasks

```bash
# Check worker status
docker compose exec celery-worker celery -A core.orchestrator.celery_app inspect ping

# Check queue depth
docker compose exec redis redis-cli -a $REDIS_PASSWORD llen celery

# Restart worker
docker compose restart celery-worker
```

### Frontend can't connect to backend

```bash
# Check CORS
curl -X OPTIONS http://localhost:8000/health \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: GET"

# Verify NEXT_PUBLIC_API_URL in .env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### Models not loading (adapter missing)

```bash
# Check what's in the models volume
docker compose exec backend ls -la /models/

# Expected:
# clip_image_detector/
# dinov2_image_detector/
# siglip_image_detector/
# aasist3_audio_detector/
# wav2vec2_xls_r_audio_detector/
# videomae_video_detector/
# altfree_video_detector/
# timesformer_video_detector/
# ecapa_audio_detector/
# calibration/ (if fitted)
# clip_lora_image_adapter/ (if trained)
# dinov2_image_adapter/ (if trained)
```

### Complete Reset

```bash
# Stop everything
docker compose down -v

# Remove all volumes (deletes models, data, logs)
docker volume rm argus-backend-models argus-mongodb-data argus-redis-data argus-minio-data

# Rebuild from scratch
docker compose up -d --build
```

---

## Appendix A: Complete File Structure

```
Argus_Core-main/
├── .env.example                    # Environment template
├── .github/workflows/ci.yml       # CI/CD pipeline
├── CHANGELOG.md                   # Full version history
├── TRAINING.md                    # Fine-tuning guide
├── VALIDATION.md                  # Validation protocol
├── RUNBOOK.md                     # This document
├── docker-compose.yml             # 8-service Docker Compose
├── prometheus/prometheus.yml      # Prometheus scrape config
├── grafana/
│   ├── dashboards/argus-platform.json
│   └── provisioning/
│       ├── datasources/prometheus.yml
│       └── dashboards/dashboards.yml
├── scripts/
│   ├── validate_system.py         # Master validation orchestrator
│   ├── test_end_to_end.py         # 28-stage E2E tests
│   ├── test_endpoints.py          # 12 endpoints × 10 scenarios
│   ├── simulate_failures.py       # 12 chaos scenarios
│   ├── test_regression.py         # 11-metric regression
│   ├── verify_cpu_only.py         # CPU-only proof
│   ├── verify_reproducibility.py  # Reproducibility proof
│   ├── train_lora_adapters.py     # LoRA training pipeline
│   ├── benchmark_sota.py          # SOTA benchmark harness
│   ├── benchmark_adversarial.py   # Adversarial robustness
│   ├── fit_calibration.py         # Temperature + Conformal fitting
│   └── dataset_download.py        # Dataset helper
├── backend/
│   ├── Dockerfile                 # Multi-stage CUDA/CPU build
│   ├── docker-entrypoint.sh       # Model pull on startup
│   ├── requirements.txt           # All Python dependencies
│   ├── config.py                  # 40+ config flags
│   ├── server.py                  # FastAPI app + /health/detailed
│   ├── models/
│   │   ├── manifest.yaml          # 12 pinned HF models
│   │   ├── registry.py            # Model registry (18 models)
│   │   └── downloader.py          # Deterministic HF puller
│   ├── detectors/                 # 9 SOTA detectors + ensemble
│   ├── analyzers/                 # Image + audio + video pipelines
│   ├── defenses/                  # RPS + Gate + RS-lite + Certified
│   ├── calibration/               # Temperature + Conformal + Audit
│   ├── monitoring/                # Drift + Reference store
│   ├── continuous_learning/       # Feedback + Retrain + A/B
│   ├── security/                  # Watermarking + Fingerprinting
│   ├── observability/             # 15 Prometheus metrics
│   ├── inference/                 # Multi-GPU + Memory guard
│   ├── modes/                     # Lite/Balanced/Research manager
│   └── core/                      # Engine + Fusion + XAI + Post-processing
└── frontend/
    ├── Dockerfile
    └── src/
        ├── app/                   # Next.js pages
        ├── components/            # UI + XAI panels
        └── types/                 # TypeScript types
```

## Appendix B: Quick Command Reference

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Stop + delete data
docker compose down -v

# Logs
docker compose logs -f backend
docker compose logs -f celery-worker

# Shell into backend
docker compose exec backend bash

# Check health
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed | python3 -m json.tool

# Metrics
curl http://localhost:8000/metrics | grep argus_

# Validate
python scripts/validate_system.py --suite all

# Switch mode
EXECUTION_MODE=lite docker compose up -d      # CPU
EXECUTION_MODE=balanced docker compose up -d   # GPU
EXECUTION_MODE=research docker compose up -d   # Max accuracy

# Train
python scripts/train_lora_adapters.py --modality image --backbone clip ...

# Benchmark
python scripts/benchmark_sota.py --modality image --test-set celebdf_v2 ...

# Calibrate
python scripts/fit_calibration.py --modality image --calibration-json ...
```
