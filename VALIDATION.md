# Argus Core — System Validation & Continuous Verification Protocol

This document describes the complete validation framework for the Argus
Core deepfake detection platform.

---

## Overview

The platform is **never considered complete** solely because it compiles
or appears functional. After every architectural modification, feature
addition, bug fix, refactor, model update, or dependency upgrade, a
complete end-to-end validation is executed.

---

## Quick Start

### Run Full Validation

```bash
# Start the stack first
docker compose up -d
sleep 60  # Wait for models to download

# Run all validation suites
python scripts/validate_system.py --suite all --api-url http://localhost:8000

# View the report
cat /tmp/argus_validation_report.md
```

### Run Individual Suites

```bash
# End-to-end user flow (28 stages)
python scripts/validate_system.py --suite e2e

# Endpoint validation (12 endpoints × 10 scenarios each)
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
EXECUTION_MODE=lite CUDA_VISIBLE_DEVICES="" python scripts/verify_cpu_only.py
```

### Reproducibility Verification

```bash
python scripts/verify_reproducibility.py --runs 5 --tolerance 1e-4
```

---

## Validation Suites

### 1. End-to-End User Flow (`test_end_to_end.py`)

Simulates the complete 28-stage user lifecycle:

| Stage | What's validated |
|-------|-----------------|
| 01 | Frontend reachable |
| 02 | Backend /health responds |
| 03 | /health/detailed returns subsystems |
| 04 | OpenAPI docs available |
| 05 | Authentication login |
| 06 | Invalid auth rejected |
| 07 | Valid file upload (image) |
| 08 | Invalid file upload rejected |
| 09 | Large file handling |
| 10 | Analysis creation |
| 11 | Analysis status polling |
| 12 | WebSocket progress |
| 13 | Preprocessing pipeline |
| 14 | Model selection (ModeManager) |
| 15 | Image inference |
| 16 | Post-processing (calibration + conformal) |
| 17 | XAI explanation generation |
| 18 | Result aggregation |
| 19 | Database storage |
| 20 | API response format |
| 21 | Report generation |
| 22 | /metrics endpoint |
| 23 | Logging |
| 24 | Prometheus metrics content |
| 25 | Audit trail |
| 26 | Concurrent requests (10 parallel) |
| 27 | Cleanup |
| 28 | Recovery after restart |

### 2. Endpoint Validation (`test_endpoints.py`)

Tests 12 endpoints against 10 scenarios each (120 total tests):

**Endpoints tested:**
- GET /health
- GET /health/detailed
- GET /metrics
- GET /docs
- GET /api/v1/analyses
- GET /api/v1/analyze/{id}
- POST /api/v1/analyze
- POST /api/v1/upload
- POST /api/v1/feedback
- GET /api/v1/feedback/stats
- POST /api/v1/retrain/image
- GET /api/v1/ab_test/image

**Scenarios per endpoint:**
1. Success (valid input)
2. Invalid input (malformed JSON)
3. Empty input
4. Large file (file endpoints only)
5. Unsupported format (file endpoints only)
6. Corrupted file (file endpoints only)
7. Timeout (1ms client timeout)
8. Authentication failure (no token)
9. Authorization failure (invalid token)
10. Concurrent requests (10 parallel)
11. Queue overload (50 rapid requests)

### 3. Failure Simulation (`simulate_failures.py`)

Chaos engineering — 12 failure modes:

| Failure | How it's simulated | Expected behavior |
|---------|-------------------|-------------------|
| Redis unavailable | Check /health still responds | Health doesn't depend on Redis |
| Celery worker crash | Check analysis queues, not crashes | Requests queue, don't fail |
| Database offline | Check /health responds | Read-only mode or graceful error |
| GPU unavailable | Check health/detailed reports GPU | ModeManager auto-detects, falls back |
| Model load failure | Check neutral score fallback | Returns 0.5 score, logs warning |
| Corrupted weights | Check manifest checksum | Manifest verification rejects |
| Full disk | Check upload rejection | Returns 413/507 error |
| High memory | Check MemoryGuard triggers | FP16/INT8 fallback or LRU eviction |
| High CPU | Check platform responds | Slower but functional |
| Network interruption | 1ms timeout | Timeout exception handled |
| Slow inference | Check Celery time limit | task_soft_time_limit=300s catches |
| Queue backlog | 100 rapid requests | 80%+ success rate |

### 4. Regression Testing (`test_regression.py`)

Captures 11 metrics and compares against baseline:

| Metric | Target | How measured |
|--------|--------|-------------|
| Latency p50 | < 100ms | 20 health-check requests |
| Latency p95 | < 500ms | 20 health-check requests |
| Latency p99 | < 1000ms | 20 health-check requests |
| Throughput | > 10 req/s | 50 concurrent requests |
| Memory usage | < 8GB | process_resident_memory_bytes metric |
| CPU utilization | tracked | process_cpu_seconds_total metric |
| XAI output | present | /health/detailed shows models |
| Auth enforced | 401/403 without token | Protected endpoint without auth |
| CORS not wildcard | no `*` in production | OPTIONS request check |
| HTTPS | enabled or behind proxy | URL scheme check |
| Metrics complete | all 9 required | /metrics contains all argus_ metrics |

**Baseline capture:**
```bash
python scripts/test_regression.py --baseline --output /tmp/baseline.json
```

**Comparison:**
```bash
python scripts/test_regression.py --compare /tmp/baseline.json
```

---

## CI/CD Pipeline (`.github/workflows/ci.yml`)

### Triggers
- **Push** to main/develop: Python tests + frontend tests + Docker build + security scan
- **Pull request** to main: Same as push
- **Daily schedule** (02:00 UTC): Full integration tests + regression baseline

### Jobs

| Job | What it does | When |
|-----|-------------|------|
| `python-tests` | Syntax check + pytest + CPU-only + reproducibility | Every push/PR |
| `frontend-tests` | TypeScript check + vitest | Every push/PR |
| `docker-build` | Build backend + frontend images | Every push/PR |
| `security-scan` | Bandit + secret check + safety (CVEs) | Every push/PR |
| `integration-tests` | Full Docker Compose + validate_system.py | Daily + main pushes |
| `regression-baseline` | Compare against previous baseline | Daily |

### Artifacts
- `validation-report` — JSON + Markdown validation report from each run

---

## Acceptance Criteria

A feature is considered complete **only if ALL** of the following pass:

- [ ] Functional tests pass (`--suite unit`)
- [ ] Integration tests pass (`--suite e2e`)
- [ ] Endpoint tests pass (`--suite endpoints`)
- [ ] Failure simulation passes (`--suite failures`)
- [ ] Regression tests pass (`--suite regression`)
- [ ] Security checks pass (bandit + safety + CORS + auth)
- [ ] Performance targets met (latency p95 < 500ms, throughput > 10 req/s)
- [ ] Monitoring operational (Prometheus metrics present)
- [ ] Documentation updated (CHANGELOG.md + relevant docs)
- [ ] No regression in image, audio, or video pipelines

---

## Regression Policy

If any regression is detected:

1. **Identify root cause** — use the validation report's failure detail
2. **Propose least disruptive fix** — prefer config changes over code changes
3. **Revalidate the entire platform** — run `validate_system.py --suite all`
4. **Do not proceed** until regression is resolved or explicitly justified

### Regression Metrics Monitored

| Metric | Regression threshold |
|--------|---------------------|
| Detection Accuracy | > 2% drop |
| Precision | > 2% drop |
| Recall | > 2% drop |
| F1 Score | > 2% drop |
| Latency p95 | > 50% increase |
| Throughput | > 30% decrease |
| Memory Usage | > 50% increase |
| GPU Utilization | unexpected spike/drop |
| CPU Utilization | > 80% sustained |
| Explainability | XAI output missing |
| Security | auth bypass / CORS wildcard |

---

## Continuous Improvement Loop

After each validation cycle:

1. **Analyze weaknesses** — review failed tests + near-misses
2. **Identify bottlenecks** — latency p99, memory spikes, queue depth
3. **Compare with SOTA** — check latest papers/benchmarks for new techniques
4. **Propose improvements** — evidence-based, with expected ROI
5. **Implement** — only evidence-based enhancements
6. **Repeat validation** — full `validate_system.py` run
7. **Document** — update CHANGELOG + TRAINING + this file

Continue until no significant architectural, performance, reliability,
security, or usability improvements remain.

---

## Observability

All important events are observable via:

### Prometheus Metrics (15 Argus-specific)
- `argus_inference_total{modality, verdict}`
- `argus_inference_latency_seconds{modality}` (histogram)
- `argus_drift_score{modality}` / `argus_drift_severity{modality}`
- `argus_drift_psi{modality}` / `argus_drift_mmd{modality}`
- `argus_retrain_total{modality, status}` / `argus_retrain_samples{modality}`
- `argus_ab_test_accuracy{modality, is_candidate}`
- `argus_calibration_ece{modality}` / `argus_calibration_brier{modality}`
- `argus_adversarial_flagged_total{modality, defense}`
- `argus_conformal_route_to_human_total{modality}`
- `argus_feedback_buffer_size{modality}`
- `argus_certified_robustness_radius{modality}` (histogram)
- `argus_watermark_embedded_total{adapter_name}`

### Grafana Dashboard
- 11-panel dashboard at `http://localhost:3030`
- Auto-provisioned from `grafana/dashboards/argus-platform.json`

### Health Endpoints
- `GET /health` — basic liveness
- `GET /health/detailed` — full subsystem status (drift, retrain, A/B, calibration, models, defenses)

---

## File Index

| File | Purpose |
|------|---------|
| `scripts/validate_system.py` | Master orchestrator — runs all suites |
| `scripts/test_end_to_end.py` | 28-stage user flow simulation |
| `scripts/test_endpoints.py` | 12 endpoints × 10 scenarios |
| `scripts/simulate_failures.py` | 12 failure mode simulations |
| `scripts/test_regression.py` | 11-metric regression baseline + compare |
| `scripts/verify_cpu_only.py` | CPU-only functionality proof |
| `scripts/verify_reproducibility.py` | Same input → same output proof |
| `.github/workflows/ci.yml` | CI/CD pipeline (6 jobs) |
| `VALIDATION.md` | This document |
