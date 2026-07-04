# Argus Core — Production Readiness Audit

**Date:** 2026-07-02
**Target:** Real-user deployment
**Verdict:** **NOT production-ready.** Estimated 4-6 weeks of work to close all gaps, assuming models + LoRA training happen in parallel on your Mac/Colab.

---

## How to Read This Audit

Each finding is scored:
- **🔴 CRITICAL** — will cause an incident within 7 days of launch
- **🟠 HIGH** — will cause an incident within 30 days of launch
- **🟡 MEDIUM** — will cause an incident eventually or under load
- **🟢 LOW** — quality issue, not an incident risk

Each finding has:
- **What** — the gap
- **Why it matters** — the failure mode
- **Fix** — what to do
- **Where** — file/location
- **Who can do it** — me (in this env), you (on Mac/Colab), or both

---

## Category 1: Data Safety 🔴

### 1.1 🔴 MongoDB has no backup strategy
**What:** The `analyses` collection is the system of record. If the MongoDB container dies and its volume is corrupted, every analysis ever run is gone.
**Why it matters:** Real users will upload evidence they need to retrieve later. Losing the DB = losing user data = legal liability.
**Fix:** Daily `mongodump` to MinIO + weekly full backup to offsite storage. Test restore quarterly.
**Where:** New `scripts/backup_mongodb.sh`, `scripts/restore_mongodb.sh`
**Who:** Me (scripts) + you (test restore on Mac)

### 1.2 🔴 MinIO has no backup strategy
**What:** Uploaded files, preprocessed frames, and PDF reports live in MinIO. No backup.
**Why it matters:** Same as 1.1 — user-uploaded evidence is irreplaceable.
**Fix:** `mc mirror` to a second MinIO instance or S3 nightly. Versioned bucket.
**Where:** New `scripts/backup_minio.sh`
**Who:** Me (scripts) + you (test)

### 1.3 🔴 No database migration framework
**What:** Schema changes are manual. The `analyses` collection has no version field. If you change `schemas.py`, existing documents break silently.
**Why it matters:** First schema change post-launch = corrupted production data.
**Fix:** Add `alembic`-style migration framework for MongoDB. Add `schema_version` field to every document. Write migration scripts.
**Where:** New `migrations/` directory, `backend/storage/db.py`
**Who:** Me

### 1.4 🟠 No data retention policy
**What:** Analyses, audit logs, and uploaded files grow forever. The 500MB file upload limit × 1000 users/day = 500GB/day of uploads.
**Why it matters:** Disk fills up in weeks. Legal/privacy compliance requires retention limits.
**Fix:** TTL index on `analyses` (e.g., 90 days). Cron job to purge MinIO objects older than retention. GDPR delete endpoint.
**Where:** `backend/storage/db.py`, new `scripts/purge_old_data.py`
**Who:** Me

---

## Category 2: Security 🔴

### 2.1 🔴 No TLS termination
**What:** All traffic (API, WebSocket, MinIO S3, MongoDB) is plaintext HTTP.
**Why it matters:** JWT tokens, uploaded files, and credentials flow in cleartext. Any network observer can steal sessions. **This is illegal under GDPR/CCPA for real user data.**
**Fix:** Nginx reverse proxy with Let's Encrypt TLS in front of all services. Internal traffic can stay HTTP.
**Where:** New `nginx/nginx.conf`, `docker-compose.prod.yml`
**Who:** Me (config) + you (DNS + cert provisioning)

### 2.2 🔴 Secrets in `.env` file committed to repo
**What:** `.env` contains `JWT_SECRET`, `MINIO_SECRET_KEY`, `MONGO_PASSWORD`. If committed, anyone with repo access has prod credentials.
**Why it matters:** Credential leak = total system compromise.
**Fix:** `.env` in `.gitignore` (verify). Use Docker secrets or a real secrets manager (Vault, AWS Secrets Manager). Template `.env.example` only.
**Where:** `.gitignore`, new `.env.example`, `docker-compose.prod.yml`
**Who:** Me

### 2.3 🔴 All ports exposed to host
**What:** `docker-compose.yml` exposes MongoDB (27017), Redis (6379), MinIO (9000/9001) to the host. In production, these should be internal-only.
**Why it matters:** Anyone on the network can connect to your DB directly, bypassing the app.
**Fix:** Remove `ports:` mappings for internal services in prod compose. Only expose nginx (80/443).
**Where:** `docker-compose.prod.yml`
**Who:** Me

### 2.4 🔴 No per-user rate limiting
**What:** Current rate limiter is per-IP, 100 req/min. One authenticated user can submit 100 analyses/min = DoS.
**Why it matters:** A single abusive user can starve the Celery worker pool and block all other users.
**Fix:** Per-user rate limit (e.g., 10 analyses/hour for free tier, 100/hour for paid). Redis-based sliding window.
**Where:** `backend/api/middleware.py`
**Who:** Me

### 2.5 🔴 No file upload security validation
**What:** The 500MB upload limit exists, but:
- No magic-byte verification beyond initial check
- No virus scanning
- No content-type enforcement
- No filename sanitization (path traversal possible)
- `await file.read()` buffers entire 500MB in RAM = memory DoS
**Why it matters:** Malicious uploads can crash the server or exploit parser bugs (Pillow/PIL has had RCE CVEs).
**Fix:** Streaming upload to MinIO (don't buffer in RAM). ClamAV scan. Strict allowlist of file types. Sanitize filenames.
**Where:** `backend/api/router.py`, `backend/processing/sanitize.py`
**Who:** Me

### 2.6 🟠 JWT secret mismatch fixed but not tested in prod config
**What:** I fixed the `JWT_SECRET` vs `SECRET_KEY` naming. But the prod `docker-compose.yml` may still pass the wrong env var.
**Why it matters:** If prod uses a different env var name than dev, JWT validation fails silently and auth breaks.
**Fix:** Audit `docker-compose.prod.yml` env vars. Add a startup check that logs which secret is in use.
**Where:** `docker-compose.prod.yml`, `backend/server.py`
**Who:** Me

### 2.7 🟠 CORS wildcard in some configs
**What:** I made `cors_origins_list` reject `*` in production. But the default is still `http://localhost:3000` which is wrong for prod.
**Why it matters:** Either CORS blocks your real frontend, or you fall back to wildcard.
**Fix:** Prod compose must set `CORS_ORIGINS=https://yourdomain.com` explicitly.
**Where:** `docker-compose.prod.yml`
**Who:** Me

### 2.8 🟡 No CSRF protection
**What:** Cookie-based auth would need CSRF tokens. Current JWT-in-header is CSRF-safe, but if you add cookie auth later, this breaks.
**Why it matters:** Not urgent now, but a landmine.
**Fix:** Document that auth is header-only. If cookies added, add CSRF middleware.
**Where:** Docs
**Who:** Me

### 2.9 🟡 No security headers audit
**What:** `SecurityHeadersMiddleware` exists but hasn't been tested with a real browser. HSTS, CSP, X-Frame-Options may be missing or wrong.
**Why it matters:** Clickjacking, XSS, MIME-sniffing attacks.
**Fix:** Add `securityheaders.com` scan to CI. Hardened CSP.
**Where:** `backend/api/middleware.py`
**Who:** Me

---

## Category 3: Reliability 🔴

### 3.1 🔴 No graceful shutdown
**What:** When the backend restarts, in-flight HTTP requests are killed. When Celery restarts, in-flight analyses are lost (marked PENDING forever).
**Why it matters:** Every deploy = lost user work. Every crash = stuck analyses that never complete.
**Fix:** FastAPI `lifespan` shutdown handler drains connections (10s grace). Celery `worker_shutdown` retries in-flight tasks. Stuck-task reaper cron.
**Where:** `backend/server.py`, `backend/core/orchestrator.py`
**Who:** Me

### 3.2 🔴 No request timeouts
**What:** API endpoints have no timeout. A slow analysis can hold an HTTP connection for 6 minutes (Celery hard limit). WebSocket connections have no idle timeout.
**Why it matters:** Connection pool exhaustion = all users blocked.
**Fix:** Per-endpoint timeouts (30s for uploads, 5s for status checks). WebSocket idle timeout (5 min).
**Where:** `backend/api/router.py`, `backend/api/websocket.py`
**Who:** Me

### 3.3 🔴 Celery worker death = stuck analyses
**What:** If a Celery worker crashes mid-analysis, the task is marked STARTED forever. No visibility, no retry, no user notification.
**Why it matters:** User sees "analyzing..." for hours, gives up, loses trust.
**Fix:** Stuck-task reaper: every 5 min, find tasks in STARTED > 10 min, mark FAILED, notify user via WebSocket.
**Where:** New `backend/core/stuck_task_reaper.py`, Celery beat schedule
**Who:** Me

### 3.4 🔴 Permanent MinIO fallback
**What:** Per engineering review REL-1: after 3 MinIO failures, storage permanently switches to local filesystem. Never recovers.
**Why it matters:** A transient MinIO outage degrades storage forever until manual restart.
**Fix:** I partially fixed this (fallback retry interval). Verify the recovery actually works. Add a "storage mode" health check that alerts when in fallback.
**Where:** `backend/storage/storage.py`, `backend/api/health.py`
**Who:** Me

### 3.5 🟠 No circuit breaker for MongoDB
**What:** If MongoDB goes down, every API call fails. No fallback, no degradation.
**Why it matters:** DB outage = total outage.
**Fix:** Circuit breaker: after N failures, return cached results or 503 with retry-after. Auto-recover when DB returns.
**Where:** `backend/storage/db.py`
**Who:** Me

### 3.6 🟠 No circuit breaker for Redis
**What:** If Redis (broker) goes down, Celery can't dispatch tasks. API accepts uploads but they never process.
**Why it matters:** Silent failure — users upload, nothing happens.
**Fix:** Redis health check in upload path. If Redis down, return 503 immediately.
**Where:** `backend/api/router.py`
**Who:** Me

### 3.7 🟠 No memory limits on containers
**What:** `docker-compose.yml` has no `mem_limit`. One bad analysis (huge video) can OOM the host and kill all services.
**Why it matters:** One user uploading a 500MB 4K video can take down the entire platform.
**Fix:** `mem_limit: 4g` on backend, `2g` on celery, `1g` on each stateful service. OOM-killer redirects.
**Where:** `docker-compose.prod.yml`
**Who:** Me

### 3.8 🟡 No Celery worker prefetch tuning
**What:** `worker_prefetch_multiplier=1` is good for memory-heavy work, but means only 4 concurrent analyses (one per prefork worker).
**Why it matters:** Throughput ceiling. 4 users analyzing simultaneously = everyone waits.
**Fix:** Tune based on load testing. Consider `--pool=threads` for I/O-bound work.
**Where:** `backend/core/orchestrator.py`
**Who:** Me (config) + you (load test)

---

## Category 4: Observability 🟠

### 4.1 🟠 No alerting
**What:** Prometheus scrapes metrics, Grafana has a dashboard, but no Alertmanager. The system can be down and nobody knows.
**Why it matters:** Users find out before you do.
**Fix:** Alertmanager with rules: error rate > 1%, latency p95 > 30s, queue depth > 50, any service unhealthy > 1 min. PagerDuty/Slack integration.
**Where:** New `alertmanager/alertmanager.yml`, `alertmanager/rules.yml`
**Who:** Me

### 4.2 🟠 No error tracking
**What:** `sentry-sdk` is in requirements but not configured. Errors go to logs, not a dashboard.
**Why it matters:** Errors get lost in log noise. No stack traces, no breadcrumbs, no release tracking.
**Fix:** Configure Sentry DSN via env var. Add before-send hook to scrub PII.
**Where:** `backend/server.py`, `backend/config.py`
**Who:** Me (config) + you (Sentry account)

### 4.3 🟠 No structured logging in production
**What:** `structlog` is configured but logs go to stdout as text. No log aggregation.
**Why it matters:** Debugging prod issues means `docker logs | grep` across 8 containers.
**Fix:** JSON logs to stdout, collected by Filebeat or Fluentd, shipped to Elasticsearch/Loki. OR use a managed service (Datadog, New Relic).
**Where:** `backend/utils/logging.py`
**Who:** Me (JSON format) + you (log aggregator choice)

### 4.4 🟡 No distributed tracing
**What:** No OpenTelemetry. A request spans FastAPI → Celery → analyzers → storage, but there's no trace ID connecting them.
**Why it matters:** "Why was this analysis slow?" is unanswerable.
**Fix:** OpenTelemetry instrumentation. Jaeger or Honeycomb backend.
**Where:** `backend/server.py`, `backend/api/middleware.py`
**Who:** Me (instrumentation) + you (backend choice)

### 4.5 🟡 No audit log retention
**What:** Audit log has a TTL index (per engineering review DB-1) but it's not enforced. Logs grow forever.
**Why it matters:** Disk fill + compliance issues.
**Fix:** Enforce TTL. 90-day retention default, configurable.
**Where:** `backend/storage/db.py`
**Who:** Me

---

## Category 5: Performance 🟡

### 5.1 🟡 No response caching
**What:** Every `GET /api/v1/analyze/{id}` hits MongoDB. No Redis cache.
**Why it matters:** Popular analyses (e.g., user refreshing status page) hammer the DB.
**Fix:** Redis cache with 60s TTL for status, 5 min for detail. Invalidate on update.
**Where:** `backend/api/router.py`
**Who:** Me

### 5.2 🟡 No connection pooling in Celery
**What:** Each Celery task creates a fresh MongoDB connection. No pool reuse.
**Why it matters:** Connection setup overhead per task = slower + more DB connections.
**Fix:** Celery `worker_process_init` signal creates a shared client.
**Where:** `backend/core/orchestrator.py`
**Who:** Me

### 5.3 🟡 Synchronous ONNX in async (mostly fixed)
**What:** I verified `loop.run_in_executor` is used for ONNX calls. But the thread pool is only 4 workers.
**Why it matters:** 5 concurrent analyses = 5th waits for ONNX.
**Fix:** Load test to find right pool size. Probably 8-16 on M1 Max.
**Where:** `backend/core/engine.py`
**Who:** Me (config) + you (load test)

### 5.4 🟡 No preprocessed-frame caching
**What:** Same video uploaded twice = preprocessed twice.
**Why it matters:** Wasted CPU on duplicate uploads.
**Fix:** Hash preprocessed output, cache in MinIO with content hash key.
**Where:** `backend/processing/preprocess.py`
**Who:** Me

---

## Category 6: Operability 🟠

### 6.1 🟠 No runbook
**What:** `RUNBOOK.md` exists but is developer-focused, not ops-focused. No "what to do when X breaks."
**Why it matters:** 3 AM incident, nobody knows what to do.
**Fix:** Runbook with: common incidents, diagnosis steps, recovery procedures, escalation contacts.
**Where:** `RUNBOOK.md`
**Who:** Me

### 6.2 🟠 No CI/CD
**What:** No GitHub Actions. Deploys are manual `docker compose up`.
**Why it matters:** Manual deploys = human error. No test gate before prod.
**Fix:** GitHub Actions: lint → test → build → push to registry → deploy to staging. Manual promote to prod.
**Where:** `.github/workflows/ci.yml`
**Who:** Me

### 6.3 🟡 No blue-green deployment
**What:** Deploys require downtime. `docker compose up -d --build` restarts everything.
**Why it matters:** Every deploy = 30s of 502s.
**Fix:** Two backend instances behind nginx. Deploy to inactive, swap, drain old.
**Where:** `docker-compose.prod.yml`, `nginx/nginx.conf`
**Who:** Me

### 6.4 🟡 No log rotation
**What:** Docker logs grow forever. No `max-size` / `max-file`.
**Why it matters:** Disk fill in weeks.
**Fix:** `logging.options.max-size: 100m, max-file: 5` in compose.
**Where:** `docker-compose.prod.yml`
**Who:** Me

### 6.5 🟡 No health check alerting
**What:** Docker healthchecks exist but no alerting when unhealthy.
**Why it matters:** Container restarts silently, nobody investigates root cause.
**Fix:** Healthcheck status → Prometheus → Alertmanager.
**Where:** `prometheus/prometheus.yml`, alert rules
**Who:** Me

---

## Category 7: Models & ML 🔴 (only you can do these)

### 7.1 🔴 Models not downloaded
**What:** `/models/` is empty. All detectors return 0.5 (placeholder).
**Fix:** Run `python -m models.bootstrap` on your Mac. Verify all 23 models load on MPS.
**Who:** You

### 7.2 🔴 LoRA adapters not trained
**What:** SOTA detectors fall back to zero-shot. Accuracy will be poor.
**Fix:** Train image/audio/video LoRA on Colab (~$3-20). See my earlier Phase 4 instructions.
**Who:** You

### 7.3 🔴 Platt calibration not fit
**What:** `platt_params.json` doesn't exist. Trust scores are uncalibrated.
**Fix:** Run `scripts/fit_calibration.py` on 200+ labeled images per modality.
**Who:** You

### 7.4 🔴 `purdue_m2.onnx` known corrupt
**What:** Per `AGENTS.md` line 30. Audio detection silently fails.
**Fix:** Replace with AASIST from `clovaai/aasist-l`.
**Who:** You

### 7.5 🔴 End-to-end pipeline never run with real models
**What:** All my testing was without models. Real inference will surface bugs.
**Fix:** After 7.1-7.4, run `scripts/test_end_to_end.py` with real media. Fix what breaks.
**Who:** You

---

## Category 8: Documentation 🟡

### 8.1 🟡 No README for users
**What:** `AGENTS.md` is for developers. No "how to use this" for end users.
**Fix:** README with: what it does, quickstart, architecture overview, deployment.
**Who:** Me

### 8.2 🟡 No ARCHITECTURE.md
**What:** Architecture is documented in `ENGINEERING_REVIEW.md` (900 lines, too dense).
**Fix:** 2-page architecture overview with the system diagram.
**Who:** Me

### 8.3 🟡 No API documentation beyond OpenAPI
**What:** `/docs` exists but no usage guide.
**Fix:** `docs/API.md` with examples for each endpoint.
**Who:** Me

---

## Priority Order for "Real Users" Launch

### Week 1-2: Critical (do in parallel)

**Track A (me, in this environment):**
- Security: nginx + TLS, secrets, upload hardening, per-user rate limit (2.1-2.5)
- Data: backup/restore scripts, migration framework, retention policy (1.1-1.4)
- Reliability: graceful shutdown, timeouts, stuck-task reaper (3.1-3.3)
- Observability: alerting, Sentry, structured logging (4.1-4.3)

**Track B (you, on Mac + Colab):**
- Models: download all 23, verify on MPS, replace corrupt purdue_m2 (7.1, 7.4)
- LoRA: train image + audio + video adapters on Colab (7.2)
- Calibration: fit Platt on labeled validation set (7.3)
- E2E: run `test_end_to_end.py` with real media, fix bugs (7.5)

### Week 3: Hardening

- Circuit breakers (3.5, 3.6)
- Memory limits, log rotation (3.7, 6.4)
- CI/CD pipeline (6.2)
- Runbook, README, ARCHITECTURE.md (6.1, 8.1, 8.2)

### Week 4: Load testing + chaos

- Load test with Locust (find throughput ceiling)
- Chaos testing: kill each service, verify recovery
- Performance tuning based on load test results

### Week 5: Staging + canary

- Deploy to staging
- Canary with 5% of traffic
- Monitor for 48h
- Full rollout

---

## What I'm Starting On Now

Given you have time and want production-ready, I'll execute Track A in this environment. Starting with:

1. **This audit** (done)
2. **Security hardening** (nginx + TLS + secrets + upload limits) — highest incident risk
3. **Backup/restore scripts** — highest data-loss risk
4. **Graceful shutdown + timeouts** — highest reliability risk
5. **Alerting + Sentry** — highest "nobody knows it's broken" risk
6. **CI/CD** — eliminates manual deploy errors
7. **Runbook + README + ARCHITECTURE.md** — operability

You execute Track B in parallel. We meet in the middle at week 3 for hardening.

**Starting now.**
