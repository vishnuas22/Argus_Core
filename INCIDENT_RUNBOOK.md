# Argus Core — Operations Incident Runbook

**Audience:** On-call engineers, ops team
**Purpose:** What to do when Argus Core breaks in production

> This is the **incident response** runbook. For initial deployment, see [RUNBOOK.md](RUNBOOK.md). For architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## On-Call Quick Reference

| Symptom | Jump to |
|---|---|
| Site is down / 502 errors | [Incident A](#incident-a--site-down-502-errors) |
| Uploads fail with 500 | [Incident B](#incident-b--uploads-failing) |
| Analyses stuck in "processing" | [Incident C](#incident-c--analyses-stuck) |
| Verdicts are all "uncertain" | [Incident D](#incident-d--all-verdicts-uncertain) |
| Disk full alert | [Incident E](#incident-e--disk-full) |
| MongoDB connection errors | [Incident F](#incident-f--mongodb-down) |
| MinIO errors / storage fallback | [Incident G](#incident-g--minio-down) |
| Celery worker not processing | [Incident H](#incident-h--celery-worker-down) |
| High memory / OOM kills | [Incident I](#incident-i--oom-kills) |
| Drift alert fired | [Incident J](#incident-j--drift-detected) |

---

## General Diagnostic Commands

```bash
# Service status (all)
docker compose -f docker-compose.prod.yml ps

# Logs (last 100 lines, follow)
docker compose -f docker-compose.prod.yml logs -f --tail 100 backend
docker compose -f docker-compose.prod.yml logs -f --tail 100 celery-worker
docker compose -f docker-compose.prod.yml logs -f --tail 100 nginx

# Health check
curl -s https://your-domain/api/v1/health | jq

# Container resource usage
docker stats --no-stream

# MongoDB shell
docker exec -it argus-mongodb mongosh -u $MONGO_USER -p $MONGO_PASSWORD --authenticationDatabase admin argus_core

# Redis CLI
docker exec -it argus-redis redis-cli -a $REDIS_PASSWORD

# MinIO CLI
docker exec -it argus-minio mc alias set local http://localhost:9000 $MINIO_ACCESS_KEY $MINIO_SECRET_KEY
docker exec -it argus-minio mc ls local/
```

---

## Incident A — Site Down (502 Errors)

**Symptom:** Users see 502 Bad Gateway. Nginx can't reach backend.

**Diagnosis:**
```bash
# Is backend container running?
docker compose -f docker-compose.prod.yml ps backend

# If not running, check why:
docker compose -f docker-compose.prod.yml logs --tail 200 backend
```

**Most common causes:**

1. **Backend crashed** → container restarts but crashes again
   - Fix: see logs, likely OOM or model load failure
   - Quick fix: `docker compose -f docker-compose.prod.yml restart backend`

2. **Backend stuck in startup** (model download slow)
   - Check: `docker compose logs backend | grep "Starting up"`
   - Fix: wait, or set `DOWNLOAD_ON_STARTUP=false` and pre-download models

3. **Nginx can't resolve upstream**
   - Check: `docker compose logs nginx | grep "host not found"`
   - Fix: `docker compose -f docker-compose.prod.yml restart nginx`

**Recovery:**
```bash
docker compose -f docker-compose.prod.yml restart backend
sleep 30
curl -s https://your-domain/api/v1/health
```

**If unrecoverable in 5 min:** roll back to previous image version.
```bash
export ARGUS_VERSION=<previous-good-version>
docker compose -f docker-compose.prod.yml up -d backend
```

---

## Incident B — Uploads Failing

**Symptom:** `POST /api/v1/analyze` returns 500 or 413.

**Diagnosis:**
```bash
# Check backend logs for upload errors
docker compose -f docker-compose.prod.yml logs backend | grep -i "analyze\|upload\|error" | tail -50

# Check MinIO health
docker exec -it argus-minio curl -s localhost:9000/minio/health/live

# Check bucket exists
docker exec -it argus-minio mc ls local/argus-uploads
```

**Common causes:**

1. **File too large (413)** → increase `client_max_body_size` in nginx.conf and `max_file_size_mb` in .env
2. **MinIO down** → see [Incident G](#incident-g--minio-down)
3. **MongoDB down** → see [Incident F](#incident-f--mongodb-down)
4. **Redis down** (can't enqueue Celery task) → `docker compose restart redis`
5. **Invalid file type** → user error, not an incident

**Recovery:**
```bash
# Restart dependent services
docker compose -f docker-compose.prod.yml restart redis minio
sleep 10
# Test upload
curl -X POST https://your-domain/api/v1/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test.jpg"
```

---

## Incident C — Analyses Stuck

**Symptom:** Analyses show "processing" for >10 minutes.

**Diagnosis:**
```bash
# Check Celery worker status
docker exec -it argus-celery-worker celery -A core.orchestrator.celery_app inspect active
docker exec -it argus-celery-worker celery -A core.orchestrator.celery_app inspect stats

# Check queue depth
docker exec -it argus-redis redis-cli -a $REDIS_PASSWORD llen celery

# Check for stuck tasks in DB
docker exec -it argus-mongodb mongosh -u $MONGO_USER -p $MONGO_PASSWORD --authenticationDatabase admin argus_core --eval '
  db.analyses.find({
    status: { $in: ["preprocessing", "analyzing", "aggregating"] },
    updated_at: { $lt: new Date(Date.now() - 10*60*1000) }
  }).count()
'
```

**Common causes:**

1. **Celery worker died** → see [Incident H](#incident-h--celery-worker-down)
2. **Model inference hung** (ONNX session deadlock) → restart worker
3. **Stuck-task reaper not running** → check celery-beat is up

**Recovery:**
```bash
# The stuck-task reaper (runs every 5 min) should auto-fail these.
# If it hasn't, run it manually:
docker exec -it argus-celery-beat celery -A core.orchestrator.celery_app call argus_tasks.reap_stuck_tasks

# Or restart the worker (in-flight tasks will be retried)
docker compose -f docker-compose.prod.yml restart celery-worker
```

---

## Incident D — All Verdicts "Uncertain"

**Symptom:** Every analysis returns trust score 40-59 (uncertain).

**Diagnosis:**
```bash
# Check loaded models
curl -s https://your-domain/api/v1/models | jq

# Check model inference error rate
docker exec -it argus-prometheus promtool query instant http://localhost:9090 'rate(argus_model_inference_total{success="false"}[10m])'
```

**Common causes:**

1. **Models not loaded** → all detectors return 0.5 (placeholder), fusion produces uncertain
   - Fix: restart backend, check model download logs
   ```bash
   docker compose -f docker-compose.prod.yml logs backend | grep -i "model.*load\|model.*fail"
   ```

2. **Model files corrupted** → re-download
   ```bash
   docker exec -it argus-backend python -m models.bootstrap --force
   ```

3. **All detectors failing** → check for shared dependency issue
   ```bash
   docker compose -f docker-compose.prod.yml logs celery-worker | grep -i "error\|exception" | tail -50
   ```

**Recovery:**
```bash
# Restart to reload models
docker compose -f docker-compose.prod.yml restart backend celery-worker
sleep 60
# Verify models loaded
curl -s https://your-domain/api/v1/models | jq '.loaded_models | length'
```

---

## Incident E — Disk Full

**Symptom:** Alertmanager fires `HostDiskFull`. Services start failing.

**Diagnosis:**
```bash
# Check disk usage
df -h

# Find biggest space consumers
docker system df
docker compose -f docker-compose.prod.yml exec mongodb du -sh /data/db
docker compose -f docker-compose.prod.yml exec minio du -sh /data
```

**Recovery:**
```bash
# 1. Clean old Docker images
docker image prune -a --filter "until=168h"

# 2. Clean old logs
docker compose -f docker-compose.prod.yml logs --tail 0 -f &  # just to truncate
# Or truncate log files:
sudo truncate -s 0 /var/lib/docker/containers/*/*-json.log

# 3. Purge old analyses (if retention allows)
docker exec -it argus-backend python scripts/purge_old_data.py --days 30

# 4. Compact MongoDB
docker exec -it argus-mongodb mongosh -u $MONGO_USER -p $MONGO_PASSWORD --authenticationDatabase admin argus_core --eval 'db.runCommand({compact: "analyses"})'

# 5. If still full, expand disk volume (cloud provider)
```

---

## Incident F — MongoDB Down

**Symptom:** Backend can't read/write analyses. All API calls fail.

**Diagnosis:**
```bash
docker compose -f docker-compose.prod.yml ps mongodb
docker compose -f docker-compose.prod.yml logs --tail 100 mongodb
```

**Common causes:**
1. **Disk full** → see [Incident E](#incident-e--disk-full)
2. **OOM** → MongoDB killed by OOM-killer
3. **Corrupted WAL** → needs repair

**Recovery:**
```bash
# 1. Try simple restart
docker compose -f docker-compose.prod.yml restart mongodb
sleep 30
docker exec -it argus-mongodb mongosh -u $MONGO_USER -p $MONGO_PASSWORD --authenticationDatabase admin --eval 'db.adminCommand("ping")'

# 2. If corrupted, repair
docker compose -f docker-compose.prod.yml stop mongodb
docker run --rm -v argus-core_mongodb_data:/data/db mongo:7 mongod --repair --dbpath /data/db
docker compose -f docker-compose.prod.yml start mongodb

# 3. If unrecoverable, restore from backup
./scripts/restore_mongodb.sh --latest
```

**Post-recovery:** verify data integrity
```bash
docker exec -it argus-mongodb mongosh -u $MONGO_USER -p $MONGO_PASSWORD --authenticationDatabase admin argus_core --eval 'db.analyses.countDocuments()'
```

---

## Incident G — MinIO Down

**Symptom:** Uploads fail, storage falls back to local filesystem.

**Diagnosis:**
```bash
docker compose -f docker-compose.prod.yml ps minio
docker compose -f docker-compose.prod.yml logs --tail 100 minio
docker exec -it argus-minio curl -s localhost:9000/minio/health/live
```

**Recovery:**
```bash
# 1. Restart MinIO
docker compose -f docker-compose.prod.yml restart minio
sleep 15
docker exec -it argus-minio curl -s localhost:9000/minio/health/live

# 2. Verify buckets
docker exec -it argus-minio mc ls local/

# 3. If buckets missing, recreate
docker exec -it argus-minio mc mb local/argus-uploads
docker exec -it argus-minio mc mb local/argus-preprocessed
docker exec -it argus-minio mc mb local/argus-results
docker exec -it argus-minio mc mb local/argus-backups

# 4. Backend auto-recovers from local fallback on next successful MinIO call
```

---

## Incident H — Celery Worker Down

**Symptom:** Analyses queued but not processing. `celery_workers_online = 0`.

**Diagnosis:**
```bash
docker compose -f docker-compose.prod.yml ps celery-worker
docker compose -f docker-compose.prod.yml logs --tail 200 celery-worker
```

**Common causes:**
1. **OOM** → model loading exceeds memory limit
2. **Model load failure** → corrupt weights
3. **ONNX runtime crash** → incompatible model

**Recovery:**
```bash
# 1. Restart worker
docker compose -f docker-compose.prod.yml restart celery-worker
sleep 60

# 2. Verify worker is registered
docker exec -it argus-celery-worker celery -A core.orchestrator.celery_app inspect ping

# 3. Check it's processing
docker exec -it argus-redis redis-cli -a $REDIS_PASSWORD llen celery
# Should decrease over time

# 4. If repeated OOM, increase memory limit in docker-compose.prod.yml
#    celery-worker deploy.resources.limits.memory: 6G
```

---

## Incident I — OOM Kills

**Symptom:** Containers dying, `docker logs` shows "Killed" or OOM messages.

**Diagnosis:**
```bash
# Check OOM events
dmesg | grep -i "out of memory\|oom"

# Check container memory limits vs usage
docker stats --no-stream
```

**Common causes:**
1. **Large video upload** → 500MB file + frame extraction = memory spike
2. **Multiple concurrent analyses** → each loads models
3. **Model memory leak** → ONNX session not releasing

**Recovery:**
```bash
# 1. Increase memory limits in docker-compose.prod.yml
#    backend: 6G, celery-worker: 8G

# 2. Reduce concurrency
#    CELERY_CONCURRENCY=2 (was 4)

# 3. Restart
docker compose -f docker-compose.prod.yml up -d --force-recreate backend celery-worker

# 4. Long-term: implement streaming frame extraction (don't hold all frames in RAM)
```

---

## Incident J — Drift Detected

**Symptom:** Alertmanager fires `DistributionDriftDetected` (PSI > 0.25).

**What it means:** A new deepfake generator may have appeared. Detection accuracy on this generator is unknown.

**Action:**
1. **Don't panic** — the system is still working, just less accurately on new content
2. **Check the drift dashboard** in Grafana → which modality is drifting?
3. **Collect samples** of the drifting content for analysis
4. **Retrain LoRA adapters** on a GPU machine (Colab/RunPod):
   ```bash
   python scripts/train_lora_adapters.py --modality <drifting-modality> --dataset <new-samples>
   ```
5. **A/B test** the new adapter before promoting:
   ```bash
   # New adapter goes to 10% of traffic automatically
   # Monitor accuracy for 24h
   # Promote when confident:
   curl -X POST https://your-domain/api/v1/admin/promote-adapter -d '{"modality":"image"}' -H "Authorization: Bearer $ADMIN_TOKEN"
   ```

---

## Post-Incident Checklist

After resolving any incident:

- [ ] All services healthy (`curl https://your-domain/api/v1/health`)
- [ ] No alerts firing in Alertmanager
- [ ] Error rate back to baseline (<1%)
- [ ] Latency back to baseline (p95 <30s)
- [ ] Incident documented in #argus-incidents Slack channel
- [ ] Postmortem scheduled (if user-facing)
- [ ] Runbook updated with any new failure mode discovered

---

## Escalation

| Severity | Response time | Escalate to |
|---|---|---|
| Critical (site down, data loss) | 5 min | On-call → Tech Lead → CTO |
| High (feature broken, no workaround) | 30 min | On-call → Tech Lead |
| Medium (feature broken, workaround exists) | 4 hours | On-call |
| Low (cosmetic, minor bug) | Next business day | Ticket queue |

---

## Backup & Restore

### Backup schedule
- **MongoDB**: daily at 00:15 UTC (automatic via Celery Beat)
- **MinIO**: weekly full mirror (manual: `./scripts/backup_minio.sh`)

### Test restore quarterly
```bash
# 1. Spin up a test instance
docker compose -f docker-compose.prod.yml -p argus-test up -d mongodb minio

# 2. Restore latest MongoDB backup
./scripts/restore_mongodb.sh --latest

# 3. Verify data
docker exec -it argus-test-mongodb mongosh -u $MONGO_USER -p $MONGO_PASSWORD --authenticationDatabase admin argus_core --eval 'db.analyses.countDocuments()'

# 4. Tear down test instance
docker compose -f docker-compose.prod.yml -p argus-test down -v
```

---

## Maintenance Windows

### Deploy (no downtime with blue-green)
1. Push to `main` → CI/CD builds and deploys to staging
2. Verify staging for 30 min
3. Tag release → CI/CD deploys to production (manual approval)
4. Monitor for 1 hour

### Database migration
1. Backup first: `./scripts/backup_mongodb.sh`
2. Run migration: `python -m migrations.<name>`
3. Verify: `python -m migrations.<name> --verify`
4. If broken: `./scripts/restore_mongodb.sh --latest`

### Model update
1. Download new models: `docker exec -it argus-backend python -m models.bootstrap --force`
2. Reload without restart:
   ```bash
   curl -X POST https://your-domain/api/v1/admin/reload-models -H "Authorization: Bearer $ADMIN_TOKEN"
   ```
3. Monitor error rate for 10 min
