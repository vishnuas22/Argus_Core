# Argus Core - Infrastructure Configuration

## Service Architecture

This document describes the complete service infrastructure for Argus Core, including automatic startup configuration.

### Service Components

#### 1. Core Services (Managed by Supervisor)

All services are configured for automatic startup via Supervisor and will restart automatically if they fail.

**Backend API (FastAPI)**
- Port: 8001
- Workers: 1 (with hot reload)
- Command: `uvicorn server:app --host 0.0.0.0 --port 8001 --workers 1 --reload`
- Directory: `/app/backend`
- Logs: `/var/log/supervisor/backend.{out,err}.log`

**Frontend UI (Next.js)**
- Port: 3000
- Command: `yarn start`
- Directory: `/app/frontend`
- Logs: `/var/log/supervisor/frontend.{out,err}.log`

**MongoDB**
- Port: 27017
- Command: `/usr/bin/mongod --bind_ip_all`
- Database Name: `argus_core`
- Logs: `/var/log/mongodb.{out,err}.log`

**Redis**
- Port: 6379
- Command: `/usr/bin/redis-server --bind 127.0.0.1 --port 6379`
- Used for: WebSocket pub/sub, Celery broker
- Logs: `/var/log/supervisor/redis.{out,err}.log`

**MinIO (Object Storage)**
- Port: 9000 (API), 9001 (Console)
- Command: `/usr/local/bin/minio server /data/minio --console-address :9001`
- Access Key: `minioadmin`
- Secret Key: `minioadmin`
- Data Directory: `/data/minio`
- Buckets: `argus-uploads`, `argus-preprocessed`, `argus-results`
- Logs: `/var/log/supervisor/minio.{out,err}.log`

**Celery Worker**
- Concurrency: 2 workers
- Command: `celery -A core.orchestrator.celery_app worker --loglevel=info --concurrency=2`
- Directory: `/app/backend`
- Queues: `analysis`, `preprocessing`, `aggregation`, `reports`
- Logs: `/var/log/supervisor/celery.{out,err}.log`

#### 2. ML Models (Loaded at Startup)

All 6 deepfake detection models are automatically loaded into VRAM at backend startup:

1. **efficientnet_b3_spatial** (300MB VRAM) - Spatial artifact detection
2. **retinaface** (200MB VRAM) - Face detection and alignment
3. **purdue_m2** (250MB VRAM) - Multi-modal manipulation detection
4. **clip_vit_b16** (400MB VRAM) - Vision-language analysis
5. **siglip_deepfake** (450MB VRAM) - Signature-based detection
6. **xclip_temporal** (600MB VRAM) - Temporal consistency analysis

Total VRAM Usage: 2200MB / 3500MB limit

### Startup Sequence

Services start in the following order (managed by priority settings):

1. **MongoDB** (priority: default) - Database initialization
2. **Redis** (priority: 1) - Cache and message broker
3. **MinIO** (priority: 2) - Object storage with bucket creation
4. **Backend** (priority: default) - Loads ML models, initializes connections
5. **Celery Worker** (priority: 10) - Async task processing
6. **Frontend** (priority: default) - Next.js development server

### Health Check Endpoints

**Basic Health Check**
```bash
GET http://localhost:8001/health
Response: {"status": "healthy"}
```

**Detailed Health Check**
```bash
GET http://localhost:8001/api/v1/health
Response: {
  "status": "healthy",
  "timestamp": "2026-02-12T04:22:46.702959+00:00",
  "version": "v1",
  "components": {
    "database": "healthy",
    "storage": "healthy",
    "models": {
      "status": "healthy",
      "loaded": 6,
      "model_names": [...],
      "vram_used_mb": 2200,
      "vram_available_mb": 1300
    }
  }
}
```

### Service Management Commands

**View all service status:**
```bash
sudo supervisorctl status
```

**Restart individual services:**
```bash
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
sudo supervisorctl restart redis
sudo supervisorctl restart minio
sudo supervisorctl restart celery_worker
```

**Restart all services:**
```bash
sudo supervisorctl restart all
```

**View service logs:**
```bash
# Backend logs
tail -f /var/log/supervisor/backend.out.log
tail -f /var/log/supervisor/backend.err.log

# Celery worker logs
tail -f /var/log/supervisor/celery.out.log

# Redis logs
tail -f /var/log/supervisor/redis.out.log

# MinIO logs
tail -f /var/log/supervisor/minio.out.log
```

**Check Redis connectivity:**
```bash
redis-cli ping
# Expected response: PONG
```

**Check MinIO connectivity:**
```bash
curl http://localhost:9000/minio/health/live
```

### Verification Script

Run the automated startup verification script:
```bash
/app/scripts/startup_verification.sh
```

Or run the comprehensive end-to-end validation:
```bash
cd /app/backend && python test_e2e_validation.py
```

### Configuration Files

**Supervisor Configuration:**
- Main config: `/etc/supervisor/conf.d/supervisord.conf` (READONLY)
- Services config: `/etc/supervisor/conf.d/argus_services.conf`

**Backend Configuration:**
- Environment: `/app/backend/.env`
- Config module: `/app/backend/config.py`

**Frontend Configuration:**
- Environment: `/app/frontend/.env`
- Next.js config: `/app/frontend/next.config.js`

### Environment Variables

**Backend (.env):**
```bash
# Database
MONGO_URL=mongodb://localhost:27017
DB_NAME=argus_core

# Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_UPLOADS=argus-uploads
MINIO_BUCKET_PREPROCESSED=argus-preprocessed
MINIO_BUCKET_RESULTS=argus-results

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# ML Configuration
MODEL_CACHE_DIR=/models
USE_GPU=false
GPU_MEMORY_LIMIT_MB=3500
```

**Frontend (.env):**
```bash
REACT_APP_BACKEND_URL=http://localhost:8001
NEXT_PUBLIC_WS_URL=ws://localhost:8001
NEXT_PUBLIC_API_URL=/api
```

### Troubleshooting

**If a service fails to start:**
1. Check service logs in `/var/log/supervisor/`
2. Verify configuration in `/etc/supervisor/conf.d/`
3. Restart the service: `sudo supervisorctl restart <service_name>`
4. Check health endpoint: `curl http://localhost:8001/api/v1/health`

**If health check shows degraded:**
- Check which component is unhealthy
- Verify the service is running: `sudo supervisorctl status`
- Check service logs for errors
- Restart the problematic service

**Common Issues:**

1. **Redis connection refused:**
   ```bash
   sudo supervisorctl restart redis
   ```

2. **MinIO bucket errors:**
   - Buckets are created automatically on first backend startup
   - Verify MinIO is running and accessible

3. **Celery worker not processing tasks:**
   ```bash
   sudo supervisorctl restart celery_worker
   tail -f /var/log/supervisor/celery.out.log
   ```

4. **Models not loading:**
   - Check VRAM usage in health endpoint
   - Verify MODEL_CACHE_DIR exists
   - Check backend logs for model loading errors

### Performance Monitoring

**Check service resource usage:**
```bash
ps aux | grep -E "(mongo|redis|minio|celery|uvicorn|node)"
```

**Monitor Redis:**
```bash
redis-cli info stats
```

**Check Celery task queue:**
```bash
cd /app/backend
celery -A core.orchestrator.celery_app inspect active
celery -A core.orchestrator.celery_app inspect stats
```

### API Endpoints

**Available Endpoints:**
- `GET /` - API information
- `GET /health` - Basic health check
- `GET /api/v1/health` - Detailed health check with component status
- `GET /api/v1/models` - List available ML models
- `GET /api/v1/stats` - Analysis statistics
- `POST /api/v1/analyze` - Submit media for analysis
- `GET /api/v1/analyze/{analysis_id}` - Get analysis status
- `GET /api/v1/analyze/{analysis_id}/detail` - Get detailed results
- `GET /api/v1/analyze/{analysis_id}/heatmaps` - Get visualization heatmaps
- `GET /api/v1/analyze/{analysis_id}/report` - Download PDF report
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /openapi.json` - OpenAPI specification

### WebSocket Support

**Real-time progress updates:**
```javascript
const ws = new WebSocket('ws://localhost:8001/api/v1/ws/{analysis_id}');
ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(progress);
};
```

Progress messages are published via Redis pub/sub and distributed to connected WebSocket clients.

---

**Last Updated:** 2026-02-12
**System Status:** All services operational and healthy
**Version:** v1
