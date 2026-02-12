# Argus Core - Auto-Start Services Configuration

## Overview
This document details the automatic startup configuration for all Argus Core services.

## Services Configured for Auto-Start

### Core Application Services
1. **Backend (FastAPI)** - Port 8001
   - Auto-starts via supervisor
   - Includes Redis/MinIO health checks on startup
   - Loads 6 AI models automatically
   - Priority: 50 (starts after infrastructure)

2. **Frontend (Next.js)** - Port 3000
   - Auto-starts via supervisor
   - Hot reload enabled for development
   - Priority: 50

3. **MongoDB** - Port 27017
   - Auto-starts via supervisor
   - Database for analysis storage
   - Priority: 10

### Infrastructure Services (New)
4. **Redis** - Port 6379
   - Auto-starts via supervisor
   - Used for: caching, rate limiting, WebSocket pub/sub, Celery broker
   - Priority: 10 (starts first)

5. **MinIO** - Ports 9000 (API), 9001 (Console)
   - Auto-starts via supervisor
   - S3-compatible object storage for files
   - Credentials: minioadmin/minioadmin
   - Priority: 10 (starts first)

6. **Celery Worker** - Background tasks
   - Auto-starts via supervisor
   - Processes async analysis jobs
   - Concurrency: 2 workers
   - Priority: 20 (starts after Redis)

## Configuration Files

### Supervisor Configuration
- **Main Config**: `/etc/supervisor/conf.d/supervisord.conf` (READ-ONLY)
- **Services Config**: `/etc/supervisor/conf.d/argus_services.conf` (Auto-generated)

### Backend Startup Checks
Modified `/app/backend/api/deps.py` to include:
- `wait_for_redis()` - Retries connection up to 10 times
- `wait_for_minio()` - Ensures buckets are initialized
- Enhanced `startup_dependencies()` - Fails fast if services unavailable

### Health Check Enhancements
Enhanced `/app/backend/api/router.py` health endpoint:
- Added Redis connectivity check
- Added Celery worker status check
- Comprehensive component health reporting

## Service Management

### View Status
```bash
sudo supervisorctl status
```

### Restart All Services
```bash
sudo supervisorctl restart all
```

### Restart Individual Services
```bash
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
sudo supervisorctl restart redis
sudo supervisorctl restart minio
sudo supervisorctl restart celery
```

### View Logs
```bash
# Backend
tail -f /var/log/supervisor/backend.out.log
tail -f /var/log/supervisor/backend.err.log

# Redis
tail -f /var/log/supervisor/redis.out.log

# MinIO
tail -f /var/log/supervisor/minio.out.log

# Celery
tail -f /var/log/supervisor/celery.out.log
```

## Startup Sequence

1. **Infrastructure Services Start** (Priority 10)
   - Redis starts and binds to port 6379
   - MinIO starts and creates default buckets
   - MongoDB continues running

2. **Celery Worker Starts** (Priority 20)
   - Connects to Redis broker
   - Registers task handlers

3. **Application Services Start** (Priority 50)
   - Backend waits for Redis (max 20s)
   - Backend waits for MinIO (max 20s)
   - Backend connects to MongoDB
   - Backend loads 6 AI models
   - Frontend starts and connects to backend

## Health Check Endpoint

### Endpoint
```
GET http://localhost:8001/api/v1/health
```

### Response
```json
{
  "status": "healthy",
  "timestamp": "2026-02-12T11:00:30.458788+00:00",
  "version": "v1",
  "components": {
    "database": "healthy",
    "storage": "healthy",
    "redis": "healthy",
    "celery": {
      "status": "healthy",
      "active_workers": 1
    },
    "models": {
      "status": "healthy",
      "loaded": 6,
      "model_names": ["efficientnet_b3_spatial", "retinaface", "purdue_m2", 
                      "clip_vit_b16", "siglip_deepfake", "xclip_temporal"],
      "vram_used_mb": 2200,
      "vram_available_mb": 1300
    }
  }
}
```

## End-to-End Validation

### Run Validation Script
```bash
cd /app/backend
python test_e2e_validation.py
```

### Validation Tests
1. **Infrastructure Services**
   - Redis connectivity
   - MongoDB connectivity
   - MinIO accessibility
   - Celery worker status

2. **Backend API & Models**
   - Health endpoint check
   - AI models loading status

3. **Frontend Application**
   - Accessibility check
   - Rendering validation

4. **Integration Tests**
   - File upload workflow

## Troubleshooting

### Services Not Starting
```bash
# Check supervisor logs
sudo supervisorctl tail -f redis
sudo supervisorctl tail -f minio
sudo supervisorctl tail -f celery

# Force restart
sudo supervisorctl restart all
```

### Backend Fails to Start
- Check if Redis is running: `redis-cli ping`
- Check if MinIO is running: `curl http://localhost:9000/minio/health/live`
- Check backend logs: `tail -f /var/log/supervisor/backend.err.log`

### Celery Worker Issues
- Verify Redis connection: `redis-cli ping`
- Check celery logs: `tail -f /var/log/supervisor/celery.err.log`
- Restart celery: `sudo supervisorctl restart celery`

## MinIO Console Access
- **URL**: http://localhost:9001
- **Username**: minioadmin
- **Password**: minioadmin

## Default Buckets
- `argus-uploads` - Uploaded media files
- `argus-preprocessed` - Processed frames/chunks
- `argus-results` - Analysis results and reports

## Environment Variables

### Backend (.env)
```bash
# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

## Auto-Start on System Boot

All services configured in supervisor will automatically start when the container/system boots due to the `autostart=true` configuration.

## Notes
- Redis is required for WebSocket real-time updates
- MinIO is required for file storage and retrieval
- Celery is required for async background processing
- Backend will fail to start if Redis or MinIO are unavailable
- All services have automatic restart enabled (`autorestart=true`)
