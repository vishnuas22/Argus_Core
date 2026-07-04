#!/usr/bin/env bash
# ============================================================
# Argus Core — MinIO Backup Script
# ============================================================
# Mirrors all MinIO buckets to a backup MinIO instance or S3.
# Run weekly via cron.
#
# Usage:
#   ./scripts/backup_minio.sh
#
# Env vars (from .env):
#   MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_ENDPOINT
#   BACKUP_S3_ENDPOINT (optional — defaults to local dir)
#   BACKUP_S3_ACCESS_KEY, BACKUP_S3_SECRET_KEY (optional)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

: "${MINIO_ACCESS_KEY:?MINIO_ACCESS_KEY must be set}"
: "${MINIO_SECRET_KEY:?MINIO_SECRET_KEY must be set}"
: "${MINIO_ENDPOINT:?MINIO_ENDPOINT must be set}"

BUCKETS=("argus-uploads" "argus-preprocessed" "argus-results")
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

echo "[$(date -u)] Starting MinIO backup..."

# Configure mc inside the minio container
docker exec -i argus-minio mc alias set src "http://localhost:9000" \
    "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" 2>/dev/null || true

for bucket in "${BUCKETS[@]}"; do
    echo "[$(date -u)] Backing up $bucket..."

    if [ -n "${BACKUP_S3_ENDPOINT:-}" ]; then
        # Remote S3 backup
        docker exec -i argus-minio mc alias set dst "${BACKUP_S3_ENDPOINT}" \
            "${BACKUP_S3_ACCESS_KEY}" "${BACKUP_S3_SECRET_KEY}" 2>/dev/null || true
        docker exec -i argus-minio mc mirror --overwrite "src/$bucket" "dst/$bucket-$TIMESTAMP"
    else
        # Local filesystem backup
        BACKUP_DIR="${LOCAL_BACKUP_DIR:-$PROJECT_ROOT/backups/minio}/$bucket-$TIMESTAMP"
        mkdir -p "$BACKUP_DIR"
        docker exec -i argus-minio mc mirror --overwrite "src/$bucket" "/backup"
        # Note: this requires mounting /backup in the container. For a
        # proper setup, use a remote S3 endpoint or a MinIO replication
        # setup. This script is a starting point.
        echo "  (Local backup mode — configure BACKUP_S3_* for remote backup)"
    fi
done

echo "[$(date -u)] MinIO backup complete."
