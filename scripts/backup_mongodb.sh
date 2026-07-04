#!/usr/bin/env bash
# ============================================================
# Argus Core — MongoDB Backup Script
# ============================================================
# Creates a compressed mongodump and uploads to MinIO.
# Run daily via cron or Celery Beat.
#
# Usage:
#   ./scripts/backup_mongodb.sh
#
# Env vars (from .env):
#   MONGO_USER, MONGO_PASSWORD, MONGO_URL, DB_NAME
#   MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_ENDPOINT
#
# Retention: 30 daily backups kept in MinIO.
# Restore: see scripts/restore_mongodb.sh
# ============================================================
set -euo pipefail

# Load .env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Required vars
: "${MONGO_USER:?MONGO_USER must be set}"
: "${MONGO_PASSWORD:?MONGO_PASSWORD must be set}"
: "${DB_NAME:=argus_core}"
: "${MINIO_ACCESS_KEY:?MINIO_ACCESS_KEY must be set}"
: "${MINIO_SECRET_KEY:?MINIO_SECRET_KEY must be set}"
: "${MINIO_ENDPOINT:?MINIO_ENDPOINT must be set}"

BACKUP_DIR="/tmp/mongodb_backup_$$"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
ARCHIVE="/tmp/argus_mongodb_${TIMESTAMP}.tar.gz"
RETENTION_DAYS=30
BUCKET="argus-backups"
OBJECT="mongodb/argus_mongodb_${TIMESTAMP}.tar.gz"

echo "[$(date -u)] Starting MongoDB backup..."

# 1. Dump the database
mkdir -p "$BACKUP_DIR"
echo "[$(date -u)] Dumping $DB_NAME to $BACKUP_DIR..."
mongodump \
    --uri="mongodb://${MONGO_USER}:${MONGO_PASSWORD}@mongodb:27017/${DB_NAME}?authSource=admin" \
    --out="$BACKUP_DIR" \
    --gzip

# 2. Compress
echo "[$(date -u)] Compressing to $ARCHIVE..."
tar -czf "$ARCHIVE" -C "$BACKUP_DIR" .

# 3. Upload to MinIO
echo "[$(date -u)] Uploading to MinIO s3://$BUCKET/$OBJECT..."
export AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY"
export AWS_ENDPOINT_URL="http://${MINIO_ENDPOINT}"

# Ensure bucket exists
aws s3 mb "s3://$BUCKET" 2>/dev/null || true

# Upload
aws s3 cp "$ARCHIVE" "s3://$BUCKET/$OBJECT" \
    --endpoint-url "http://${MINIO_ENDPOINT}"

# 4. Set lifecycle policy (delete after RETENTION_DAYS)
# Run once; idempotent
cat > /tmp/lifecycle.json << EOF
{
  "Rules": [
    {
      "ID": "DeleteOldBackups",
      "Status": "Enabled",
      "Filter": {"Prefix": "mongodb/"},
      "Expiration": {"Days": ${RETENTION_DAYS}}
    }
  ]
}
EOF
aws s3api put-bucket-lifecycle-configuration \
    --bucket "$BUCKET" \
    --lifecycle-configuration file:///tmp/lifecycle.json \
    --endpoint-url "http://${MINIO_ENDPOINT}" 2>/dev/null || true

# 5. Cleanup local files
rm -rf "$BACKUP_DIR" "$ARCHIVE" /tmp/lifecycle.json

# 6. List recent backups
echo "[$(date -u)] Recent backups in s3://$BUCKET/mongodb/:"
aws s3 ls "s3://$BUCKET/mongodb/" \
    --endpoint-url "http://${MINIO_ENDPOINT}" \
    | tail -5

echo "[$(date -u)] MongoDB backup complete."
