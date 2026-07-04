#!/usr/bin/env bash
# ============================================================
# Argus Core — MongoDB Restore Script
# ============================================================
# Restores MongoDB from a MinIO backup.
#
# Usage:
#   # List available backups:
#   ./scripts/restore_mongodb.sh --list
#
#   # Restore latest:
#   ./scripts/restore_mongodb.sh --latest
#
#   # Restore specific backup:
#   ./scripts/restore_mongodb.sh --file argus_mongodb_20260702T120000Z.tar.gz
#
# WARNING: This OVERWRITES the current database.
# Always test on a staging instance first.
# ============================================================
set -euo pipeFail

# Load .env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

: "${MONGO_USER:?MONGO_USER must be set}"
: "${MONGO_PASSWORD:?MONGO_PASSWORD must be set}"
: "${DB_NAME:=argus_core}"
: "${MINIO_ACCESS_KEY:?MINIO_ACCESS_KEY must be set}"
: "${MINIO_SECRET_KEY:?MINIO_SECRET_KEY must be set}"
: "${MINIO_ENDPOINT:?MINIO_ENDPOINT must be set}"
: "${BUCKET:=argus-backups}"

export AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY"
export AWS_ENDPOINT_URL="http://${MINIO_ENDPOINT}"

list_backups() {
    echo "Available MongoDB backups in s3://$BUCKET/mongodb/:"
    aws s3 ls "s3://$BUCKET/mongodb/" \
        --endpoint-url "http://${MINIO_ENDPOINT}" \
        | awk '{print $4}' | sort -r
}

restore_file() {
    local file="$1"
    local archive="/tmp/argus_restore_$$_$(basename "$file")"
    local restore_dir="/tmp/mongodb_restore_$$"

    echo "[$(date -u)] Downloading $file..."
    aws s3 cp "s3://$BUCKET/mongodb/$file" "$archive" \
        --endpoint-url "http://${MINIO_ENDPOINT}"

    echo "[$(date -u)] Extracting to $restore_dir..."
    mkdir -p "$restore_dir"
    tar -xzf "$archive" -C "$restore_dir"

    # Confirm before overwriting
    echo ""
    echo "========================================"
    echo "  WARNING: ABOUT TO OVERWRITE DATABASE"
    echo "========================================"
    echo "  Target:   $DB_NAME on mongodb:27017"
    echo "  Source:   $file"
    echo "  This will DROP existing data in $DB_NAME."
    echo ""
    read -p "Type 'CONFIRM' to proceed: " confirm
    if [ "$confirm" != "CONFIRM" ]; then
        echo "Aborted."
        rm -rf "$restore_dir" "$archive"
        exit 1
    fi

    echo "[$(date -u)] Dropping existing $DB_NAME..."
    mongosh \
        --uri "mongodb://${MONGO_USER}:${MONGO_PASSWORD}@mongodb:27017/?authSource=admin" \
        --eval "db.getSiblingDB('${DB_NAME}').dropDatabase()"

    echo "[$(date -u)] Restoring from $restore_dir..."
    mongorestore \
        --uri "mongodb://${MONGO_USER}:${MONGO_PASSWORD}@mongodb:27017/?authSource=admin" \
        --db "$DB_NAME" \
        --drop \
        "$restore_dir/$DB_NAME" \
        --gzip

    rm -rf "$restore_dir" "$archive"
    echo "[$(date -u)] Restore complete."
}

# Parse args
case "${1:-}" in
    --list)
        list_backups
        ;;
    --latest)
        latest=$(list_backups | head -1)
        if [ -z "$latest" ]; then
            echo "No backups found."
            exit 1
        fi
        echo "Latest backup: $latest"
        restore_file "$latest"
        ;;
    --file)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 --file <filename>"
            exit 1
        fi
        restore_file "$2"
        ;;
    *)
        echo "Usage: $0 [--list | --latest | --file <filename>]"
        echo ""
        echo "Available backups:"
        list_backups
        exit 1
        ;;
esac
