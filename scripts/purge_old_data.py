"""
Argus Core — Data Retention Purge
==================================

Deletes old analyses, audit logs, and MinIO objects per the retention
policy. Run daily via Celery Beat or cron.

Retention (configurable via env):
  - analyses: 90 days
  - audit_log: 90 days
  - MinIO uploads: 30 days
  - MinIO preprocessed: 7 days
  - MinIO results: 90 days

Usage:
    python scripts/purge_old_data.py                    # uses defaults
    python scripts/purge_old_data.py --days 30          # override analyses retention
    python scripts/purge_old_data.py --dry-run          # show what would be deleted
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone, timedelta


async def purge_analyses(days: int, dry_run: bool = False) -> int:
    """Delete analyses older than `days`. Returns count deleted."""
    from storage.db import get_db_client
    db = await get_db_client()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    if dry_run:
        count = await db._db.analyses.count_documents({"created_at": {"$lt": cutoff.isoformat()}})
        print(f"[DRY RUN] Would delete {count} analyses older than {days} days")
        return count

    result = await db._db.analyses.delete_many({"created_at": {"$lt": cutoff.isoformat()}})
    print(f"Deleted {result.deleted_count} analyses older than {days} days")
    return result.deleted_count


async def purge_audit_log(days: int, dry_run: bool = False) -> int:
    """Delete audit log entries older than `days`."""
    from storage.db import get_db_client
    db = await get_db_client()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    if dry_run:
        count = await db._db.audit_log.count_documents({"timestamp": {"$lt": cutoff}})
        print(f"[DRY RUN] Would delete {count} audit log entries older than {days} days")
        return count

    result = await db._db.audit_log.delete_many({"timestamp": {"$lt": cutoff}})
    print(f"Deleted {result.deleted_count} audit log entries older than {days} days")
    return result.deleted_count


async def purge_minio(bucket: str, days: int, dry_run: bool = False) -> int:
    """Delete MinIO objects older than `days` in the given bucket."""
    from minio import Minio
    from minio.deleteobjects import DeleteObject
    from config import config

    client = Minio(
        config.minio_endpoint,
        access_key=config.minio_access_key,
        secret_key=config.minio_secret_key,
        secure=config.minio_secure,
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted_count = 0

    objects_to_delete = []
    for obj in client.list_objects(bucket, recursive=True):
        if obj.last_modified.replace(tzinfo=timezone.utc) < cutoff:
            objects_to_delete.append(DeleteObject(obj.object_name))
            deleted_count += 1

    if dry_run:
        print(f"[DRY RUN] Would delete {deleted_count} objects from {bucket} older than {days} days")
        return deleted_count

    if objects_to_delete:
        errors = client.remove_objects(bucket, objects_to_delete)
        for err in errors:
            print(f"Error deleting {err.object_name}: {err.error_message}")

    print(f"Deleted {deleted_count} objects from {bucket} older than {days} days")
    return deleted_count


async def main():
    parser = argparse.ArgumentParser(description="Purge old data per retention policy")
    parser.add_argument("--days", type=int, default=90, help="Analyses retention (days)")
    parser.add_argument("--audit-days", type=int, default=90, help="Audit log retention (days)")
    parser.add_argument("--uploads-days", type=int, default=30, help="MinIO uploads retention (days)")
    parser.add_argument("--preprocessed-days", type=int, default=7, help="MinIO preprocessed retention (days)")
    parser.add_argument("--results-days", type=int, default=90, help="MinIO results retention (days)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()

    print(f"=== Data Retention Purge {'(DRY RUN)' if args.dry_run else ''} ===")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()

    await purge_analyses(args.days, args.dry_run)
    await purge_audit_log(args.audit_days, args.dry_run)

    try:
        await purge_minio("argus-uploads", args.uploads_days, args.dry_run)
        await purge_minio("argus-preprocessed", args.preprocessed_days, args.dry_run)
        await purge_minio("argus-results", args.results_days, args.dry_run)
    except Exception as e:
        print(f"MinIO purge skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
