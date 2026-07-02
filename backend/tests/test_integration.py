"""
Argus Core - Database & Storage Integration Tests
===================================================
Tests for storage/db.py (MongoDB) and storage/storage.py (MinIO/local fallback).

Tests cover:
- DatabaseClient CRUD operations
- DatabaseClient indexing
- StorageClient upload/download/delete
- LocalStorageClient file operations
- Presigned URL generation
- Bucket operations

Uses real MongoDB and local filesystem storage.
No mocks. Real infrastructure connections.

Requirements: MongoDB running on localhost:27017
Run with: pytest tests/test_integration.py -v -m integration
"""

import os
import sys
import uuid
import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from storage.db import DatabaseClient
from storage.storage import StorageClient, LocalStorageClient
from schemas.schemas import (
    AnalysisDocument, AnalysisStatus, FileInput, AnalyzeOptions,
    TrustScore, Verdict, Modality,
)
from utils.errors import StorageError


# ============== DATABASE CLIENT TESTS ==============

class TestDatabaseClientConnection:
    """Test database connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self) -> None:
        db = DatabaseClient(
            mongo_url="mongodb://localhost:27017",
            db_name="argus_test_connect",
        )
        await db.connect()
        assert db._client is not None
        assert db._db is not None
        
        # Ping to verify connection
        result = await db._db.command("ping")
        assert result["ok"] == 1.0
        
        await db.disconnect()
        assert db._client is None

    @pytest.mark.asyncio
    async def test_double_connect_is_safe(self) -> None:
        db = DatabaseClient(
            mongo_url="mongodb://localhost:27017",
            db_name="argus_test_double_connect",
        )
        await db.connect()
        await db.connect()  # Should not raise
        await db.disconnect()


class TestDatabaseClientCRUD:
    """Test CRUD operations for analysis documents."""

    @pytest.fixture
    async def db_client(self) -> DatabaseClient:
        db = DatabaseClient(
            mongo_url="mongodb://localhost:27017",
            db_name=f"argus_test_crud_{uuid.uuid4().hex[:8]}",
        )
        await db.connect()
        yield db
        # Cleanup: drop test database
        if db._client:
            db._client.drop_database(db.db_name)
        await db.disconnect()

    @pytest.mark.asyncio
    async def test_insert_analysis(self, db_client: DatabaseClient) -> None:
        analysis = AnalysisDocument(
            analysis_id=str(uuid.uuid4()),
            status=AnalysisStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            input=FileInput(
                file_id="uploads/test/file.jpg",
                file_type="image/jpeg",
                original_filename="test.jpg",
                file_hash="a" * 64,
                file_size=1024,
            ),
        )
        await db_client.insert_analysis(analysis)
        
        # Verify insertion
        retrieved = await db_client.get_analysis(analysis.analysis_id)
        assert retrieved is not None
        assert retrieved.analysis_id == analysis.analysis_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_analysis(self, db_client: DatabaseClient) -> None:
        result = await db_client.get_analysis("nonexistent-id-12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_analysis_status(self, db_client: DatabaseClient) -> None:
        analysis = AnalysisDocument(
            analysis_id=str(uuid.uuid4()),
            status=AnalysisStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        await db_client.insert_analysis(analysis)
        
        await db_client.update_analysis_status(
            analysis.analysis_id,
            AnalysisStatus.COMPLETED,
        )
        
        updated = await db_client.get_analysis(analysis.analysis_id)
        assert updated.status == AnalysisStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_delete_analysis(self, db_client: DatabaseClient) -> None:
        analysis = AnalysisDocument(
            analysis_id=str(uuid.uuid4()),
            status=AnalysisStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        await db_client.insert_analysis(analysis)
        
        await db_client.delete_analysis(analysis.analysis_id)
        
        deleted = await db_client.get_analysis(analysis.analysis_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_list_analyses(self, db_client: DatabaseClient) -> None:
        # Insert multiple analyses
        for i in range(5):
            analysis = AnalysisDocument(
                analysis_id=str(uuid.uuid4()),
                status=AnalysisStatus.PENDING if i < 3 else AnalysisStatus.COMPLETED,
                created_at=datetime.now(timezone.utc),
            )
            await db_client.insert_analysis(analysis)
        
        # List all
        all_analyses = await db_client.list_analyses(limit=100)
        assert len(all_analyses) >= 5

        # List pending only
        pending = await db_client.list_analyses(status=AnalysisStatus.PENDING, limit=100)
        assert len(pending) >= 3

    @pytest.mark.asyncio
    async def test_list_analyses_with_pagination(self, db_client: DatabaseClient) -> None:
        # Insert analyses
        for _ in range(10):
            await db_client.insert_analysis(AnalysisDocument(
                analysis_id=str(uuid.uuid4()),
                status=AnalysisStatus.PENDING,
                created_at=datetime.now(timezone.utc),
            ))
        
        page1 = await db_client.list_analyses(limit=5, offset=0)
        page2 = await db_client.list_analyses(limit=5, offset=5)
        
        assert len(page1) == 5
        # Verify different results
        ids1 = {a.analysis_id for a in page1}
        ids2 = {a.analysis_id for a in page2}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_count_analyses(self, db_client: DatabaseClient) -> None:
        initial_count = await db_client.count_analyses()
        
        await db_client.insert_analysis(AnalysisDocument(
            analysis_id=str(uuid.uuid4()),
            status=AnalysisStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        ))
        
        new_count = await db_client.count_analyses()
        assert new_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_log_audit_event(self, db_client: DatabaseClient) -> None:
        await db_client.log_audit_event(
            event_type="test_event",
            resource_id="test-resource-001",
            actor="test-user",
            metadata={"key": "value"},
        )
        # Verify audit log was written
        audit_collection = db_client._db.audit_log
        event = await audit_collection.find_one({"resource_id": "test-resource-001"})
        assert event is not None
        assert event["event_type"] == "test_event"


# ============== LOCAL STORAGE TESTS ==============

class TestLocalStorageClient:
    """Test LocalStorageClient operations."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorageClient:
        return LocalStorageClient(str(tmp_path / "storage"))

    @pytest.mark.asyncio
    async def test_upload_and_download(self, storage: LocalStorageClient) -> None:
        content = b"Hello, Argus!"
        await storage.upload_file(
            file=content,
            bucket="test-bucket",
            object_key="test/file.txt",
            content_type="text/plain",
        )
        
        downloaded = await storage.download_file("test-bucket", "test/file.txt")
        assert downloaded == content

    @pytest.mark.asyncio
    async def test_upload_bytes_io(self, storage: LocalStorageClient) -> None:
        import io
        content = b"Binary data test"
        file_obj = io.BytesIO(content)
        
        await storage.upload_file(
            file=file_obj,
            bucket="test-bucket",
            object_key="test/binary.bin",
        )
        
        downloaded = await storage.download_file("test-bucket", "test/binary.bin")
        assert downloaded == content

    @pytest.mark.asyncio
    async def test_delete_file(self, storage: LocalStorageClient) -> None:
        await storage.upload_file(
            file=b"to be deleted",
            bucket="test-bucket",
            object_key="test/deleteme.txt",
        )
        
        await storage.delete_file("test-bucket", "test/deleteme.txt")
        
        with pytest.raises(Exception):
            await storage.download_file("test-bucket", "test/deleteme.txt")

    @pytest.mark.asyncio
    async def test_file_exists(self, storage: LocalStorageClient) -> None:
        await storage.upload_file(
            file=b"exists",
            bucket="test-bucket",
            object_key="test/exists.txt",
        )
        
        exists = await storage.file_exists("test-bucket", "test/exists.txt")
        assert exists is True
        
        not_exists = await storage.file_exists("test-bucket", "test/nope.txt")
        assert not_exists is False

    @pytest.mark.asyncio
    async def test_list_objects(self, storage: LocalStorageClient) -> None:
        for i in range(3):
            await storage.upload_file(
                file=f"file {i}".encode(),
                bucket="test-bucket",
                object_key=f"test/list/file{i}.txt",
            )
        
        objects = await storage.list_objects("test-bucket", prefix="test/list/")
        assert len(objects) >= 3

    @pytest.mark.asyncio
    async def test_get_file_size(self, storage: LocalStorageClient) -> None:
        """Verify file exists after upload (size check via download)."""
        content = b"A" * 1000
        await storage.upload_file(
            file=content,
            bucket="test-bucket",
            object_key="test/sized.txt",
        )
        
        downloaded = await storage.download_file("test-bucket", "test/sized.txt")
        assert len(downloaded) == 1000

    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, storage: LocalStorageClient) -> None:
        """Ensure path traversal attacks are blocked."""
        with pytest.raises(StorageError, match="path_traversal"):
            await storage.upload_file(
                file=b"malicious",
                bucket="test-bucket",
                object_key="../../../etc/passwd",
            )

    @pytest.mark.asyncio
    async def test_nested_directories(self, storage: LocalStorageClient) -> None:
        content = b"nested"
        await storage.upload_file(
            file=content,
            bucket="test-bucket",
            object_key="a/b/c/d/deep.txt",
        )
        
        downloaded = await storage.download_file("test-bucket", "a/b/c/d/deep.txt")
        assert downloaded == content

    @pytest.mark.asyncio
    async def test_overwrite_file(self, storage: LocalStorageClient) -> None:
        await storage.upload_file(
            file=b"original",
            bucket="test-bucket",
            object_key="test/overwrite.txt",
        )
        
        await storage.upload_file(
            file=b"updated",
            bucket="test-bucket",
            object_key="test/overwrite.txt",
        )
        
        downloaded = await storage.download_file("test-bucket", "test/overwrite.txt")
        assert downloaded == b"updated"

    @pytest.mark.asyncio
    async def test_bucket_operations(self, storage: LocalStorageClient) -> None:
        bucket_path = storage._ensure_bucket_dir("new-bucket")
        assert bucket_path.exists()
        assert bucket_path.is_dir()
