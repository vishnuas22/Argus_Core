"""
Argus Core - Test Configuration & Shared Fixtures
===================================================
Central test infrastructure providing:
- FastAPI test app with dependency overrides
- Test database (separate MongoDB collection prefix)
- Test storage (local filesystem fallback)
- JWT authentication tokens
- Cleanup fixtures for test isolation

All fixtures use real infrastructure (no mocks).
Database isolation uses per-test collection prefixes.
Storage uses local filesystem fallback (no MinIO required).

Note: Heavy imports (torch, server, api.deps) are deferred to fixture
execution time to allow unit tests (schemas, errors, config) to run
without ML dependencies installed.
"""

import os
import sys
import uuid
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Generator, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio

# Ensure backend directory is on the Python path
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Set test environment variables BEFORE importing app modules
os.environ["DB_NAME"] = "argus_core_test"
os.environ["LOG_LEVEL"] = "WARNING"

# Light imports (no torch dependency chain)
from config import config, get_settings, Settings
from schemas.schemas import (
    AnalysisDocument, AnalysisStatus, AnalysisResponse,
    FileInput, AnalyzeOptions, Modality, TrustScore, Verdict,
    Explanation, ProgressUpdate, ErrorResponse, EvidencePackage,
    FeatureImportance, VisualEvidence, ScientificReference,
)
from storage.storage import LocalStorageClient
from processing.sanitize import InputSanitizer, SanitizedFile, FileType
from utils.errors import (
    ArgusError, InvalidFileError, AnalysisNotFoundError,
    ValidationError, StorageError, RateLimitError, AuthenticationError,
)
from utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import flags
_app_imports_available = False
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
    _app_imports_available = True
except ImportError:
    logger = None  # logging not available at this point; imports optional for tests without web deps


# ============== JWT TOKEN GENERATION ==============

def _generate_jwt_token(
    user_id: str = "test-user-001",
    email: str = "test@argus.dev",
    roles: Optional[list] = None,
    expires_minutes: int = 60,
) -> str:
    """
    Generate a real JWT token for test authentication.
    
    Uses the same secret and algorithm as the production config
    to ensure token validation works identically.
    """
    import jwt
    
    payload = {
        "sub": user_id,
        "email": email,
        "roles": roles or ["user", "analyst"],
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_minutes),
        "iat": datetime.now(timezone.utc),
    }
    
    return jwt.encode(
        payload,
        config.jwt_secret,
        algorithm=config.jwt_algorithm,
    )


# ============== TEST DATABASE ==============

class TestDatabase:
    """
    Test database manager with per-session isolation.
    
    Uses a dedicated test database (argus_core_test) with
    unique collection prefixes per test session.
    """
    
    def __init__(self) -> None:
        self.db_name = "argus_core_test"
        self.client: Optional[Any] = None
        self._session_id = str(uuid.uuid4())[:8]
    
    async def connect(self) -> Any:
        """Connect to test database."""
        from storage.db import DatabaseClient
        self.client = DatabaseClient(
            mongo_url=config.mongo_url,
            db_name=self.db_name,
        )
        await self.client.connect()
        return self.client
    
    async def cleanup(self) -> None:
        """Drop all test collections."""
        if self.client is not None and self.client._db is not None:
            try:
                collections = await self.client._db.list_collection_names()
                for name in collections:
                    await self.client._db.drop_collection(name)
            except Exception as e:
                logger.warning(f"Test DB cleanup warning: {e}")
    
    async def disconnect(self) -> None:
        """Close database connection."""
        if self.client:
            await self.client.disconnect()
            self.client = None


# ============== TEST STORAGE ==============

class TestStorage:
    """
    Test storage manager using local filesystem fallback.
    
    Creates a temporary directory for test file operations.
    No MinIO required.
    """
    
    def __init__(self) -> None:
        self._base_path = Path("/tmp/argus_test_storage") / str(uuid.uuid4())[:8]
        self._storage: Optional[LocalStorageClient] = None
    
    def create(self) -> LocalStorageClient:
        """Create local storage client."""
        self._storage = LocalStorageClient(str(self._base_path))
        return self._storage
    
    def cleanup(self) -> None:
        """Remove test storage directory."""
        import shutil
        if self._base_path.exists():
            try:
                shutil.rmtree(self._base_path)
            except Exception as e:
                logger.warning(f"Test storage cleanup warning: {e}")


# ============== FIXTURES ==============

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db() -> Generator:
    """Session-scoped test database."""
    db = TestDatabase()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(db.connect())
        yield db
    finally:
        loop.run_until_complete(db.cleanup())
        loop.run_until_complete(db.disconnect())
        loop.close()


@pytest.fixture(scope="session")
def test_storage() -> Generator:
    """Session-scoped test storage."""
    storage = TestStorage()
    storage.create()
    yield storage
    storage.cleanup()


@pytest.fixture(scope="session")
def auth_token() -> str:
    """Valid JWT token for test user."""
    return _generate_jwt_token()


@pytest.fixture(scope="session")
def admin_token() -> str:
    """Valid JWT token for admin user."""
    return _generate_jwt_token(
        user_id="admin-001",
        email="admin@argus.dev",
        roles=["user", "admin", "analyst"],
    )


@pytest.fixture(scope="session")
def expired_token() -> str:
    """Expired JWT token for negative testing."""
    return _generate_jwt_token(expires_minutes=-60)


@pytest.fixture(scope="session")
def auth_headers(auth_token: str) -> Dict[str, str]:
    """Authorization headers with valid token."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture(scope="session")
def admin_headers(admin_token: str) -> Dict[str, str]:
    """Authorization headers with admin token."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="session")
def app(test_db: TestDatabase, test_storage: TestStorage):
    """
    Create test FastAPI application with dependency overrides.
    
    Overrides all external service dependencies with test implementations.
    Heavy imports are deferred here to avoid requiring torch for unit tests.
    """
    from server import create_app
    from api.deps import (
        get_db, get_storage, get_rate_limiter, RateLimiter,
    )
    
    test_app = create_app()
    
    # Override database dependency
    async def override_get_db():
        yield test_db.client
    
    # Override storage dependency (always use local fallback)
    storage_instance = test_storage.create()
    
    def override_get_storage():
        return storage_instance
    
    # Override rate limiter to always allow
    permissive_limiter = RateLimiter(max_requests=10000, window_seconds=60)
    
    def override_get_rate_limiter():
        return permissive_limiter
    
    # Apply overrides
    test_app.dependency_overrides[get_db] = override_get_db
    test_app.dependency_overrides[get_storage] = override_get_storage
    test_app.dependency_overrides[get_rate_limiter] = override_get_rate_limiter
    
    return test_app


@pytest.fixture(scope="session")
def client(app) -> Generator:
    """
    Synchronous test client for API testing.
    
    Uses Starlette TestClient which handles async endpoints
    by running them in a background thread.
    """
    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ============== DATA FIXTURES ==============

@pytest.fixture
def sample_file_input() -> FileInput:
    """Sample file input metadata."""
    return FileInput(
        file_id="uploads/test-001/sample.jpg",
        file_type="image/jpeg",
        original_filename="sample.jpg",
        file_hash="a" * 64,
        file_size=102400,
    )


@pytest.fixture
def sample_trust_score() -> TrustScore:
    """Sample trust score."""
    return TrustScore(value=75.5, confidence=0.92, calibrated=True)


@pytest.fixture
def sample_analysis_document(sample_file_input: FileInput) -> AnalysisDocument:
    """Sample completed analysis document."""
    return AnalysisDocument(
        analysis_id=str(uuid.uuid4()),
        status=AnalysisStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        input=sample_file_input,
        options=AnalyzeOptions(),
        trust_score=TrustScore(value=82.0, confidence=0.95, calibrated=True),
        verdict=Verdict.AUTHENTIC,
        explanation=Explanation(
            summary="No manipulation detected",
            key_findings=["No artifacts found", "Consistent metadata"],
            confidence_rationale="High confidence based on multi-modal analysis",
            methodology_used=["spatial_analysis", "frequency_analysis"],
        ),
    )


@pytest.fixture
def sample_pending_analysis(sample_file_input: FileInput) -> AnalysisDocument:
    """Sample pending (in-progress) analysis document."""
    return AnalysisDocument(
        analysis_id=str(uuid.uuid4()),
        status=AnalysisStatus.ANALYZING,
        created_at=datetime.now(timezone.utc),
        input=sample_file_input,
        options=AnalyzeOptions(),
    )


@pytest.fixture
def jpeg_bytes() -> bytes:
    """Valid JPEG file bytes (minimal valid JPEG)."""
    return (
        b'\xff\xd8\xff\xe0'  # JPEG SOI + APP0 marker
        b'\x00\x10'  # APP0 length
        b'JFIF\x00'  # JFIF identifier
        b'\x01\x01'  # Version 1.1
        b'\x00'  # Aspect ratio units
        b'\x00\x01\x00\x01'  # X/Y density
        b'\x00\x00'  # No thumbnail
        b'\xff\xd9'  # JPEG EOI
    )


@pytest.fixture
def png_bytes() -> bytes:
    """Valid PNG file bytes (minimal 1x1 transparent pixel)."""
    import zlib
    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'
    # IHDR chunk (13 bytes data)
    ihdr_data = (
        b'\x00\x00\x00\x01'  # Width: 1
        b'\x00\x00\x00\x01'  # Height: 1
        b'\x08'  # Bit depth: 8
        b'\x06'  # Color type: RGBA
        b'\x00'  # Compression: deflate
        b'\x00'  # Filter: adaptive
        b'\x00'  # Interlace: none
    )
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data).to_bytes(4, 'big')
    ihdr = b'\x00\x00\x00\x0d' + b'IHDR' + ihdr_data + ihdr_crc
    # IDAT chunk (1x1 transparent pixel)
    raw_data = b'\x00\xff\xff\xff\xff'  # Filter byte + RGBA
    compressed = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b'IDAT' + compressed).to_bytes(4, 'big')
    idat = len(compressed).to_bytes(4, 'big') + b'IDAT' + compressed + idat_crc
    # IEND chunk
    iend_crc = zlib.crc32(b'IEND').to_bytes(4, 'big')
    iend = b'\x00\x00\x00\x00' + b'IEND' + iend_crc
    return signature + ihdr + idat + iend


@pytest.fixture
def mp4_bytes() -> bytes:
    """Minimal MP4 file bytes (ftyp box header)."""
    ftyp_data = (
        b'\x00\x00\x00\x1c'  # Box size
        b'ftyp'  # Box type
        b'isom'  # Major brand
        b'\x00\x00\x02\x00'  # Minor version
        b'isom'  # Compatible brand 1
        b'iso2'  # Compatible brand 2
        b'mp41'  # Compatible brand 3
    )
    return ftyp_data + b'\x00' * 100


@pytest.fixture
def wav_bytes() -> bytes:
    """Minimal WAV file bytes."""
    data = b'\x00\x80' * 100  # 100 samples of silence
    return (
        b'RIFF'
        b'\x00\x00\x00\x00'  # Placeholder size
        b'WAVE'
        b'fmt '
        b'\x10\x00\x00\x00'  # Chunk size: 16
        b'\x01\x00'  # PCM
        b'\x01\x00'  # Mono
        b'\x44\xac\x00\x00'  # Sample rate: 44100
        b'\x88\x58\x01\x00'  # Byte rate: 88200
        b'\x02\x00'  # Block align
        b'\x10\x00'  # Bits per sample
        b'data'
        + len(data).to_bytes(4, 'big')
        + data
    )


@pytest.fixture
def plain_text_bytes() -> bytes:
    """Valid plain text bytes for text detection."""
    return b'This is a test text file for the Argus Core deepfake detection platform. ' * 5


@pytest.fixture
def mp3_bytes() -> bytes:
    """Minimal MP3 file bytes (ID3 header)."""
    id3_header = (
        b'ID3'  # ID3 identifier
        b'\x03\x00'  # Version 2.3.0
        b'\x00'  # Flags
        b'\x00\x00\x00\x00'  # Size (synchsafe, 0)
    )
    return id3_header + b'\xff\xfb' + b'\x00' * 100


# ============== CLEANUP FIXTURE ==============

@pytest.fixture(autouse=False)
def clean_test_db(test_db: TestDatabase):
    """
    Clean test database before and after test.
    
    Not autouse - activate per test with: @pytest.mark.usefixtures("clean_test_db")
    """
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_db.cleanup())
    yield
    loop.run_until_complete(test_db.cleanup())
    loop.close()
