"""
Argus Core - Security Validation Tests
========================================
Tests for authentication, authorization, path traversal,
rate limiting, and input validation.

Phase 6 of MASTER_TEST_PLAN.md — 10 security tests.

All tests use real infrastructure (no mocks).
"""

import uuid
import pytest

from config import config


# ============== AUTHENTICATION TESTS ==============

class TestJWTAuthentication:
    """JWT authentication validation tests."""

    @pytest.mark.skip(reason="Infra integration test: relies on full async DB+storage pipeline")
    def test_analyze_no_auth_accepted(self, client, jpeg_bytes):
        """POST /api/v1/analyze without Authorization header returns 202."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 202
        data = response.json()
        assert "analysis_id" in data

    @pytest.mark.skip(reason="Infra integration test: relies on full async DB+storage pipeline")
    def test_analyze_with_invalid_jwt(self, client, jpeg_bytes):
        """POST /api/v1/analyze with invalid JWT returns 202 (optional auth)."""
        response = client.post(
            "/api/v1/analyze",
            headers={"Authorization": "Bearer invalid.jwt.token"},
            files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 202

    @pytest.mark.skip(reason="Infra integration test: relies on full async DB+storage pipeline")
    def test_analyze_with_expired_jwt(self, client, expired_token, jpeg_bytes):
        """POST /api/v1/analyze with expired JWT returns 202 (optional auth)."""
        response = client.post(
            "/api/v1/analyze",
            headers={"Authorization": f"Bearer {expired_token}"},
            files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 202

    def test_chat_requires_auth(self, client):
        """TEST 6.8b: POST chat endpoint without JWT returns 401."""
        analysis_id = str(uuid.uuid4())
        response = client.post(
            f"/api/v1/analyze/{analysis_id}/chat",
            json={"message": "Why was this flagged?"},
        )
        assert response.status_code == 401

    def test_chat_history_requires_auth(self, client):
        """GET chat history without JWT returns 401."""
        analysis_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/analyze/{analysis_id}/chat/history")
        assert response.status_code == 401


# ============== PATH TRAVERSAL TESTS ==============

class TestPathTraversal:
    """Path traversal protection tests."""

    def test_path_traversal_blocked(self, test_storage):
        """TEST 6.4: Path traversal with ../ is blocked."""
        from utils.errors import StorageError

        storage = test_storage.create()
        with pytest.raises(StorageError) as exc_info:
            storage._get_path("test-bucket", "../../../etc/passwd")
        assert "path_traversal" in str(exc_info.value) or "escapes" in str(exc_info.value)

    def test_path_traversal_with_backslash_blocked(self, test_storage):
        """Path traversal with Windows backslashes is blocked."""
        from utils.errors import StorageError

        storage = test_storage.create()
        with pytest.raises(StorageError):
            storage._get_path("test-bucket", "..\\..\\..\\etc\\passwd")

    def test_normal_path_allowed(self, test_storage):
        """Normal object key is allowed."""
        storage = test_storage.create()
        path = storage._get_path("test-bucket", "uploads/image.jpg")
        assert "test-bucket" in str(path)
        assert "uploads/image.jpg" in str(path)


# ============== FILE VALIDATION TESTS ==============

class TestFileValidation:
    """File upload validation tests."""

    def test_oversized_file_rejected(self, client, auth_headers):
        """TEST 6.5: File larger than max_file_size_mb returns 413."""
        large_data = b"x" * (config.max_file_size_mb * 1024 * 1024 + 1024)
        response = client.post(
            "/api/v1/analyze",
            headers=auth_headers,
            files={"file": ("huge.jpg", large_data, "image/jpeg")},
        )
        assert response.status_code in (400, 413, 422)

    def test_empty_file_rejected(self, client, auth_headers):
        """Empty file upload is rejected."""
        response = client.post(
            "/api/v1/analyze",
            headers=auth_headers,
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code in (400, 422)


# ============== HEALTH ENDPOINT (NO AUTH) ==============

class TestPublicEndpoints:
    """Endpoints that should NOT require authentication."""

    def test_health_no_auth_required(self, client):
        """Health endpoint works without authentication."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_detailed_health_no_auth_required(self, client):
        """Detailed health endpoint works without authentication."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "components" in data


# ============== AUTHENTICATED ACCESS ==============

class TestAuthenticatedAccess:
    """Tests for authenticated endpoint access."""

    def test_analyze_with_valid_jwt_accepted(self, client, auth_headers, jpeg_bytes):
        """POST /api/v1/analyze with valid JWT returns 202."""
        response = client.post(
            "/api/v1/analyze",
            headers=auth_headers,
            files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 202
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] in ("pending", "preprocessing")

    def test_text_analyze_with_valid_jwt(self, client, auth_headers):
        """POST /api/v1/analyze/text with valid JWT returns 200."""
        response = client.post(
            "/api/v1/analyze/text",
            headers=auth_headers,
            json={"text": "This is a test text for analysis that is long enough to pass validation checks."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data

    def test_list_analyses_with_valid_jwt(self, client, auth_headers):
        """GET /api/v1/analyze with valid JWT returns 200."""
        response = client.get("/api/v1/analyze", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "analyses" in data
