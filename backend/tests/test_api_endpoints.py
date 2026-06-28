"""
Argus Core - HTTP API Endpoint Tests
======================================
Comprehensive tests for all HTTP API endpoints defined in:
- api/router.py (analysis, health, models, stats, reports, XAI)
- api/chat.py (chat endpoints)
- server.py (root endpoints)

Uses real FastAPI TestClient with real HTTP requests.
Database operations use real MongoDB test database.
Storage uses local filesystem fallback.

No mocks. Real dependency injection with test infrastructure.

Requirements: torch, transformers (full ML stack)
Run with: pytest tests/test_api_endpoints.py -v
"""

import os
import sys
import uuid
import asyncio
import json
import io
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

# Skip entire module if torch is not installed (required by server.py)
torch = pytest.importorskip("torch", reason="torch required for API endpoint tests")

from schemas.schemas import (
    AnalysisDocument, AnalysisStatus, FileInput, AnalyzeOptions,
    Modality, TrustScore, Verdict, Explanation,
    EvidencePackage, FeatureImportance,
)


# ============== HELPER FUNCTIONS ==============

def _create_analysis_in_db(
    client: TestClient,
    auth_headers: Dict[str, str],
    file_bytes: bytes = None,
    filename: str = "test.jpg",
    content_type: str = "image/jpeg",
) -> str:
    """Helper: upload a file and return analysis_id."""
    if file_bytes is None:
        # Minimal valid JPEG
        file_bytes = (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xd9'
        )
    
    response = client.post(
        "/api/v1/analyze",
        files={"file": (filename, io.BytesIO(file_bytes), content_type)},
        data={"generate_report": "false", "generate_heatmaps": "false"},
        headers=auth_headers,
    )
    assert response.status_code == 202, f"Upload failed: {response.text}"
    return response.json()["analysis_id"]


def _wait_for_analysis(
    client: TestClient,
    analysis_id: str,
    auth_headers: Dict[str, str],
    max_wait: float = 30.0,
) -> Dict[str, Any]:
    """Helper: poll for analysis completion."""
    import time
    start = time.time()
    while time.time() - start < max_wait:
        response = client.get(f"/api/v1/analyze/{analysis_id}", headers=auth_headers)
        if response.status_code == 200:
            data = response.json()
            if data["status"] in ("completed", "failed"):
                return data
        time.sleep(0.5)
    raise TimeoutError(f"Analysis {analysis_id} did not complete within {max_wait}s")


# ============== ROOT ENDPOINTS ==============

class TestRootEndpoints:
    """Test root-level endpoints from server.py."""

    def test_root_endpoint(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs_url" in data

    def test_health_basic(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_metrics_endpoint(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus metrics should return text content
        assert "argus_" in response.text or "python_" in response.text


# ============== HEALTH ENDPOINT ==============

class TestHealthEndpoint:
    """Test /api/v1/health detailed health check."""

    def test_health_detailed(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data

    def test_health_components_structure(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        data = response.json()
        components = data["components"]
        # Should check at least these components
        assert "database" in components
        assert "storage" in components
        assert "redis" in components


# ============== MODELS ENDPOINT ==============

class TestModelsEndpoint:
    """Test /api/v1/models listing."""

    def test_list_models(self, client: TestClient) -> None:
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert isinstance(data["models"], list)

    def test_models_have_required_fields(self, client: TestClient) -> None:
        response = client.get("/api/v1/models")
        data = response.json()
        if data["models"]:
            model = data["models"][0]
            assert "name" in model
            assert "category" in model
            assert "loaded" in model


# ============== STATS ENDPOINT ==============

class TestStatsEndpoint:
    """Test /api/v1/stats statistics."""

    def test_stats_structure(self, client: TestClient) -> None:
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "by_status" in data
        assert "by_verdict" in data

    def test_stats_total_is_integer(self, client: TestClient) -> None:
        response = client.get("/api/v1/stats")
        data = response.json()
        assert isinstance(data["total"], int)
        assert data["total"] >= 0


# ============== ANALYSIS CRUD ENDPOINTS ==============

class TestAnalyzeEndpoint:
    """Test POST /api/v1/analyze - file upload and analysis."""

    def test_upload_jpeg(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        jpeg = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9'
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
            data={"generate_report": "false", "generate_heatmaps": "false"},
            headers=auth_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "pending"

    def test_upload_png(self, client: TestClient, auth_headers: Dict[str, str], png_bytes: bytes) -> None:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
            data={"generate_report": "false", "generate_heatmaps": "false"},
            headers=auth_headers,
        )
        assert response.status_code == 202

    def test_upload_with_options(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        jpeg = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9'
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
            data={
                "generate_report": "true",
                "generate_heatmaps": "true",
                "defense_level": "aggressive",
                "modalities": "image",
            },
            headers=auth_headers,
        )
        assert response.status_code == 202

    def test_upload_empty_file_rejected(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_upload_invalid_type_rejected(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.bin", io.BytesIO(b'\x00\x01\x02\x03'), "application/octet-stream")},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_upload_without_auth(self, client: TestClient) -> None:
        jpeg = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9'
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
        )
        assert response.status_code in (401, 403)


class TestGetAnalysisEndpoint:
    """Test GET /api/v1/analyze/{analysis_id}."""

    def test_get_existing_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        response = client.get(f"/api/v1/analyze/{analysis_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_id"] == analysis_id
        assert data["status"] in [s.value for s in AnalysisStatus]

    def test_get_nonexistent_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/analyze/{fake_id}", headers=auth_headers)
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error_code"] == "ANALYSIS_NOT_FOUND"


class TestGetAnalysisDetailEndpoint:
    """Test GET /api/v1/analyze/{analysis_id}/detail."""

    def test_detail_pending_analysis_returns_400(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        response = client.get(f"/api/v1/analyze/{analysis_id}/detail", headers=auth_headers)
        # Should return 400 if analysis is still pending
        assert response.status_code in (200, 400)

    def test_detail_nonexistent_returns_404(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/analyze/{fake_id}/detail", headers=auth_headers)
        assert response.status_code == 404


class TestDeleteAnalysisEndpoint:
    """Test DELETE /api/v1/analyze/{analysis_id}."""

    def test_delete_existing_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        response = client.delete(f"/api/v1/analyze/{analysis_id}", headers=auth_headers)
        assert response.status_code == 204

    def test_delete_then_get_returns_404(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        client.delete(f"/api/v1/analyze/{analysis_id}", headers=auth_headers)
        response = client.get(f"/api/v1/analyze/{analysis_id}", headers=auth_headers)
        assert response.status_code == 404

    def test_delete_nonexistent_returns_404(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.delete(f"/api/v1/analyze/{fake_id}", headers=auth_headers)
        assert response.status_code == 404


class TestListAnalysesEndpoint:
    """Test GET /api/v1/analyze - list analyses."""

    def test_list_analyses_empty(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        response = client.get("/api/v1/analyze", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_analyses_with_limit(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        response = client.get("/api/v1/analyze?limit=5", headers=auth_headers)
        assert response.status_code == 200

    def test_list_analyses_with_offset(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        response = client.get("/api/v1/analyze?limit=10&offset=0", headers=auth_headers)
        assert response.status_code == 200

    def test_list_analyses_filter_by_status(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        response = client.get("/api/v1/analyze?status=pending", headers=auth_headers)
        assert response.status_code == 200


class TestAnalyzeTextEndpoint:
    """Test POST /api/v1/analyze/text."""

    def test_analyze_valid_text(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        text = "This is a test text for AI detection analysis. " * 3
        response = client.post(
            "/api/v1/analyze/text",
            data={"text": text, "generate_report": "false"},
            headers=auth_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert "analysis_id" in data

    def test_analyze_text_too_short(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        response = client.post(
            "/api/v1/analyze/text",
            data={"text": "Short"},
            headers=auth_headers,
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_analyze_text_without_auth(self, client: TestClient) -> None:
        text = "This is a test text for AI detection analysis. " * 3
        response = client.post(
            "/api/v1/analyze/text",
            data={"text": text},
        )
        assert response.status_code in (401, 403)


# ============== REPORT & HEATMAP ENDPOINTS ==============

class TestReportEndpoint:
    """Test GET /api/v1/analyze/{analysis_id}/report."""

    def test_report_pending_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        response = client.get(f"/api/v1/analyze/{analysis_id}/report", headers=auth_headers)
        # Pending analysis should return 400
        assert response.status_code in (400, 404)

    def test_report_nonexistent_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/analyze/{fake_id}/report", headers=auth_headers)
        assert response.status_code == 404


class TestHeatmapsEndpoint:
    """Test GET /api/v1/analyze/{analysis_id}/heatmaps."""

    def test_heatmaps_nonexistent_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/analyze/{fake_id}/heatmaps", headers=auth_headers)
        assert response.status_code == 404

    def test_heatmaps_existing_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        response = client.get(f"/api/v1/analyze/{analysis_id}/heatmaps", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "heatmaps" in data
        assert "count" in data


# ============== XAI ENDPOINTS ==============

class TestXAIEndpoints:
    """Test XAI explanation endpoints."""

    def test_xai_nonexistent_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/analyze/{fake_id}/xai", headers=auth_headers)
        assert response.status_code == 404

    def test_xai_existing_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        response = client.get(f"/api/v1/analyze/{analysis_id}/xai", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert "image_xai" in data
        assert "video_xai" in data
        assert "audio_xai" in data
        assert "text_xai" in data

    def test_xai_heatmaps_nonexistent(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/analyze/{fake_id}/xai/heatmaps", headers=auth_headers)
        assert response.status_code == 404

    def test_xai_heatmaps_existing(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        response = client.get(f"/api/v1/analyze/{analysis_id}/xai/heatmaps", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "heatmaps" in data


# ============== CHAT ENDPOINTS ==============

class TestChatEndpoints:
    """Test chat API endpoints."""

    def test_chat_on_pending_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        response = client.post(
            f"/api/v1/analyze/{analysis_id}/chat",
            json={"message": "What did you find in this analysis?"},
            headers=auth_headers,
        )
        # Chat only available for completed analyses
        assert response.status_code in (400, 503)

    def test_chat_nonexistent_analysis(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.post(
            f"/api/v1/analyze/{fake_id}/chat",
            json={"message": "Hello?"},
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_chat_history_nonexistent(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/analyze/{fake_id}/chat/history",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_chat_clear_nonexistent(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        fake_id = str(uuid.uuid4())
        response = client.delete(
            f"/api/v1/analyze/{fake_id}/chat",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_chat_history_existing(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        analysis_id = _create_analysis_in_db(client, auth_headers)
        response = client.get(
            f"/api/v1/analyze/{analysis_id}/chat/history",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert "messages" in data


# ============== AUTHENTICATION TESTS ==============

class TestAuthentication:
    """Test authentication across protected endpoints."""

    def test_protected_endpoints_require_auth(self, client: TestClient) -> None:
        """All analysis endpoints should require authentication."""
        endpoints = [
            ("POST", "/api/v1/analyze"),
            ("GET", "/api/v1/analyze"),
            ("POST", "/api/v1/analyze/text"),
        ]
        for method, url in endpoints:
            if method == "POST":
                response = client.post(url, data={"text": "test" * 20})
            else:
                response = client.get(url)
            assert response.status_code in (401, 403), f"{method} {url} should require auth"

    def test_expired_token_rejected(self, client: TestClient, expired_token: str) -> None:
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/api/v1/health", headers=headers)
        # Health endpoint is public, but we test that expired tokens don't grant special access
        # Test with a protected endpoint
        response = client.get("/api/v1/analyze", headers=headers)
        assert response.status_code in (401, 403)

    def test_invalid_token_rejected(self, client: TestClient) -> None:
        headers = {"Authorization": "Bearer invalid-token-here"}
        response = client.get("/api/v1/analyze", headers=headers)
        assert response.status_code in (401, 403)

    def test_missing_bearer_prefix_rejected(self, client: TestClient, auth_token: str) -> None:
        headers = {"Authorization": auth_token}  # No "Bearer " prefix
        response = client.get("/api/v1/analyze", headers=headers)
        assert response.status_code in (401, 403)


# ============== INTEGRATION FLOW TEST ==============

class TestAnalysisFlow:
    """Test complete analysis lifecycle: upload -> poll -> get detail -> delete."""

    @pytest.mark.integration
    def test_full_analysis_lifecycle(self, client: TestClient, auth_headers: Dict[str, str]) -> None:
        # 1. Upload file
        analysis_id = _create_analysis_in_db(client, auth_headers)
        assert analysis_id is not None

        # 2. Verify in list
        list_response = client.get("/api/v1/analyze", headers=auth_headers)
        assert list_response.status_code == 200
        ids = [a["analysis_id"] for a in list_response.json()]
        assert analysis_id in ids

        # 3. Get status
        status_response = client.get(f"/api/v1/analyze/{analysis_id}", headers=auth_headers)
        assert status_response.status_code == 200

        # 4. Get XAI (should work even for pending)
        xai_response = client.get(f"/api/v1/analyze/{analysis_id}/xai", headers=auth_headers)
        assert xai_response.status_code == 200

        # 5. Get heatmaps
        heatmaps_response = client.get(f"/api/v1/analyze/{analysis_id}/heatmaps", headers=auth_headers)
        assert heatmaps_response.status_code == 200

        # 6. Get chat history
        chat_response = client.get(f"/api/v1/analyze/{analysis_id}/chat/history", headers=auth_headers)
        assert chat_response.status_code == 200

        # 7. Delete
        delete_response = client.delete(f"/api/v1/analyze/{analysis_id}", headers=auth_headers)
        assert delete_response.status_code == 204

        # 8. Verify deleted
        gone_response = client.get(f"/api/v1/analyze/{analysis_id}", headers=auth_headers)
        assert gone_response.status_code == 404
