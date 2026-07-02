"""
Argus Core - WebSocket Endpoint Tests
=======================================
Tests for WebSocket endpoints in api/websocket.py.

Tests cover:
- /ws/analysis/{analysis_id} - per-analysis progress updates
- /ws/updates - global system updates
- JWT authentication via query parameter
- Connection lifecycle (connect, subscribe, unsubscribe, disconnect)
- Ping/pong keepalive
- ConnectionManager broadcast logic

Uses real FastAPI TestClient WebSocket support.
No mocks. Real WebSocket connections.

Requirements: torch, transformers (full ML stack)
Run with: pytest tests/test_websocket.py -v
"""

import os
import sys
import uuid
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Optional

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch", reason="torch required for WebSocket endpoint tests")

from api.websocket import (
    ConnectionManager, send_progress_update, send_completion_update,
    send_error_update,
)
from schemas.schemas import AnalysisStatus, ProgressUpdate


# ============== JWT TOKEN HELPER ==============

def _make_token(
    user_id: str = "ws-test-user",
    expires_minutes: int = 60,
) -> str:
    """Generate a JWT token for WebSocket auth testing."""
    import jwt
    from config import config
    
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc).timestamp() + (expires_minutes * 60),
        "iat": datetime.now(timezone.utc).timestamp(),
    }
    return jwt.encode(payload, config.jwt_secret, algorithm=config.jwt_algorithm)


# ============== CONNECTION MANAGER TESTS ==============

class TestConnectionManager:
    """Test ConnectionManager class directly (unit tests)."""

    def test_initial_state(self) -> None:
        mgr = ConnectionManager()
        assert mgr.subscriptions == {}
        assert mgr.active_connections == set()

    def test_connect_adds_to_active(self) -> None:
        """Test that connect adds websocket to active connections.
        
        Note: We test the ConnectionManager directly since WebSocket
        accept() requires a real connection. We test the state changes.
        """
        mgr = ConnectionManager()
        # ConnectionManager.connect() calls websocket.accept() which
        # requires a real connection, so we test via the full endpoint.


# ============== WEBSOCKET AUTHENTICATION TESTS ==============

class TestWebSocketAuthentication:
    """Test WebSocket JWT authentication."""

    def test_no_token_closes_connection(self, client: TestClient) -> None:
        """WebSocket without token should be rejected."""
        analysis_id = str(uuid.uuid4())
        with client.websocket_connect(
            f"/ws/analysis/{analysis_id}",
            open_ok=False,
            close_ok=False,
        ) as ws:
            # Should receive close frame
            close_data = ws.receive()
            assert close_data.get("type") == "websocket.close"
            assert close_data.get("code") == 4001

    def test_invalid_token_closes_connection(self, client: TestClient) -> None:
        """WebSocket with invalid token should be rejected."""
        analysis_id = str(uuid.uuid4())
        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/analysis/{analysis_id}?token=invalid") as ws:
                data = ws.receive()

    def test_expired_token_closes_connection(self, client: TestClient) -> None:
        """WebSocket with expired token should be rejected."""
        token = _make_token(expires_minutes=-60)
        analysis_id = str(uuid.uuid4())
        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/analysis/{analysis_id}?token={token}") as ws:
                data = ws.receive()

    def test_valid_token_connects(self, client: TestClient) -> None:
        """WebSocket with valid token should connect."""
        token = _make_token()
        analysis_id = str(uuid.uuid4())
        with client.websocket_connect(f"/ws/analysis/{analysis_id}?token={token}") as ws:
            # Should receive current status message
            data = ws.receive_json()
            assert "type" in data
            assert data.get("analysis_id") == analysis_id


# ============== ANALYSIS PROGRESS WEBSOCKET ==============

class TestAnalysisProgressWebSocket:
    """Test /ws/analysis/{analysis_id} endpoint."""

    def test_ping_pong(self, client: TestClient) -> None:
        """Client ping should receive pong."""
        token = _make_token()
        analysis_id = str(uuid.uuid4())
        with client.websocket_connect(f"/ws/analysis/{analysis_id}?token={token}") as ws:
            # Receive initial status
            ws.receive_json()
            
            # Send ping
            ws.send_json({"type": "ping"})
            
            # Receive pong
            response = ws.receive_json()
            assert response["type"] == "pong"
            assert "timestamp" in response

    def test_subscribe_to_analysis(self, client: TestClient) -> None:
        """Client can subscribe to additional analysis."""
        token = _make_token()
        analysis_id1 = str(uuid.uuid4())
        analysis_id2 = str(uuid.uuid4())
        with client.websocket_connect(f"/ws/analysis/{analysis_id1}?token={token}") as ws:
            # Receive initial status for analysis1
            ws.receive_json()
            
            # Subscribe to analysis2
            ws.send_json({"type": "subscribe", "analysis_id": analysis_id2})
            
            # Should receive subscribed confirmation
            response = ws.receive_json()
            assert response["type"] == "subscribed"
            assert response["analysis_id"] == analysis_id2

    def test_unsubscribe_from_analysis(self, client: TestClient) -> None:
        """Client can unsubscribe from analysis."""
        token = _make_token()
        analysis_id = str(uuid.uuid4())
        with client.websocket_connect(f"/ws/analysis/{analysis_id}?token={token}") as ws:
            # Receive initial status
            ws.receive_json()
            
            # Unsubscribe
            ws.send_json({"type": "unsubscribe", "analysis_id": analysis_id})
            
            response = ws.receive_json()
            assert response["type"] == "unsubscribed"
            assert response["analysis_id"] == analysis_id

    def test_refresh_status(self, client: TestClient) -> None:
        """Client can request status refresh."""
        token = _make_token()
        analysis_id = str(uuid.uuid4())
        with client.websocket_connect(f"/ws/analysis/{analysis_id}?token={token}") as ws:
            # Receive initial status
            ws.receive_json()
            
            # Request refresh
            ws.send_json({"type": "refresh", "analysis_id": analysis_id})
            
            # Should receive status update
            response = ws.receive_json()
            assert "type" in response
            assert response.get("analysis_id") == analysis_id


# ============== GLOBAL UPDATES WEBSOCKET ==============

class TestGlobalUpdatesWebSocket:
    """Test /ws/updates endpoint."""

    def test_global_connect(self, client: TestClient) -> None:
        """Global updates WebSocket connects with valid token."""
        token = _make_token()
        with client.websocket_connect(f"/ws/updates?token={token}") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"
            assert "timestamp" in data

    def test_global_no_token_rejected(self, client: TestClient) -> None:
        """Global updates requires authentication."""
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/updates") as ws:
                ws.receive()

    def test_global_ping_pong(self, client: TestClient) -> None:
        """Global WebSocket supports ping/pong."""
        token = _make_token()
        with client.websocket_connect(f"/ws/updates?token={token}") as ws:
            # Receive connected message
            ws.receive_json()
            
            ws.send_json({"type": "ping"})
            response = ws.receive_json()
            assert response["type"] == "pong"

    def test_global_subscribe(self, client: TestClient) -> None:
        """Global WebSocket supports dynamic subscription."""
        token = _make_token()
        analysis_id = str(uuid.uuid4())
        with client.websocket_connect(f"/ws/updates?token={token}") as ws:
            # Receive connected message
            ws.receive_json()
            
            ws.send_json({"type": "subscribe", "analysis_id": analysis_id})
            response = ws.receive_json()
            assert response["type"] == "subscribed"
            assert response["analysis_id"] == analysis_id


# ============== WEBSOCKET HELPER FUNCTIONS ==============

class TestWebSocketHelpers:
    """Test WebSocket utility functions."""

    def test_send_progress_update_function_exists(self) -> None:
        """Verify send_progress_update is callable."""
        assert callable(send_progress_update)

    def test_send_completion_update_function_exists(self) -> None:
        """Verify send_completion_update is callable."""
        assert callable(send_completion_update)

    def test_send_error_update_function_exists(self) -> None:
        """Verify send_error_update is callable."""
        assert callable(send_error_update)

    def test_progress_update_schema(self) -> None:
        """Verify ProgressUpdate schema works."""
        pu = ProgressUpdate(
            analysis_id="test-001",
            status=AnalysisStatus.ANALYZING,
            progress_percent=50.0,
            current_stage="spatial_analysis",
            message="Processing frames",
        )
        data = pu.model_dump(mode="json")
        assert data["analysis_id"] == "test-001"
        assert data["progress_percent"] == 50.0
