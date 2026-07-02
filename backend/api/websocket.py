"""
Argus Core - WebSocket Handlers
===============================
Real-time analysis progress updates via WebSocket.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - api/websocket.py

SOTA Algorithm: Pub/Sub pattern via Redis

Role: WebSocket handlers for real-time analysis progress updates.

Integration:
- Imports: storage/db.py, config.py
- Inputs: WebSocket connections, analysis_id subscriptions
- Outputs: Real-time progress events (JSON)

Why this approach: WebSockets provide low-latency updates without polling.
Redis Pub/Sub enables multi-worker broadcasting.
"""

import json
import asyncio
from typing import Optional, Set, Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from fastapi.websockets import WebSocketState

from config import config
from schemas.schemas import AnalysisStatus, ProgressUpdate
from storage.db import DatabaseClient, get_db_client
from utils.logging import get_logger
from utils.errors import AnalysisNotFoundError

logger = get_logger(__name__)

# WebSocket router
router = APIRouter(tags=["websocket"])

# Persistent Redis connection pool for publishing (avoids connection-per-call)
_redis_publish_pool: Optional[Any] = None


async def _get_redis_publish_client() -> Any:
    """
    Get or create a persistent async Redis client for publishing.
    
    Uses a module-level singleton to avoid creating new connections
    on every progress update, which would exhaust file descriptors
    under load.
    """
    global _redis_publish_pool
    if _redis_publish_pool is None:
        try:
            import redis.asyncio as aioredis
            _redis_publish_pool = aioredis.from_url(
                config.redis_url,
                max_connections=10,
                decode_responses=True,
            )
        except Exception as exc:
            logger.error(f"Failed to create Redis publish pool: {exc}")
            return None
    return _redis_publish_pool


# ============== CONNECTION MANAGER ==============

class ConnectionManager:
    """
    Manages WebSocket connections and subscriptions.
    
    Handles:
    - Connection lifecycle
    - Analysis ID subscriptions
    - Broadcasting to subscribed clients
    - Redis Pub/Sub integration
    """
    
    def __init__(self):
        """Initialize connection manager."""
        # Active connections by analysis_id
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        
        # All active connections
        self.active_connections: Set[WebSocket] = set()
        
        # Redis subscriber task
        self._redis_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def connect(
        self,
        websocket: WebSocket,
        analysis_id: Optional[str] = None
    ):
        """
        Accept WebSocket connection and optionally subscribe to analysis.
        
        Args:
            websocket: WebSocket connection
            analysis_id: Optional analysis ID to subscribe to
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        
        if analysis_id:
            await self.subscribe(websocket, analysis_id)
        
        logger.info(
            f"WebSocket connected, "
            f"total={len(self.active_connections)}, "
            f"subscribed_to={analysis_id}"
        )
        
    async def disconnect(self, websocket: WebSocket):
        """
        Handle WebSocket disconnection.
        
        Removes from all subscriptions and active connections.
        
        Args:
            websocket: Disconnected WebSocket
        """
        self.active_connections.discard(websocket)
        
        # Remove from all subscriptions
        for analysis_id, connections in self.subscriptions.items():
            connections.discard(websocket)
        
        # Clean up empty subscription sets
        self.subscriptions = {
            k: v for k, v in self.subscriptions.items() if v
        }
        
        logger.info(f"WebSocket disconnected, total={len(self.active_connections)}")
        
    async def subscribe(self, websocket: WebSocket, analysis_id: str):
        """
        Subscribe WebSocket to analysis updates.
        
        Args:
            websocket: WebSocket connection
            analysis_id: Analysis ID to subscribe to
        """
        if analysis_id not in self.subscriptions:
            self.subscriptions[analysis_id] = set()
        
        self.subscriptions[analysis_id].add(websocket)
        
        # Send current status immediately
        await self._send_current_status(websocket, analysis_id)
        
        logger.debug(f"WebSocket subscribed to analysis: {analysis_id}")
        
    async def unsubscribe(self, websocket: WebSocket, analysis_id: str):
        """
        Unsubscribe WebSocket from analysis updates.
        
        Args:
            websocket: WebSocket connection
            analysis_id: Analysis ID to unsubscribe from
        """
        if analysis_id in self.subscriptions:
            self.subscriptions[analysis_id].discard(websocket)
            
            # Clean up empty set
            if not self.subscriptions[analysis_id]:
                del self.subscriptions[analysis_id]
                
        logger.debug(f"WebSocket unsubscribed from analysis: {analysis_id}")
        
    async def broadcast_to_analysis(
        self,
        analysis_id: str,
        message: dict
    ):
        """
        Broadcast message to all subscribers of an analysis.
        
        Args:
            analysis_id: Analysis ID
            message: Message dict to broadcast
        """
        if analysis_id not in self.subscriptions:
            return
            
        connections = list(self.subscriptions[analysis_id])
        
        for websocket in connections:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                await self.disconnect(websocket)
                
    async def broadcast_all(self, message: dict):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message dict to broadcast
        """
        connections = list(self.active_connections)
        
        for websocket in connections:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast: {e}")
                await self.disconnect(websocket)
                
    async def _send_current_status(
        self,
        websocket: WebSocket,
        analysis_id: str
    ):
        """
        Send current analysis status to newly subscribed client.
        
        Args:
            websocket: WebSocket connection
            analysis_id: Analysis ID
        """
        try:
            db = await get_db_client()
            analysis = await db.get_analysis(analysis_id)
            
            if analysis:
                # Determine progress based on status
                progress_map = {
                    AnalysisStatus.PENDING: 0.0,
                    AnalysisStatus.PREPROCESSING: 15.0,
                    AnalysisStatus.ANALYZING: 50.0,
                    AnalysisStatus.AGGREGATING: 85.0,
                    AnalysisStatus.COMPLETED: 100.0,
                    AnalysisStatus.FAILED: 0.0
                }
                
                message = {
                    "type": "status",
                    "analysis_id": analysis_id,
                    "status": analysis.status.value,
                    "progress_percent": progress_map.get(analysis.status, 0.0),
                    "current_stage": analysis.status.value,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                if analysis.status == AnalysisStatus.COMPLETED and analysis.trust_score:
                    message["trust_score"] = analysis.trust_score.model_dump(mode="json")
                    message["verdict"] = analysis.verdict.value if analysis.verdict else None
                    
                await websocket.send_json(message)
            else:
                await websocket.send_json({
                    "type": "error",
                    "error_code": "ANALYSIS_NOT_FOUND",
                    "message": f"Analysis not found: {analysis_id}"
                })
                
        except Exception as e:
            logger.error(f"Failed to send current status: {e}")
            await websocket.send_json({
                "type": "error",
                "error_code": "STATUS_FETCH_FAILED",
                "message": str(e)
            })
            
    async def start_redis_listener(self):
        """
        Start Redis Pub/Sub listener for cross-worker messages.
        
        Listens to progress channels and broadcasts to WebSocket clients.
        """
        if self._running:
            return
            
        self._running = True
        self._redis_task = asyncio.create_task(self._redis_listener_loop())
        logger.info("Redis Pub/Sub listener started")
        
    async def stop_redis_listener(self):
        """Stop Redis Pub/Sub listener."""
        self._running = False
        
        if self._redis_task:
            self._redis_task.cancel()
            try:
                await self._redis_task
            except asyncio.CancelledError:
                logger.debug("Redis listener task cancelled")
            self._redis_task = None
            
        logger.info("Redis Pub/Sub listener stopped")
        
    async def _redis_listener_loop(self):
        """
        Main Redis Pub/Sub listener loop.
        
        Subscribes to progress channels and forwards to WebSocket clients.
        """
        import redis.asyncio as aioredis
        
        while self._running:
            redis_client = None
            pubsub = None
            try:
                # Connect to Redis
                redis_client = aioredis.from_url(config.redis_url)
                pubsub = redis_client.pubsub()
                
                # Subscribe to progress pattern
                await pubsub.psubscribe("argus:progress:*")
                
                logger.info("Subscribed to Redis progress channels")
                
                async for message in pubsub.listen():
                    if not self._running:
                        break
                        
                    if message["type"] == "pmessage":
                        try:
                            # Extract analysis_id from channel
                            channel = message["channel"]
                            if isinstance(channel, bytes):
                                channel = channel.decode("utf-8")
                            
                            # Channel format: argus:progress:{analysis_id}
                            analysis_id = channel.split(":")[-1]
                            
                            # Parse message data
                            data = message["data"]
                            if isinstance(data, bytes):
                                data = data.decode("utf-8")
                            
                            progress_data = json.loads(data)
                            progress_data["type"] = "progress"
                            
                            # Broadcast to subscribers
                            await self.broadcast_to_analysis(analysis_id, progress_data)
                            
                        except Exception as e:
                            logger.warning(f"Failed to process Redis message: {e}")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Redis listener error: {e}")
                await asyncio.sleep(5)  # Retry after delay
            finally:
                # Always clean up connections to prevent leaks
                if pubsub is not None:
                    try:
                        await pubsub.unsubscribe()
                    except Exception:
                        pass
                if redis_client is not None:
                    try:
                        await redis_client.close()
                    except Exception:
                        pass


# Global connection manager
manager = ConnectionManager()


# ============== WEBSOCKET ENDPOINTS ==============

@router.websocket("/ws/analysis/{analysis_id}")
async def analysis_progress(
    websocket: WebSocket,
    analysis_id: str,
    token: Optional[str] = None,
):
    """
    WebSocket endpoint for analysis progress updates.
    
    Requires JWT authentication via query parameter: ?token=<jwt_token>
    
    Streams real-time progress updates for a specific analysis.
    
    Message types sent:
    - status: Initial status when connecting
    - progress: Progress updates during analysis
    - completed: Final results when analysis completes
    - error: Error messages
    
    Args:
        websocket: WebSocket connection
        analysis_id: Analysis ID to subscribe to
        token: JWT token from query parameter
    """
    authenticated = False
    user_id = None
    
    if token:
        try:
            import jwt
            from config import config as cfg
            payload = jwt.decode(
                token,
                cfg.jwt_secret,
                algorithms=[cfg.jwt_algorithm]
            )
            user_id = payload.get("sub")
            authenticated = True
        except jwt.ExpiredSignatureError:
            logger.warning("WebSocket auth failed: token expired")
            authenticated = False
        except jwt.InvalidTokenError as exc:
            logger.warning(f"WebSocket auth failed: invalid token - {exc}")
            authenticated = False
        except Exception as exc:
            logger.error(f"WebSocket auth unexpected error: {exc}")
            authenticated = False
    
    if not authenticated:
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    await manager.connect(websocket, analysis_id)
    
    try:
        while True:
            # Wait for client messages (keep-alive pings, unsubscribe requests)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=60.0  # 1 minute timeout for pings
                )
                
                message_type = data.get("type", "unknown")
                
                if message_type == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
                elif message_type == "subscribe":
                    # Subscribe to additional analysis
                    new_analysis_id = data.get("analysis_id")
                    if new_analysis_id:
                        await manager.subscribe(websocket, new_analysis_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "analysis_id": new_analysis_id
                        })
                        
                elif message_type == "unsubscribe":
                    # Unsubscribe from analysis
                    unsub_analysis_id = data.get("analysis_id")
                    if unsub_analysis_id:
                        await manager.unsubscribe(websocket, unsub_analysis_id)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "analysis_id": unsub_analysis_id
                        })
                        
                elif message_type == "refresh":
                    # Request status refresh
                    refresh_id = data.get("analysis_id", analysis_id)
                    await manager._send_current_status(websocket, refresh_id)
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                except Exception:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for analysis: {analysis_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)


@router.websocket("/ws/updates")
async def global_updates(
    websocket: WebSocket,
    token: Optional[str] = None,
):
    """
    WebSocket endpoint for global system updates.
    
    Requires JWT authentication via query parameter: ?token=<jwt_token>
    
    Streams updates about all analyses (for admin dashboards).
    Clients can subscribe/unsubscribe to specific analyses dynamically.
    
    Args:
        websocket: WebSocket connection
        token: JWT token from query parameter
    """
    authenticated = False
    
    if token:
        try:
            import jwt
            from config import config as cfg
            payload = jwt.decode(
                token,
                cfg.jwt_secret,
                algorithms=[cfg.jwt_algorithm]
            )
            authenticated = True
        except jwt.ExpiredSignatureError:
            logger.warning("WebSocket auth failed: token expired")
            authenticated = False
        except jwt.InvalidTokenError as exc:
            logger.warning(f"WebSocket auth failed: invalid token - {exc}")
            authenticated = False
        except Exception as exc:
            logger.error(f"WebSocket auth unexpected error: {exc}")
            authenticated = False
    
    if not authenticated:
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Argus global updates",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=60.0
                )
                
                message_type = data.get("type", "unknown")
                
                if message_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
                elif message_type == "subscribe":
                    analysis_id = data.get("analysis_id")
                    if analysis_id:
                        await manager.subscribe(websocket, analysis_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "analysis_id": analysis_id
                        })
                        
                elif message_type == "unsubscribe":
                    analysis_id = data.get("analysis_id")
                    if analysis_id:
                        await manager.unsubscribe(websocket, analysis_id)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "analysis_id": analysis_id
                        })
                        
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                except Exception:
                    break
                    
    except WebSocketDisconnect:
        logger.info("Global updates WebSocket disconnected")
    except Exception as e:
        logger.error(f"Global updates WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)


# ============== UTILITY FUNCTIONS ==============

async def send_progress_update(
    analysis_id: str,
    status: AnalysisStatus,
    progress_percent: float,
    current_stage: str,
    message: Optional[str] = None
):
    """
    Helper function to send progress update to subscribers.
    
    Can be called from anywhere in the application to notify
    WebSocket clients of progress updates.
    
    Args:
        analysis_id: Analysis ID
        status: Current analysis status
        progress_percent: Progress percentage (0-100)
        current_stage: Current processing stage
        message: Optional human-readable message
    """
    update = ProgressUpdate(
        analysis_id=analysis_id,
        status=status,
        progress_percent=progress_percent,
        current_stage=current_stage,
        message=message
    )
    
    # Broadcast via WebSocket manager
    await manager.broadcast_to_analysis(
        analysis_id,
        {
            "type": "progress",
            **update.model_dump(mode="json"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
    
    # Also publish to Redis for cross-worker delivery (persistent pool)
    try:
        r = await _get_redis_publish_client()
        if r:
            await r.publish(
                f"argus:progress:{analysis_id}",
                json.dumps(update.model_dump(mode="json"))
            )
    except Exception as e:
        logger.warning(f"Failed to publish to Redis: {e}")


async def send_completion_update(
    analysis_id: str,
    trust_score: float,
    verdict: str,
    report_url: Optional[str] = None
):
    """
    Send analysis completion update.
    
    Args:
        analysis_id: Analysis ID
        trust_score: Final trust score (0-100)
        verdict: Final verdict
        report_url: Optional URL to generated report
    """
    update = {
        "type": "completed",
        "analysis_id": analysis_id,
        "status": AnalysisStatus.COMPLETED.value,
        "progress_percent": 100.0,
        "current_stage": "completed",
        "trust_score": trust_score,
        "verdict": verdict,
        "report_url": report_url,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    await manager.broadcast_to_analysis(analysis_id, update)


async def send_error_update(
    analysis_id: str,
    error_code: str,
    error_message: str
):
    """
    Send analysis error update.
    
    Args:
        analysis_id: Analysis ID
        error_code: Error code
        error_message: Human-readable error message
    """
    update = {
        "type": "error",
        "analysis_id": analysis_id,
        "status": AnalysisStatus.FAILED.value,
        "error_code": error_code,
        "message": error_message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    await manager.broadcast_to_analysis(analysis_id, update)


# ============== LIFECYCLE HOOKS ==============

async def startup_websocket():
    """Initialize WebSocket manager on application startup."""
    try:
        await manager.start_redis_listener()
        logger.info("WebSocket manager initialized")
    except Exception as e:
        logger.warning(f"Redis listener failed to start (non-critical): {e}")


async def shutdown_websocket():
    """Cleanup WebSocket manager on application shutdown."""
    global _redis_publish_pool
    
    await manager.stop_redis_listener()
    
    # Close Redis publish pool
    if _redis_publish_pool is not None:
        try:
            await _redis_publish_pool.aclose()
            _redis_publish_pool = None
        except Exception:
            logger.warning("Failed to close Redis publish pool")
    
    # Close all active connections
    for websocket in list(manager.active_connections):
        try:
            await websocket.close()
        except Exception:
            pass
            
    logger.info("WebSocket manager shutdown complete")


# Export
__all__ = [
    "router",
    "manager",
    "ConnectionManager",
    "send_progress_update",
    "send_completion_update",
    "send_error_update",
    "startup_websocket",
    "shutdown_websocket"
]
