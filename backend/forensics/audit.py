"""
Argus Core - Audit Trail Logger
================================
Immutable audit trail logging for forensic chain of custody.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - forensics/audit.py

SOTA Algorithm: Append-only log with cryptographic chaining

Role: Chain of custody tracking for legal proceedings.

Events Logged:
- File upload
- Analysis started
- Analysis completed
- Report generated
- File accessed
- File deleted

Each entry includes:
- Timestamp (UTC)
- Event type
- Actor (user/system)
- Resource ID
- Cryptographic hash of previous entry (chain)

Why this approach: Cryptographic chaining provides tamper-evidence for legal proceedings.
"""

import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from utils.logging import get_logger

logger = get_logger(__name__)


# ============== ENUMS ==============

class AuditEventType(str, Enum):
    """Audit event types for chain of custody tracking."""
    # File lifecycle
    FILE_UPLOAD = "file_upload"
    FILE_ACCESSED = "file_accessed"
    FILE_DELETED = "file_deleted"
    
    # Analysis lifecycle
    ANALYSIS_CREATED = "analysis_created"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    ANALYSIS_DELETED = "analysis_deleted"
    
    # Report lifecycle
    REPORT_GENERATED = "report_generated"
    REPORT_ACCESSED = "report_accessed"
    
    # User actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


# ============== SCHEMAS ==============

class AuditEntry(BaseModel):
    """
    Single audit log entry with cryptographic chain link.
    
    The chain_hash links this entry to the previous one,
    creating a tamper-evident log.
    """
    entry_id: str = Field(..., description="Unique entry identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType
    resource_id: str = Field(..., description="ID of affected resource")
    resource_type: str = Field(default="analysis", description="Type of resource")
    actor: str = Field(default="system", description="User ID or 'system'")
    actor_ip: Optional[str] = Field(default=None, description="Client IP address")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chain_hash: str = Field(..., description="SHA-256 hash of previous entry")
    entry_hash: str = Field(default="", description="SHA-256 hash of this entry")
    
    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of this entry.
        
        Hash is computed over all fields except entry_hash itself.
        """
        data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "actor": self.actor,
            "actor_ip": self.actor_ip,
            "metadata": self.metadata,
            "chain_hash": self.chain_hash
        }
        
        # Deterministic JSON serialization
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode()).hexdigest()


class AuditQuery(BaseModel):
    """Query parameters for audit log searches."""
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    event_type: Optional[AuditEventType] = None
    actor: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class ChainValidationResult(BaseModel):
    """Result of audit chain validation."""
    valid: bool = Field(..., description="Whether chain is valid")
    entries_checked: int = Field(default=0)
    first_invalid_entry: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)


# ============== AUDIT LOGGER ==============

class AuditLogger:
    """
    Immutable audit trail for forensic chain of custody.
    
    Implements cryptographic chaining where each entry contains
    a hash of the previous entry, creating a tamper-evident log.
    
    Features:
    - Append-only log (no updates or deletes)
    - Cryptographic hash chain
    - Structured event types
    - MongoDB backend with indexes
    - Chain validation
    
    Usage:
        audit = AuditLogger()
        await audit.log_event(
            event_type=AuditEventType.ANALYSIS_CREATED,
            resource_id=analysis_id,
            actor="user123",
            metadata={"filename": "video.mp4"}
        )
    """
    
    # Genesis block hash (starting point for chain)
    GENESIS_HASH = "0" * 64
    
    def __init__(self):
        """Initialize audit logger."""
        self._db = None
        self._collection_name = "audit_log"
        self._initialized = False
    
    async def _get_db(self):
        """Get database client lazily."""
        if self._db is None:
            from storage.db import get_db_client
            self._db = await get_db_client()
        return self._db
    
    async def _ensure_initialized(self):
        """Ensure indexes are created."""
        if self._initialized:
            return
        
        try:
            db = await self._get_db()
            collection = db.db[self._collection_name]
            
            # Create indexes for efficient queries
            await collection.create_index("entry_id", unique=True)
            await collection.create_index("resource_id")
            await collection.create_index("event_type")
            await collection.create_index("actor")
            await collection.create_index("timestamp")
            await collection.create_index([
                ("resource_id", 1),
                ("timestamp", -1)
            ])
            
            self._initialized = True
            logger.info("Audit log indexes created")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit log: {e}")
    
    async def _get_last_entry_hash(self) -> str:
        """
        Get hash of the most recent entry.
        
        Returns genesis hash if no entries exist.
        """
        try:
            db = await self._get_db()
            collection = db.db[self._collection_name]
            
            # Get most recent entry
            cursor = collection.find().sort("timestamp", -1).limit(1)
            entries = await cursor.to_list(length=1)
            
            if entries:
                return entries[0].get("entry_hash", self.GENESIS_HASH)
            
            return self.GENESIS_HASH
            
        except Exception as e:
            logger.error(f"Failed to get last entry hash: {e}")
            return self.GENESIS_HASH
    
    async def log_event(
        self,
        event_type: AuditEventType,
        resource_id: str,
        actor: str = "system",
        actor_ip: Optional[str] = None,
        resource_type: str = "analysis",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event.
        
        Creates a new entry linked to the previous entry via hash chain.
        
        Args:
            event_type: Type of event
            resource_id: ID of affected resource
            actor: User ID or 'system'
            actor_ip: Client IP address (optional)
            resource_type: Type of resource (default: 'analysis')
            metadata: Additional event metadata
            
        Returns:
            Entry ID of created log entry
            
        Note:
            This operation is atomic per entry. For high-volume
            scenarios, consider batching.
        """
        import uuid
        
        await self._ensure_initialized()
        
        try:
            # Generate entry ID
            entry_id = str(uuid.uuid4())
            
            # Get previous hash for chain
            chain_hash = await self._get_last_entry_hash()
            
            # Create entry
            entry = AuditEntry(
                entry_id=entry_id,
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                resource_id=resource_id,
                resource_type=resource_type,
                actor=actor,
                actor_ip=actor_ip,
                metadata=metadata or {},
                chain_hash=chain_hash
            )
            
            # Compute entry hash
            entry.entry_hash = entry.compute_hash()
            
            # Insert into database
            db = await self._get_db()
            collection = db.db[self._collection_name]
            
            # Convert to dict for MongoDB
            entry_dict = entry.model_dump(mode="json")
            entry_dict["timestamp"] = entry.timestamp  # Keep as datetime
            
            await collection.insert_one(entry_dict)
            
            logger.debug(
                f"Audit event logged: {event_type.value}",
                extra={
                    "entry_id": entry_id,
                    "resource_id": resource_id,
                    "actor": actor
                }
            )
            
            return entry_id
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise
    
    async def get_entries(
        self,
        query: Optional[AuditQuery] = None
    ) -> List[AuditEntry]:
        """
        Query audit log entries.
        
        Args:
            query: Query parameters (optional)
            
        Returns:
            List of matching audit entries
        """
        await self._ensure_initialized()
        
        if query is None:
            query = AuditQuery()
        
        try:
            db = await self._get_db()
            collection = db.db[self._collection_name]
            
            # Build MongoDB query
            mongo_query: Dict[str, Any] = {}
            
            if query.resource_id:
                mongo_query["resource_id"] = query.resource_id
            
            if query.resource_type:
                mongo_query["resource_type"] = query.resource_type
            
            if query.event_type:
                mongo_query["event_type"] = query.event_type.value
            
            if query.actor:
                mongo_query["actor"] = query.actor
            
            if query.start_time or query.end_time:
                mongo_query["timestamp"] = {}
                if query.start_time:
                    mongo_query["timestamp"]["$gte"] = query.start_time
                if query.end_time:
                    mongo_query["timestamp"]["$lte"] = query.end_time
            
            # Execute query
            cursor = collection.find(mongo_query)
            cursor = cursor.sort("timestamp", -1)
            cursor = cursor.skip(query.offset).limit(query.limit)
            
            entries = []
            async for doc in cursor:
                # Remove MongoDB _id
                doc.pop("_id", None)
                entries.append(AuditEntry(**doc))
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to query audit log: {e}")
            return []
    
    async def get_resource_history(
        self,
        resource_id: str,
        resource_type: str = "analysis"
    ) -> List[AuditEntry]:
        """
        Get complete audit history for a resource.
        
        Args:
            resource_id: Resource identifier
            resource_type: Type of resource
            
        Returns:
            List of audit entries in chronological order
        """
        query = AuditQuery(
            resource_id=resource_id,
            resource_type=resource_type,
            limit=1000
        )
        
        entries = await self.get_entries(query)
        
        # Return in chronological order
        return list(reversed(entries))
    
    async def validate_chain(
        self,
        resource_id: Optional[str] = None,
        limit: int = 1000
    ) -> ChainValidationResult:
        """
        Validate the cryptographic chain integrity.
        
        Checks that each entry's chain_hash matches the previous
        entry's entry_hash, detecting any tampering.
        
        Args:
            resource_id: Validate only entries for this resource (optional)
            limit: Maximum entries to validate
            
        Returns:
            ChainValidationResult with validation status
        """
        await self._ensure_initialized()
        
        try:
            db = await self._get_db()
            collection = db.db[self._collection_name]
            
            # Build query
            mongo_query: Dict[str, Any] = {}
            if resource_id:
                mongo_query["resource_id"] = resource_id
            
            # Get entries in chronological order
            cursor = collection.find(mongo_query).sort("timestamp", 1).limit(limit)
            
            entries_checked = 0
            expected_chain_hash = self.GENESIS_HASH
            
            async for doc in cursor:
                entries_checked += 1
                
                # Verify chain link
                if doc.get("chain_hash") != expected_chain_hash:
                    return ChainValidationResult(
                        valid=False,
                        entries_checked=entries_checked,
                        first_invalid_entry=doc.get("entry_id"),
                        error_message=f"Chain break at entry {doc.get('entry_id')}: "
                                     f"expected {expected_chain_hash[:16]}..., "
                                     f"got {doc.get('chain_hash', '')[:16]}..."
                    )
                
                # Verify entry hash
                doc.pop("_id", None)
                entry = AuditEntry(**doc)
                computed_hash = entry.compute_hash()
                
                if computed_hash != entry.entry_hash:
                    return ChainValidationResult(
                        valid=False,
                        entries_checked=entries_checked,
                        first_invalid_entry=entry.entry_id,
                        error_message=f"Entry hash mismatch at {entry.entry_id}"
                    )
                
                # Update expected hash for next entry
                expected_chain_hash = entry.entry_hash
            
            return ChainValidationResult(
                valid=True,
                entries_checked=entries_checked
            )
            
        except Exception as e:
            logger.error(f"Chain validation failed: {e}")
            return ChainValidationResult(
                valid=False,
                entries_checked=0,
                error_message=str(e)
            )
    
    async def get_chain_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the audit log.
        
        Returns:
            Dict with entry counts by type, date range, etc.
        """
        await self._ensure_initialized()
        
        try:
            db = await self._get_db()
            collection = db.db[self._collection_name]
            
            # Count total entries
            total_count = await collection.count_documents({})
            
            # Get event type distribution
            pipeline = [
                {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            type_counts = {}
            async for doc in collection.aggregate(pipeline):
                type_counts[doc["_id"]] = doc["count"]
            
            # Get date range
            first_entry = await collection.find_one(
                {},
                sort=[("timestamp", 1)]
            )
            last_entry = await collection.find_one(
                {},
                sort=[("timestamp", -1)]
            )
            
            return {
                "total_entries": total_count,
                "by_event_type": type_counts,
                "first_entry_at": first_entry.get("timestamp") if first_entry else None,
                "last_entry_at": last_entry.get("timestamp") if last_entry else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get chain summary: {e}")
            return {"error": str(e)}


# ============== SINGLETON ==============

_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Get singleton audit logger instance.
    
    Returns:
        AuditLogger instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# ============== CONVENIENCE FUNCTIONS ==============

async def log_file_upload(
    resource_id: str,
    actor: str,
    filename: str,
    file_hash: str,
    file_size: int,
    actor_ip: Optional[str] = None
) -> str:
    """Log file upload event."""
    audit = get_audit_logger()
    return await audit.log_event(
        event_type=AuditEventType.FILE_UPLOAD,
        resource_id=resource_id,
        actor=actor,
        actor_ip=actor_ip,
        metadata={
            "filename": filename,
            "file_hash": file_hash,
            "file_size": file_size
        }
    )


async def log_analysis_created(
    analysis_id: str,
    actor: str,
    file_id: str,
    options: Dict[str, Any],
    actor_ip: Optional[str] = None
) -> str:
    """Log analysis creation event."""
    audit = get_audit_logger()
    return await audit.log_event(
        event_type=AuditEventType.ANALYSIS_CREATED,
        resource_id=analysis_id,
        actor=actor,
        actor_ip=actor_ip,
        metadata={
            "file_id": file_id,
            "options": options
        }
    )


async def log_analysis_completed(
    analysis_id: str,
    trust_score: float,
    verdict: str,
    processing_time: float
) -> str:
    """Log analysis completion event."""
    audit = get_audit_logger()
    return await audit.log_event(
        event_type=AuditEventType.ANALYSIS_COMPLETED,
        resource_id=analysis_id,
        actor="system",
        metadata={
            "trust_score": trust_score,
            "verdict": verdict,
            "processing_time_seconds": processing_time
        }
    )


async def log_report_generated(
    analysis_id: str,
    report_url: str
) -> str:
    """Log report generation event."""
    audit = get_audit_logger()
    return await audit.log_event(
        event_type=AuditEventType.REPORT_GENERATED,
        resource_id=analysis_id,
        actor="system",
        resource_type="report",
        metadata={
            "report_url": report_url
        }
    )


# Export
__all__ = [
    "AuditEventType",
    "AuditEntry",
    "AuditQuery",
    "ChainValidationResult",
    "AuditLogger",
    "get_audit_logger",
    "log_file_upload",
    "log_analysis_created",
    "log_analysis_completed",
    "log_report_generated"
]
