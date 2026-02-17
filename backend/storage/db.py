"""
Argus Core - MongoDB Database Client
====================================
MongoDB async client with connection pooling for analysis storage.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - storage/db.py

Collections:
- analyses: Main analysis records
- jobs: Celery job tracking
- audit_log: Immutable audit trail
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId

from config import config
from schemas import (
    AnalysisDocument,
    AnalysisStatus,
    AnalysisResponse,
    AnalysisDetailResponse
)
from utils.errors import AnalysisNotFoundError, StorageError
from utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseClient:
    """
    MongoDB async client with connection pooling.
    
    Provides CRUD operations for analysis records
    and audit logging functionality.
    """
    
    def __init__(
        self,
        mongo_url: Optional[str] = None,
        db_name: Optional[str] = None
    ):
        """
        Initialize MongoDB client.
        
        Args:
            mongo_url: MongoDB connection URL
            db_name: Database name
        """
        self.mongo_url = mongo_url or config.mongo_url
        self.db_name = db_name or config.db_name
        
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self) -> None:
        """Establish database connection."""
        if self._client is None:
            self._client = AsyncIOMotorClient(
                self.mongo_url,
                maxPoolSize=50,
                minPoolSize=10
            )
            self._db = self._client[self.db_name]
            
            # Create indexes
            await self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {self.db_name}")
    
    async def disconnect(self) -> None:
        """Close database connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self) -> None:
        """Create database indexes for performance."""
        # Analyses collection indexes
        await self._db.analyses.create_index("analysis_id", unique=True)
        await self._db.analyses.create_index("status")
        await self._db.analyses.create_index([("created_at", DESCENDING)])
        await self._db.analyses.create_index("input.file_hash")
        
        # Jobs collection indexes
        await self._db.jobs.create_index("job_id", unique=True)
        await self._db.jobs.create_index("analysis_id")
        await self._db.jobs.create_index("status")
        
        # Audit log collection - TTL index for retention
        await self._db.audit_log.create_index([("timestamp", DESCENDING)])
        await self._db.audit_log.create_index("resource_id")
        
        logger.info("Database indexes created")
    
    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Get database instance."""
        if self._db is None:
            raise StorageError("database", "Not connected to database")
        return self._db
    
    # ============== ANALYSIS OPERATIONS ==============
    
    async def insert_analysis(
        self,
        analysis: AnalysisDocument
    ) -> str:
        """
        Insert new analysis record.
        
        Args:
            analysis: Analysis document to insert
            
        Returns:
            Inserted analysis_id
        """
        try:
            doc = analysis.model_dump(mode="json")
            # Convert datetime to ISO string for JSON serialization
            if doc.get("created_at"):
                doc["created_at"] = analysis.created_at.isoformat()
            
            await self.db.analyses.insert_one(doc)
            logger.info(f"Inserted analysis: {analysis.analysis_id}")
            return analysis.analysis_id
            
        except Exception as e:
            logger.error(f"Insert analysis failed: {e}")
            raise StorageError("insert_analysis", str(e))
    
    async def get_analysis(
        self,
        analysis_id: str
    ) -> Optional[AnalysisDocument]:
        """
        Retrieve analysis by ID.
        
        Args:
            analysis_id: Analysis ID to retrieve
            
        Returns:
            AnalysisDocument if found, None otherwise
        """
        try:
            doc = await self.db.analyses.find_one(
                {"analysis_id": analysis_id},
                {"_id": 0}  # Exclude MongoDB _id
            )
            
            if doc is None:
                return None
            
            return AnalysisDocument(**doc)
            
        except Exception as e:
            logger.error(f"Get analysis failed: {e}")
            raise StorageError("get_analysis", str(e))
    
    async def update_analysis(
        self,
        analysis_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update analysis with new data.
        
        Args:
            analysis_id: Analysis ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update succeeded
        """
        try:
            result = await self.db.analyses.update_one(
                {"analysis_id": analysis_id},
                {"$set": updates}
            )
            
            if result.matched_count == 0:
                raise AnalysisNotFoundError(analysis_id)
            
            logger.info(f"Updated analysis: {analysis_id}")
            return result.modified_count > 0
            
        except AnalysisNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Update analysis failed: {e}")
            raise StorageError("update_analysis", str(e))
    
    async def update_analysis_status(
        self,
        analysis_id: str,
        status: AnalysisStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update analysis status.
        
        Args:
            analysis_id: Analysis ID
            status: New status
            error_message: Optional error message for failed status
            
        Returns:
            True if update succeeded
        """
        updates = {"status": status.value}
        
        if status == AnalysisStatus.COMPLETED:
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        if error_message:
            updates["error_message"] = error_message
        
        return await self.update_analysis(analysis_id, updates)
    
    async def delete_analysis(
        self,
        analysis_id: str
    ) -> bool:
        """
        Delete analysis record.
        
        Args:
            analysis_id: Analysis ID to delete
            
        Returns:
            True if deletion succeeded
        """
        try:
            result = await self.db.analyses.delete_one(
                {"analysis_id": analysis_id}
            )
            
            if result.deleted_count > 0:
                logger.info(f"Deleted analysis: {analysis_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Delete analysis failed: {e}")
            raise StorageError("delete_analysis", str(e))
    
    async def list_analyses(
        self,
        status: Optional[AnalysisStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AnalysisResponse]:
        """
        List analyses with optional filtering.
        
        Args:
            status: Filter by status
            limit: Maximum results
            offset: Skip count for pagination
            
        Returns:
            List of AnalysisResponse
        """
        try:
            query = {}
            if status:
                query["status"] = status.value
            
            cursor = self.db.analyses.find(
                query,
                {"_id": 0}
            ).sort(
                "created_at", DESCENDING
            ).skip(offset).limit(limit)
            
            results = []
            async for doc in cursor:
                results.append(AnalysisResponse(**doc))
            
            return results
            
        except Exception as e:
            logger.error(f"List analyses failed: {e}")
            raise StorageError("list_analyses", str(e))
    
    # ============== AUDIT LOG OPERATIONS ==============
    
    async def log_audit_event(
        self,
        event_type: str,
        resource_id: str,
        actor: str,
        metadata: Optional[Dict[str, Any]] = None,
        previous_hash: Optional[str] = None
    ) -> str:
        """
        Log immutable audit event.
        
        Args:
            event_type: Type of event (upload, analysis, etc.)
            resource_id: ID of affected resource
            actor: User or system performing action
            metadata: Additional event data
            previous_hash: Hash of previous entry for chaining
            
        Returns:
            Audit entry ID
        """
        try:
            import hashlib
            import json
            
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "resource_id": resource_id,
                "actor": actor,
                "metadata": metadata or {},
                "previous_hash": previous_hash
            }
            
            # Generate hash for chain integrity
            entry_str = json.dumps(entry, sort_keys=True)
            entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
            
            result = await self.db.audit_log.insert_one(entry)
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Audit log failed: {e}")
            # Don't raise - audit failures shouldn't break operations
            return ""
    
    # ============== JOB TRACKING ==============
    
    async def create_job(
        self,
        job_id: str,
        analysis_id: str,
        job_type: str
    ) -> str:
        """
        Create job tracking record.
        
        Args:
            job_id: Celery job ID
            analysis_id: Associated analysis ID
            job_type: Type of job (preprocess, analyze, etc.)
            
        Returns:
            Job ID
        """
        try:
            doc = {
                "job_id": job_id,
                "analysis_id": analysis_id,
                "job_type": job_type,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "completed_at": None,
                "error": None
            }
            
            # Use update_one with upsert to handle duplicate key errors on retry
            result = await self.db.jobs.update_one(
                {"job_id": job_id},
                {"$setOnInsert": doc},
                upsert=True
            )
            
            # If document was already inserted (matched existing), just return the job_id
            # This handles Celery task retries gracefully
            if result.matched_count > 0:
                logger.info(f"Job {job_id} already exists, continuing with existing job")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Create job failed: {e}")
            raise StorageError("create_job", str(e))
    
    async def update_job_status(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None
    ) -> bool:
        """
        Update job status.
        
        Args:
            job_id: Job ID
            status: New status
            error: Error message if failed
            
        Returns:
            True if update succeeded
        """
        try:
            updates = {"status": status}
            
            if status == "running":
                updates["started_at"] = datetime.now(timezone.utc).isoformat()
            elif status in ("completed", "failed"):
                updates["completed_at"] = datetime.now(timezone.utc).isoformat()
            
            if error:
                updates["error"] = error
            
            result = await self.db.jobs.update_one(
                {"job_id": job_id},
                {"$set": updates}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Update job failed: {e}")
            raise StorageError("update_job", str(e))


# Singleton instance
_db_client: Optional[DatabaseClient] = None


async def get_db_client() -> DatabaseClient:
    """Get singleton database client instance."""
    global _db_client
    if _db_client is None:
        _db_client = DatabaseClient()
        await _db_client.connect()
    return _db_client


async def close_db_client() -> None:
    """Close database client connection."""
    global _db_client
    if _db_client is not None:
        await _db_client.disconnect()
        _db_client = None
