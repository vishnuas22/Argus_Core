"""
Argus Core - MinIO Storage Client with Local Fallback
======================================================
S3-compatible object storage client for media files with automatic
fallback to local filesystem storage when MinIO is unavailable.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - storage/storage.py

Buckets:
- argus-uploads: Raw uploaded files
- argus-preprocessed: Extracted frames, audio
- argus-results: Heatmaps, reports

Features:
- Automatic retry with exponential backoff
- Health checks with auto-reconnection
- Guaranteed bucket creation on startup
- Local filesystem fallback when MinIO unavailable
"""

import io
import os
import time
import shutil
from typing import Union, BinaryIO, Optional, AsyncIterator, List, Dict, Any
from datetime import timedelta
from pathlib import Path
import asyncio
from functools import partial
import urllib3

from config import config
from interfaces.storage import IStorage
from utils.errors import StorageError
from utils.logging import get_logger

logger = get_logger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0
RETRY_DELAY_MAX = 10.0

# Local storage fallback directory - use current working directory or temp
LOCAL_STORAGE_BASE = os.environ.get("LOCAL_STORAGE_PATH", os.path.join(os.getcwd(), "storage_fallback"))


class LocalStorageClient:
    """
    Local filesystem storage fallback when MinIO is unavailable.
    
    Provides same interface as MinIO StorageClient but uses local disk.
    """
    
    def __init__(self, base_path: str = LOCAL_STORAGE_BASE):
        """Initialize local storage with base directory."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.bucket_uploads = "argus-uploads"
        self.bucket_preprocessed = "argus-preprocessed"
        self.bucket_results = "argus-results"
        logger.info(f"Local storage fallback initialized at: {self.base_path}")
    
    def _get_path(self, bucket: str, object_key: str) -> Path:
        """Get full path for object with path traversal protection."""
        sanitized_key = object_key.replace("\\", "/").lstrip("/")
        full_path = (self.base_path / bucket / sanitized_key).resolve()
        base_resolved = self.base_path.resolve()
        if not str(full_path).startswith(str(base_resolved)):
            raise StorageError(
                "path_traversal",
                f"Invalid object key: path escapes storage base directory"
            )
        return full_path
    
    def _ensure_bucket_dir(self, bucket: str) -> Path:
        """Ensure bucket directory exists."""
        bucket_path = self.base_path / bucket
        bucket_path.mkdir(parents=True, exist_ok=True)
        return bucket_path
    
    async def upload_file(
        self,
        file: Union[bytes, BinaryIO],
        bucket: str,
        object_key: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """Upload file to local storage."""
        self._ensure_bucket_dir(bucket)
        file_path = self._get_path(bucket, object_key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(file, bytes):
            file_path.write_bytes(file)
        else:
            file.seek(0)
            file_path.write_bytes(file.read())
        
        logger.debug(f"Local storage: uploaded {object_key} to {bucket}")
        return object_key
    
    async def download_file(self, bucket: str, object_key: str) -> bytes:
        """Download file from local storage."""
        file_path = self._get_path(bucket, object_key)
        if not file_path.exists():
            raise StorageError("download_file", f"File not found: {object_key}")
        return file_path.read_bytes()
    
    async def delete_file(self, bucket: str, object_key: str) -> None:
        """Delete file from local storage."""
        file_path = self._get_path(bucket, object_key)
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Local storage: deleted {object_key} from {bucket}")
    
    async def file_exists(self, bucket: str, object_key: str) -> bool:
        """Check if file exists in local storage."""
        return self._get_path(bucket, object_key).exists()
    
    async def list_objects(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        recursive: bool = True
    ) -> List[str]:
        """List objects in local storage bucket."""
        bucket_path = self.base_path / bucket
        if not bucket_path.exists():
            return []
        
        if prefix:
            search_path = bucket_path / prefix
            if not search_path.exists():
                return []
            pattern = "**/*" if recursive else "*"
        else:
            search_path = bucket_path
            pattern = "**/*" if recursive else "*"
        
        objects = []
        for path in search_path.glob(pattern):
            if path.is_file():
                rel_path = path.relative_to(bucket_path)
                objects.append(str(rel_path))
        return objects
    
    async def get_presigned_url(
        self,
        bucket: str,
        object_key: str,
        expires_seconds: int = 3600
    ) -> str:
        """Return local file path as URL (for local fallback)."""
        file_path = self._get_path(bucket, object_key)
        return f"file://{file_path}"
    
    async def ensure_default_buckets(self) -> None:
        """Ensure default bucket directories exist on local storage."""
        for bucket in ["argus-uploads", "argus-preprocessed", "argus-results"]:
            self._ensure_bucket_dir(bucket)

    async def health_check(self) -> Dict[str, Any]:
        """Check local storage health."""
        try:
            test_file = self.base_path / ".health_check"
            test_file.write_text("ok")
            test_file.unlink()
            return {
                "status": "healthy",
                "mode": "local_fallback",
                "path": str(self.base_path)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "mode": "local_fallback",
                "error": str(e)
            }


class StorageClient(IStorage):
    """
    MinIO object storage client with automatic retry, health checks,
    and local filesystem fallback.
    
    Provides async wrapper around MinIO SDK for
    file upload, download, and management operations.
    Falls back to local storage when MinIO is unavailable.
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: bool = False
    ):
        """
        Initialize MinIO client with local fallback.
        
        Args:
            endpoint: MinIO server endpoint
            access_key: Access key
            secret_key: Secret key
            secure: Use HTTPS
        """
        self.endpoint = endpoint or config.minio_endpoint
        self.access_key = access_key or config.minio_access_key
        self.secret_key = secret_key or config.minio_secret_key
        self.secure = secure or config.minio_secure
        self.external_endpoint = os.environ.get("MINIO_EXTERNAL_ENDPOINT", "localhost:9000")
        
        self._client = None
        self._initialized = False
        self._buckets_created = False
        self._use_local_fallback = False
        self._fallback_retry_interval = 60
        self._fallback_until = 0.0
        
        # Local storage fallback
        self._local_storage = LocalStorageClient()
        
        # Default buckets from config
        self.bucket_uploads = config.minio_bucket_uploads
        self.bucket_preprocessed = config.minio_bucket_preprocessed
        self.bucket_results = config.minio_bucket_results
        
        # Try to initialize MinIO client
        self._create_client()
    
    def _create_client(self) -> None:
        """Create MinIO client instance with proper timeout settings."""
        try:
            from minio import Minio
            from minio.error import S3Error
            
            # Store S3Error for later use
            self._S3Error = S3Error
            
            # Configure HTTP client with timeouts
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=2.0, read=10.0),
                maxsize=10,
                retries=urllib3.Retry(
                    total=1,
                    backoff_factor=0.2,
                    status_forcelist=[500, 502, 503, 504]
                )
            )
            
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                http_client=http_client
            )
            logger.info(f"MinIO client created for endpoint: {self.endpoint}")
        except ImportError:
            logger.warning("MinIO library not available, using local storage fallback")
            self._use_local_fallback = True
            self._client = None
            self._S3Error = Exception
    
    def _reconnect(self) -> None:
        """Reconnect to MinIO server."""
        logger.warning("Reconnecting to MinIO...")
        self._create_client()
        self._buckets_created = False
        if self._client is not None:
            self._use_local_fallback = False
            self._fallback_until = 0.0
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous MinIO operation in executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(func, *args, **kwargs)
        )
    
    async def _retry_operation(
        self,
        operation_name: str,
        func,
        *args,
        max_retries: int = MAX_RETRIES,
        **kwargs
    ):
        """
        Execute operation with retry logic and exponential backoff.
        Falls back to local storage if MinIO consistently fails.
        
        Args:
            operation_name: Name of operation for logging
            func: Function to execute
            *args: Function arguments
            max_retries: Maximum retry attempts
            **kwargs: Function keyword arguments
            
        Returns:
            Operation result
            
        Raises:
            StorageError: If all retries fail
        """
        # If using local fallback, attempt recovery after retry interval
        if self._use_local_fallback:
            now = time.time()
            if now < self._fallback_until:
                raise StorageError(operation_name, "MinIO unavailable, using local fallback")
            logger.info("Attempting MinIO recovery after fallback period...")
            self._reconnect()
        
        last_error = None
        S3Error = self._S3Error
        
        for attempt in range(max_retries + 1):
            try:
                return await self._run_sync(func, *args, **kwargs)
            except S3Error as e:
                last_error = e
                if attempt < max_retries:
                    delay = min(RETRY_DELAY_BASE * (2 ** attempt), RETRY_DELAY_MAX)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    
                    # Reconnect on connection errors
                    if "Connection" in str(e) or "timeout" in str(e).lower():
                        self._reconnect()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = min(RETRY_DELAY_BASE * (2 ** attempt), RETRY_DELAY_MAX)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    self._reconnect()
        
        # Switch to local fallback after exhausting retries
        logger.warning(f"{operation_name} failed after {max_retries + 1} attempts, switching to local fallback")
        self._use_local_fallback = True
        self._fallback_until = time.time() + self._fallback_retry_interval
        raise StorageError(operation_name, str(last_error))
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check storage health and connectivity.
        
        Returns:
            Dict with health status and details
        """
        # If using local fallback, check that instead
        if self._use_local_fallback:
            return await self._local_storage.health_check()
        
        try:
            start_time = time.time()
            
            # Try to list buckets as health check
            buckets = await self._run_sync(self._client.list_buckets)
            
            latency_ms = (time.time() - start_time) * 1000
            bucket_names = [b.name for b in buckets]
            
            return {
                "status": "healthy",
                "mode": "minio",
                "latency_ms": round(latency_ms, 2),
                "endpoint": self.endpoint,
                "buckets": bucket_names,
                "buckets_ready": all(
                    b in bucket_names for b in [
                        self.bucket_uploads,
                        self.bucket_preprocessed,
                        self.bucket_results
                    ]
                )
            }
        except Exception as e:
            logger.warning(f"MinIO health check failed: {e}, switching to local fallback")
            self._use_local_fallback = True
            return await self._local_storage.health_check()
    
    async def wait_for_ready(self, timeout: float = 5.0, interval: float = 1.0) -> bool:
        """
        Wait for storage to become ready (MinIO or local fallback).
        
        Args:
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds
            
        Returns:
            True if storage is ready
        """
        # If already using local fallback, it's ready
        if self._use_local_fallback:
            logger.info("Using local storage fallback (MinIO unavailable)")
            return True
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                health = await self.health_check()
                if health["status"] == "healthy":
                    if health.get("mode") == "local_fallback":
                        logger.info("Storage ready (local fallback mode)")
                    else:
                        logger.info(f"MinIO ready after {time.time() - start_time:.1f}s")
                    return True
            except Exception:
                logger.debug("MinIO health check failed, retrying...")
            
            await asyncio.sleep(interval)
        
        # Switch to local fallback after timeout
        logger.warning(f"MinIO not ready after {timeout}s, using local storage fallback")
        self._use_local_fallback = True
        return True  # Local fallback is always "ready"
    
    async def ensure_bucket(self, bucket: str) -> None:
        """
        Ensure bucket exists, create if not.
        Uses local directory if in fallback mode.
        
        Args:
            bucket: Bucket name to ensure
        """
        if self._use_local_fallback:
            # Local fallback: just ensure directory exists
            self._local_storage._ensure_bucket_dir(bucket)
            logger.debug(f"Local bucket directory ensured: {bucket}")
            return
        
        try:
            exists = await self._retry_operation(
                f"check_bucket_{bucket}",
                self._client.bucket_exists,
                bucket
            )
            if not exists:
                await self._retry_operation(
                    f"create_bucket_{bucket}",
                    self._client.make_bucket,
                    bucket
                )
                logger.info(f"Created bucket: {bucket}")
            else:
                logger.debug(f"Bucket exists: {bucket}")
        except StorageError:
            # Fall back to local storage
            self._use_local_fallback = True
            self._local_storage._ensure_bucket_dir(bucket)
            logger.info(f"Using local storage fallback for bucket: {bucket}")
        except Exception as e:
            self._use_local_fallback = True
            self._local_storage._ensure_bucket_dir(bucket)
            logger.warning(f"Bucket ensure failed, using local fallback: {e}")
    
    async def ensure_default_buckets(self) -> None:
        """
        Ensure all default buckets exist (MinIO or local directories).
        
        This is called on startup and ensures storage is ready.
        """
        if self._buckets_created:
            logger.debug("Buckets already verified")
            return
        
        # Quick check if MinIO is available, otherwise use local
        ready = await self.wait_for_ready(timeout=5.0)
        if not ready and not self._use_local_fallback:
            self._use_local_fallback = True
            logger.info("Using local storage fallback for all operations")
        
        buckets = [
            self.bucket_uploads,
            self.bucket_preprocessed,
            self.bucket_results
        ]
        
        for bucket in buckets:
            await self.ensure_bucket(bucket)
        
        self._buckets_created = True
        logger.info(f"All default buckets ready: {buckets}")
    
    async def _ensure_bucket_exists(self, bucket: str) -> None:
        """
        Internal method to ensure bucket exists before operation.
        
        Args:
            bucket: Bucket name to verify
        """
        if not self._buckets_created:
            await self.ensure_default_buckets()
    
    async def upload_file(
        self,
        file: Union[bytes, BinaryIO],
        bucket: str,
        object_key: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload file to MinIO with automatic retry and local fallback.
        
        Args:
            file: File content as bytes or binary stream
            bucket: Target bucket name
            object_key: Object key/path within bucket
            content_type: MIME type of file
            
        Returns:
            Object key for retrieval
        """
        # Use local storage if MinIO is unavailable
        if self._use_local_fallback:
            return await self._local_storage.upload_file(
                file, bucket, object_key, content_type
            )
        
        # Ensure bucket exists
        await self._ensure_bucket_exists(bucket)
        
        # Convert bytes to BytesIO if needed
        if isinstance(file, bytes):
            file_data = io.BytesIO(file)
            file_size = len(file)
        else:
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            file_data = file
        
        # Upload with retry, fall back to local on failure
        try:
            await self._retry_operation(
                f"upload_{object_key}",
                self._client.put_object,
                bucket,
                object_key,
                file_data,
                file_size,
                content_type=content_type
            )
        except StorageError:
            # Retry operation already set _use_local_fallback = True
            if isinstance(file, bytes):
                file_data = io.BytesIO(file)
            else:
                file.seek(0)
                file_data = file
            return await self._local_storage.upload_file(
                file_data, bucket, object_key, content_type
            )
        
        logger.info(f"Uploaded {object_key} to {bucket} ({file_size} bytes)")
        return object_key
    
    async def download_file(
        self,
        bucket: str,
        object_key: str
    ) -> bytes:
        """
        Download file from MinIO with automatic retry and local fallback.
        
        Args:
            bucket: Source bucket name
            object_key: Object key/path within bucket
            
        Returns:
            File content as bytes
        """
        # Use local storage if MinIO is unavailable
        if self._use_local_fallback:
            return await self._local_storage.download_file(bucket, object_key)
        
        await self._ensure_bucket_exists(bucket)
        
        try:
            response = await self._retry_operation(
                f"download_{object_key}",
                self._client.get_object,
                bucket,
                object_key
            )
        except StorageError:
            # Fallback to local storage
            return await self._local_storage.download_file(bucket, object_key)
        
        try:
            data = response.read()
            return data
        finally:
            response.close()
            response.release_conn()
    
    async def download_stream(
        self,
        bucket: str,
        object_key: str,
        chunk_size: int = 1024 * 1024
    ) -> AsyncIterator[bytes]:
        """
        Stream download file in chunks.
        
        Args:
            bucket: Source bucket name
            object_key: Object key/path within bucket
            chunk_size: Size of each chunk (default 1MB)
            
        Yields:
            File content chunks
        """
        await self._ensure_bucket_exists(bucket)
        
        response = await self._retry_operation(
            f"stream_{object_key}",
            self._client.get_object,
            bucket,
            object_key
        )
        
        try:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            response.close()
            response.release_conn()
    
    async def get_presigned_url(
        self,
        bucket: str,
        object_key: str,
        expires_seconds: int = 3600
    ) -> str:
        """
        Generate presigned URL for direct client access.
        
        Rewrites the internal Docker hostname to the external endpoint
        so that browser clients can access the URL.
        
        Args:
            bucket: Bucket name
            object_key: Object key
            expires_seconds: URL validity duration (default 1 hour)
            
        Returns:
            Presigned download URL accessible from browser
        """
        await self._ensure_bucket_exists(bucket)
        
        url = await self._retry_operation(
            f"presigned_{object_key}",
            self._client.presigned_get_object,
            bucket,
            object_key,
            expires=timedelta(seconds=expires_seconds)
        )
        
        # Rewrite internal Docker endpoint to external endpoint for browser access
        if self.endpoint != self.external_endpoint:
            url = url.replace(self.endpoint, self.external_endpoint)
            # Also fix http:// to match the external access pattern
            if self.external_endpoint.startswith("localhost"):
                url = url.replace("http://localhost", "http://localhost")
        
        return url
    
    async def delete_file(
        self,
        bucket: str,
        object_key: str
    ) -> None:
        """
        Delete file from MinIO.
        
        Args:
            bucket: Bucket name
            object_key: Object key to delete
        """
        await self._ensure_bucket_exists(bucket)
        
        await self._retry_operation(
            f"delete_{object_key}",
            self._client.remove_object,
            bucket,
            object_key
        )
        logger.info(f"Deleted {object_key} from {bucket}")
    
    async def file_exists(
        self,
        bucket: str,
        object_key: str
    ) -> bool:
        """
        Check if file exists in storage.
        
        Args:
            bucket: Bucket name
            object_key: Object key to check
            
        Returns:
            True if file exists
        """
        await self._ensure_bucket_exists(bucket)
        
        try:
            await self._retry_operation(
                f"stat_{object_key}",
                self._client.stat_object,
                bucket,
                object_key
            )
            return True
        except StorageError as e:
            if "NoSuchKey" in str(e):
                return False
            raise
    
    async def list_objects(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        recursive: bool = True
    ) -> List[str]:
        """
        List objects in bucket with optional prefix filter.
        
        Args:
            bucket: Bucket name
            prefix: Optional path prefix to filter
            recursive: List recursively (default True)
            
        Returns:
            List of object keys
        """
        await self._ensure_bucket_exists(bucket)
        
        try:
            objects = self._client.list_objects(
                bucket,
                prefix=prefix,
                recursive=recursive
            )
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"List objects failed: {e}")
            raise StorageError("list_objects", str(e))
    
    async def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str
    ) -> str:
        """
        Copy object between buckets/keys.
        
        Args:
            source_bucket: Source bucket
            source_key: Source object key
            dest_bucket: Destination bucket
            dest_key: Destination object key
            
        Returns:
            Destination object key
        """
        await self._ensure_bucket_exists(source_bucket)
        await self._ensure_bucket_exists(dest_bucket)
        
        from minio.commonconfig import CopySource
        
        await self._retry_operation(
            f"copy_{source_key}_to_{dest_key}",
            self._client.copy_object,
            dest_bucket,
            dest_key,
            CopySource(source_bucket, source_key)
        )
        
        logger.info(f"Copied {source_bucket}/{source_key} to {dest_bucket}/{dest_key}")
        return dest_key
    
    async def get_object_info(
        self,
        bucket: str,
        object_key: str
    ) -> Dict[str, Any]:
        """
        Get object metadata/info.
        
        Args:
            bucket: Bucket name
            object_key: Object key
            
        Returns:
            Dict with object metadata
        """
        await self._ensure_bucket_exists(bucket)
        
        stat = await self._retry_operation(
            f"info_{object_key}",
            self._client.stat_object,
            bucket,
            object_key
        )
        
        return {
            "bucket": bucket,
            "object_key": object_key,
            "size": stat.size,
            "content_type": stat.content_type,
            "last_modified": stat.last_modified.isoformat() if stat.last_modified else None,
            "etag": stat.etag,
            "metadata": dict(stat.metadata) if stat.metadata else {}
        }


# Singleton instance
_storage_client: Optional[StorageClient] = None


def get_storage_client() -> StorageClient:
    """Get singleton storage client instance."""
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient()
    return _storage_client


async def init_storage() -> StorageClient:
    """
    Initialize storage client and ensure buckets exist.
    
    Call this on application startup to guarantee MinIO is ready.
    
    Returns:
        Initialized StorageClient
    """
    client = get_storage_client()
    await client.ensure_default_buckets()
    return client
