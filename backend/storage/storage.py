"""
Argus Core - MinIO Storage Client
=================================
S3-compatible object storage client for media files.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - storage/storage.py

Buckets:
- argus-uploads: Raw uploaded files
- argus-preprocessed: Extracted frames, audio
- argus-results: Heatmaps, reports
"""

import io
from typing import Union, BinaryIO, Optional, AsyncIterator, List
from datetime import timedelta
from minio import Minio
from minio.error import S3Error
import asyncio
from functools import partial

from config import config
from interfaces.storage import IStorage
from utils.errors import StorageError
from utils.logging import get_logger

logger = get_logger(__name__)


class StorageClient(IStorage):
    """
    MinIO object storage client.
    
    Provides async wrapper around MinIO SDK for
    file upload, download, and management operations.
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: bool = False
    ):
        """
        Initialize MinIO client.
        
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
        
        self._client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
        # Default buckets from config
        self.bucket_uploads = config.minio_bucket_uploads
        self.bucket_preprocessed = config.minio_bucket_preprocessed
        self.bucket_results = config.minio_bucket_results
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous MinIO operation in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(func, *args, **kwargs)
        )
    
    async def ensure_bucket(self, bucket: str) -> None:
        """
        Ensure bucket exists, create if not.
        
        Args:
            bucket: Bucket name to ensure
        """
        try:
            exists = await self._run_sync(
                self._client.bucket_exists,
                bucket
            )
            if not exists:
                await self._run_sync(
                    self._client.make_bucket,
                    bucket
                )
                logger.info(f"Created bucket: {bucket}")
        except S3Error as e:
            raise StorageError("ensure_bucket", str(e))
    
    async def ensure_default_buckets(self) -> None:
        """Ensure all default buckets exist."""
        for bucket in [
            self.bucket_uploads,
            self.bucket_preprocessed,
            self.bucket_results
        ]:
            await self.ensure_bucket(bucket)
    
    async def upload_file(
        self,
        file: Union[bytes, BinaryIO],
        bucket: str,
        object_key: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload file to MinIO.
        
        Args:
            file: File content as bytes or binary stream
            bucket: Target bucket name
            object_key: Object key/path within bucket
            content_type: MIME type of file
            
        Returns:
            Object key for retrieval
        """
        try:
            # Convert bytes to BytesIO if needed
            if isinstance(file, bytes):
                file_data = io.BytesIO(file)
                file_size = len(file)
            else:
                file.seek(0, 2)  # Seek to end
                file_size = file.tell()
                file.seek(0)  # Reset to beginning
                file_data = file
            
            await self._run_sync(
                self._client.put_object,
                bucket,
                object_key,
                file_data,
                file_size,
                content_type=content_type
            )
            
            logger.info(f"Uploaded {object_key} to {bucket}")
            return object_key
            
        except S3Error as e:
            logger.error(f"Upload failed: {e}")
            raise StorageError("upload", str(e))
    
    async def download_file(
        self,
        bucket: str,
        object_key: str
    ) -> bytes:
        """
        Download file from MinIO.
        
        Args:
            bucket: Source bucket name
            object_key: Object key/path within bucket
            
        Returns:
            File content as bytes
        """
        try:
            response = await self._run_sync(
                self._client.get_object,
                bucket,
                object_key
            )
            data = response.read()
            response.close()
            response.release_conn()
            return data
            
        except S3Error as e:
            logger.error(f"Download failed: {e}")
            raise StorageError("download", str(e))
    
    async def download_stream(
        self,
        bucket: str,
        object_key: str,
        chunk_size: int = 1024 * 1024  # 1MB chunks
    ) -> AsyncIterator[bytes]:
        """
        Stream download file in chunks.
        
        Args:
            bucket: Source bucket name
            object_key: Object key/path within bucket
            chunk_size: Size of each chunk
            
        Yields:
            File content chunks
        """
        try:
            response = await self._run_sync(
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
                
        except S3Error as e:
            logger.error(f"Stream download failed: {e}")
            raise StorageError("stream_download", str(e))
    
    async def get_presigned_url(
        self,
        bucket: str,
        object_key: str,
        expires_seconds: int = 3600
    ) -> str:
        """
        Generate presigned URL for direct client access.
        
        Args:
            bucket: Bucket name
            object_key: Object key
            expires_seconds: URL validity duration (default 1 hour)
            
        Returns:
            Presigned download URL
        """
        try:
            url = await self._run_sync(
                self._client.presigned_get_object,
                bucket,
                object_key,
                expires=timedelta(seconds=expires_seconds)
            )
            return url
            
        except S3Error as e:
            logger.error(f"Presigned URL generation failed: {e}")
            raise StorageError("presigned_url", str(e))
    
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
        try:
            await self._run_sync(
                self._client.remove_object,
                bucket,
                object_key
            )
            logger.info(f"Deleted {object_key} from {bucket}")
            
        except S3Error as e:
            logger.error(f"Delete failed: {e}")
            raise StorageError("delete", str(e))
    
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
        try:
            await self._run_sync(
                self._client.stat_object,
                bucket,
                object_key
            )
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise StorageError("file_exists", str(e))
    
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
        try:
            from minio.commonconfig import CopySource
            
            await self._run_sync(
                self._client.copy_object,
                dest_bucket,
                dest_key,
                CopySource(source_bucket, source_key)
            )
            return dest_key
            
        except S3Error as e:
            logger.error(f"Copy failed: {e}")
            raise StorageError("copy", str(e))


# Singleton instance
_storage_client: Optional[StorageClient] = None


def get_storage_client() -> StorageClient:
    """Get singleton storage client instance."""
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient()
    return _storage_client
