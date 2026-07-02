"""
Argus Core - Storage Interface
==============================
Abstract base class defining the contract for storage implementations.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - storage/storage.py
"""

from abc import ABC, abstractmethod
from typing import Union, BinaryIO, Optional, AsyncIterator


class IStorage(ABC):
    """
    Abstract base class for object storage.
    
    Defines contract for MinIO or any S3-compatible storage backend.
    
    Buckets:
    - argus-uploads: Raw uploaded files
    - argus-preprocessed: Extracted frames, audio
    - argus-results: Heatmaps, reports
    """
    
    @abstractmethod
    async def upload_file(
        self,
        file: Union[bytes, BinaryIO],
        bucket: str,
        object_key: str,
        content_type: str
    ) -> str:
        """
        Upload file to object storage.
        
        Args:
            file: File content as bytes or binary stream
            bucket: Target bucket name
            object_key: Object key/path within bucket
            content_type: MIME type of file
            
        Returns:
            Object key for retrieval
            
        Raises:
            StorageError: If upload fails
        """
        pass
    
    @abstractmethod
    async def download_file(
        self,
        bucket: str,
        object_key: str
    ) -> bytes:
        """
        Download file from object storage.
        
        Args:
            bucket: Source bucket name
            object_key: Object key/path within bucket
            
        Returns:
            File content as bytes
            
        Raises:
            StorageError: If download fails
            NotFoundError: If object doesn't exist
        """
        pass
    
    @abstractmethod
    async def download_stream(
        self,
        bucket: str,
        object_key: str
    ) -> AsyncIterator[bytes]:
        """
        Stream download file in chunks.
        
        Args:
            bucket: Source bucket name
            object_key: Object key/path within bucket
            
        Yields:
            File content chunks
        """
        pass
    
    @abstractmethod
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
            expires_seconds: URL validity duration
            
        Returns:
            Presigned download URL
        """
        pass
    
    @abstractmethod
    async def delete_file(
        self,
        bucket: str,
        object_key: str
    ) -> None:
        """
        Delete file from object storage.
        
        Args:
            bucket: Bucket name
            object_key: Object key to delete
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def list_objects(
        self,
        bucket: str,
        prefix: Optional[str] = None
    ) -> list:
        """
        List objects in bucket with optional prefix filter.
        
        Args:
            bucket: Bucket name
            prefix: Optional path prefix to filter
            
        Returns:
            List of object keys
        """
        pass
    
    @abstractmethod
    async def ensure_bucket(self, bucket: str) -> None:
        """
        Ensure bucket exists, create if not.
        
        Args:
            bucket: Bucket name to ensure
        """
        pass
