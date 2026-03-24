"""
Argus Core - Model Download Script
==================================
Download and verify ML models from HuggingFace and GitHub.

Implements: plans/MODEL_ARCHITECTURE_REALIGNMENT.md - Phase 2.2

Features:
- Async download with progress tracking
- SHA256 checksum verification
- Automatic retry on failure
- Model registry integration

Usage:
    python -m models.download_models [--model MODEL_NAME] [--all] [--verify]
"""

import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    sys.exit(1)

from models.registry import get_model_registry, ModelMetadata, DEFAULT_MODELS
from utils.logging import get_logger
from config import config

logger = get_logger(__name__)


@dataclass
class DownloadResult:
    """Result of a model download attempt."""
    model_name: str
    success: bool
    message: str
    download_time_seconds: float = 0.0
    file_size_mb: float = 0.0
    checksum_verified: bool = False
    error: Optional[str] = None


class ModelDownloader:
    """
    Download and verify ML models from remote sources.
    
    Supports:
    - HuggingFace model hub
    - GitHub releases
    - Direct URLs
    
    Features:
    - Progress tracking
    - Checksum verification
    - Automatic retry
    - Concurrent downloads
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: int = 300,
        max_concurrent: int = 3
    ):
        """
        Initialize downloader.
        
        Args:
            model_dir: Directory to save models (defaults to config.model_cache_dir)
            max_retries: Maximum retry attempts per download
            timeout_seconds: Download timeout in seconds
            max_concurrent: Maximum concurrent downloads
        """
        self.model_dir = Path(model_dir or config.model_cache_dir)
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelDownloader initialized with model_dir={self.model_dir}")
    
    def compute_sha256(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal SHA256 checksum
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def verify_checksum(
        self,
        file_path: Path,
        expected_checksum: str
    ) -> bool:
        """
        Verify file checksum against expected value.
        
        Args:
            file_path: Path to downloaded file
            expected_checksum: Expected SHA256 checksum (hex string)
            
        Returns:
            True if checksum matches, False otherwise
        """
        if not expected_checksum:
            logger.warning(f"No checksum provided for {file_path.name}, skipping verification")
            return True  # No checksum to verify against
        
        actual_checksum = self.compute_sha256(file_path)
        
        if actual_checksum.lower() == expected_checksum.lower():
            logger.info(f"Checksum verified for {file_path.name}")
            return True
        else:
            logger.error(
                f"Checksum mismatch for {file_path.name}: "
                f"expected={expected_checksum[:16]}... actual={actual_checksum[:16]}..."
            )
            return False
    
    async def download_model(
        self,
        model: ModelMetadata,
        force: bool = False
    ) -> DownloadResult:
        """
        Download a single model with verification.
        
        Args:
            model: Model metadata from registry
            force: Force re-download even if file exists
            
        Returns:
            DownloadResult with download status
        """
        async with self._semaphore:
            start_time = datetime.now()
            target_path = self.model_dir / f"{model.name}.onnx"
            
            # Check if file already exists
            if target_path.exists() and not force:
                # Verify existing file
                if model.checksum_sha256:
                    if self.verify_checksum(target_path, model.checksum_sha256):
                        file_size_mb = target_path.stat().st_size / (1024 * 1024)
                        return DownloadResult(
                            model_name=model.name,
                            success=True,
                            message="Model already exists and verified",
                            file_size_mb=file_size_mb,
                            checksum_verified=True
                        )
                    else:
                        logger.warning(f"Existing file has wrong checksum, re-downloading: {model.name}")
                else:
                    file_size_mb = target_path.stat().st_size / (1024 * 1024)
                    return DownloadResult(
                        model_name=model.name,
                        success=True,
                        message="Model already exists (no checksum to verify)",
                        file_size_mb=file_size_mb,
                        checksum_verified=False
                    )
            
            # Check if download URL is available
            if not model.download_url:
                return DownloadResult(
                    model_name=model.name,
                    success=False,
                    message="No download URL available for this model",
                    error="missing_download_url"
                )
            
            # Download with retry
            last_error = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info(f"Downloading {model.name} (attempt {attempt}/{self.max_retries})")
                    logger.debug(f"URL: {model.download_url}")
                    
                    async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                        response = await client.get(model.download_url, follow_redirects=True)
                        response.raise_for_status()
                        
                        # Write to temporary file first
                        temp_path = target_path.with_suffix(".tmp")
                        temp_path.write_bytes(response.content)
                        
                        # Verify checksum
                        checksum_verified = True
                        if model.checksum_sha256:
                            checksum_verified = self.verify_checksum(temp_path, model.checksum_sha256)
                            if not checksum_verified:
                                temp_path.unlink()
                                return DownloadResult(
                                    model_name=model.name,
                                    success=False,
                                    message="Downloaded file failed checksum verification",
                                    error="checksum_mismatch"
                                )
                        
                        # Move to final location
                        temp_path.rename(target_path)
                        
                        # Calculate stats
                        download_time = (datetime.now() - start_time).total_seconds()
                        file_size_mb = target_path.stat().st_size / (1024 * 1024)
                        
                        logger.info(
                            f"Successfully downloaded {model.name}: "
                            f"{file_size_mb:.1f}MB in {download_time:.1f}s"
                        )
                        
                        return DownloadResult(
                            model_name=model.name,
                            success=True,
                            message="Download successful",
                            download_time_seconds=download_time,
                            file_size_mb=file_size_mb,
                            checksum_verified=checksum_verified
                        )
                        
                except httpx.HTTPStatusError as e:
                    last_error = f"HTTP {e.response.status_code}: {e.response.text[:100]}"
                    logger.warning(f"Download failed (attempt {attempt}): {last_error}")
                except httpx.RequestError as e:
                    last_error = f"Request error: {str(e)}"
                    logger.warning(f"Download failed (attempt {attempt}): {last_error}")
                except Exception as e:
                    last_error = f"Unexpected error: {str(e)}"
                    logger.error(f"Download failed (attempt {attempt}): {last_error}")
                
                # Wait before retry
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            return DownloadResult(
                model_name=model.name,
                success=False,
                message=f"Download failed after {self.max_retries} attempts",
                error=last_error
            )
    
    async def download_all(
        self,
        models: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict[str, DownloadResult]:
        """
        Download multiple models concurrently.
        
        Args:
            models: List of model names to download (None = all with download URLs)
            force: Force re-download
            
        Returns:
            Dict mapping model names to download results
        """
        registry = get_model_registry()
        
        # Determine which models to download
        if models:
            model_names = models
        else:
            # Download all models that have download URLs
            model_names = [
                name for name, meta in DEFAULT_MODELS.items()
                if meta.download_url
            ]
        
        logger.info(f"Starting download of {len(model_names)} models")
        
        # Create tasks
        tasks = []
        for name in model_names:
            try:
                metadata = registry.get_model_metadata(name)
                tasks.append(self.download_model(metadata, force=force))
            except Exception as e:
                logger.error(f"Failed to get metadata for {name}: {e}")
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # Build result dict
        return {r.model_name: r for r in results}
    
    def verify_all(self) -> Dict[str, bool]:
        """
        Verify checksums of all downloaded models.
        
        Returns:
            Dict mapping model names to verification status
        """
        registry = get_model_registry()
        results = {}
        
        for name, meta in DEFAULT_MODELS.items():
            model_path = self.model_dir / f"{name}.onnx"
            
            if not model_path.exists():
                results[name] = False
                logger.warning(f"Model not found: {name}")
                continue
            
            if meta.checksum_sha256:
                results[name] = self.verify_checksum(model_path, meta.checksum_sha256)
            else:
                results[name] = True  # No checksum to verify
                logger.info(f"Model exists (no checksum): {name}")
        
        return results
    
    def print_status_report(self, results: Dict[str, DownloadResult]) -> None:
        """Print a formatted status report."""
        print("\n" + "=" * 60)
        print("MODEL DOWNLOAD REPORT")
        print("=" * 60)
        
        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        
        for name, result in results.items():
            status = "✓" if result.success else "✗"
            size_str = f"{result.file_size_mb:.1f}MB" if result.file_size_mb > 0 else "N/A"
            checksum_str = "✓" if result.checksum_verified else "✗"
            
            print(f"\n{status} {name}")
            print(f"  Status: {result.message}")
            print(f"  Size: {size_str}")
            print(f"  Checksum verified: {checksum_str}")
            
            if result.error:
                print(f"  Error: {result.error}")
        
        print("\n" + "-" * 60)
        print(f"SUMMARY: {success_count}/{total_count} models downloaded successfully")
        print("=" * 60 + "\n")


async def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ML models for Argus")
    parser.add_argument(
        "--model", "-m",
        help="Download a specific model",
        type=str,
        default=None
    )
    parser.add_argument(
        "--all", "-a",
        help="Download all models with download URLs",
        action="store_true"
    )
    parser.add_argument(
        "--verify", "-v",
        help="Verify existing model checksums",
        action="store_true"
    )
    parser.add_argument(
        "--force", "-f",
        help="Force re-download even if file exists",
        action="store_true"
    )
    parser.add_argument(
        "--list", "-l",
        help="List available models",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.list:
        print("\nAvailable Models:")
        print("-" * 60)
        for name, meta in DEFAULT_MODELS.items():
            has_url = "✓" if meta.download_url else "✗"
            print(f"  {name}")
            print(f"    Category: {meta.category.value}")
            print(f"    Description: {meta.description[:50]}...")
            print(f"    Download URL: {has_url}")
            print(f"    Source: {meta.source or 'N/A'}")
            print()
        return
    
    if args.verify:
        print("\nVerifying model checksums...")
        results = downloader.verify_all()
        
        print("\nVerification Results:")
        print("-" * 40)
        for name, verified in results.items():
            status = "✓" if verified else "✗"
            print(f"  {status} {name}")
        
        verified_count = sum(1 for v in results.values() if v)
        print(f"\n{verified_count}/{len(results)} models verified")
        return
    
    if args.model:
        registry = get_model_registry()
        try:
            metadata = registry.get_model_metadata(args.model)
            result = await downloader.download_model(metadata, force=args.force)
            downloader.print_status_report({args.model: result})
        except Exception as e:
            print(f"Error: Model '{args.model}' not found in registry")
            print(f"Available models: {list(DEFAULT_MODELS.keys())}")
            sys.exit(1)
        return
    
    if args.all:
        results = await downloader.download_all(force=args.force)
        downloader.print_status_report(results)
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
