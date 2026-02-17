"""
Argus Core - Model Initialization Module
========================================
Automatic model download and initialization on first use.

Implements: plans/MODEL_ARCHITECTURE_REALIGNMENT.md - Phase 2.2

Features:
- Automatic model download on first access
- Background model preloading
- Model health checks
- Integration with InferenceEngine
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import threading
import logging

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelStatus:
    """Status of a model in the system."""
    name: str
    available: bool
    file_size_mb: float
    is_placeholder: bool
    last_checked: datetime
    download_attempted: bool = False
    download_success: bool = False
    error_message: Optional[str] = None


class ModelInitializer:
    """
    Manages automatic model download and initialization.
    
    Features:
    - Checks model availability on startup
    - Downloads missing models automatically
    - Validates model integrity
    - Background preloading for faster first inference
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for consistent model management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        auto_download: bool = True,
        background_download: bool = True
    ):
        """
        Initialize the model initializer.
        
        Args:
            model_dir: Directory containing models
            auto_download: Whether to automatically download missing models
            background_download: Whether to download in background thread
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.model_dir = Path(model_dir or config.model_cache_dir)
        self.auto_download = auto_download
        self.background_download = background_download
        self._model_status: Dict[str, ModelStatus] = {}
        self._download_in_progress: Set[str] = set()
        self._initialized = True
        
        logger.info(f"ModelInitializer created: model_dir={self.model_dir}")
    
    def check_model_availability(self, model_name: str) -> ModelStatus:
        """
        Check if a model is available and valid.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            ModelStatus with availability information
        """
        model_path = self.model_dir / f"{model_name}.onnx"
        
        if model_path.exists():
            file_size = model_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            # A real ONNX model should be at least 1KB
            # Placeholders are typically ~130 bytes
            is_placeholder = file_size < 1024
            
            status = ModelStatus(
                name=model_name,
                available=True,
                file_size_mb=file_size_mb,
                is_placeholder=is_placeholder,
                last_checked=datetime.now()
            )
            
            if is_placeholder:
                logger.warning(
                    f"Model {model_name} appears to be a placeholder "
                    f"({file_size} bytes). Real model needed."
                )
        else:
            status = ModelStatus(
                name=model_name,
                available=False,
                file_size_mb=0,
                is_placeholder=False,
                last_checked=datetime.now()
            )
            logger.info(f"Model {model_name} not found at {model_path}")
        
        self._model_status[model_name] = status
        return status
    
    def check_all_models(self, model_names: List[str]) -> Dict[str, ModelStatus]:
        """
        Check availability of multiple models.
        
        Args:
            model_names: List of model names to check
            
        Returns:
            Dict mapping model names to their status
        """
        for name in model_names:
            self.check_model_availability(name)
        return self._model_status
    
    def download_model(
        self,
        model_name: str,
        force: bool = False
    ) -> bool:
        """
        Download a model if missing.
        
        Args:
            model_name: Name of the model to download
            force: Force re-download even if exists
            
        Returns:
            True if download successful, False otherwise
        """
        if model_name in self._download_in_progress:
            logger.info(f"Download already in progress for {model_name}")
            return False
        
        self._download_in_progress.add(model_name)
        
        try:
            # Check current status
            status = self.check_model_availability(model_name)
            
            if status.available and not status.is_placeholder and not force:
                logger.info(f"Model {model_name} already available ({status.file_size_mb:.1f}MB)")
                return True
            
            # Attempt download
            logger.info(f"Downloading model {model_name}...")
            
            try:
                from models.model_downloader import ProductionModelDownloader
                
                downloader = ProductionModelDownloader(model_dir=str(self.model_dir))
                result = downloader.download_model(model_name, force=force)
                
                if result.get("success"):
                    logger.info(f"Successfully downloaded {model_name}: {result.get('message')}")
                    
                    # Update status
                    self._model_status[model_name] = ModelStatus(
                        name=model_name,
                        available=True,
                        file_size_mb=result.get("file_size_mb", 0),
                        is_placeholder=False,
                        last_checked=datetime.now(),
                        download_attempted=True,
                        download_success=True
                    )
                    return True
                else:
                    logger.warning(f"Failed to download {model_name}: {result.get('message')}")
                    
                    self._model_status[model_name] = ModelStatus(
                        name=model_name,
                        available=status.available,
                        file_size_mb=status.file_size_mb,
                        is_placeholder=status.is_placeholder,
                        last_checked=datetime.now(),
                        download_attempted=True,
                        download_success=False,
                        error_message=result.get("message")
                    )
                    return False
                    
            except ImportError as e:
                logger.error(f"Model downloader not available: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False
        finally:
            self._download_in_progress.discard(model_name)
    
    def download_models_async(
        self,
        model_names: List[str],
        callback=None
    ) -> threading.Thread:
        """
        Download models in a background thread.
        
        Args:
            model_names: List of model names to download
            callback: Optional callback function(status_dict)
            
        Returns:
            Thread running the download
        """
        def download_thread():
            results = {}
            for name in model_names:
                results[name] = self.download_model(name)
            
            if callback:
                callback(results)
        
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()
        return thread
    
    def ensure_models_available(
        self,
        model_names: List[str],
        timeout_seconds: int = 300
    ) -> Dict[str, bool]:
        """
        Ensure all specified models are available.
        
        Downloads missing models synchronously.
        
        Args:
            model_names: List of model names to ensure
            timeout_seconds: Maximum time to wait for downloads
            
        Returns:
            Dict mapping model names to availability status
        """
        results = {}
        
        for name in model_names:
            status = self.check_model_availability(name)
            
            if not status.available or status.is_placeholder:
                if self.auto_download:
                    success = self.download_model(name)
                    results[name] = success
                else:
                    results[name] = False
            else:
                results[name] = True
        
        return results
    
    def get_missing_models(self, model_names: List[str]) -> List[str]:
        """
        Get list of models that are missing or are placeholders.
        
        Args:
            model_names: List of model names to check
            
        Returns:
            List of model names that need to be downloaded
        """
        missing = []
        for name in model_names:
            status = self.check_model_availability(name)
            if not status.available or status.is_placeholder:
                missing.append(name)
        return missing
    
    def get_status_report(self) -> Dict[str, Dict]:
        """
        Get a status report for all checked models.
        
        Returns:
            Dict with status information for each model
        """
        return {
            name: {
                "available": status.available,
                "file_size_mb": status.file_size_mb,
                "is_placeholder": status.is_placeholder,
                "download_attempted": status.download_attempted,
                "download_success": status.download_success,
                "error": status.error_message
            }
            for name, status in self._model_status.items()
        }
    
    def print_status_report(self) -> None:
        """Print a formatted status report."""
        print("\n" + "=" * 60)
        print("MODEL STATUS REPORT")
        print("=" * 60)
        
        for name, status in self._model_status.items():
            if status.available and not status.is_placeholder:
                icon = "✓"
                status_text = f"Available ({status.file_size_mb:.1f}MB)"
            elif status.available and status.is_placeholder:
                icon = "⚠"
                status_text = f"Placeholder ({status.file_size_mb:.3f}MB)"
            else:
                icon = "✗"
                status_text = "Missing"
            
            print(f"{icon} {name}: {status_text}")
            
            if status.error_message:
                print(f"  Error: {status.error_message}")
        
        print("=" * 60 + "\n")


# Global initializer instance
_initializer: Optional[ModelInitializer] = None


def get_model_initializer(
    model_dir: Optional[str] = None,
    auto_download: bool = True
) -> ModelInitializer:
    """
    Get the global model initializer instance.
    
    Args:
        model_dir: Directory containing models
        auto_download: Whether to auto-download missing models
        
    Returns:
        ModelInitializer instance
    """
    global _initializer
    if _initializer is None:
        _initializer = ModelInitializer(
            model_dir=model_dir,
            auto_download=auto_download
        )
    return _initializer


def ensure_models_for_analyzer(
    analyzer_type: str,
    model_names: List[str]
) -> Dict[str, bool]:
    """
    Ensure models are available for a specific analyzer.
    
    This is the main entry point for analyzers to ensure
    their required models are available.
    
    Args:
        analyzer_type: Type of analyzer (audio, video, text, image)
        model_names: List of required model names
        
    Returns:
        Dict mapping model names to availability
    """
    initializer = get_model_initializer()
    
    logger.info(f"Ensuring models for {analyzer_type} analyzer: {model_names}")
    
    return initializer.ensure_models_available(model_names)


def check_model_status(model_name: str) -> ModelStatus:
    """
    Check the status of a specific model.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        ModelStatus for the model
    """
    initializer = get_model_initializer()
    return initializer.check_model_availability(model_name)


# Convenience function for quick model check
def is_model_ready(model_name: str) -> bool:
    """
    Check if a model is ready for use.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if model is available and not a placeholder
    """
    status = check_model_status(model_name)
    return status.available and not status.is_placeholder
