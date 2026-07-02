"""
Argus Core - Memory Guard (Iteration 8)
========================================
Monitors VRAM/RAM usage and triggers automatic fallback when memory
constraints are hit.

Engineering rule: "Memory constraints should trigger automatic fallback
to smaller or quantized models."

Algorithm:
1. Before loading a model, check available memory.
2. If the model's estimated VRAM exceeds available memory:
   a. Try FP16 quantization (halves memory).
   b. Try INT8 quantization (quarters memory).
   c. Try the CPU alternative model (from manifest).
   d. If all fail, skip the model and log a warning.
3. After loading, periodically check memory usage.
4. If memory pressure is detected during inference:
   a. Evict the least-recently-used model from the cache.
   b. Retry the inference.

Strict-compat: pure-additive. Detectors that don't use MemoryGuard
continue to work as before.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryInfo:
    """Memory status snapshot."""
    total_mb: int
    used_mb: int
    free_mb: int
    is_gpu: bool


class MemoryGuard:
    """
    Monitors memory and triggers fallback when constraints are hit.
    """

    def __init__(self, limit_mb: int = 0, check_interval_s: float = 30.0):
        self._limit_mb = limit_mb
        self._check_interval = check_interval_s
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._cached_info: Optional[MemoryInfo] = None
        self._eviction_callbacks: Dict[str, Callable] = {}
        logger.info(
            "MemoryGuard initialized: limit=%dMB, interval=%.0fs",
            limit_mb, check_interval_s,
        )

    # ------------------------------------------------------------------
    def get_memory_info(self, device: str = "cuda") -> MemoryInfo:
        """
        Get current memory info for a device.

        Args:
            device: "cuda" or "cpu"

        Returns:
            MemoryInfo with total/used/free in MB.
        """
        if device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    return self._get_cpu_memory()
                idx = 0
                if ":" in device:
                    idx = int(device.split(":")[1])
                total = torch.cuda.get_device_properties(idx).total_memory
                reserved = torch.cuda.memory_reserved(idx)
                allocated = torch.cuda.memory_allocated(idx)
                free = total - allocated
                return MemoryInfo(
                    total_mb=int(total // (1024 * 1024)),
                    used_mb=int(allocated // (1024 * 1024)),
                    free_mb=int(free // (1024 * 1024)),
                    is_gpu=True,
                )
            except Exception:
                return self._get_cpu_memory()
        return self._get_cpu_memory()

    def _get_cpu_memory(self) -> MemoryInfo:
        """Get system RAM info."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return MemoryInfo(
                total_mb=int(mem.total // (1024 * 1024)),
                used_mb=int(mem.used // (1024 * 1024)),
                free_mb=int(mem.available // (1024 * 1024)),
                is_gpu=False,
            )
        except ImportError:
            # Fallback: read /proc/meminfo (Linux only)
            try:
                with open("/proc/meminfo", "r") as fh:
                    lines = fh.readlines()
                info = {}
                for line in lines:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = int(parts[1].strip().split()[0]) // 1024  # KB→MB
                        info[key] = val
                return MemoryInfo(
                    total_mb=info.get("MemTotal", 0),
                    used_mb=info.get("MemTotal", 0) - info.get("MemAvailable", 0),
                    free_mb=info.get("MemAvailable", 0),
                    is_gpu=False,
                )
            except Exception:
                return MemoryInfo(total_mb=0, used_mb=0, free_mb=0, is_gpu=False)

    # ------------------------------------------------------------------
    def can_load_model(
        self,
        model_vram_mb: int,
        device: str = "cuda",
    ) -> bool:
        """
        Check if a model can be loaded given current memory constraints.

        Args:
            model_vram_mb: Estimated VRAM needed by the model.
            device: Target device.

        Returns:
            True if the model can be loaded, False otherwise.
        """
        info = self.get_memory_info(device)
        if info.free_mb < model_vram_mb:
            logger.warning(
                "Insufficient memory for model (%dMB needed, %dMB free on %s)",
                model_vram_mb, info.free_mb, device,
            )
            return False
        return True

    # ------------------------------------------------------------------
    def get_fallback_precision(
        self,
        model_vram_mb: int,
        device: str = "cuda",
    ) -> Optional[str]:
        """
        Determine the best fallback precision if the model doesn't fit.

        Args:
            model_vram_mb: Estimated VRAM needed at FP32.
            device: Target device.

        Returns:
            "fp16" if FP16 would fit, "int8" if INT8 would fit,
            None if neither fits.
        """
        info = self.get_memory_info(device)
        # FP16 halves memory
        if info.free_mb >= model_vram_mb // 2:
            return "fp16"
        # INT8 quarters memory
        if info.free_mb >= model_vram_mb // 4:
            return "int8"
        return None

    # ------------------------------------------------------------------
    def register_eviction_callback(
        self,
        model_name: str,
        callback: Callable,
    ) -> None:
        """Register a callback to evict a model from cache."""
        with self._lock:
            self._eviction_callbacks[model_name] = callback

    def evict_lru(self) -> bool:
        """
        Trigger LRU eviction of cached models.

        Returns:
            True if a model was evicted, False if none registered.
        """
        with self._lock:
            if not self._eviction_callbacks:
                return False
            # Evict the first registered model (in production, use LRU order)
            model_name, callback = next(iter(self._eviction_callbacks.items()))
        try:
            logger.info("Memory pressure: evicting model %s", model_name)
            callback()
            with self._lock:
                self._eviction_callbacks.pop(model_name, None)
            return True
        except Exception as e:
            logger.warning("Eviction of %s failed: %s", model_name, e)
            return False

    # ------------------------------------------------------------------
    def check_and_evict_if_needed(self, device: str = "cuda") -> bool:
        """
        Check memory pressure and evict if needed.

        Returns:
            True if eviction occurred, False otherwise.
        """
        if self._limit_mb == 0:
            return False
        info = self.get_memory_info(device)
        if info.used_mb > self._limit_mb:
            logger.warning(
                "Memory pressure: %dMB used > %dMB limit; triggering eviction",
                info.used_mb, self._limit_mb,
            )
            return self.evict_lru()
        return False


# ---------------------------------------------------------------------
_default_guard: Optional[MemoryGuard] = None


def get_default_memory_guard() -> MemoryGuard:
    global _default_guard
    if _default_guard is None:
        from config import config
        # Use the mode manager's memory limit
        try:
            from modes import get_current_mode
            mode_cfg = get_current_mode()
            limit = mode_cfg.memory_limit_mb
        except Exception:
            limit = getattr(config, "gpu_memory_limit_mb", 4096)
        _default_guard = MemoryGuard(limit_mb=limit)
    return _default_guard
