"""
Argus Core - Multi-GPU Sharding Implementation (Iteration 6)
==============================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int


class MultiGPUSharder:
    """
    Manages multi-GPU model sharding via HuggingFace Accelerate.
    """

    def __init__(self, prefer_multi_gpu: bool = True):
        self.prefer_multi_gpu = prefer_multi_gpu
        self._available_gpus: List[GPUInfo] = []
        self._detect_gpus()

    # ------------------------------------------------------------------
    def _detect_gpus(self) -> None:
        """Detect available GPUs."""
        try:
            import torch
            n = torch.cuda.device_count()
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                # Free memory via memory_allocated (approximate)
                free_mb = props.total_memory // (1024 * 1024)
                self._available_gpus.append(GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_mb=free_mb,
                    free_memory_mb=free_mb,
                ))
            if self._available_gpus:
                logger.info(
                    "MultiGPUSharder: detected %d GPU(s): %s",
                    len(self._available_gpus),
                    [(g.name, f"{g.total_memory_mb}MB") for g in self._available_gpus],
                )
            else:
                logger.info("MultiGPUSharder: no GPUs detected; using CPU")
        except Exception as e:
            logger.warning("GPU detection failed: %s", e)

    # ------------------------------------------------------------------
    @property
    def num_gpus(self) -> int:
        return len(self._available_gpus)

    @property
    def has_multi_gpu(self) -> bool:
        return len(self._available_gpus) > 1

    # ------------------------------------------------------------------
    def get_device_map(self, model_size_mb: int = 0) -> Optional[Dict[str, int]]:
        """
        Get a device_map for sharding a model across GPUs.

        Args:
            model_size_mb: Estimated model size in MB (for planning).

        Returns:
            device_map dict for HuggingFace from_pretrained(device_map=...),
            or None if single-GPU/CPU.
        """
        if not self._available_gpus:
            return None
        if len(self._available_gpus) == 1 or not self.prefer_multi_gpu:
            return None  # Single GPU — no sharding needed

        # Use Accelerate's auto device_map
        try:
            from accelerate import infer_auto_device_map
            # We can't call infer_auto_device_map without the model,
            # but we can return "auto" which tells from_pretrained to
            # use the built-in sharding.
            return "auto"
        except ImportError:
            logger.warning(
                "accelerate not installed; cannot shard across GPUs. "
                "Install with: pip install accelerate"
            )
            return None

    # ------------------------------------------------------------------
    def load_model_sharded(
        self,
        model_loader_fn: Any,
        model_id: str,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Load a model with automatic multi-GPU sharding.

        Args:
            model_loader_fn: The from_pretrained class method (e.g.
                AutoModel.from_pretrained).
            model_id: HuggingFace model ID.
            cache_dir: Cache directory.
            **kwargs: Additional kwargs passed to from_pretrained.

        Returns:
            Loaded model (sharded across GPUs if multi-GPU).
        """
        device_map = self.get_device_map()
        load_kwargs = {"cache_dir": cache_dir}
        if device_map:
            load_kwargs["device_map"] = device_map
            logger.info(
                "Loading %s with device_map=%s (multi-GPU sharding)",
                model_id, device_map,
            )
        else:
            # Single GPU or CPU
            if self._available_gpus:
                load_kwargs["device"] = f"cuda:{self._available_gpus[0].index}"
            else:
                load_kwargs["device"] = "cpu"
            logger.info(
                "Loading %s on %s (single device)",
                model_id, load_kwargs["device"],
            )

        load_kwargs.update(kwargs)
        return model_loader_fn(model_id, **load_kwargs)

    # ------------------------------------------------------------------
    def get_shard_info(self) -> Dict[str, Any]:
        """Return summary info about the current sharding setup."""
        return {
            "num_gpus": len(self._available_gpus),
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_memory_mb": g.total_memory_mb,
                }
                for g in self._available_gpus
            ],
            "multi_gpu_enabled": self.has_multi_gpu and self.prefer_multi_gpu,
        }


# ---------------------------------------------------------------------
def get_device_map(model_size_mb: int = 0) -> Optional[Dict[str, int]]:
    """Convenience function: get a device_map from the default sharder."""
    return get_default_sharder().get_device_map(model_size_mb)


_default_sharder: Optional[MultiGPUSharder] = None


def get_default_sharder() -> MultiGPUSharder:
    global _default_sharder
    if _default_sharder is None:
        prefer = os.environ.get("ARGUS_PREFER_MULTI_GPU", "true").lower() == "true"
        _default_sharder = MultiGPUSharder(prefer_multi_gpu=prefer)
    return _default_sharder
