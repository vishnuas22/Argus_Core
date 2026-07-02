"""
Argus Core - Multi-GPU Sharding (Iteration 6)
===============================================
Automatic device_map sharding for large models across multiple GPUs.

Research grounding:
- HuggingFace Accelerate `device_map="auto"` (HuggingFace, 2022-2026):
  automatically shards a model across available GPUs based on layer
  sizes + GPU memory. The model's forward pass transparently moves
  intermediate tensors between GPUs.
- Pipeline parallelism (Huang et al., "GPipe: Efficient Training of
  Giant Neural Networks using Pipeline Parallelism", NeurIPS 2019):
  the theoretical foundation for device_map sharding.
- For Argus: enables running Wav2Vec2-XLS-R-300M (1.2GB) + VideoMAE
  (350MB) + CLIP (600MB) + DINOv2 (350MB) + SigLIP (400MB) +
  TimeSformer (300MB) + ECAPA-TDNN (80MB) = ~3.3GB total across 2 T4s
  (16GB each) with room for batch processing.

Algorithm:
1. Detect available GPUs via torch.cuda.device_count().
2. If 1 GPU: load model on that GPU (no sharding).
3. If >1 GPU: use Accelerate's device_map="auto" to shard.
4. If 0 GPUs: fall back to CPU.

Strict-compat: pure-additive. Detectors that don't use this helper
continue to load on a single device as before.
"""

from inference.multi_gpu_sharding import (
    MultiGPUSharder,
    get_device_map,
    get_default_sharder,
)
from inference.memory_guard import (
    MemoryGuard,
    MemoryInfo,
    get_default_memory_guard,
)

__all__ = [
    "MultiGPUSharder",
    "get_device_map",
    "get_default_sharder",
    # Iteration 8: Memory guard
    "MemoryGuard", "MemoryInfo",
    "get_default_memory_guard",
]
