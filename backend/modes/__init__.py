"""
Argus Core - Execution Mode Manager (Iteration 8)
===================================================
Unified 3-mode execution system: Lite / Balanced / Research.

Engineering rule: "No code changes should be required to switch modes—
only configuration."

Mode characteristics:
- Lite (CPU): quantized ONNX, batch_size=1, FP32, no SOTA detectors
  unless they have CPU alternatives. Target: <2s per image on laptop.
- Balanced: GPU if available, else CPU. FP16 on GPU, FP32 on CPU.
  SOTA detectors enabled with auto-fallback. Target: <500ms per image
  on T4, <5s on CPU.
- Research/SOTA: GPU required. FP16 mixed precision, batch inference,
  full ensemble (9 detectors), full XAI, full defenses. Target:
  maximum accuracy, latency secondary.

Mode selection priority:
1. EXECUTION_MODE env var (lite | balanced | research)
2. Auto-detect: if torch.cuda.is_available() and >=8GB VRAM → research;
   if GPU available → balanced; else lite.
3. Default: balanced

Strict-compat: pure-additive. Existing code continues to work; the
ModeManager just provides a unified config snapshot that detectors
can query.
"""

from modes.mode_manager import (
    ExecutionMode,
    ModeConfig,
    ModeManager,
    get_mode_manager,
    get_current_mode,
)

__all__ = [
    "ExecutionMode",
    "ModeConfig",
    "ModeManager",
    "get_mode_manager",
    "get_current_mode",
]
