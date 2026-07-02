"""
Argus Core - Execution Mode Manager Implementation (Iteration 8)
==================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


class ExecutionMode(str, Enum):
    """The three execution modes."""
    LITE = "lite"           # CPU-only, quantized, minimal
    BALANCED = "balanced"   # GPU if available, else CPU
    RESEARCH = "research"   # GPU required, maximum accuracy


@dataclass
class ModeConfig:
    """Configuration snapshot for the active execution mode."""
    mode: ExecutionMode
    # Device
    device: str = "cpu"             # "cpu" | "cuda" | "cuda:0"
    use_gpu: bool = False
    # Precision
    precision: str = "fp32"         # "fp32" | "fp16" | "int8"
    mixed_precision: bool = False
    # Batch
    batch_size: int = 1
    # Detectors
    enable_sota_detectors: bool = True
    enable_timesformer: bool = True   # cc-by-nc-4.0, heavy
    enable_ecapa: bool = True
    # Defenses
    enable_rps: bool = True
    enable_adversarial_gate: bool = False  # slow
    enable_rs_lite: bool = False           # slow
    enable_certified_robustness: bool = False  # very slow
    # XAI
    enable_xai_attribution_output: bool = True
    enable_attn_lrp: bool = True
    # Calibration
    enable_calibration: bool = True
    # Memory guard
    enable_memory_guard: bool = True
    memory_limit_mb: int = 0  # 0 = auto-detect
    # Targets
    target_latency_ms: int = 5000
    target_accuracy: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "device": self.device,
            "use_gpu": self.use_gpu,
            "precision": self.precision,
            "mixed_precision": self.mixed_precision,
            "batch_size": self.batch_size,
            "enable_sota_detectors": self.enable_sota_detectors,
            "enable_timesformer": self.enable_timesformer,
            "enable_ecapa": self.enable_ecapa,
            "enable_rps": self.enable_rps,
            "enable_adversarial_gate": self.enable_adversarial_gate,
            "enable_rs_lite": self.enable_rs_lite,
            "enable_certified_robustness": self.enable_certified_robustness,
            "enable_xai_attribution_output": self.enable_xai_attribution_output,
            "enable_attn_lrp": self.enable_attn_lrp,
            "enable_calibration": self.enable_calibration,
            "enable_memory_guard": self.enable_memory_guard,
            "memory_limit_mb": self.memory_limit_mb,
            "target_latency_ms": self.target_latency_ms,
            "target_accuracy": self.target_accuracy,
        }


class ModeManager:
    """
    Manages the 3-mode execution system.

    Auto-detects the best mode based on hardware, or uses the
    EXECUTION_MODE env var if set. Provides a ModeConfig snapshot
    that all detectors and analyzers query.
    """

    # Per-mode defaults
    MODE_DEFAULTS: Dict[ExecutionMode, Dict[str, Any]] = {
        ExecutionMode.LITE: {
            "device": "cpu",
            "use_gpu": False,
            "precision": "int8",
            "mixed_precision": False,
            "batch_size": 1,
            "enable_sota_detectors": False,  # Use legacy ONNX only
            "enable_timesformer": False,
            "enable_ecapa": False,
            "enable_rps": True,
            "enable_adversarial_gate": False,
            "enable_rs_lite": False,
            "enable_certified_robustness": False,
            "enable_xai_attribution_output": True,
            "enable_attn_lrp": False,  # LXT needs gradients, slow on CPU
            "enable_calibration": True,
            "enable_memory_guard": True,
            "target_latency_ms": 2000,
            "target_accuracy": 0.85,
        },
        ExecutionMode.BALANCED: {
            "device": "cuda",  # will fall back to cpu if unavailable
            "use_gpu": True,
            "precision": "fp16",  # fp16 on GPU, fp32 on CPU
            "mixed_precision": True,
            "batch_size": 4,
            "enable_sota_detectors": True,
            "enable_timesformer": True,
            "enable_ecapa": True,
            "enable_rps": True,
            "enable_adversarial_gate": False,  # off by default
            "enable_rs_lite": False,
            "enable_certified_robustness": False,
            "enable_xai_attribution_output": True,
            "enable_attn_lrp": True,
            "enable_calibration": True,
            "enable_memory_guard": True,
            "target_latency_ms": 500,
            "target_accuracy": 0.95,
        },
        ExecutionMode.RESEARCH: {
            "device": "cuda",
            "use_gpu": True,
            "precision": "fp16",
            "mixed_precision": True,
            "batch_size": 16,
            "enable_sota_detectors": True,
            "enable_timesformer": True,
            "enable_ecapa": True,
            "enable_rps": True,
            "enable_adversarial_gate": True,   # full defenses
            "enable_rs_lite": True,
            "enable_certified_robustness": True,
            "enable_xai_attribution_output": True,
            "enable_attn_lrp": True,
            "enable_calibration": True,
            "enable_memory_guard": True,
            "target_latency_ms": 10000,  # latency secondary
            "target_accuracy": 0.98,
        },
    }

    def __init__(self):
        self._mode: Optional[ExecutionMode] = None
        self._config: Optional[ModeConfig] = None
        self._gpu_info: Dict[str, Any] = {}
        self._detect_hardware()

    # ------------------------------------------------------------------
    def _detect_hardware(self) -> None:
        """Detect available GPU hardware."""
        try:
            import torch
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                props = torch.cuda.get_device_properties(0)
                self._gpu_info = {
                    "available": True,
                    "count": n,
                    "name": props.name,
                    "total_memory_mb": props.total_memory // (1024 * 1024),
                }
                logger.info(
                    "GPU detected: %s (%dMB)",
                    self._gpu_info["name"],
                    self._gpu_info["total_memory_mb"],
                )
            else:
                self._gpu_info = {"available": False}
                logger.info("No GPU detected; CPU-only mode")
        except ImportError:
            self._gpu_info = {"available": False}
            logger.info("torch not installed; CPU-only mode")
        except Exception as e:
            self._gpu_info = {"available": False}
            logger.warning("GPU detection failed: %s", e)

    # ------------------------------------------------------------------
    def auto_detect_mode(self) -> ExecutionMode:
        """Auto-detect the best mode based on hardware."""
        if not self._gpu_info.get("available", False):
            return ExecutionMode.LITE
        vram_mb = self._gpu_info.get("total_memory_mb", 0)
        if vram_mb >= 16000:  # 16GB+ → research
            return ExecutionMode.RESEARCH
        return ExecutionMode.BALANCED

    # ------------------------------------------------------------------
    def initialize(self, force_mode: Optional[str] = None) -> ModeConfig:
        """
        Initialize the mode manager. Called once at startup.

        Args:
            force_mode: Optional mode override ("lite" | "balanced" | "research").
                        If None, reads EXECUTION_MODE env var, then auto-detects.

        Returns:
            ModeConfig snapshot for the active mode.
        """
        # Determine mode
        if force_mode:
            mode_str = force_mode.lower()
        else:
            mode_str = os.environ.get("EXECUTION_MODE", "").lower()

        if mode_str in ("lite", "balanced", "research"):
            mode = ExecutionMode(mode_str)
            logger.info("Execution mode set via config: %s", mode.value)
        else:
            mode = self.auto_detect_mode()
            logger.info("Execution mode auto-detected: %s", mode.value)

        # Build config from defaults
        defaults = self.MODE_DEFAULTS[mode].copy()

        # Adjust device based on actual hardware
        if mode == ExecutionMode.LITE:
            defaults["device"] = "cpu"
            defaults["use_gpu"] = False
        elif mode == ExecutionMode.BALANCED:
            if not self._gpu_info.get("available", False):
                # No GPU — fall back to CPU but keep balanced settings
                defaults["device"] = "cpu"
                defaults["use_gpu"] = False
                defaults["precision"] = "fp32"
                defaults["mixed_precision"] = False
                defaults["batch_size"] = 1
                logger.info(
                    "Balanced mode: GPU unavailable, using CPU with FP32"
                )
            else:
                defaults["device"] = "cuda:0"
                defaults["use_gpu"] = True
        elif mode == ExecutionMode.RESEARCH:
            if not self._gpu_info.get("available", False):
                logger.warning(
                    "Research mode requires GPU but none available. "
                    "Falling back to balanced CPU mode. Accuracy will be limited."
                )
                # Degrade to balanced-CPU
                mode = ExecutionMode.BALANCED
                defaults = self.MODE_DEFAULTS[ExecutionMode.BALANCED].copy()
                defaults["device"] = "cpu"
                defaults["use_gpu"] = False
                defaults["precision"] = "fp32"
                defaults["mixed_precision"] = False
                defaults["batch_size"] = 1
            else:
                defaults["device"] = "cuda:0"
                defaults["use_gpu"] = True

        # Auto-detect memory limit if not set
        if defaults.get("memory_limit_mb", 0) == 0:
            if self._gpu_info.get("available"):
                # Leave 20% headroom
                defaults["memory_limit_mb"] = int(
                    self._gpu_info["total_memory_mb"] * 0.8
                )
            else:
                # CPU: use 4GB as default limit
                defaults["memory_limit_mb"] = 4096

        self._mode = mode
        self._config = ModeConfig(mode=mode, **defaults)

        logger.info(
            "ModeManager initialized: mode=%s, device=%s, precision=%s, "
            "batch_size=%d, sota_detectors=%s",
            self._mode.value,
            self._config.device,
            self._config.precision,
            self._config.batch_size,
            self._config.enable_sota_detectors,
        )
        return self._config

    # ------------------------------------------------------------------
    @property
    def mode(self) -> ExecutionMode:
        if self._mode is None:
            self.initialize()
        return self._mode

    @property
    def config(self) -> ModeConfig:
        if self._config is None:
            self.initialize()
        return self._config

    @property
    def gpu_info(self) -> Dict[str, Any]:
        return self._gpu_info

    # ------------------------------------------------------------------
    def get_device_for_detector(self, detector_name: str) -> str:
        """
        Get the device for a specific detector.

        In Lite mode, all detectors use CPU.
        In Balanced/Research, SOTA detectors use GPU (if available),
        legacy detectors stay on CPU.
        """
        cfg = self.config
        if cfg.mode == ExecutionMode.LITE:
            return "cpu"
        # Balanced/Research: GPU for SOTA, CPU for legacy
        sota_detectors = {
            "CLIPLoRAImageDetector", "DINOv2ImageDetector", "SigLIPImageDetector",
            "SBIDetector", "UCFCrossForgeryDetector",
            "AASIST3AudioDetector", "Wav2Vec2XLSRMoELoRADetector",
            "ECAPATDNNAudioDetector", "CDPMambaDetector",
            "VideoMAEDetector", "AltFreeVideoDetector", "TimeSformerVideoDetector",
        }
        if detector_name in sota_detectors and cfg.use_gpu:
            return cfg.device
        return "cpu"

    def get_batch_size(self, detector_name: str) -> int:
        """Get batch size for a detector, scaled by mode."""
        cfg = self.config
        if cfg.mode == ExecutionMode.LITE:
            return 1
        return cfg.batch_size

    def should_enable_feature(self, feature: str) -> bool:
        """
        Check if a feature should be enabled in the current mode.

        Args:
            feature: One of: sota_detectors, timesformer, ecapa, rps,
                     adversarial_gate, rs_lite, certified_robustness,
                     xai_attribution_output, attn_lrp, calibration,
                     memory_guard.

        Returns:
            True if the feature should be enabled.
        """
        cfg = self.config
        feature_map = {
            "sota_detectors": cfg.enable_sota_detectors,
            "timesformer": cfg.enable_timesformer,
            "ecapa": cfg.enable_ecapa,
            "rps": cfg.enable_rps,
            "adversarial_gate": cfg.enable_adversarial_gate,
            "rs_lite": cfg.enable_rs_lite,
            "certified_robustness": cfg.enable_certified_robustness,
            "xai_attribution_output": cfg.enable_xai_attribution_output,
            "attn_lrp": cfg.enable_attn_lrp,
            "calibration": cfg.enable_calibration,
            "memory_guard": cfg.enable_memory_guard,
        }
        return feature_map.get(feature, False)


# ---------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------

_mode_manager: Optional[ModeManager] = None


def get_mode_manager() -> ModeManager:
    global _mode_manager
    if _mode_manager is None:
        _mode_manager = ModeManager()
        _mode_manager.initialize()
    return _mode_manager


def get_current_mode() -> ModeConfig:
    """Convenience: get the current ModeConfig."""
    return get_mode_manager().config
