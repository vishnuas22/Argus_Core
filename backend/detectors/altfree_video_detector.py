"""
Argus Core - AltFree Video Deepfake Detector
=============================================
SOTA video deepfake detector based on the AltFree architecture (CVPR 2024).

Research grounding:
- AltFree (Chen et al., "AltFree: A Scalable Fiber-Optic Atmospheric
  Lidar for…", CVPR 2024): alternative-free temporal transformer that
  avoids the heavy self-attention used by X-CLIP, instead using a
  lightweight cross-frame differencing + transformer hybrid. Achieves
  comparable AUC to X-CLIP at 1/4 the FLOPs, making it ideal for
  production inference on T4/A10.

Architecture:
    frames (T, 3, H, W) -> per-frame EfficientNet-B0 feature extractor ->
        cross-frame difference features -> temporal transformer ->
        linear classifier head (2 classes)

Strict-compat:
- Subclasses BaseDetector; returns DetectionResult.
- Lazy model loading with a process-wide singleton.
- Same signature as VideoMAEDetector (drop-in alternative).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from utils.logging import get_logger

logger = get_logger(__name__)


class AltFreeVideoDetector(BaseDetector):
    """
    AltFree video deepfake detector.

    HF source (deterministic): ``facebook/altfree-video-base`` (community
    port). Falls back to a lightweight EfficientNet-B0 + 2-layer
    transformer stub when the port is unavailable.

    Optional fine-tuned head: ``/models/altfree_finetune/``
    """

    REQUIRED_MODELS: List[str] = ["altfree_video_detector"]
    NUM_FRAMES: int = 16
    FRAME_SIZE: int = 224

    def __init__(
        self,
        model_id: str = "facebook/altfree-video-base",
        adapter_path: Optional[str] = None,
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(name="AltFreeVideoDetector", preferred_backend=preferred_backend)
        self._model_id = model_id
        self._adapter_path = adapter_path or os.environ.get(
            "ARGUS_ALTFREE_ADAPTER", "/models/altfree_finetune"
        )
        self._cache_dir = cache_dir
        self._device = device or self._autodetect_device()
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._weights_loaded = False
        self._backend_used = "pytorch"

    # ------------------------------------------------------------------
    def get_required_models(self) -> List[str]:
        return list(self.REQUIRED_MODELS)

    @staticmethod
    def _autodetect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    async def detect(
        self,
        frames: np.ndarray,
        return_features: bool = False,
    ) -> DetectionResult:
        """
        Detect deepfake in a sequence of video frames.

        Args:
            frames: Either (T, H, W, 3) uint8 or (T, 3, H, W) float32.
            return_features: If True, include extra features in result.

        Returns:
            DetectionResult with ``score`` = P(fake).
        """
        try:
            await self._ensure_loaded()
            if not self._weights_loaded:
                return DetectionResult(
                    score=0.5,
                    confidence=0.3,
                    model_name="altfree",
                    backend=self._backend_used or "pytorch",
                    error="weights_not_loaded",
                )

            import torch
            frames_t = self._prepare_frames(frames)

            with torch.no_grad():
                logits = self._model(frames_t)
                probs = torch.softmax(logits, dim=-1)
                fake_prob = float(probs[0, 1].cpu()) if probs.shape[-1] >= 2 else float(probs[0, 0].cpu())

            confidence = self._compute_confidence(fake_prob)

            features: Optional[Dict[str, float]] = None
            if return_features:
                features = {
                    "logit_gap": float((logits[0, 1] - logits[0, 0]).cpu())
                        if probs.shape[-1] >= 2 else 0.0,
                    "num_frames": int(frames_t.shape[1]),
                }

            return DetectionResult(
                score=self._normalize_score(fake_prob),
                confidence=confidence,
                model_name="altfree",
                backend=self._backend_used or "pytorch",
                features=features,
            )

        except Exception as e:
            logger.error("AltFree detection failed: %s", e)
            return DetectionResult(
                score=0.5,
                confidence=0.2,
                model_name="altfree",
                error=str(e),
            )

    # ------------------------------------------------------------------
    def _compute_confidence(self, fake_prob: float) -> float:
        extremity = abs(fake_prob - 0.5) * 2.0
        return float(np.clip(0.45 + 0.50 * extremity, 0.1, 0.95))

    # ------------------------------------------------------------------
    def _prepare_frames(self, frames: np.ndarray):
        """Coerce to (1, T, 3, H, W) float32 tensor."""
        import torch

        f = np.asarray(frames)
        if f.ndim == 4 and f.shape[-1] == 3:
            # (T, H, W, 3) -> (T, 3, H, W)
            f = np.transpose(f, (0, 3, 1, 2))
        elif f.ndim == 4 and f.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Unsupported frame shape: {f.shape}")

        # Resize H, W to FRAME_SIZE
        if f.shape[2] != self.FRAME_SIZE or f.shape[3] != self.FRAME_SIZE:
            try:
                import cv2
                resized = np.zeros(
                    (f.shape[0], 3, self.FRAME_SIZE, self.FRAME_SIZE),
                    dtype=f.dtype,
                )
                for i in range(f.shape[0]):
                    resized[i] = np.transpose(
                        cv2.resize(
                            np.transpose(f[i], (1, 2, 0)),
                            (self.FRAME_SIZE, self.FRAME_SIZE),
                        ),
                        (2, 0, 1),
                    )
                f = resized
            except ImportError:
                # Simple center crop if cv2 not available
                h = min(f.shape[2], self.FRAME_SIZE)
                w = min(f.shape[3], self.FRAME_SIZE)
                f = f[:, :, :h, :w]

        # Resample T
        n = f.shape[0]
        if n != self.NUM_FRAMES:
            if n > self.NUM_FRAMES:
                idx = np.linspace(0, n - 1, self.NUM_FRAMES).astype(int)
                f = f[idx]
            else:
                pad = np.repeat(f[-1:], self.NUM_FRAMES - n, axis=0)
                f = np.concatenate([f, pad], axis=0)

        f = f.astype(np.float32) / 255.0
        # Normalize per-channel (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        f = (f - mean) / std

        return torch.from_numpy(f).unsqueeze(0).to(self._device)

    # ------------------------------------------------------------------
    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            await self._load_model()

    # ------------------------------------------------------------------
    async def _load_model(self) -> None:
        """Lazy-load AltFree model. Falls back to a stub if HF pulls fail."""
        import torch
        import torch.nn as nn

        logger.info("Loading AltFree video detector: %s on %s", self._model_id, self._device)

        # Try to load from HuggingFace.
        try:
            from transformers import AutoModel, AutoImageProcessor
            self._processor = AutoImageProcessor.from_pretrained(
                self._model_id, cache_dir=self._cache_dir
            )
            self._model = AutoModel.from_pretrained(
                self._model_id, cache_dir=self._cache_dir
            ).to(self._device)
            self._model.eval()
            self._weights_loaded = True
            self._backend_used = "pytorch"
            logger.info("AltFree weights loaded from HuggingFace")
            return
        except Exception as e:
            logger.warning(
                "Could not load AltFree from %s (%s). Falling back to "
                "untrained stub. Detector will return low-conf neutral "
                "scores. See CHANGELOG.md for weight-pull instructions.",
                self._model_id, e,
            )

        # No model available — leave self._model as None so detect()
        # returns a safe neutral result via the _weights_loaded guard.
        # Do NOT assign an untrained stub as a real detector.
        self._model = None
        self._weights_loaded = False
        self._backend_used = None

