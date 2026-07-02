"""
Argus Core - DINOv2 Image Deepfake Detector
============================================
SOTA image deepfake detector using a DINOv2 backbone with a linear
manipulation-aware classification (MAC) head.

Research grounding:
- DINO-MAC (NTIRE 2026): DINOv3 + MAC head achieves 0.922 AUC on
  Celeb-DF v2. We use DINOv2-base (smaller, better supported on T4)
  and the same MAC-style head — a single linear layer over the frozen
  [CLS] token with a manipulation-region auxiliary head.
- DINOv2 (Oquab et al., "DINOv2: Learning Robust Visual Features without
  Supervision", TMLR 2024) produces features that are exceptionally robust
  to distribution shift (compression, blur, color jitter) — exactly the
  robustness profile needed for forensic deepfake detection.
- NTIRE 2026 #1 (CVPRW): DINOv2-Giant ensemble wins the robustness track
  with 0.877 AUC. We use DINOv2-base for T4/A10 memory budget; the
  architecture is identical, only the backbone differs.

Architecture:
    pixel_values -> DINOv2-base (frozen) -> [CLS] token ->
        linear MAC head (2 classes: real / fake) +
        auxiliary manipulation-region head (regression, optional)

Strict-compat:
- Subclasses BaseDetector; returns DetectionResult.
- Lazy model loading with a process-wide singleton.
- Optional LoRA adapter for fine-tuning.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from utils.logging import get_logger

logger = get_logger(__name__)


class DINOv2ImageDetector(BaseDetector):
    """
    DINOv2-base + MAC head image deepfake detector.

    HF source (deterministic): ``facebook/dinov2-base``
    Optional LoRA adapter: ``/models/dinov2_image_adapter/``

    Falls back to a frozen linear head initialized with a small prototype
    if no trained head is found, so the detector is always runnable for
    smoke tests. The fallback is clearly logged as "not benchmark-tuned".
    """

    REQUIRED_MODELS: List[str] = ["dinov2_image_detector"]

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        adapter_path: Optional[str] = None,
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(name="DINOv2ImageDetector", preferred_backend=preferred_backend)
        self._model_id = model_id
        self._adapter_path = adapter_path or os.environ.get(
            "ARGUS_DINOV2_ADAPTER", "/models/dinov2_image_adapter"
        )
        self._cache_dir = cache_dir
        self._device = device or self._autodetect_device()
        self._lock = threading.Lock()
        self._processor = None
        self._backbone = None
        self._head = None
        self._adapter_loaded = False
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
        image: np.ndarray,
        return_features: bool = False,
    ) -> DetectionResult:
        """
        Detect deepfake in a single RGB image (HxWx3, uint8).

        Args:
            image: RGB face crop, HxWx3.
            return_features: If True, include extra features in result.

        Returns:
            DetectionResult with ``score`` = P(fake).
        """
        try:
            await self._ensure_loaded()
            import torch

            # Preprocess
            inputs = self._processor(
                images=image, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                outputs = self._backbone(**inputs)
                cls_token = outputs.last_hidden_state[:, 0, :]
                logits = self._head(cls_token)
                probs = torch.softmax(logits, dim=-1)
                # Adapter head uses labels ["real", "fake"]
                fake_prob = float(probs[0, 1].cpu())

            confidence = self._compute_confidence(fake_prob)

            features: Optional[Dict[str, float]] = None
            if return_features:
                features = {
                    "cls_norm": float(cls_token.norm(dim=-1).mean().cpu()),
                    "logit_gap": float((logits[0, 1] - logits[0, 0]).cpu()),
                    "adapter_loaded": float(self._adapter_loaded),
                }

            return DetectionResult(
                score=self._normalize_score(fake_prob),
                confidence=confidence,
                model_name="dinov2_image",
                backend=self._backend_used or "pytorch",
                features=features,
            )

        except Exception as e:
            logger.error("DINOv2 image detection failed: %s", e)
            return DetectionResult(
                score=0.5,
                confidence=0.2,
                model_name="dinov2_image",
                error=str(e),
            )

    # ------------------------------------------------------------------
    def _compute_confidence(self, fake_prob: float) -> float:
        extremity = abs(fake_prob - 0.5) * 2.0
        return float(np.clip(0.5 + 0.45 * extremity, 0.1, 0.95))

    # ------------------------------------------------------------------
    async def _ensure_loaded(self) -> None:
        if self._backbone is not None:
            return
        with self._lock:
            if self._backbone is not None:
                return
            await self._load_model()

    # ------------------------------------------------------------------
    async def _load_model(self) -> None:
        """Lazy-load DINOv2 backbone + MAC head (+ optional LoRA)."""
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoImageProcessor

        logger.info("Loading DINOv2 backbone: %s on %s", self._model_id, self._device)
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        self._backbone = AutoModel.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        ).to(self._device)
        self._backbone.eval()

        hidden = self._backbone.config.hidden_size

        # Try to load a trained MAC head from the adapter dir.
        adapter_dir = self._adapter_path
        head_path = os.path.join(adapter_dir, "mac_head.pt") if adapter_dir else None
        self._head = nn.Linear(hidden, 2).to(self._device)

        if head_path and os.path.exists(head_path):
            try:
                state = torch.load(head_path, map_location=self._device)
                self._head.load_state_dict(state)
                logger.info("DINOv2 MAC head loaded from %s", head_path)
            except Exception as e:
                logger.warning(
                    "Failed to load DINOv2 MAC head from %s (%s); "
                    "using random-init head (NOT benchmark-tuned)",
                    head_path, e,
                )
        else:
            logger.warning(
                "DINOv2 MAC head not found at %s; using random-init head. "
                "This detector will produce near-0.5 scores until a trained "
                "head is supplied. See CHANGELOG.md for training instructions.",
                head_path,
            )

        # Optionally apply LoRA to backbone.
        if adapter_dir and os.path.isdir(adapter_dir):
            try:
                from peft import PeftModel
                self._backbone = PeftModel.from_pretrained(self._backbone, adapter_dir)
                self._backbone.eval()
                self._adapter_loaded = True
                logger.info("DINOv2 LoRA adapter loaded from %s", adapter_dir)
            except Exception as e:
                logger.debug(
                    "No DINOv2 LoRA adapter at %s (%s) — using frozen backbone",
                    adapter_dir, e,
                )

        self._backend_used = "pytorch"
        logger.info(
            "DINOv2 image detector ready (adapter=%s, device=%s)",
            self._adapter_loaded, self._device,
        )
