"""
Argus Core - TimeSformer Video Deepfake Detector (Iteration 4)
===============================================================
SOTA video deepfake detector using TimeSformer backbone.

Research grounding:
- TimeSformer (Bertasius et al., "Is Space-Time Attention All You Need
  for Video Understanding?", ICML 2021): factorized space-time attention
  transformer. Verified public HF repo:
  facebook/timesformer-base-finetuned-k400 (13k downloads).
- TimeSformer's space-time attention captures different temporal patterns
  than VideoMAE's tubelet masking, providing ensemble DIVERSITY.
- LICENSE: cc-by-nc-4.0 (non-commercial). The backbone is loaded as a
  feature extractor with a fresh 2-class head trained by Argus. The
  non-commercial restriction applies to the backbone weights; operators
  must verify their use case complies. For commercial deployment, use
  VideoMAE+AltFree ensemble only (set ENABLE_TIMESFORMER=false).

Architecture:
    frames (T=8, 3, 224, 224) -> TimeSformer backbone (frozen) ->
        [CLS] token -> linear classifier head (2 classes: real / fake)

Strict-compat: subclasses BaseDetector; returns DetectionResult.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from utils.logging import get_logger

logger = get_logger(__name__)


class TimeSformerVideoDetector(BaseDetector):
    """
    TimeSformer-base video deepfake detector.

    HF source (deterministic): ``facebook/timesformer-base-finetuned-k400``
    License: cc-by-nc-4.0 (NON-COMMERCIAL) — see note above.
    Optional fine-tuned head: ``/models/timesformer_finetune/classifier.pt``

    Falls back to K400 action-classification head (useless for deepfake)
    if no fine-tuned head is found, with a clear log warning.
    """

    REQUIRED_MODELS: List[str] = ["timesformer_video_detector"]
    NUM_FRAMES: int = 8  # TimeSformer uses 8 frames (not 16 like VideoMAE)
    FRAME_SIZE: int = 224

    def __init__(
        self,
        model_id: str = "facebook/timesformer-base-finetuned-k400",
        adapter_path: Optional[str] = None,
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(name="TimeSformerVideoDetector", preferred_backend=preferred_backend)
        self._model_id = model_id
        self._adapter_path = adapter_path or os.environ.get(
            "ARGUS_TIMESFORMER_ADAPTER", "/models/timesformer_finetune"
        )
        self._cache_dir = cache_dir
        self._device = device or self._autodetect_device()
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._classifier = None
        self._head_loaded = False
        self._backend_used = "pytorch"

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

    async def detect(
        self,
        frames: np.ndarray,
        return_features: bool = False,
    ) -> DetectionResult:
        """
        Detect deepfake in a sequence of video frames.

        Args:
            frames: Either (T, H, W, 3) uint8 or (T, 3, H, W) float32.
                    Will be resampled to (8, 3, 224, 224).
            return_features: If True, include extra features.

        Returns:
            DetectionResult with ``score`` = P(fake).
        """
        try:
            await self._ensure_loaded()
            import torch

            frames_t = self._prepare_frames(frames)

            with torch.no_grad():
                outputs = self._model(**frames_t)
                # TimeSformer returns last_hidden_state with [CLS] at index 0
                cls_token = outputs.last_hidden_state[:, 0, :]
                logits = self._classifier(cls_token)
                probs = torch.softmax(logits, dim=-1)
                fake_prob = float(probs[0, 1].cpu()) if probs.shape[-1] >= 2 else float(probs[0, 0].cpu())

            confidence = self._compute_confidence(fake_prob)

            features: Optional[Dict[str, float]] = None
            if return_features:
                features = {
                    "cls_norm": float(cls_token.norm(dim=-1).mean().cpu()),
                    "logit_gap": float((logits[0, 1] - logits[0, 0]).cpu())
                        if probs.shape[-1] >= 2 else 0.0,
                    "head_loaded": float(self._head_loaded),
                    "num_frames": int(frames_t["pixel_values"].shape[1] if "pixel_values" in frames_t else 8),
                }

            return DetectionResult(
                score=self._normalize_score(fake_prob),
                confidence=confidence,
                model_name="timesformer",
                backend=self._backend_used or "pytorch",
                features=features,
            )

        except Exception as e:
            logger.error("TimeSformer detection failed: %s", e)
            return DetectionResult(
                score=0.5,
                confidence=0.2,
                model_name="timesformer",
                error=str(e),
            )

    def _compute_confidence(self, fake_prob: float) -> float:
        extremity = abs(fake_prob - 0.5) * 2.0
        return float(np.clip(0.45 + 0.50 * extremity, 0.1, 0.95))

    def _prepare_frames(self, frames: np.ndarray):
        """Coerce to TimeSformer input format."""
        import torch

        f = np.asarray(frames)
        if f.ndim == 4 and f.shape[-1] == 3:
            f = np.transpose(f, (0, 3, 1, 2))  # (T, 3, H, W)
        elif f.ndim == 4 and f.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Unsupported frame shape: {f.shape}")

        # Resize
        if f.shape[2] != self.FRAME_SIZE or f.shape[3] != self.FRAME_SIZE:
            try:
                import cv2
                resized = np.zeros((f.shape[0], 3, self.FRAME_SIZE, self.FRAME_SIZE), dtype=f.dtype)
                for i in range(f.shape[0]):
                    resized[i] = np.transpose(
                        cv2.resize(np.transpose(f[i], (1, 2, 0)), (self.FRAME_SIZE, self.FRAME_SIZE)),
                        (2, 0, 1),
                    )
                f = resized
            except ImportError:
                h = min(f.shape[2], self.FRAME_SIZE)
                w = min(f.shape[3], self.FRAME_SIZE)
                f = f[:, :, :h, :w]

        # Resample T to NUM_FRAMES=8
        n = f.shape[0]
        if n != self.NUM_FRAMES:
            if n > self.NUM_FRAMES:
                idx = np.linspace(0, n - 1, self.NUM_FRAMES).astype(int)
                f = f[idx]
            else:
                pad = np.repeat(f[-1:], self.NUM_FRAMES - n, axis=0)
                f = np.concatenate([f, pad], axis=0)

        f = f.astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        f = (f - mean) / std

        # TimeSformer expects (B, T, C, H, W) via processor or direct tensor
        pixel_values = torch.from_numpy(f).unsqueeze(0).to(self._device)
        return {"pixel_values": pixel_values}

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            await self._load_model()

    async def _load_model(self) -> None:
        """Lazy-load TimeSformer backbone + classifier head."""
        import torch
        import torch.nn as nn

        logger.info("Loading TimeSformer backbone: %s on %s", self._model_id, self._device)
        try:
            from transformers import TimesformerModel, AutoImageProcessor
            try:
                self._processor = AutoImageProcessor.from_pretrained(
                    self._model_id, cache_dir=self._cache_dir
                )
            except Exception:
                self._processor = None  # We'll prepare frames manually

            self._model = TimesformerModel.from_pretrained(
                self._model_id, cache_dir=self._cache_dir
            ).to(self._device)
            self._model.eval()
        except Exception as e:
            logger.error("Failed to load TimeSformer from %s: %s", self._model_id, e)
            raise

        hidden = self._model.config.hidden_size
        self._classifier = nn.Linear(hidden, 2).to(self._device)

        # Try to load fine-tuned head
        adapter_dir = self._adapter_path
        head_path = os.path.join(adapter_dir, "classifier.pt") if adapter_dir else None
        if head_path and os.path.exists(head_path):
            try:
                state = torch.load(head_path, map_location=self._device)
                self._classifier.load_state_dict(state)
                self._head_loaded = True
                logger.info("TimeSformer classifier head loaded from %s", head_path)
            except Exception as e:
                logger.warning(
                    "Failed to load TimeSformer head from %s (%s); "
                    "using random-init head (NOT benchmark-tuned)",
                    head_path, e,
                )
        else:
            logger.warning(
                "TimeSformer classifier head not found at %s; using random-init head. "
                "Detector will produce near-0.5 scores until a trained head is supplied. "
                "NOTE: backbone license is cc-by-nc-4.0 (non-commercial).",
                head_path,
            )

        # Optional LoRA
        if adapter_dir and os.path.isdir(adapter_dir):
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, adapter_dir)
                self._model.eval()
                logger.info("TimeSformer LoRA adapter loaded from %s", adapter_dir)
            except Exception as e:
                logger.debug("No TimeSformer LoRA adapter at %s (%s)", adapter_dir, e)

        self._backend_used = "pytorch"
        logger.info(
            "TimeSformer detector ready (head=%s, device=%s)",
            self._head_loaded, self._device,
        )
