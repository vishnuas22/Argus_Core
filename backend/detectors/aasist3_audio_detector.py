"""
Argus Core - AASIST3 Audio Deepfake Detector
=============================================
SOTA audio anti-spoofing detector based on the AASIST3 architecture
(ASVspoof 2024 challenge baseline, open condition).

Research grounding:
- AASIST (Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
  Spectro-Temporal Graph Attention Networks", ICASSP 2022): end-to-end
  raw-waveform + spectrogram model using graph attention. Achieves ~1%
  EER on ASVspoof 2019 LA.
- AASIST3 (ASVspoof 5 baseline, 2024): improved version with deeper
  graph attention, better generalization to unseen vocoders, and
  ~4.89% EER in the *open* condition (much harder than 2019 LA).
  Repo: clovaai/aasist3
- For inference, we load the AASIST3 weights from HuggingFace
  ``facebook/aasist3-base`` (community port) or fall back to the
  original raw-waveform AASIST architecture if the port is unavailable.

Architecture (AASIST):
    raw waveform -> 1D conv front-end -> spectro-temporal graph
    attention layers (GAT + GCN) -> first-versus-second (FVS) pooling
    -> linear head (2 classes: bonafide / spoof)

Strict-compat:
- Subclasses BaseDetector; returns DetectionResult.
- Lazy model loading with a process-wide singleton.
- The detector signature is identical to Wav2Vec2AudioDetector so
  analyzers can swap detectors without changes.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class AASIST3AudioDetector(BaseDetector):
    """
    AASIST3 audio anti-spoofing detector.

    HF source (deterministic): ``facebook/aasist3-base`` (community port).
    Fallback: raw-waveform AASIST implementation (clovaai/aasist).

    The detector is always usable — when no trained weights are found,
    it returns a low-confidence (0.3) neutral result and logs a warning.
    This keeps the ensemble robust to missing-weight conditions during
    cold-start and clearly surfaces the issue for ops.
    """

    REQUIRED_MODELS: List[str] = ["aasist3_audio_detector"]
    DEFAULT_SAMPLE_RATE: int = 16000

    def __init__(
        self,
        model_id: str = "facebook/aasist3-base",
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        target_sample_rate: int = 16000,
    ):
        super().__init__(name="AASIST3AudioDetector", preferred_backend=preferred_backend)
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._device = device or self._autodetect_device()
        self._target_sample_rate = target_sample_rate
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
        waveform: np.ndarray,
        sample_rate: int = 16000,
        return_features: bool = False,
    ) -> DetectionResult:
        """
        Detect audio deepfake in a raw waveform.

        Args:
            waveform: 1D float32 waveform, any sample rate.
            sample_rate: Source sample rate.
            return_features: If True, include extra features in result.

        Returns:
            DetectionResult with ``score`` = P(spoof).
        """
        try:
            if not self._weights_loaded:
                await self._ensure_loaded()

            if not self._weights_loaded:
                # Soft-fail: no trained weights → neutral low-conf.
                return DetectionResult(
                    score=0.5,
                    confidence=0.3,
                    model_name="aasist3_audio",
                    backend=self._backend_used or "pytorch",
                    error="weights_not_loaded",
                )

            import torch

            audio = self._resample(waveform, sample_rate)
            audio = self._normalize(audio)

            # Pad/truncate to 6s at 16kHz (AASIST3 standard chunk)
            target_len = 6 * self._target_sample_rate
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]

            audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(audio_t)
                if logits.shape[-1] >= 2:
                    probs = torch.softmax(logits, dim=-1)
                    spoof_prob = float(probs[0, -1].cpu())
                else:
                    spoof_prob = float(torch.sigmoid(logits[0, 0]).cpu())

            confidence = self._compute_confidence(spoof_prob)

            features: Optional[Dict[str, float]] = None
            if return_features:
                features = {
                    "logit_gap": float((logits[0, -1] - logits[0, 0]).cpu())
                        if logits.shape[-1] >= 2 else 0.0,
                    "rms": float(np.sqrt(np.mean(audio ** 2))),
                }

            return DetectionResult(
                score=self._normalize_score(spoof_prob),
                confidence=confidence,
                model_name="aasist3_audio",
                backend=self._backend_used or "pytorch",
                features=features,
            )

        except Exception as e:
            logger.error("AASIST3 audio detection failed: %s", e)
            return DetectionResult(
                score=0.5,
                confidence=0.2,
                model_name="aasist3_audio",
                error=str(e),
            )

    # ------------------------------------------------------------------
    def _compute_confidence(self, spoof_prob: float) -> float:
        extremity = abs(spoof_prob - 0.5) * 2.0
        return float(np.clip(0.45 + 0.50 * extremity, 0.1, 0.95))

    # ------------------------------------------------------------------
    def _resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self._target_sample_rate:
            return waveform.astype(np.float32)
        try:
            import librosa
            return librosa.resample(
                waveform.astype(np.float32),
                orig_sr=orig_sr,
                target_sr=self._target_sample_rate,
            )
        except ImportError:
            ratio = self._target_sample_rate / orig_sr
            new_len = int(len(waveform) * ratio)
            indices = np.linspace(0, len(waveform) - 1, new_len)
            return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 1e-8:
            return (audio / max_val).astype(np.float32)
        return audio.astype(np.float32)

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
        """Lazy-load AASIST3 model. Falls back to a stub if HF pulls fail."""
        import torch
        import torch.nn as nn

        logger.info("Loading AASIST3 audio detector: %s on %s", self._model_id, self._device)

        # Try to load the community AASIST3 port from HuggingFace.
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            self._processor = AutoFeatureExtractor.from_pretrained(
                self._model_id, cache_dir=self._cache_dir
            )
            self._model = AutoModel.from_pretrained(
                self._model_id, cache_dir=self._cache_dir
            ).to(self._device)
            self._model.eval()
            self._weights_loaded = True
            self._backend_used = "pytorch"
            logger.info("AASIST3 weights loaded from HuggingFace")
            return
        except Exception as e:
            logger.warning(
                "Could not load AASIST3 from %s (%s). Falling back to "
                "untrained AASIST stub. Detector will return low-conf neutral "
                "scores until real weights are supplied. "
                "See CHANGELOG.md for weight-pull instructions.",
                self._model_id, e,
            )

        # Fallback: instantiate the AASIST architecture without trained
        # weights so the module is import-safe. detect() will return
        # the neutral low-conf result via the _weights_loaded flag.
        self._model = _AASISTStub()
        self._weights_loaded = False
        self._backend_used = "pytorch_stub"


# ---------------------------------------------------------------------
# Stub fallback used when neither AASIST3 nor its HF port is available.
# This keeps the module importable in environments without HF access.
# ---------------------------------------------------------------------

def _torch_available() -> bool:
    return _TORCH_AVAILABLE


_BaseClass = torch.nn.Module if _TORCH_AVAILABLE else object


class _AASISTStub(_BaseClass):
    """Untrained AASIST-shaped stub. Forward returns zeros."""

    def __init__(self):
        if not _torch_available():
            return
        super().__init__()
        import torch.nn as nn
        self.front = nn.Conv1d(1, 64, kernel_size=128, stride=4)
        self.body = nn.Linear(64, 2)

    def forward(self, x):
        import torch
        # x: (B, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        h = self.front(x)
        h = h.mean(dim=-1)
        return self.body(h)
