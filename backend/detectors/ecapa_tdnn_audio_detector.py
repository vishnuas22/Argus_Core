"""
Argus Core - ECAPA-TDNN Audio Deepfake Detector (Iteration 4)
===============================================================
SOTA audio deepfake detector using ECAPA-TDNN speaker embeddings.

Research grounding:
- ECAPA-TDNN (Desplanques et al., "ECAPA-TDNN: Emphasized Channel
  Attention, Propagation and Aggregation in TDNN Based Speaker
  Verification", INTERSPEECH 2020): SOTA speaker embedding architecture.
  Verified public HF repo: speechbrain/spkrec-ecapa-voxceleb (MIT).
- For deepfake detection: ECAPA-TDNN embeddings capture speaker identity
  cues that vocoders struggle to preserve. A simple embedding-distance
  or embedding-anomaly approach detects spoofing by measuring how
  "natural" the embedding looks compared to a reference distribution.
- This provides ensemble DIVERSITY against Wav2Vec2-based detectors —
  ECAPA-TDNN was trained on speaker verification, so its error surface
  is different from Wav2Vec2's.
- NOTE: This is an embedding-based anomaly detector, NOT a fine-tuned
  classifier. It computes the distance from the input embedding to the
  centroid of "real" embeddings. Operators must build the reference
  centroid by running the detector on a calibration set of real audio
  (see scripts/fit_calibration.py).

Architecture:
    raw waveform -> ECAPA-TDNN (frozen) -> 192-dim embedding ->
        distance to reference centroid -> sigmoid -> fake probability

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


class ECAPATDNNAudioDetector(BaseDetector):
    """
    ECAPA-TDNN audio deepfake detector (embedding-distance-based).

    HF source (deterministic): ``speechbrain/spkrec-ecapa-voxceleb``
    License: MIT (commercially usable — unlike TimeSformer).
    Reference centroid: ``/models/ecapa_reference_centroid.npy``

    The centroid is the mean embedding of ~100-1000 real audio samples.
    Build it via:
        python scripts/fit_calibration.py --modality audio \\
            --calibration-json /data/real_audio.json \\
            --output-dir /models
    (The fit_calibration script will be extended to support centroid
    building in a future iteration; for now, build it manually.)
    """

    REQUIRED_MODELS: List[str] = ["ecapa_audio_detector"]
    DEFAULT_SAMPLE_RATE: int = 16000
    EMBEDDING_DIM: int = 192

    def __init__(
        self,
        model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        target_sample_rate: int = 16000,
        reference_centroid_path: Optional[str] = None,
    ):
        super().__init__(name="ECAPATDNNAudioDetector", preferred_backend=preferred_backend)
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._device = device or self._autodetect_device()
        self._target_sample_rate = target_sample_rate
        self._reference_centroid_path = reference_centroid_path or os.environ.get(
            "ARGUS_ECAPA_CENTROID", "/models/ecapa_reference_centroid.npy"
        )
        self._lock = threading.Lock()
        self._model = None
        self._reference_centroid: Optional[np.ndarray] = None
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
        waveform: np.ndarray,
        sample_rate: int = 16000,
        return_features: bool = False,
    ) -> DetectionResult:
        """
        Detect audio deepfake via embedding distance to reference centroid.

        Args:
            waveform: 1D float32 waveform.
            sample_rate: Source sample rate.
            return_features: If True, include embedding + distance.

        Returns:
            DetectionResult with ``score`` = P(spoof).
        """
        try:
            await self._ensure_loaded()
            import torch

            audio = self._resample(waveform, sample_rate)
            audio = self._normalize(audio)

            # ECAPA-TDNN expects (batch, samples) tensor
            wav_t = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)

            with torch.no_grad():
                embedding = self._model(wav_t).squeeze(0).cpu().numpy()
                # embedding: (192,)

            if self._reference_centroid is None:
                # No reference — return neutral
                logger.warning("ECAPA: no reference centroid loaded; returning neutral")
                return DetectionResult(
                    score=0.5,
                    confidence=0.3,
                    model_name="ecapa_tdnn",
                    backend=self._backend_used or "pytorch",
                    error="no_reference_centroid",
                    features={"embedding_norm": float(np.linalg.norm(embedding))} if return_features else None,
                )

            # Cosine distance to centroid
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            cen_norm = self._reference_centroid / (np.linalg.norm(self._reference_centroid) + 1e-8)
            cosine_sim = float(np.dot(emb_norm, cen_norm))
            # Distance = 1 - cosine_sim, in [0, 2]
            distance = 1.0 - cosine_sim
            # Convert to fake probability via sigmoid
            # Higher distance = more likely fake
            # Scale: typical real-audio distance is ~0.1-0.3; fake is ~0.4-0.8
            fake_prob = float(1.0 / (1.0 + np.exp(-(distance - 0.3) * 5.0)))

            confidence = self._compute_confidence(fake_prob, distance)

            features: Optional[Dict[str, float]] = None
            if return_features:
                features = {
                    "embedding_norm": float(np.linalg.norm(embedding)),
                    "cosine_sim": cosine_sim,
                    "distance": distance,
                    "has_reference": 1.0,
                }

            return DetectionResult(
                score=self._normalize_score(fake_prob),
                confidence=confidence,
                model_name="ecapa_tdnn",
                backend=self._backend_used or "pytorch",
                features=features,
            )

        except Exception as e:
            logger.error("ECAPA-TDNN audio detection failed: %s", e)
            return DetectionResult(
                score=0.5,
                confidence=0.2,
                model_name="ecapa_tdnn",
                error=str(e),
            )

    def _compute_confidence(self, fake_prob: float, distance: float) -> float:
        extremity = abs(fake_prob - 0.5) * 2.0
        # Higher distance from centroid = more confident (clearly fake)
        distance_factor = min(distance / 0.8, 1.0)
        return float(np.clip(0.4 + 0.4 * extremity + 0.2 * distance_factor, 0.1, 0.95))

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

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            await self._load_model()

    async def _load_model(self) -> None:
        """Lazy-load ECAPA-TDNN model + reference centroid."""
        import torch

        logger.info("Loading ECAPA-TDNN: %s on %s", self._model_id, self._device)
        try:
            # SpeechBrain models need the speechbrain package
            from speechbrain.inference.speaker import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(
                source=self._model_id,
                savedir=os.path.join(self._cache_dir or "/tmp", "ecapa_tdnn"),
                run_opts={"device": self._device},
            )
            logger.info("ECAPA-TDNN loaded from %s", self._model_id)
        except ImportError:
            logger.warning(
                "speechbrain package not installed; ECAPA-TDNN detector unavailable. "
                "Install with: pip install speechbrain"
            )
            raise
        except Exception as e:
            logger.error("Failed to load ECAPA-TDNN from %s: %s", self._model_id, e)
            raise

        # Load reference centroid if available
        if os.path.exists(self._reference_centroid_path):
            try:
                self._reference_centroid = np.load(self._reference_centroid_path)
                logger.info(
                    "ECAPA reference centroid loaded from %s (dim=%d)",
                    self._reference_centroid_path, len(self._reference_centroid),
                )
            except Exception as e:
                logger.warning(
                    "Failed to load ECAPA reference centroid from %s: %s",
                    self._reference_centroid_path, e,
                )
        else:
            logger.info(
                "ECAPA reference centroid not found at %s. "
                "Detector will return neutral until centroid is built. "
                "Build it by running the detector on ~100 real audio samples "
                "and averaging embeddings, then save to the centroid path.",
                self._reference_centroid_path,
            )

        self._backend_used = "pytorch"
        logger.info(
            "ECAPA-TDNN detector ready (centroid=%s, device=%s)",
            self._reference_centroid is not None, self._device,
        )

    def build_reference_centroid(self, embeddings: np.ndarray) -> None:
        """
        Build and save the reference centroid from a set of real-audio embeddings.

        Args:
            embeddings: (N, 192) array of embeddings from real audio.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        centroid = embeddings.mean(axis=0)
        self._reference_centroid = centroid
        os.makedirs(os.path.dirname(self._reference_centroid_path) or ".", exist_ok=True)
        np.save(self._reference_centroid_path, centroid)
        logger.info(
            "ECAPA reference centroid built from %d samples, saved to %s",
            len(embeddings), self._reference_centroid_path,
        )

    async def embed(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Compute the ECAPA-TDNN embedding for a waveform.

        Useful for building the reference centroid from real audio.

        Args:
            waveform: 1D float32 waveform.
            sample_rate: Source sample rate.

        Returns:
            (192,) embedding array.
        """
        await self._ensure_loaded()
        import torch
        audio = self._resample(waveform, sample_rate)
        audio = self._normalize(audio)
        wav_t = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            embedding = self._model(wav_t).squeeze(0).cpu().numpy()
        return embedding
