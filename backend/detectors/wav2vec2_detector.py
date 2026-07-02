import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from utils.logging import get_logger

logger = get_logger(__name__)


class Wav2Vec2AudioDetector(BaseDetector):
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base-960h",
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        target_sample_rate: int = 16000,
    ):
        super().__init__(
            name="Wav2Vec2AudioDetector", preferred_backend=preferred_backend
        )
        self._model_id = model_id
        self._device = device
        self._cache_dir = cache_dir
        self._target_sample_rate = target_sample_rate
        self._processor = None
        self._classifier = None
        self._backend_used = None

    def get_required_models(self) -> List[str]:
        return ["wav2vec2_base"]

    async def detect(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        return_features: bool = False,
    ) -> DetectionResult:
        try:
            if self._model is None:
                await self._load_model()

            import torch
            audio = self._resample(waveform, sample_rate)
            audio = self._normalize(audio)

            inputs = self._processor(
                audio,
                sampling_rate=self._target_sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(self._device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._device)

            with torch.no_grad():
                outputs = self._model(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    logits = outputs.logits
                    pooled = outputs.logits.mean(dim=1) if outputs.logits.dim() > 1 else outputs.logits
                elif hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state
                    pooled = hidden.mean(dim=1)
                    logits = self._classifier(pooled)
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden = outputs.hidden_states[-1]
                    pooled = hidden.mean(dim=1)
                    logits = self._classifier(pooled)
                else:
                    raise AttributeError("No usable output found from Wav2Vec2 model")

                probs = torch.softmax(logits, dim=-1)

            spoof_prob = float(probs[0, 1].cpu()) if probs.shape[-1] >= 2 else float(probs[0, 0].cpu())
            confidence = self._compute_confidence(spoof_prob, probs)

            features_dict = None
            if return_features:
                features_dict = {
                    "pooled_norm": float(pooled.norm(dim=-1).mean().cpu()),
                    "logit_gap": float((logits[0, 1] - logits[0, 0]).cpu()),
                }

            return DetectionResult(
                score=self._normalize_score(spoof_prob),
                confidence=confidence,
                model_name=self._model_id.split("/")[-1],
                backend=self._backend_used or "pytorch",
                features=features_dict,
            )

        except Exception as e:
            logger.error(f"Wav2Vec2 audio detection failed: {e}")
            return DetectionResult(
                score=0.5,
                confidence=0.3,
                model_name=self._model_id.split("/")[-1],
                error=str(e),
            )

    def _compute_confidence(self, spoof_prob: float, probs: "torch.Tensor") -> float:
        import torch
        max_prob = float(torch.max(probs[0]).cpu())
        extremity = abs(spoof_prob - 0.5) * 2
        return float(np.clip(0.4 + 0.4 * (max_prob - 0.5) * 2 + 0.2 * extremity, 0.1, 0.95))

    def _resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self._target_sample_rate:
            return waveform
        try:
            import librosa
            return librosa.resample(
                waveform, orig_sr=orig_sr, target_sr=self._target_sample_rate
            )
        except ImportError:
            ratio = self._target_sample_rate / orig_sr
            new_len = int(len(waveform) * ratio)
            indices = np.linspace(0, len(waveform) - 1, new_len)
            return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return (audio / max_val).astype(np.float32)
        return audio.astype(np.float32)

    async def _load_model(self):
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

        logger.info(f"Loading Wav2Vec2 audio detector: {self._model_id}")
        self._processor = Wav2Vec2Processor.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        self._model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self._model_id,
            cache_dir=self._cache_dir,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        self._model.to(self._device)
        self._model.eval()
        self._classifier = self._model.classifier
        self._backend_used = "pytorch"
        logger.info(f"Wav2Vec2 audio detector loaded on {self._device}")
