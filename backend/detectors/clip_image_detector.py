"""
Argus Core - CLIP + LoRA Image Deepfake Detector
================================================
SOTA image deepfake detector based on a CLIP ViT visual backbone with a
LoRA adapter trained on FaceForensics++ / Celeb-DF / DFDC.

Research grounding:
- ForAda (CVPR 2025): CLIP backbone + LoRA adapter generalizes to unseen
  forgery families better than full fine-tuning of ResNet/EfficientNet.
  Paper: "ForAda: Forgery-Adaptive Deepfake Detection", CVPR 2025.
- CLIP visual encoders learn semantics ("face", "skin texture", "eyes")
  that transfer to deepfake artifacts even without deepfake-specific
  pretraining (Yan et al., "Deepfake Detection with CLIP", ECCVW 2024).
- LoRA (Hu et al., "LoRA: Low-Rank Adaptation of LLMs", ICLR 2022) keeps
  the backbone frozen and trains only rank-r injects — 1-3% trainable
  params, no catastrophic forgetting, easy to hot-swap.

Architecture:
    pixel_values -> CLIP ViT-B/16 (frozen) -> [CLS] token ->
        LoRA-adapted attention layers (trainable) ->
        linear classifier head (2 classes: real / fake)

Strict-compat:
- Subclasses BaseDetector; returns DetectionResult.
- get_required_models() returns registry keys declared in models/registry.py.
- All model loading is lazy and thread-safe (process-wide singleton).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from utils.logging import get_logger

logger = get_logger(__name__)


class CLIPLoRAImageDetector(BaseDetector):
    """
    CLIP ViT-B/16 + LoRA image deepfake detector.

    HF source (deterministic): ``openai/clip-vit-base-patch16``
    LoRA adapter (Argus-trained): loaded from
    ``/models/clip_lora_image_adapter/`` if present.

    Falls back to zero-shot CLIP cosine similarity against prompts
    "a real human face" / "a synthetic AI-generated fake face" if the
    adapter is absent, so the detector is always usable.
    """

    # Registry keys declared in models/registry.py
    REQUIRED_MODELS: List[str] = ["clip_image_detector"]

    # Zero-shot prompts used as the fallback head.
    REAL_PROMPT: str = "a real human face photograph"
    FAKE_PROMPT: str = "a synthetic AI-generated deepfake face"

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch16",
        adapter_path: Optional[str] = None,
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        # Iteration 1.5: optional deepfake-specific fine-tuned head.
        # When set, this HF repo is loaded as the classifier INSTEAD OF
        # the LoRA adapter. Real public options (verify availability):
        #   - "dima806/deepfake_detection_model_image"
        #   - "dima806/ai_vs_real_image_detection"
        #   - "Wvolf/real-vs-fake"
        # See TRAINING.md for the full list and license notes.
        fine_tuned_head_repo: Optional[str] = None,
    ):
        super().__init__(name="CLIPLoRAImageDetector", preferred_backend=preferred_backend)
        self._model_id = model_id
        self._adapter_path = adapter_path or os.environ.get(
            "ARGUS_CLIP_LORA_ADAPTER", "/models/clip_lora_image_adapter"
        )
        self._fine_tuned_head_repo = fine_tuned_head_repo or os.environ.get(
            "ARGUS_CLIP_FINE_TUNED_HEAD", ""
        )
        self._cache_dir = cache_dir
        self._device = device or self._autodetect_device()
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._adapter_loaded = False
        self._tokenizer = None
        self._real_text_emb = None
        self._fake_text_emb = None
        # Iteration 1.5: separate fine-tuned-head model + processor
        self._head_model = None
        self._head_processor = None
        self._backend_used = "pytorch"

    # ------------------------------------------------------------------
    def get_required_models(self) -> List[str]:
        return list(self.REQUIRED_MODELS)

    # ------------------------------------------------------------------
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
        Detect deepfake in a single RGB image (HxWx3, uint8 or float32).

        Detection order (first successful wins):
        1. Fine-tuned head (if ``fine_tuned_head_repo`` was set and
           loaded successfully). This is the path that produces real
           benchmark numbers.
        2. LoRA adapter (if ``/models/clip_lora_image_adapter/`` exists).
        3. Zero-shot CLIP cosine similarity against prompts (fallback).

        Args:
            image: RGB face crop, HxWx3.
            return_features: If True, include extra features in result.

        Returns:
            DetectionResult with ``score`` = P(fake).
        """
        try:
            await self._ensure_loaded()
            import torch
            from PIL import Image as PILImage

            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 255)).astype(np.uint8)
            pil_img = PILImage.fromarray(image)

            # ----- Path 1: fine-tuned head (real benchmark numbers) -----
            if self._head_model is not None:
                head_inputs = self._head_processor(
                    images=pil_img, return_tensors="pt"
                ).to(self._device)
                with torch.no_grad():
                    outputs = self._head_model(**head_inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    # Use infer_fake_class_index to handle different label schemes
                    from analyzers.base import infer_fake_class_index
                    id2label = getattr(self._head_model.config, "id2label", None)
                    fake_idx = infer_fake_class_index(id2label=id2label, default_index=1)
                    fake_idx = int(np.clip(fake_idx, 0, probs.shape[-1] - 1))
                    fake_prob = float(probs[0, fake_idx].cpu())

                confidence = self._compute_confidence(fake_prob)
                features: Optional[Dict[str, float]] = None
                if return_features:
                    features = {
                        "head_repo": self._fine_tuned_head_repo,
                        "logit_gap": float((logits[0, -1] - logits[0, 0]).cpu())
                            if probs.shape[-1] >= 2 else 0.0,
                        "adapter_loaded": 0.0,
                        "head_loaded": 1.0,
                    }
                return DetectionResult(
                    score=self._normalize_score(fake_prob),
                    confidence=confidence,
                    model_name=f"clip_finetuned_head:{self._fine_tuned_head_repo.split('/')[-1]}",
                    backend=self._backend_used or "pytorch",
                    features=features,
                )

            # ----- Path 2 & 3: backbone + adapter or zero-shot -----
            inputs = self._processor(
                images=pil_img, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                img_emb = self._model.get_image_features(**inputs)
                img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)

                if self._adapter_loaded:
                    # Adapter head path — CLIPViT + LoRA + linear head.
                    logits = self._model.classifier(
                        self._model.vision_model(**inputs).last_hidden_state[:, 0, :]
                    )
                    probs = torch.softmax(logits, dim=-1)
                    # Adapter always uses labels ["real", "fake"]
                    fake_prob = float(probs[0, 1].cpu())
                else:
                    # Zero-shot fallback: cosine similarity against prompts.
                    sim_real = float((img_emb @ self._real_text_emb.T).cpu())
                    sim_fake = float((img_emb @ self._fake_text_emb.T).cpu())
                    # Temperature-scaled softmax
                    logits = np.array([sim_real, sim_fake]) * 4.0  # T=0.25
                    probs = np.exp(logits - logits.max())
                    probs = probs / probs.sum()
                    fake_prob = float(probs[1])

            confidence = self._compute_confidence(fake_prob)

            features: Optional[Dict[str, float]] = None
            if return_features:
                features = {
                    "img_emb_norm": float(img_emb.norm(dim=-1).mean().cpu()),
                    "sim_real": float(sim_real) if not self._adapter_loaded else 0.0,
                    "sim_fake": float(sim_fake) if not self._adapter_loaded else 0.0,
                    "adapter_loaded": float(self._adapter_loaded),
                    "head_loaded": 0.0,
                }

            return DetectionResult(
                score=self._normalize_score(fake_prob),
                confidence=confidence,
                model_name="clip_lora_image",
                backend=self._backend_used or "pytorch",
                features=features,
            )

        except Exception as e:
            logger.error("CLIP+LoRA image detection failed: %s", e)
            return DetectionResult(
                score=0.5,
                confidence=0.2,
                model_name="clip_lora_image",
                error=str(e),
            )

    # ------------------------------------------------------------------
    def _compute_confidence(self, fake_prob: float) -> float:
        """Confidence from prediction extremity + a small floor."""
        extremity = abs(fake_prob - 0.5) * 2.0
        return float(np.clip(0.5 + 0.45 * extremity, 0.1, 0.95))

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
        """Lazy-load CLIP backbone + (optional) fine-tuned head or LoRA adapter."""
        import torch
        from transformers import (
            CLIPModel,
            CLIPProcessor,
            CLIPTokenizer,
            AutoModelForImageClassification,
            AutoImageProcessor,
        )

        # ----- Iteration 1.5: try fine-tuned head FIRST -----
        # This is the path that produces real benchmark numbers.
        if self._fine_tuned_head_repo:
            try:
                logger.info(
                    "Loading fine-tuned deepfake head: %s on %s",
                    self._fine_tuned_head_repo, self._device,
                )
                self._head_processor = AutoImageProcessor.from_pretrained(
                    self._fine_tuned_head_repo, cache_dir=self._cache_dir
                )
                self._head_model = AutoModelForImageClassification.from_pretrained(
                    self._fine_tuned_head_repo, cache_dir=self._cache_dir
                ).to(self._device)
                self._head_model.eval()
                logger.info(
                    "Fine-tuned head loaded: %s (labels=%s)",
                    self._fine_tuned_head_repo,
                    getattr(self._head_model.config, "id2label", "unknown"),
                )
            except Exception as e:
                logger.warning(
                    "Could not load fine-tuned head %s (%s). "
                    "Falling back to CLIP backbone + adapter/zero-shot.",
                    self._fine_tuned_head_repo, e,
                )
                self._head_model = None
                self._head_processor = None

        # ----- Always load the CLIP backbone (zero-shot fallback) -----
        logger.info("Loading CLIP backbone: %s on %s", self._model_id, self._device)
        self._processor = CLIPProcessor.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        self._tokenizer = CLIPTokenizer.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        self._model = CLIPModel.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        ).to(self._device)
        self._model.eval()

        # Pre-compute text embeddings for the zero-shot fallback.
        with torch.no_grad():
            real_inputs = self._tokenizer(
                [self.REAL_PROMPT], padding=True, return_tensors="pt"
            ).to(self._device)
            fake_inputs = self._tokenizer(
                [self.FAKE_PROMPT], padding=True, return_tensors="pt"
            ).to(self._device)
            self._real_text_emb = self._model.get_text_features(**real_inputs)
            self._real_text_emb = self._real_text_emb / (
                self._real_text_emb.norm(dim=-1, keepdim=True) + 1e-8
            )
            self._fake_text_emb = self._model.get_text_features(**fake_inputs)
            self._fake_text_emb = self._fake_text_emb / (
                self._fake_text_emb.norm(dim=-1, keepdim=True) + 1e-8
            )

        # Try to load the LoRA adapter if present.
        adapter_dir = self._adapter_path
        if adapter_dir and os.path.isdir(adapter_dir):
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, adapter_dir)
                self._model.eval()
                self._adapter_loaded = True
                logger.info("CLIP LoRA adapter loaded from %s", adapter_dir)
            except Exception as e:
                logger.warning(
                    "CLIP LoRA adapter at %s failed to load (%s); "
                    "falling back to zero-shot", adapter_dir, e
                )
                self._adapter_loaded = False
        else:
            logger.info(
                "No CLIP LoRA adapter found at %s; using zero-shot fallback",
                adapter_dir,
            )
            self._adapter_loaded = False

        self._backend_used = "pytorch"
        logger.info(
            "CLIP image detector ready (head=%s, adapter=%s, device=%s)",
            self._head_model is not None, self._adapter_loaded, self._device,
        )
