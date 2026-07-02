"""
Argus Core - SigLIP Image Deepfake Detector (Iteration 3)
===========================================================
SOTA image deepfake detector using Google's SigLIP backbone.

Research grounding:
- SigLIP (Zhai et al., "Sigmoid Loss for Language Image Pre-Training",
  ICCV 2023): replaces CLIP's softmax contrastive loss with a per-pair
  sigmoid loss. Better calibration and stronger zero-shot performance
  on fine-grained tasks than CLIP.
- For deepfake detection: SigLIP's stronger fine-grained features
  provide ensemble DIVERSITY against CLIP+DINOv2 — they share less of
  their error surface, so the DiversityEnsemble combiner can down-
  weight correlated failures more effectively.
- SigLIP-base (google/siglip-base-patch16-224) is verified public on
  HuggingFace, 371M params, Apache-2.0 license.

Architecture:
    pixel_values -> SigLIP vision encoder (frozen) -> [CLS] pooled ->
        zero-shot cosine similarity against prompts (fallback) OR
        fine-tuned linear head (if available)

Strict-compat: subclasses BaseDetector; returns DetectionResult.
Same interface as CLIPLoRAImageDetector — drop-in alternative.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from utils.logging import get_logger

logger = get_logger(__name__)


class SigLIPImageDetector(BaseDetector):
    """
    SigLIP-base image deepfake detector.

    HF source (deterministic): ``google/siglip-base-patch16-224``
    Optional fine-tuned head: ``/models/siglip_image_adapter/``

    Falls back to zero-shot cosine similarity against prompts if no
    fine-tuned head is available.
    """

    REQUIRED_MODELS: List[str] = ["siglip_image_detector"]

    # Zero-shot prompts
    REAL_PROMPT: str = "a real human face photograph"
    FAKE_PROMPT: str = "a synthetic AI-generated deepfake face"

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        adapter_path: Optional[str] = None,
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        fine_tuned_head_repo: Optional[str] = None,
    ):
        super().__init__(name="SigLIPImageDetector", preferred_backend=preferred_backend)
        self._model_id = model_id
        self._adapter_path = adapter_path or os.environ.get(
            "ARGUS_SIGLIP_ADAPTER", "/models/siglip_image_adapter"
        )
        self._fine_tuned_head_repo = fine_tuned_head_repo or os.environ.get(
            "ARGUS_SIGLIP_FINE_TUNED_HEAD", ""
        )
        self._cache_dir = cache_dir
        self._device = device or self._autodetect_device()
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._adapter_loaded = False
        self._head_model = None
        self._head_processor = None
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

        Detection order:
        1. Fine-tuned head (if ``fine_tuned_head_repo`` was set and loaded).
        2. LoRA adapter (if ``/models/siglip_image_adapter/`` exists).
        3. Zero-shot SigLIP cosine similarity (fallback).
        """
        try:
            await self._ensure_loaded()
            import torch
            from PIL import Image as PILImage

            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 255)).astype(np.uint8)
            pil_img = PILImage.fromarray(image)

            # ----- Path 1: fine-tuned head -----
            if self._head_model is not None:
                head_inputs = self._head_processor(
                    images=pil_img, return_tensors="pt"
                ).to(self._device)
                with torch.no_grad():
                    outputs = self._head_model(**head_inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
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
                        "adapter_loaded": 0.0,
                        "head_loaded": 1.0,
                    }
                return DetectionResult(
                    score=self._normalize_score(fake_prob),
                    confidence=confidence,
                    model_name=f"siglip_finetuned_head:{self._fine_tuned_head_repo.split('/')[-1]}",
                    backend=self._backend_used or "pytorch",
                    features=features,
                )

            # ----- Path 2 & 3: backbone + adapter or zero-shot -----
            inputs = self._processor(
                images=pil_img, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # SigLIP returns pooler_output
                img_emb = outputs.pooler_output if hasattr(outputs, "pooler_output") \
                    else outputs.last_hidden_state[:, 0, :]
                img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)

                if self._adapter_loaded:
                    logits = self._model.classifier(img_emb)
                    probs = torch.softmax(logits, dim=-1)
                    fake_prob = float(probs[0, 1].cpu())
                else:
                    # Zero-shot fallback: cosine similarity against prompts
                    # SigLIP uses sigmoid, not softmax, for classification
                    from transformers import AutoTokenizer
                    if not hasattr(self, "_text_emb_cache"):
                        self._load_text_embeddings()
                    sim_real = float((img_emb @ self._real_text_emb.T).cpu())
                    sim_fake = float((img_emb @ self._fake_text_emb.T).cpu())
                    # SigLIP-style: sigmoid of (sim - threshold)
                    # For binary, use softmax over [sim_real, sim_fake]
                    logits_arr = np.array([sim_real, sim_fake]) * 4.0
                    probs_arr = np.exp(logits_arr - logits_arr.max())
                    probs_arr = probs_arr / probs_arr.sum()
                    fake_prob = float(probs_arr[1])

            confidence = self._compute_confidence(fake_prob)

            features: Optional[Dict[str, float]] = None
            if return_features:
                features = {
                    "img_emb_norm": float(img_emb.norm(dim=-1).mean().cpu()),
                    "adapter_loaded": float(self._adapter_loaded),
                    "head_loaded": 0.0,
                }

            return DetectionResult(
                score=self._normalize_score(fake_prob),
                confidence=confidence,
                model_name="siglip_image",
                backend=self._backend_used or "pytorch",
                features=features,
            )

        except Exception as e:
            logger.error("SigLIP image detection failed: %s", e)
            return DetectionResult(
                score=0.5,
                confidence=0.2,
                model_name="siglip_image",
                error=str(e),
            )

    # ------------------------------------------------------------------
    def _compute_confidence(self, fake_prob: float) -> float:
        extremity = abs(fake_prob - 0.5) * 2.0
        return float(np.clip(0.5 + 0.45 * extremity, 0.1, 0.95))

    # ------------------------------------------------------------------
    def _load_text_embeddings(self):
        """Pre-compute text embeddings for zero-shot fallback."""
        import torch
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(self._model_id, cache_dir=self._cache_dir)
        text_model = AutoModel.from_pretrained(self._model_id, cache_dir=self._cache_dir).to(self._device)
        text_model.eval()
        with torch.no_grad():
            real_inputs = tokenizer([self.REAL_PROMPT], padding=True, return_tensors="pt").to(self._device)
            fake_inputs = tokenizer([self.FAKE_PROMPT], padding=True, return_tensors="pt").to(self._device)
            self._real_text_emb = text_model.get_text_features(**real_inputs)
            self._real_text_emb = self._real_text_emb / (self._real_text_emb.norm(dim=-1, keepdim=True) + 1e-8)
            self._fake_text_emb = text_model.get_text_features(**fake_inputs)
            self._fake_text_emb = self._fake_text_emb / (self._fake_text_emb.norm(dim=-1, keepdim=True) + 1e-8)
        # Free the text model to save memory
        del text_model

    # ------------------------------------------------------------------
    async def _ensure_loaded(self) -> None:
        if self._model is not None or self._head_model is not None:
            return
        with self._lock:
            if self._model is not None or self._head_model is not None:
                return
            await self._load_model()

    # ------------------------------------------------------------------
    async def _load_model(self) -> None:
        """Lazy-load SigLIP backbone + (optional) fine-tuned head or LoRA adapter."""
        import torch
        from transformers import (
            AutoModel,
            AutoProcessor,
            AutoTokenizer,
            AutoModelForImageClassification,
            AutoImageProcessor,
        )

        # ----- Try fine-tuned head FIRST -----
        if self._fine_tuned_head_repo:
            try:
                logger.info(
                    "Loading fine-tuned SigLIP head: %s on %s",
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
                    "Fine-tuned SigLIP head loaded: %s (labels=%s)",
                    self._fine_tuned_head_repo,
                    getattr(self._head_model.config, "id2label", "unknown"),
                )
            except Exception as e:
                logger.warning(
                    "Could not load fine-tuned SigLIP head %s (%s). "
                    "Falling back to SigLIP backbone + adapter/zero-shot.",
                    self._fine_tuned_head_repo, e,
                )
                self._head_model = None
                self._head_processor = None

        # ----- Always load the SigLIP backbone (zero-shot fallback) -----
        logger.info("Loading SigLIP backbone: %s on %s", self._model_id, self._device)
        self._processor = AutoProcessor.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        self._model = AutoModel.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        ).to(self._device)
        self._model.eval()

        # Try to load the LoRA adapter if present.
        adapter_dir = self._adapter_path
        if adapter_dir and os.path.isdir(adapter_dir):
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, adapter_dir)
                self._model.eval()
                self._adapter_loaded = True
                logger.info("SigLIP LoRA adapter loaded from %s", adapter_dir)
            except Exception as e:
                logger.warning(
                    "SigLIP LoRA adapter at %s failed to load (%s); "
                    "falling back to zero-shot", adapter_dir, e
                )
                self._adapter_loaded = False
        else:
            logger.info(
                "No SigLIP LoRA adapter found at %s; using zero-shot fallback",
                adapter_dir,
            )
            self._adapter_loaded = False

        self._backend_used = "pytorch"
        logger.info(
            "SigLIP image detector ready (head=%s, adapter=%s, device=%s)",
            self._head_model is not None, self._adapter_loaded, self._device,
        )
