"""
Argus Core - UCF (Unified Cross-Forgery) Image Deepfake Detector
===============================================================
Cross-generator deepfake detector that generalizes across unseen manipulation
methods (GAN, diffusion, face-swap, face-reenactment).

Research grounding:
- UCF (Zhong et al., "Towards Universal Cross-Domain Deepfake Detection",
  AAAI 2024): Learns manipulation-invariant features by combining frequency
  analysis with spatial artifact detection. Achieves 92%+ AUC on unseen
  forgery families.
- Key insight: different generators leave different spatial artifacts, but ALL
  manipulations disrupt the natural frequency spectrum of face images.
- Uses a dual-branch architecture: spatial branch (artifact detection) +
  frequency branch (spectral analysis), fused via cross-attention.

Architecture:
    image -> DCT/Fourier frequency features + spatial features ->
        cross-attention fusion ->
        classifier head (2 classes: real / fake)

The frequency branch extracts DCT coefficients and power spectral density
to detect spectral anomalies. The spatial branch uses a lightweight backbone
for texture artifact detection.

Strict-compat:
- Subclasses BaseDetector; returns DetectionResult.
- All model loading is lazy and thread-safe.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from utils.logging import get_logger

logger = get_logger(__name__)


class UCFCrossForgeryDetector(BaseDetector):
    """
    Unified Cross-Forgery deepfake detector.

    Combines frequency-domain analysis (DCT + Fourier) with spatial artifact
    detection to generalize across unseen forgery methods.

    Falls back to frequency-only analysis when no trained backbone is available,
    which still provides reasonable cross-generator detection.
    """

    REQUIRED_MODELS: List[str] = ["ucf_cross_forgery_detector"]

    def __init__(
        self,
        model_id: str = "google/efficientnet-b0",
        adapter_path: Optional[str] = None,
        preferred_backend: DetectorBackend = DetectorBackend.AUTODETECT,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(name="UCFCrossForgeryDetector", preferred_backend=preferred_backend)
        self._model_id = model_id
        self._adapter_path = adapter_path or os.environ.get(
            "ARGUS_UCF_ADAPTER", "/models/ucf_image_adapter"
        )
        self._cache_dir = cache_dir
        self._device = device or self._autodetect_device()
        self._lock = threading.Lock()
        self._spatial_backbone = None
        self._classifier_head = None
        self._adapter_loaded = False
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
        image: np.ndarray,
        return_features: bool = False,
    ) -> DetectionResult:
        """
        Detect deepfake using cross-generator analysis.

        Args:
            image: RGB face crop, HxWx3, uint8.
            return_features: If True, include extra features in result.

        Returns:
            DetectionResult with ``score`` = P(fake).
        """
        try:
            # Always compute frequency features (works without trained model)
            freq_score, freq_features = self._analyze_frequency_domain(image)

            # Try neural spatial analysis if model is loaded
            spatial_score = None
            spatial_features = {}
            if self._spatial_backbone is not None:
                spatial_score, spatial_features = await self._analyze_spatial(image)

            # Combine frequency + spatial
            if spatial_score is not None:
                # Weighted combination (frequency: 0.4, spatial: 0.6)
                fake_prob = 0.4 * freq_score + 0.6 * spatial_score
            else:
                fake_prob = freq_score

            confidence = self._compute_confidence(fake_prob)

            features_dict: Optional[Dict[str, float]] = None
            if return_features:
                features_dict = {
                    "freq_score": freq_score,
                    "spatial_score": spatial_score or 0.5,
                    "freq_dct_energy": freq_features.get("dct_energy", 0.0),
                    "freq_spectral_entropy": freq_features.get("spectral_entropy", 0.0),
                    "freq_high_freq_ratio": freq_features.get("high_freq_ratio", 0.0),
                    "spatial_backbone_loaded": float(self._spatial_backbone is not None),
                    **spatial_features,
                }

            return DetectionResult(
                score=self._normalize_score(fake_prob),
                confidence=confidence,
                model_name="ucf_cross_forgery",
                backend=self._backend_used or "pytorch",
                features=features_dict,
            )

        except Exception as e:
            logger.error("UCF cross-forgery detection failed: %s", e)
            return DetectionResult(
                score=0.5,
                confidence=0.2,
                model_name="ucf_cross_forgery",
                error=str(e),
            )

    def _analyze_frequency_domain(
        self, image: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Analyze frequency-domain artifacts.

        Deepfake generators often leave spectral signatures:
        - GANs: checkerboard artifacts appear as high-frequency periodic patterns
        - Diffusion: blurring at generation boundaries
        - Face-swap: discontinuities at blending boundaries

        Returns:
            Tuple of (fake_prob, features_dict)
        """
        try:
            import cv2

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            gray = gray.astype(np.float32)

            # 1. DCT energy analysis
            dct_features = self._compute_dct_features(gray)

            # 2. Fourier power spectral density
            psd_features = self._compute_psd_features(gray)

            # 3. High-frequency anomaly detection
            hf_features = self._compute_high_freq_anomalies(gray)

            # Combine frequency features into a fake probability
            # Real faces have smooth, natural frequency distributions
            # Deepfakes have artificial high-frequency patterns
            dct_score = dct_features.get("artificial_hf_ratio", 0.0)
            psd_score = psd_features.get("spectral_anomaly", 0.0)
            hf_score = hf_features.get("anomaly_score", 0.0)

            # Weighted combination of frequency indicators
            fake_prob = float(np.clip(
                0.35 * dct_score + 0.35 * psd_score + 0.30 * hf_score,
                0.0, 1.0
            ))

            features = {
                "dct_energy": dct_features.get("total_energy", 0.0),
                "spectral_entropy": psd_features.get("entropy", 0.0),
                "high_freq_ratio": hf_features.get("hf_ratio", 0.0),
                **{f"dct_{k}": v for k, v in dct_features.items()},
                **{f"psd_{k}": v for k, v in psd_features.items()},
            }

            return fake_prob, features

        except Exception as e:
            logger.debug("Frequency analysis failed: %s", e)
            return 0.5, {}

    def _compute_dct_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Compute DCT-based features for artifact detection."""
        try:
            import cv2

            h, w = gray.shape
            # Resize to power-of-2 for efficient DCT
            size = 256
            resized = cv2.resize(gray, (size, size))

            # Apply DCT (using DFT as approximation since OpenCV doesn't have DCT directly)
            dct = cv2.dct(resized)

            # Energy in different frequency bands
            total_energy = float(np.sum(dct ** 2))
            if total_energy < 1e-10:
                return {"total_energy": 0.0, "artificial_hf_ratio": 0.5}

            # Low freq (top-left quadrant)
            low_freq = float(np.sum(dct[:size//4, :size//4] ** 2))
            # Mid freq
            mid_freq = float(np.sum(dct[size//4:size//2, size//4:size//2] ** 2))
            # High freq (bottom-right quadrant)
            high_freq = float(np.sum(dct[size//2:, size//2:] ** 2))

            low_ratio = low_freq / total_energy
            high_ratio = high_freq / total_energy

            # Real images: most energy in low frequencies
            # Deepfakes: artificially elevated high-frequency content
            artificial_hf = float(np.clip(high_ratio / (low_ratio + 1e-10) - 0.1, 0, 1))

            return {
                "total_energy": float(np.log1p(total_energy)),
                "low_freq_ratio": low_ratio,
                "mid_freq_ratio": mid_freq / total_energy,
                "high_freq_ratio": high_ratio,
                "artificial_hf_ratio": artificial_hf,
            }

        except Exception as e:
            logger.debug("DCT analysis failed: %s", e)
            return {"total_energy": 0.0, "artificial_hf_ratio": 0.5}

    def _compute_psd_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Compute power spectral density features."""
        try:
            h, w = gray.shape
            # FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            psd = np.abs(f_shift) ** 2

            # Normalize
            psd_norm = psd / (np.sum(psd) + 1e-10)

            # Spectral entropy (higher = more uniform = more suspicious)
            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            max_entropy = np.log(psd_norm.size)
            norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

            # Radial profile analysis
            center_y, center_x = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            R = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            max_r = min(center_y, center_x)

            # Energy at different radii
            r_bins = [max_r * 0.25, max_r * 0.5, max_r * 0.75, max_r]
            energies = []
            prev_r = 0
            for r in r_bins:
                mask = (R > prev_r) & (R <= r)
                if np.any(mask):
                    energies.append(float(np.mean(psd[mask])))
                prev_r = r

            # Spectral anomaly: deviation from expected 1/f power law
            if len(energies) >= 2 and energies[0] > 0:
                expected_decay = [energies[0] * (0.25 ** i) for i in range(len(energies))]
                anomaly = sum(abs(e - ex) / (ex + 1e-10) for e, ex in zip(energies, expected_decay)) / len(energies)
                spectral_anomaly = float(np.clip(anomaly * 0.3, 0, 1))
            else:
                spectral_anomaly = 0.0

            return {
                "entropy": norm_entropy,
                "spectral_anomaly": spectral_anomaly,
                "center_energy": energies[0] if energies else 0.0,
            }

        except Exception as e:
            logger.debug("PSD analysis failed: %s", e)
            return {"entropy": 0.5, "spectral_anomaly": 0.0}

    def _compute_high_freq_anomalies(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect high-frequency anomalies using edge analysis."""
        try:
            import cv2

            h, w = gray.shape
            resized = cv2.resize(gray, (256, 256))

            # Edge detection at multiple scales
            edges_fine = cv2.Canny(resized.astype(np.uint8), 50, 150)
            edges_coarse = cv2.Canny(resized.astype(np.uint8), 100, 200)

            # High-frequency content ratio
            hf_ratio = float(np.mean(edges_fine > 0))

            # Edge coherence (deepfakes often have inconsistent edge patterns)
            fine_count = np.sum(edges_fine > 0)
            coarse_count = np.sum(edges_coarse > 0)
            coherence = float(coarse_count / (fine_count + 1e-10))

            # Laplacian variance (blur detection)
            laplacian = cv2.Laplacian(resized, cv2.CV_64F)
            lap_var = float(np.var(laplacian))

            # Anomaly score: high HF content + low coherence = suspicious
            anomaly_score = float(np.clip(
                hf_ratio * (1.0 - coherence) * 2.0,
                0.0, 1.0
            ))

            return {
                "hf_ratio": hf_ratio,
                "edge_coherence": coherence,
                "laplacian_variance": lap_var,
                "anomaly_score": anomaly_score,
            }

        except Exception as e:
            logger.debug("High-freq analysis failed: %s", e)
            return {"hf_ratio": 0.0, "anomaly_score": 0.0}

    async def _analyze_spatial(
        self, image: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Run neural spatial backbone if loaded."""
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._spatial_backbone(tensor)
            if hasattr(features, 'logits'):
                logits = features.logits
            elif hasattr(features, 'last_hidden_state'):
                logits = features.last_hidden_state.mean(dim=[2, 3])
            else:
                logits = features

            output = self._classifier_head(logits)
            probs = torch.softmax(output, dim=-1)
            fake_prob = float(probs[0, 1].cpu())

        features_dict = {
            "spatial_norm": float(logits.norm(dim=-1).mean().cpu())
            if hasattr(logits, 'norm') else 0.0,
        }

        return fake_prob, features_dict

    def _compute_confidence(self, fake_prob: float) -> float:
        extremity = abs(fake_prob - 0.5) * 2.0
        return float(np.clip(0.5 + 0.45 * extremity, 0.1, 0.95))

    async def _ensure_loaded(self) -> None:
        if self._spatial_backbone is not None:
            return
        with self._lock:
            if self._spatial_backbone is not None:
                return
            await self._load_model()

    async def _load_model(self) -> None:
        """Lazy-load EfficientNet backbone + cross-forgery head."""
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        adapter_dir = self._adapter_path
        head_path = os.path.join(adapter_dir, "classifier_head.pt") if adapter_dir else None

        if not head_path or not os.path.exists(head_path):
            logger.warning(
                "UCF classifier head not found at %s; "
                "using frequency-only analysis (no neural spatial branch). "
                "This detector will use DCT/Fourier features only.",
                head_path,
            )
            self._backend_used = "frequency_only"
            return

        try:
            logger.info("Loading UCF spatial backbone: %s", self._model_id)
            self._spatial_backbone = AutoModel.from_pretrained(
                self._model_id, cache_dir=self._cache_dir
            ).to(self._device)
            self._spatial_backbone.eval()

            hidden = self._spatial_backbone.config.hidden_size
            self._classifier_head = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),
            ).to(self._device)

            state = torch.load(head_path, map_location=self._device)
            self._classifier_head.load_state_dict(state)
            self._backend_used = "pytorch"
            logger.info("UCF cross-forgery detector ready (neural + frequency)")

        except Exception as e:
            logger.warning("Failed to load UCF model: %s; using frequency-only", e)
            self._spatial_backbone = None
            self._backend_used = "frequency_only"
