"""
Argus Core - Detectors Package
==============================
Pluggable deepfake detector adapters. Each detector subclasses
``BaseDetector`` and returns ``DetectionResult``. Detectors are combined
per-modality by ``detectors.ensemble.DiversityEnsemble``.

Iteration-1 additions (SOTA detector adapters):
- CLIPLoRAImageDetector        — CLIP ViT-B/16 + LoRA (ForAda CVPR 2025 style)
- DINOv2ImageDetector          — DINOv2-base + MAC head (DINO-MAC NTIRE 2026)
- AASIST3AudioDetector         — AASIST3 end-to-end anti-spoofing (ASVspoof 2024)
- Wav2Vec2XLSRMoELoRADetector  — Wav2Vec2-XLS-R-300M + MoE-LoRA (arxiv 2025)
- VideoMAEDetector             — VideoMAE-base (NeurIPS 2022)
- AltFreeVideoDetector         — AltFree (CVPR 2024)

Iteration-3 additions (ensemble diversity expansion):
- SigLIPImageDetector          — SigLIP-base (ICCV 2023) — 3rd image detector

Iteration-4 additions (further ensemble diversity):
- TimeSformerVideoDetector     — TimeSformer-base (ICML 2021) — 3rd video detector
  NOTE: cc-by-nc-4.0 license (non-commercial). Disable for commercial use.
- ECAPATDNNAudioDetector       — ECAPA-TDNN (INTERSPEECH 2020) — 3rd audio detector
  Embedding-distance-based; MIT license; commercially usable.

All new detectors are strict-additive — the existing Wav2Vec2AudioDetector
and the existing ONNX-based analyzers continue to work unchanged.
"""

import logging

logger = logging.getLogger(__name__)

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from detectors.wav2vec2_detector import Wav2Vec2AudioDetector

# Iteration-1 SOTA adapters
try:
    from detectors.clip_image_detector import CLIPLoRAImageDetector
except Exception as e:
    CLIPLoRAImageDetector = None
    logger.warning(f"CLIPLoRAImageDetector unavailable: {e}")

try:
    from detectors.dinov2_image_detector import DINOv2ImageDetector
except Exception as e:
    DINOv2ImageDetector = None
    logger.warning(f"DINOv2ImageDetector unavailable: {e}")

try:
    from detectors.aasist3_audio_detector import AASIST3AudioDetector
except Exception as e:
    AASIST3AudioDetector = None
    logger.warning(f"AASIST3AudioDetector unavailable: {e}")

try:
    from detectors.wav2vec2_xls_r_audio_detector import Wav2Vec2XLSRMoELoRADetector
except Exception as e:
    Wav2Vec2XLSRMoELoRADetector = None
    logger.warning(f"Wav2Vec2XLSRMoELoRADetector unavailable: {e}")

try:
    from detectors.videomae_detector import VideoMAEDetector
except Exception as e:
    VideoMAEDetector = None
    logger.warning(f"VideoMAEDetector unavailable: {e}")

try:
    from detectors.altfree_video_detector import AltFreeVideoDetector
except Exception as e:
    AltFreeVideoDetector = None
    logger.warning(f"AltFreeVideoDetector unavailable: {e}")

# Iteration-3 SOTA adapters (ensemble diversity expansion)
try:
    from detectors.siglip_image_detector import SigLIPImageDetector
except Exception as e:
    SigLIPImageDetector = None
    logger.warning(f"SigLIPImageDetector unavailable: {e}")

# Iteration-4 SOTA adapters (further ensemble diversity)
try:
    from detectors.timesformer_detector import TimeSformerVideoDetector
except Exception as e:
    TimeSformerVideoDetector = None
    logger.warning(f"TimeSformerVideoDetector unavailable: {e}")

try:
    from detectors.ecapa_tdnn_audio_detector import ECAPATDNNAudioDetector
except Exception as e:
    ECAPATDNNAudioDetector = None
    logger.warning(f"ECAPATDNNAudioDetector unavailable: {e}")

# Iteration-5 SOTA adapters (SBI boundary-artifact detection)
try:
    from detectors.sbi_image_detector import SBIDetector
except Exception as e:
    SBIDetector = None
    logger.warning(f"SBIDetector unavailable: {e}")

# Iteration-6 SOTA adapters (cross-generator + state-space audio)
try:
    from detectors.ucf_cross_forgery_detector import UCFCrossForgeryDetector
except Exception as e:
    UCFCrossForgeryDetector = None
    logger.warning(f"UCFCrossForgeryDetector unavailable: {e}")

try:
    from detectors.cdp_mamba_audio_detector import CDPMambaDetector
except Exception as e:
    CDPMambaDetector = None
    logger.warning(f"CDPMambaDetector unavailable: {e}")

from detectors.ensemble import (
    DiversityEnsemble,
    EnsembleMember,
    combine_detector_results,
    get_default_ensemble,
)

_available_sota = [
    name for name, cls in [
        ("CLIPLoRA", CLIPLoRAImageDetector),
        ("DINOv2", DINOv2ImageDetector),
        ("SigLIP", SigLIPImageDetector),
        ("AASIST3", AASIST3AudioDetector),
        ("XLS-R", Wav2Vec2XLSRMoELoRADetector),
        ("ECAPA-TDNN", ECAPATDNNAudioDetector),
        ("SBI", SBIDetector),
        ("UCF", UCFCrossForgeryDetector),
        ("CDP-Mamba", CDPMambaDetector),
        ("VideoMAE", VideoMAEDetector),
        ("AltFree", AltFreeVideoDetector),
        ("TimeSformer", TimeSformerVideoDetector),
    ] if cls is not None
]
logger.info(f"SOTA detectors available: {', '.join(_available_sota) if _available_sota else 'none'}")

__all__ = [
    # Base
    "BaseDetector", "DetectionResult", "DetectorBackend",
    # Legacy
    "Wav2Vec2AudioDetector",
    # Iteration-1 SOTA image
    "CLIPLoRAImageDetector", "DINOv2ImageDetector",
    # Iteration-1 SOTA audio
    "AASIST3AudioDetector", "Wav2Vec2XLSRMoELoRADetector",
    # Iteration-1 SOTA video
    "VideoMAEDetector", "AltFreeVideoDetector",
    # Iteration-3 SOTA image (diversity)
    "SigLIPImageDetector",
    # Iteration-4 SOTA video + audio (further diversity)
    "TimeSformerVideoDetector", "ECAPATDNNAudioDetector",
    # Iteration-5 SOTA image (SBI boundary-artifact detection)
    "SBIDetector",
    # Iteration-6 SOTA image (cross-generator detection)
    "UCFCrossForgeryDetector",
    # Iteration-6 SOTA audio (state-space model)
    "CDPMambaDetector",
    # Ensemble
    "DiversityEnsemble", "EnsembleMember",
    "combine_detector_results", "get_default_ensemble",
]
