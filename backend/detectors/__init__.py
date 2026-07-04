"""
Argus Core - Detectors Package
==============================
Pluggable deepfake detector adapters. Each detector subclasses
``BaseDetector`` and returns ``DetectionResult``. Detectors are combined
per-modality by ``detectors.ensemble.DiversityEnsemble``.

Curated 2026-07-02 (see MODEL_AUDIT.md):
  Removed dead/unusable detectors:
    - AltFreeVideoDetector   (no canonical HF port — fake stub fallback)
    - CDPMambaDetector       (no public weights — placeholder source)

  Kept (all backed by real, public model sources):
    Image:  CLIPLoRAImageDetector, DINOv2ImageDetector, SigLIPImageDetector,
            SBIDetector, UCFCrossForgeryDetector
    Audio:  AASIST3AudioDetector, Wav2Vec2XLSRMoELoRADetector,
            ECAPATDNNAudioDetector, Wav2Vec2AudioDetector (legacy)
    Video:  VideoMAEDetector, TimeSformerVideoDetector (license-gated)

All detectors are lazy-loaded — the import block below only imports the
class definitions, NOT the model weights. Model weights are loaded on
first inference call via ModelManager.get_model().
"""

import logging

logger = logging.getLogger(__name__)

from detectors.base import BaseDetector, DetectionResult, DetectorBackend
from detectors.wav2vec2_detector import Wav2Vec2AudioDetector

# ============== IMAGE DETECTORS (5) ==============
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
    from detectors.siglip_image_detector import SigLIPImageDetector
except Exception as e:
    SigLIPImageDetector = None
    logger.warning(f"SigLIPImageDetector unavailable: {e}")

try:
    from detectors.sbi_image_detector import SBIDetector
except Exception as e:
    SBIDetector = None
    logger.warning(f"SBIDetector unavailable: {e}")

try:
    from detectors.ucf_cross_forgery_detector import UCFCrossForgeryDetector
except Exception as e:
    UCFCrossForgeryDetector = None
    logger.warning(f"UCFCrossForgeryDetector unavailable: {e}")

# ============== AUDIO DETECTORS (3) ==============
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
    from detectors.ecapa_tdnn_audio_detector import ECAPATDNNAudioDetector
except Exception as e:
    ECAPATDNNAudioDetector = None
    logger.warning(f"ECAPATDNNAudioDetector unavailable: {e}")

# ============== VIDEO DETECTORS (2) ==============
try:
    from detectors.videomae_detector import VideoMAEDetector
except Exception as e:
    VideoMAEDetector = None
    logger.warning(f"VideoMAEDetector unavailable: {e}")

# TimeSformer is license-gated (CC-BY-NC-4.0 — non-commercial only).
# Default disabled for commercial deployments. Enable via
# ENABLE_TIMESFORMER=true in .env for research use.
import os
_ENABLE_TIMESFORMER = os.environ.get("ENABLE_TIMESFORMER", "false").lower() in ("true", "1", "yes")
TimeSformerVideoDetector = None
if _ENABLE_TIMESFORMER:
    try:
        from detectors.timesformer_detector import TimeSformerVideoDetector
    except Exception as e:
        TimeSformerVideoDetector = None
        logger.warning(f"TimeSformerVideoDetector unavailable: {e}")
else:
    logger.info(
        "TimeSformerVideoDetector disabled (CC-BY-NC-4.0 license). "
        "Set ENABLE_TIMESFORMER=true for non-commercial research use."
    )

# ============== Ensemble combiner ==============
from detectors.ensemble import (
    DiversityEnsemble,
    EnsembleMember,
    combine_detector_results,
    get_default_ensemble,
)


# ============== Availability summary ==============
_available_sota = [
    name for name, cls in [
        ("CLIPLoRA", CLIPLoRAImageDetector),
        ("DINOv2", DINOv2ImageDetector),
        ("SigLIP", SigLIPImageDetector),
        ("SBI", SBIDetector),
        ("UCF", UCFCrossForgeryDetector),
        ("AASIST3", AASIST3AudioDetector),
        ("XLS-R", Wav2Vec2XLSRMoELoRADetector),
        ("ECAPA-TDNN", ECAPATDNNAudioDetector),
        ("VideoMAE", VideoMAEDetector),
        ("TimeSformer", TimeSformerVideoDetector),
    ] if cls is not None
]
logger.info(
    f"SOTA detectors available ({len(_available_sota)}): "
    f"{', '.join(_available_sota) if _available_sota else 'none'}"
)


__all__ = [
    # Base
    "BaseDetector", "DetectionResult", "DetectorBackend",
    # Legacy audio
    "Wav2Vec2AudioDetector",
    # Image detectors (5)
    "CLIPLoRAImageDetector", "DINOv2ImageDetector",
    "SigLIPImageDetector", "SBIDetector", "UCFCrossForgeryDetector",
    # Audio detectors (3)
    "AASIST3AudioDetector", "Wav2Vec2XLSRMoELoRADetector",
    "ECAPATDNNAudioDetector",
    # Video detectors (2 — TimeSformer gated by license)
    "VideoMAEDetector", "TimeSformerVideoDetector",
    # Ensemble
    "DiversityEnsemble", "EnsembleMember",
    "combine_detector_results", "get_default_ensemble",
]
