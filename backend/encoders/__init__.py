"""
Argus Core - Feature Encoders
=============================
Cross-modal feature encoders for the CAMME-inspired fusion pipeline.

Each encoder extracts a fixed-dimension latent feature vector from its
respective modality, ready for cross-modal cross-attention fusion.
"""

from encoders.projection_heads import ModalityProjection, SharedProjectionConfig
from encoders.visual_encoder import VisualFeatureEncoder
from encoders.audio_encoder import AudioFeatureEncoder
from encoders.text_encoder import TextFeatureEncoder

__all__ = [
    "ModalityProjection",
    "SharedProjectionConfig",
    "VisualFeatureEncoder",
    "AudioFeatureEncoder",
    "TextFeatureEncoder",
]
