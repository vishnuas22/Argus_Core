"""
Argus Core - Fusion Layers Init
================================
Cross-modal attention fusion components for the CAMME framework.
"""

from fusion_layers.cross_attention import CrossModalAttention, CrossModalAttentionBlock
from fusion_layers.self_attention import FusionSelfAttention
from fusion_layers.classification_head import DeepfakeClassificationHead

__all__ = [
    "CrossModalAttention",
    "CrossModalAttentionBlock",
    "FusionSelfAttention",
    "DeepfakeClassificationHead",
]
