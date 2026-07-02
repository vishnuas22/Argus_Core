"""
Argus Core - Projection Heads
=============================
Shared projection layer definitions for mapping modality-specific
encoder outputs to a common embedding dimension for cross-attention fusion.

Architecture:
    Each projection head maps from modality dimension d_model -> d_fused (512).
    Uses Linear + LayerNorm + GELU + Dropout for stable training and
    gradient flow through the cross-attention layers.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class SharedProjectionConfig:
    """Configuration for all modality projection dimensions."""
    d_model_visual: int = 768
    d_model_visual_temporal: int = 512
    d_model_audio_spectral: int = 512
    d_model_audio_ssl: int = 768
    d_model_text: int = 768
    d_fused: int = 512
    dropout_rate: float = 0.1


class ModalityProjection(nn.Module):
    """
    Projects modality-specific features to the shared embedding space.

    Architecture: Linear(d_in, d_out) -> LayerNorm -> GELU -> Dropout
    Xavier uniform initialization for all projection weights.

    Used by each encoder to map its output to d_fused dimension
    before cross-modal attention fusion.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize projection head.

        Args:
            d_input: Input feature dimension from the encoder
            d_output: Output dimension (shared d_fused across modalities)
            dropout_rate: Dropout probability after activation
        """
        super().__init__()

        self.projection = nn.Linear(d_input, d_output)
        self.layer_norm = nn.LayerNorm(d_output)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier uniform initialization for stable gradient flow."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project features to shared embedding space.

        Args:
            features: Input tensor [B, d_input]

        Returns:
            Projected tensor [B, d_output]
        """
        projected = self.projection(features)
        projected = self.layer_norm(projected)
        projected = self.activation(projected)
        projected = self.dropout(projected)
        return projected
