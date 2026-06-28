"""
Argus Core - Self-Attention Refinement
=======================================
Post-fusion self-attention module that refines the concatenated
cross-attended features from all modality pairs.

After cross-attention produces 6 modality-pair representations
(V->A, V->T, A->V, A->T, T->V, T->A), they are aggregated per
modality and concatenated. Self-attention over this unified
representation captures higher-order interactions between all
cross-modal features.

Architecture:
    Input: [z_v_fused; z_a_fused; z_t_fused] (3 tokens, each d_fused)
    -> Learnable modality embeddings (additive positional encoding)
    -> 2-layer Transformer encoder (8 heads, d_model=1536, d_ff=2048)
    -> Mean pool over tokens -> Output: [B, 1536] -> Linear -> [B, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ModalityPositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for modality tokens.

    Each modality (visual, audio, text) gets a unique learned embedding
    that encodes its identity in the self-attention sequence. This helps
    the model distinguish between modality representations and learn
    modality-specific interaction patterns.
    """

    def __init__(self, num_modalities: int = 3, d_model: int = 512):
        super().__init__()
        self.embeddings = nn.Embedding(num_modalities, d_model)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add modality positional embeddings.

        Args:
            x: [B, num_modalities, d_model]

        Returns:
            x with modality embeddings added: [B, num_modalities, d_model]
        """
        num_modalities = x.shape[1]
        positions = torch.arange(num_modalities, device=x.device)
        pos_emb = self.embeddings(positions)  # [num_modalities, d_model]
        return x + pos_emb.unsqueeze(0)  # Broadcast over batch


class FusionSelfAttention(nn.Module):
    """
    Post-fusion self-attention refinement module.

    Takes the concatenated cross-attended modality features and applies
    self-attention to capture higher-order interactions. This allows
    the model to reason about complex cross-modal relationships that
    may not be captured by pairwise cross-attention alone.

    For example:
    - Visual artifact + audio vocoder + low text perplexity = higher fake probability
    - Each pair alone might be uncertain, but the triplet is highly suspicious
    """

    def __init__(
        self,
        d_model: int = 512,
        num_modalities: int = 3,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize self-attention refinement.

        Args:
            d_model: Per-modality feature dimension
            num_modalities: Number of modalities (3: visual, audio, text)
            num_heads: Number of attention heads per layer
            num_layers: Number of transformer encoder layers
            d_ff: Feed-forward hidden dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.num_modalities = num_modalities
        self.d_concat = d_model * num_modalities  # 512 * 3 = 1536

        # Modality positional embeddings
        self.modality_pos_embedding = ModalityPositionalEmbedding(
            num_modalities=num_modalities,
            d_model=d_model,
        )

        # Project each modality from d_model to d_concat for cross-modal interaction
        self.input_projection = nn.Linear(d_model, self.d_concat)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_concat,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection: d_concat -> d_model
        self.output_projection = nn.Linear(self.d_concat, d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for all linear layers."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        z_visual: torch.Tensor,
        z_audio: torch.Tensor,
        z_text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine fused features through self-attention.

        Args:
            z_visual: Cross-attended visual features [B, d_model]
            z_audio: Cross-attended audio features [B, d_model]
            z_text: Cross-attended text features [B, d_model]

        Returns:
            Refined fused features [B, d_model]
        """
        batch_size = z_visual.shape[0]

        # Stack modality features: [B, 3, d_model]
        modality_tokens = torch.stack([z_visual, z_audio, z_text], dim=1)

        # Add modality positional embeddings
        modality_tokens = self.modality_pos_embedding(modality_tokens)

        # Project to concatenated dimension: [B, 3, d_concat]
        projected = self.input_projection(modality_tokens)

        # Self-attention over modality tokens
        refined = self.transformer_encoder(projected)  # [B, 3, d_concat]

        # Mean pool over modality tokens: [B, d_concat]
        pooled = refined.mean(dim=1)

        # Project back to d_model: [B, d_model]
        output = self.output_projection(pooled)
        output = self.output_norm(output)

        return output
