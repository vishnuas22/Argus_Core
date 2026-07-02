"""
Argus Core v2 - Temporal Self-Attention Module
================================================
Multi-head self-attention over the temporal (frame) dimension
for detecting temporal inconsistencies in deepfake videos.

After cross-modal attention fuses information across modalities
at each frame, temporal self-attention captures:
    - Frame-to-frame identity drift
    - Micro-expression discontinuities
    - Blinking/lip-movement pattern anomalies
    - Flickering artifacts and motion jitter
    - Temporal interpolation artifacts from face-swapping

Architecture:
    Input: [B, T, d_model] (per-frame cross-attended features)
    → Sinusoidal positional encoding (temporal ordering)
    → N layers of multi-head self-attention + FFN
    → Options: mean-pool to [B, d_model] or return [B, T, d_model]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal ordering.

    Uses the original Transformer positional encoding scheme:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This provides the model with temporal ordering information
    without consuming learnable parameters.
    """

    def __init__(self, d_model: int = 512, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: [B, T, d_model]

        Returns:
            x + positional encoding: [B, T, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalTransformerLayer(nn.Module):
    """
    Single transformer layer for temporal self-attention.

    Pre-norm architecture (LayerNorm before attention/FFN)
    for more stable training, as used in modern models.

    Components:
        1. Multi-head self-attention (captures frame dependencies)
        2. Feed-forward network (non-linear feature transformation)
        3. Residual connections + LayerNorm
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for FFN layers."""
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with pre-norm architecture.

        Args:
            x: [B, T, d_model]
            causal_mask: Optional causal attention mask [T, T]
            key_padding_mask: Optional padding mask [B, T]

        Returns:
            (output [B, T, d_model], attention_weights [B, num_heads, T, T])
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attended, attn_weights = self.self_attention(
            normed, normed, normed,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + attended

        # Feed-forward with pre-norm
        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x, attn_weights


class TemporalSelfAttention(nn.Module):
    """
    Multi-layer temporal self-attention for deepfake detection.

    Operates on frame-level features after cross-modal attention,
    modeling temporal dependencies that reveal deepfake artifacts:

    - **Identity drift**: Gradual shifts in facial identity across frames
    - **Motion discontinuity**: Unnatural jumps between frames
    - **Blinking anomaly**: Irregular blinking patterns in face-swaps
    - **Lip-sync lag**: Temporal misalignment of lip movements
    - **Flickering artifacts**: Frame-level noise from generation

    Architecture:
        Input: [B, T, d_model] from cross-modal attention
        → Sinusoidal positional encoding
        → N transformer encoder layers (pre-norm, 8 heads each)
        → Output: [B, T, d_model] or pooled [B, d_model]

    Supports:
        - Full bidirectional attention (default)
        - Causal masking (autoregressive, for real-time streaming)
        - Key padding mask (variable-length sequences)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_frames: int = 256,
    ):
        """
        Initialize temporal self-attention.

        Args:
            d_model: Feature dimension per frame
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
            max_frames: Maximum number of frames to process
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_frames,
            dropout=dropout,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Pooling projection
        self.pool_projection = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.pool_projection.weight)
        nn.init.zeros_(self.pool_projection.bias)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply temporal self-attention over frame sequences.

        Args:
            x: Frame-level features [B, T, d_model]
            causal: If True, use causal masking (for streaming)
            padding_mask: Optional [B, T] mask (True=pad, False=valid)
            return_sequence: If True, return [B, T, d_model].
                             If False, return mean-pooled [B, d_model].
            return_attention: If True, return attention weights.

        Returns:
            (features, attention_weights or None)
            features: [B, d_model] if return_sequence=False,
                       [B, T, d_model] if return_sequence=True
            attention_weights: [B, num_heads, T, T] from last layer (if requested)
        """
        # Add positional encoding
        x = self.pos_encoding(x)

        # Optional causal mask
        causal_mask = None
        if causal:
            causal_mask = self._generate_causal_mask(x.size(1), x.device)

        # Apply transformer layers
        all_attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, causal_mask=causal_mask, key_padding_mask=padding_mask)
            all_attn_weights = attn_weights  # Keep last layer's weights

        x = self.final_norm(x)

        if return_sequence:
            out = x
        else:
            # Mean pool over temporal dimension (ignoring padding)
            if padding_mask is not None:
                # Mask padded positions before pooling
                mask_expanded = (~padding_mask).unsqueeze(-1).float()  # [B, T, 1]
                x_masked = x * mask_expanded
                lengths = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
                pooled = x_masked.sum(dim=1) / lengths  # [B, d_model]
            else:
                pooled = x.mean(dim=1)  # [B, d_model]

            out = self.pool_projection(pooled)

        attn_out = all_attn_weights if return_attention else None
        return out, attn_out
