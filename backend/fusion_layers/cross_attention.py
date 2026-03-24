"""
Argus Core v2 - Cross-Modal Attention
=======================================
Multi-head cross-attention mechanism for cross-modal feature fusion.

UMFT Architecture:
    Implements the core of the CAMME framework: each modality attends
    to every other modality through learned attention weights.

    v2 Enhancement: Supports both single-token and sequence inputs:
    - Single-token: [B, d_model] — backward compatible (v1)
    - Sequence: [B, T, d_model] — for temporal cross-modal attention (v2)

Architecture per cross-attention pair (e.g., Visual → Audio):
    Query = Linear_Q(z_visual)    [B, T, d_model] or [B, d_model]
    Key   = Linear_K(z_audio)     [B, T, d_model] or [B, d_model]
    Value = Linear_V(z_audio)     [B, T, d_model] or [B, d_model]
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    Output = LayerNorm(Q + Attention(Q, K, V))

Xavier uniform initialization for all projection weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class CrossModalAttention(nn.Module):
    """
    Multi-head cross-attention between two modality feature representations.

    v2 Enhancement: Now supports sequence inputs [B, T, d_model] in addition
    to single-token [B, d_model]. When both modalities have sequence inputs,
    each token in the query sequence attends to all tokens in the key/value
    sequence, enabling frame-level cross-modal correlation detection.

    6 instances comprise full cross-modal fusion:
    - V→A: Visual queries attend to audio keys (lip-sync detection)
    - V→T: Visual queries attend to text keys (scene-text consistency)
    - A→V: Audio queries attend to visual keys (speech-face alignment)
    - A→T: Audio queries attend to text keys (voice-content consistency)
    - T→V: Text queries attend to visual keys (description verification)
    - T→A: Text queries attend to audio keys (transcript-speech match)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize cross-modal attention.

        Args:
            d_model: Feature dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout_rate: Dropout on attention weights
        """
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = math.sqrt(self.d_k)

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier uniform initialization for stable cross-modal attention training."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

        nn.init.zeros_(self.w_q.bias)
        nn.init.zeros_(self.w_k.bias)
        nn.init.zeros_(self.w_v.bias)
        nn.init.zeros_(self.w_o.bias)

    def _ensure_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is sequence format [B, T, d_model]."""
        if x.ndim == 2:
            return x.unsqueeze(1)  # [B, d_model] → [B, 1, d_model]
        return x

    def forward(
        self,
        query_features: torch.Tensor,
        key_value_features: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute cross-modal attention.

        Supports both single-token and sequence inputs.

        Args:
            query_features: [B, d_model] or [B, T_q, d_model]
            key_value_features: [B, d_model] or [B, T_kv, d_model]
            return_attention_weights: If True, return attention weights

        Returns:
            attended_features: Same shape as query_features
            attention_weights: [B, num_heads, T_q, T_kv] or None
        """
        was_2d = query_features.ndim == 2

        # Ensure sequence format
        q = self._ensure_sequence(query_features)    # [B, T_q, d_model]
        kv = self._ensure_sequence(key_value_features)  # [B, T_kv, d_model]

        batch_size, T_q = q.shape[0], q.shape[1]
        T_kv = kv.shape[1]
        residual = q

        # Project Q, K, V
        Q = self.w_q(q)  # [B, T_q, d_model]
        K = self.w_k(kv)  # [B, T_kv, d_model]
        V = self.w_v(kv)  # [B, T_kv, d_model]

        # Reshape for multi-head attention
        # [B, T, d_model] → [B, num_heads, T, d_k]
        Q = Q.view(batch_size, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, T_kv, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, T_kv, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, heads, T_q, T_kv]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights_dropped, V)  # [B, heads, T_q, d_k]

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, T_q, self.d_model)

        # Output projection
        output = self.w_o(attended)  # [B, T_q, d_model]

        # Residual connection + layer norm
        output = self.layer_norm(output + residual)

        # Collapse back to 2D if input was 2D
        if was_2d:
            output = output.squeeze(1)  # [B, d_model]

        if return_attention_weights:
            return output, attn_weights
        return output, None


class CrossModalAttentionBlock(nn.Module):
    """
    Complete cross-modal attention block with feed-forward network.

    Architecture (pre-norm for sequence, post-norm for single-token):
        CrossModalAttention(Q, K, V) → LayerNorm(Q + Attn)
        → FFN(x) → LayerNorm(x + FFN(x))

    FFN: Linear(d_model, 4*d_model) → GELU → Dropout → Linear(4*d_model, d_model)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.cross_attention = CrossModalAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self._initialize_ffn_weights()

    def _initialize_ffn_weights(self):
        """Xavier initialization for FFN layers."""
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        query_features: torch.Tensor,
        key_value_features: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full cross-attention block with FFN.

        Supports both [B, d_model] and [B, T, d_model] inputs.

        Args:
            query_features: [B, d_model] or [B, T, d_model]
            key_value_features: [B, d_model] or [B, T, d_model]
            return_attention_weights: If True, return attention weights

        Returns:
            refined_features: Same shape as query_features
            attention_weights: Attention map or None
        """
        # Cross-attention with residual + norm (handled internally)
        attended, attn_weights = self.cross_attention(
            query_features, key_value_features, return_attention_weights
        )

        # Feed-forward with residual + norm
        ffn_output = self.ffn(attended)
        output = self.ffn_norm(attended + self.ffn_dropout(ffn_output))

        return output, attn_weights
