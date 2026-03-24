"""
Argus Core v2 - Text Feature Encoder
======================================
RoBERTa-based text feature extractor for cross-modal fusion.

UMFT Architecture:
    Uses pretrained RoBERTa-base to encode transcribed audio text
    or submitted text into semantic representations.

    Captures: semantic coherence, logical consistency, perplexity
    patterns, stylistic uniformity, and burstiness patterns
    characteristic of LLM output.

Output modes:
    - Sentence-level: [B, S, d_fused] sliding window over sentences
    - Collapsed:      [B, d_fused] single [CLS] representation

Projects to d_fused=512 for cross-attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from encoders.projection_heads import ModalityProjection, SharedProjectionConfig


class TextFeatureEncoder(nn.Module):
    """
    Text feature encoder using pretrained RoBERTa-base.

    v2 Enhancement: Adds sentence-level sliding window encoding
    for long transcripts, returning [B, S, d_fused] where S is the
    number of sentence segments. This enables temporal cross-attention
    between text segments and video frames.

    Extracts the [CLS] token representation from the last hidden
    state, which serves as a sentence-level summary capturing
    semantic, syntactic, and stylistic features relevant to
    AI-generated text detection.

    Output: 512-d feature vectors ready for cross-modal attention fusion.
    """

    def __init__(
        self,
        config: Optional[SharedProjectionConfig] = None,
        pretrained_roberta: bool = True,
        freeze_roberta_layers: int = 6,
        max_segment_length: int = 128,
        segment_stride: int = 64,
    ):
        """
        Initialize text feature encoder.

        Args:
            config: Shared projection configuration
            pretrained_roberta: Load pretrained RoBERTa weights
            freeze_roberta_layers: Number of transformer layers to freeze (0-12)
            max_segment_length: Max tokens per sliding window segment
            segment_stride: Stride between segments (overlap = max_segment_length - stride)
        """
        super().__init__()

        self.config = config or SharedProjectionConfig()
        self.max_segment_length = max_segment_length
        self.segment_stride = segment_stride

        # Build RoBERTa
        self.roberta = self._build_roberta(pretrained_roberta, freeze_roberta_layers)

        # Projection: 768 → 512
        self.projection = ModalityProjection(
            d_input=self.config.d_model_text,
            d_output=self.config.d_fused,
            dropout_rate=self.config.dropout_rate,
        )

        # Segment aggregation: attention-weighted pooling over segments
        self.segment_attention = nn.Sequential(
            nn.Linear(self.config.d_fused, self.config.d_fused // 4),
            nn.Tanh(),
            nn.Linear(self.config.d_fused // 4, 1),
        )
        self._init_segment_attention()

        self.output_norm = nn.LayerNorm(self.config.d_fused)

    def _init_segment_attention(self):
        """Initialize segment attention weights."""
        for module in self.segment_attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_roberta(self, pretrained: bool, freeze_layers: int) -> nn.Module:
        """
        Build RoBERTa model from HuggingFace transformers.

        Args:
            pretrained: Load pretrained weights from roberta-base
            freeze_layers: Number of encoder layers to freeze (0-12)

        Returns:
            RoBERTa model
        """
        from transformers import RobertaModel

        model_name = "roberta-base"
        if pretrained:
            model = RobertaModel.from_pretrained(model_name)
        else:
            from transformers import RobertaConfig
            roberta_config = RobertaConfig(
                hidden_size=self.config.d_model_text,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                vocab_size=50265,
                max_position_embeddings=514,
            )
            model = RobertaModel(roberta_config)

        # Freeze early layers
        if freeze_layers > 0:
            for i, layer in enumerate(model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract text features from tokenized input (collapsed).

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len] (1=attend, 0=ignore)

        Returns:
            Text feature vector [B, d_fused]
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract [CLS] token from last hidden state
        cls_features = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        # Project to fused dimension
        projected = self.projection(cls_features)  # [B, 512]

        return self.output_norm(projected)

    def forward_segments(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract segment-level text features via sliding window.

        Splits long input into overlapping segments, encodes each
        independently via RoBERTa, and returns per-segment features
        for cross-modal temporal alignment.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Segment-level features [B, S, d_fused] where S = num segments
        """
        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # If sequence fits in a single segment, return as [B, 1, d_fused]
        if seq_len <= self.max_segment_length:
            features = self.forward(input_ids, attention_mask)  # [B, d_fused]
            return features.unsqueeze(1)  # [B, 1, d_fused]

        # Create sliding window segments
        segment_features_list = []
        start = 0
        while start < seq_len:
            end = min(start + self.max_segment_length, seq_len)

            seg_ids = input_ids[:, start:end]  # [B, seg_len]
            seg_mask = attention_mask[:, start:end]  # [B, seg_len]

            # Encode segment
            outputs = self.roberta(input_ids=seg_ids, attention_mask=seg_mask)
            cls_features = outputs.last_hidden_state[:, 0, :]  # [B, 768]
            projected = self.projection(cls_features)  # [B, d_fused]

            segment_features_list.append(projected)

            start += self.segment_stride
            if end == seq_len:
                break

        # Stack segments: [B, S, d_fused]
        segment_features = torch.stack(segment_features_list, dim=1)

        return self.output_norm(segment_features)

    def forward_with_attention_pool(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract text features with attention-weighted segment pooling.

        Useful when you need a single vector but want to process
        long texts via sliding window with learned aggregation.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Attention-pooled features [B, d_fused]
        """
        # Get per-segment features: [B, S, d_fused]
        segment_features = self.forward_segments(input_ids, attention_mask)

        # Attention-weighted aggregation
        attn_scores = self.segment_attention(segment_features)  # [B, S, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, S, 1]
        pooled = (segment_features * attn_weights).sum(dim=1)  # [B, d_fused]

        return self.output_norm(pooled)

    def get_feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.config.d_fused
