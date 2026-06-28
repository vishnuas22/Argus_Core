"""
Argus Core v2 - Visual Feature Encoder
========================================
ViT + 3D-CNN visual feature extractor for cross-modal fusion.

UMFT Architecture:
    Spatial branch: ViT-Base-Patch16-224 extracts per-frame semantic
    features via [CLS] token (768-d → projected to 512-d).

    Temporal branch: 3D ResNet-style blocks capture spatiotemporal
    patterns across frame sequences (micro-expressions, blinking,
    frame-to-frame inconsistencies).

Output modes:
    - Frame-level: [B, T, d_fused] for temporal self-attention
    - Collapsed: [B, d_fused] for backward compatibility
    - Image-only: [B, d_fused] single frame spatial features

Both branches project to d_fused=512 for cross-attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from encoders.projection_heads import ModalityProjection, SharedProjectionConfig


class SpatioTemporalBlock3D(nn.Module):
    """
    3D ResNet-style block for temporal feature extraction.

    Processes [B, C, T, H, W] video tensors using 3D convolutions
    to capture temporal dynamics (motion artifacts, micro-expressions,
    frame-to-frame inconsistencies characteristic of deepfakes).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.gelu = nn.GELU()

        self.downsample = None
        if downsample or stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.gelu(out)
        return out


class TemporalBranch3DCNN(nn.Module):
    """
    3D CNN temporal branch for video frame sequences.

    v2 Enhancement: Returns per-frame temporal features [B, T, temporal_dim]
    instead of collapsing to a single vector via global average pooling.

    Architecture:
        Input: [B, T, C, H, W] where T=num_frames, C=3, H=W=224
        → Rearrange to [B, C, T, H, W] for Conv3d
        → 4 residual 3D blocks with spatial-only downsampling
        → Adaptive spatial pooling (preserves temporal dimension)
        → Output: [B, T', temporal_dim] where T' = T (temporal preserved)

    Captures temporal inconsistencies in deepfake videos:
    - Blinking pattern anomalies
    - Lip movement synchronization artifacts
    - Micro-expression discontinuities
    - Frame-to-frame color/illumination jitter
    """

    def __init__(self, temporal_dim: int = 512):
        super().__init__()

        self.temporal_dim = temporal_dim

        # Stem: spatial downsampling only (stride=(1,2,2) preserves temporal)
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # Residual blocks: spatial downsampling via stride=(1,2,2)
        self.layer1 = SpatioTemporalBlock3D(64, 64, stride=1)
        self.layer2 = SpatioTemporalBlock3D(64, 128, stride=(1, 2, 2), downsample=True)
        self.layer3 = SpatioTemporalBlock3D(128, 256, stride=(1, 2, 2), downsample=True)
        self.layer4 = SpatioTemporalBlock3D(256, 512, stride=(1, 2, 2), downsample=True)

        # Global spatial pooling (preserves T); projects channel dim
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # [B, 512, T, 1, 1]
        self.frame_fc = nn.Linear(512, temporal_dim)

        # Collapse path for backward compatibility
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.collapse_fc = nn.Linear(512, temporal_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for conv layers, Xavier for FC."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        frames: torch.Tensor,
        return_frame_features: bool = True,
    ) -> torch.Tensor:
        """
        Extract temporal features from video frames.

        Args:
            frames: Video tensor [B, T, C, H, W] (T frames, 3 channels, 224x224)
            return_frame_features: If True, return per-frame features [B, T, d].
                                   If False, return collapsed features [B, d].

        Returns:
            If return_frame_features: [B, T, temporal_dim]
            Otherwise: [B, temporal_dim]
        """
        batch_size, num_frames = frames.shape[0], frames.shape[1]

        # Rearrange: [B, T, C, H, W] → [B, C, T, H, W]
        x = frames.permute(0, 2, 1, 3, 4)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 512, T', H', W'] — T' ≈ T since temporal stride=1

        if return_frame_features:
            # Adaptive spatial pool preserving temporal: [B, 512, T', 1, 1]
            pooled = self.spatial_pool(x)
            pooled = pooled.squeeze(-1).squeeze(-1)  # [B, 512, T']
            pooled = pooled.permute(0, 2, 1)  # [B, T', 512]

            # Project each frame
            frame_features = self.frame_fc(pooled)  # [B, T', temporal_dim]

            # Interpolate back to original T if needed
            if frame_features.shape[1] != num_frames:
                frame_features = frame_features.permute(0, 2, 1)  # [B, d, T']
                frame_features = F.interpolate(
                    frame_features, size=num_frames, mode="linear", align_corners=False
                )
                frame_features = frame_features.permute(0, 2, 1)  # [B, T, d]

            return frame_features
        else:
            # Collapse path: global average pool → single vector
            pooled = self.global_pool(x)  # [B, 512, 1, 1, 1]
            pooled = pooled.flatten(1)  # [B, 512]
            return self.collapse_fc(pooled)  # [B, temporal_dim]


class VisualFeatureEncoder(nn.Module):
    """
    Visual feature encoder combining spatial (ViT) and temporal (3D-CNN) branches.

    v2 Enhancement: Returns frame-level features [B, T, d_fused] for the
    UMFT temporal self-attention pipeline, rather than collapsing to [B, d_fused].

    Spatial branch: Pretrained ViT-Base-Patch16-224 extracts per-frame
    semantic features. The [CLS] token (768-d) captures global image content
    and manipulation artifacts.

    Temporal branch: 3D CNN captures spatiotemporal patterns across frame
    sequences, detecting temporal inconsistencies.

    Output modes:
        return_frame_features=True  → [B, T, d_fused]  (for UMFT pipeline)
        return_frame_features=False → [B, d_fused]       (backward compatible)
    """

    def __init__(
        self,
        config: Optional[SharedProjectionConfig] = None,
        pretrained_vit: bool = True,
        freeze_vit_layers: int = 8,
    ):
        """
        Initialize visual feature encoder.

        Args:
            config: Shared projection configuration
            pretrained_vit: Whether to load pretrained ViT weights
            freeze_vit_layers: Number of ViT layers to freeze (0-12)
        """
        super().__init__()

        self.config = config or SharedProjectionConfig()

        # Spatial branch: ViT-Base-Patch16-224
        self.vit = self._build_vit(pretrained_vit, freeze_vit_layers)

        # Temporal branch: 3D CNN (enhanced with frame-level output)
        self.temporal_branch = TemporalBranch3DCNN(
            temporal_dim=self.config.d_model_visual_temporal
        )

        # Projection heads
        self.spatial_projection = ModalityProjection(
            d_input=self.config.d_model_visual,
            d_output=self.config.d_fused,
            dropout_rate=self.config.dropout_rate,
        )
        self.temporal_projection = ModalityProjection(
            d_input=self.config.d_model_visual_temporal,
            d_output=self.config.d_fused,
            dropout_rate=self.config.dropout_rate,
        )

        # Fusion gate: learnable weighting of spatial vs temporal
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.config.d_fused * 2, self.config.d_fused),
            nn.GELU(),
            nn.Linear(self.config.d_fused, 1),
            nn.Sigmoid(),
        )
        self._init_fusion_gate()

        self.output_norm = nn.LayerNorm(self.config.d_fused)

    def _init_fusion_gate(self):
        """Initialize fusion gate to 0.5 (equal spatial/temporal weight)."""
        for module in self.fusion_gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_vit(self, pretrained: bool, freeze_layers: int) -> nn.Module:
        """
        Build Vision Transformer from HuggingFace transformers.

        Args:
            pretrained: Load pretrained weights from google/vit-base-patch16-224
            freeze_layers: Number of encoder layers to freeze (0-12)

        Returns:
            ViT model with classification head removed
        """
        from transformers import ViTModel

        model_name = "google/vit-base-patch16-224"
        if pretrained:
            vit = ViTModel.from_pretrained(model_name)
        else:
            from transformers import ViTConfig
            config = ViTConfig(
                hidden_size=self.config.d_model_visual,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=224,
                patch_size=16,
            )
            vit = ViTModel(config)

        # Freeze early layers to preserve pretrained features
        if freeze_layers > 0:
            for i, layer in enumerate(vit.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        return vit

    def _extract_per_frame_spatial(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame spatial features via ViT.

        Args:
            frames: [B, T, C, H, W]

        Returns:
            Per-frame features [B, T, d_fused]
        """
        batch_size, num_frames = frames.shape[0], frames.shape[1]

        # Flatten batch and time: [B*T, C, H, W]
        flat_frames = frames.reshape(batch_size * num_frames, *frames.shape[2:])

        # ViT forward on all frames
        vit_output = self.vit(pixel_values=flat_frames)
        cls_features = vit_output.last_hidden_state[:, 0, :]  # [B*T, 768]

        # Project to fused dim
        projected = self.spatial_projection(cls_features)  # [B*T, 512]

        # Reshape back: [B, T, 512]
        return projected.reshape(batch_size, num_frames, -1)

    def forward(
        self,
        frames: torch.Tensor,
        return_spatial_only: bool = False,
        return_frame_features: bool = True,
    ) -> torch.Tensor:
        """
        Extract visual features from video frames or images.

        Args:
            frames: Either [B, T, C, H, W] for video or [B, C, H, W] for image
            return_spatial_only: If True, skip temporal branch (for single images)
            return_frame_features: If True, return [B, T, d_fused] for UMFT.
                                   If False, return [B, d_fused] (backward compat).

        Returns:
            If image or return_spatial_only: [B, d_fused]
            If video + return_frame_features: [B, T, d_fused]
            If video + not return_frame_features: [B, d_fused]
        """
        is_image = frames.ndim == 4

        if is_image:
            # Single image: [B, C, H, W] → ViT → [B, d_fused]
            vit_output = self.vit(pixel_values=frames)
            spatial_features = vit_output.last_hidden_state[:, 0, :]  # [B, 768]
            spatial_projected = self.spatial_projection(spatial_features)  # [B, 512]
            return self.output_norm(spatial_projected)

        # === Video path: [B, T, C, H, W] ===

        if return_spatial_only:
            # Middle frame only, collapsed
            mid_idx = frames.shape[1] // 2
            spatial_input = frames[:, mid_idx, :, :, :]
            vit_output = self.vit(pixel_values=spatial_input)
            spatial_features = vit_output.last_hidden_state[:, 0, :]
            spatial_projected = self.spatial_projection(spatial_features)
            return self.output_norm(spatial_projected)

        if return_frame_features:
            # === UMFT path: per-frame features [B, T, d_fused] ===
            # Per-frame spatial features
            spatial_per_frame = self._extract_per_frame_spatial(frames)  # [B, T, 512]

            # Per-frame temporal features from 3D CNN
            temporal_per_frame = self.temporal_branch(frames, return_frame_features=True)  # [B, T, 512]
            temporal_projected = self.temporal_projection(
                temporal_per_frame.reshape(-1, temporal_per_frame.shape[-1])
            ).reshape(temporal_per_frame.shape[0], temporal_per_frame.shape[1], -1)  # [B, T, 512]

            # Learnable fusion gate per frame
            concat = torch.cat([spatial_per_frame, temporal_projected], dim=-1)  # [B, T, 1024]
            gate = self.fusion_gate(concat)  # [B, T, 1]
            combined = gate * spatial_per_frame + (1.0 - gate) * temporal_projected  # [B, T, 512]

            return self.output_norm(combined)
        else:
            # === Backward-compatible collapsed path: [B, d_fused] ===
            mid_idx = frames.shape[1] // 2
            spatial_input = frames[:, mid_idx, :, :, :]
            vit_output = self.vit(pixel_values=spatial_input)
            spatial_features = vit_output.last_hidden_state[:, 0, :]
            spatial_projected = self.spatial_projection(spatial_features)

            temporal_features = self.temporal_branch(frames, return_frame_features=False)
            temporal_projected = self.temporal_projection(temporal_features)

            combined = (spatial_projected + temporal_projected) / 2.0
            return self.output_norm(combined)

    def get_feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.config.d_fused
