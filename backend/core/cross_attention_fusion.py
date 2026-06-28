"""
Argus Core v2 - Unified Multimodal Forensic Transformer (UMFT)
================================================================
End-to-end multimodal deepfake detection model.

This is the CENTRAL MODULE of the Argus v2 architecture. It replaces
the disconnected ONNX/PyTorch paths with a single, unified, end-to-end
trainable pipeline:

    Raw Media → Encoders → Cross-Modal Attention → Temporal Self-Attention
    → Lip-Sync Consistency → Classification → P(fake)

Grounded in SOTA research:
    - CAMME (Aug 2024): Pairwise bi-modal cross-attention fusion
    - AVPL (AAAI 2025): Multi-task audio-visual prompt learning
    - Emotion-Aware (2026): Text-guided semantic anchor for A-V fusion

Modality Handling:
    The model gracefully handles missing modalities. If only video is
    available (no audio/text), only the relevant cross-attention pairs
    and branches are activated. This enables the same model for:
    - Image-only analysis (social media images)
    - Video-only analysis (no audio track)
    - Video + Audio analysis (most video deepfakes)
    - Full multimodal analysis (video + audio + transcript)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from encoders.projection_heads import SharedProjectionConfig
from encoders.visual_encoder import VisualFeatureEncoder
from encoders.audio_encoder import AudioFeatureEncoder
from encoders.text_encoder import TextFeatureEncoder
from fusion_layers.cross_attention import CrossModalAttentionBlock
from fusion_layers.temporal_attention import TemporalSelfAttention
from fusion_layers.lip_sync_module import LipSyncConsistencyModule
from fusion_layers.classification_head import DeepfakeClassificationHead


@dataclass
class UMFTConfig:
    """Configuration for the Unified Multimodal Forensic Transformer."""

    # Encoder settings
    d_fused: int = 512
    pretrained_encoders: bool = True
    freeze_vit_layers: int = 8
    freeze_wav2vec2_feature_extractor: bool = True
    freeze_roberta_layers: int = 6

    # Cross-attention settings
    num_cross_attn_heads: int = 8
    cross_attn_d_ff: int = 2048
    cross_attn_dropout: float = 0.1

    # Temporal attention settings
    num_temporal_layers: int = 3
    num_temporal_heads: int = 8
    temporal_d_ff: int = 2048
    temporal_dropout: float = 0.1
    max_frames: int = 256

    # Lip-sync settings
    lip_sync_kernel: int = 5
    lip_sync_hidden: int = 64

    # Classification head
    classifier_d_hidden_1: int = 256
    classifier_d_hidden_2: int = 128
    classifier_dropout_1: float = 0.3
    classifier_dropout_2: float = 0.2
    use_temperature_scaling: bool = True


# Backward compatibility alias
CrossAttentionConfig = UMFTConfig


@dataclass
class UMFTOutput:
    """Structured output from the UMFT forward pass."""
    # Primary output
    fake_probability: torch.Tensor  # [B, 1] — P(deepfake)

    # Per-modality contributions
    modality_scores: Dict[str, torch.Tensor]  # Per cross-attention pair scores

    # Lip-sync
    lip_sync_score: Optional[torch.Tensor] = None  # [B, 1]
    lip_sync_per_frame: Optional[torch.Tensor] = None  # [B, T]

    # Attention weights for explainability
    cross_attention_weights: Optional[Dict[str, torch.Tensor]] = None
    temporal_attention_weights: Optional[torch.Tensor] = None

    # Raw logit (for loss computation)
    logit: Optional[torch.Tensor] = None


class CrossModalCrossAttentionFusion(nn.Module):
    """
    Unified Multimodal Forensic Transformer (UMFT).

    End-to-end trainable model for multimodal deepfake detection.

    Pipeline:
        1. Encoders extract features from each modality
           - Visual: ViT + 3D-CNN → [B, T, 512]
           - Audio: Wav2Vec2 + SpecRNet → [B, T, 512]
           - Text: RoBERTa → [B, S, 512]

        2. Pairwise cross-modal attention (up to 6 pairs)
           discovers inconsistencies between modalities

        3. Temporal self-attention models frame-to-frame
           consistency patterns

        4. Lip-sync module computes audio-visual synchrony

        5. Classification head produces calibrated P(fake)

    Handles any combination of modalities gracefully.
    """

    def __init__(self, config: Optional[UMFTConfig] = None):
        """
        Initialize the UMFT model.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        super().__init__()

        self.config = config or UMFTConfig()
        self.d_fused = self.config.d_fused

        proj_config = SharedProjectionConfig(d_fused=self.d_fused)

        # === Encoders ===
        self.visual_encoder = VisualFeatureEncoder(
            config=proj_config,
            pretrained_vit=self.config.pretrained_encoders,
            freeze_vit_layers=self.config.freeze_vit_layers,
        )
        self.audio_encoder = AudioFeatureEncoder(
            config=proj_config,
            pretrained_wav2vec2=self.config.pretrained_encoders,
            freeze_wav2vec2_feature_extractor=self.config.freeze_wav2vec2_feature_extractor,
        )
        self.text_encoder = TextFeatureEncoder(
            config=proj_config,
            pretrained_roberta=self.config.pretrained_encoders,
            freeze_roberta_layers=self.config.freeze_roberta_layers,
        )

        # === Cross-Modal Attention Blocks (6 pairs) ===
        cross_attn_args = dict(
            d_model=self.d_fused,
            num_heads=self.config.num_cross_attn_heads,
            d_ff=self.config.cross_attn_d_ff,
            dropout_rate=self.config.cross_attn_dropout,
        )
        self.cross_attn_v_to_a = CrossModalAttentionBlock(**cross_attn_args)
        self.cross_attn_v_to_t = CrossModalAttentionBlock(**cross_attn_args)
        self.cross_attn_a_to_v = CrossModalAttentionBlock(**cross_attn_args)
        self.cross_attn_a_to_t = CrossModalAttentionBlock(**cross_attn_args)
        self.cross_attn_t_to_v = CrossModalAttentionBlock(**cross_attn_args)
        self.cross_attn_t_to_a = CrossModalAttentionBlock(**cross_attn_args)

        # === Temporal Self-Attention ===
        self.temporal_attention = TemporalSelfAttention(
            d_model=self.d_fused,
            num_heads=self.config.num_temporal_heads,
            num_layers=self.config.num_temporal_layers,
            d_ff=self.config.temporal_d_ff,
            dropout=self.config.temporal_dropout,
            max_frames=self.config.max_frames,
        )

        # === Lip-Sync Consistency Module ===
        self.lip_sync_module = LipSyncConsistencyModule(
            d_model=self.d_fused,
            consistency_kernel=self.config.lip_sync_kernel,
            d_consistency_hidden=self.config.lip_sync_hidden,
        )

        # === Classification Head ===
        # Input dim depends on available modalities:
        # Max: 6 cross-attention pairs * d_fused + temporal d_fused + lip_sync 1
        # = 6*512 + 512 + 1 = 3585
        self._max_classifier_input = 6 * self.d_fused + self.d_fused + 1

        self.classifier = DeepfakeClassificationHead(
            d_input=self._max_classifier_input,
            d_hidden_1=self.config.classifier_d_hidden_1,
            d_hidden_2=self.config.classifier_d_hidden_2,
            dropout_1=self.config.classifier_dropout_1,
            dropout_2=self.config.classifier_dropout_2,
            use_temperature_scaling=self.config.use_temperature_scaling,
        )

        # Adaptive input projection for variable modality combinations
        # Maps any subset of features to the max classifier input dim
        self._modality_projection = nn.Linear(self.d_fused, self.d_fused)
        nn.init.xavier_uniform_(self._modality_projection.weight)
        nn.init.zeros_(self._modality_projection.bias)

    def _encode_visual(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Encode visual input to frame-level features [B, T, d_fused]."""
        return self.visual_encoder(frames, return_frame_features=True)

    def _encode_audio(
        self,
        waveform: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Encode audio input to frame-aligned features [B, T, d_fused]."""
        return self.audio_encoder.forward_temporal(waveform, num_frames)

    def _encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text input to segment features [B, S, d_fused]."""
        return self.text_encoder.forward_segments(input_ids, attention_mask)

    def _pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Mean pool a sequence [B, T, d_model] → [B, d_model]."""
        return x.mean(dim=1)

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> UMFTOutput:
        """
        Unified forward pass for multimodal deepfake detection.

        Accepts any combination of modalities and produces a calibrated
        deepfake probability with per-modality contribution scores.

        Args:
            frames: Video [B, T, C, H, W] or Image [B, C, H, W]
            waveform: Audio [B, num_samples] at 16kHz
            input_ids: Text token IDs [B, seq_len]
            attention_mask: Text attention mask [B, seq_len]
            return_attention_weights: Return attention maps for XAI

        Returns:
            UMFTOutput with fake_probability, modality_scores,
            lip_sync_score, and optional attention weights.
        """
        device = self._get_device()
        has_visual = frames is not None
        has_audio = waveform is not None
        has_text = input_ids is not None
        is_image = has_visual and frames.ndim == 4

        # Determine batch size from first available modality
        batch_size = self._get_batch_size(frames, waveform, input_ids)

        # === Step 1: Encode each modality ===
        z_visual = None  # [B, T, d_fused] or [B, 1, d_fused] for images
        z_audio = None   # [B, T, d_fused]
        z_text = None    # [B, S, d_fused]
        num_frames = 1

        if has_visual:
            if is_image:
                # Image: [B, C, H, W] → [B, 1, d_fused]
                z_visual = self.visual_encoder(frames).unsqueeze(1)
                num_frames = 1
            else:
                # Video: [B, T, C, H, W] → [B, T, d_fused]
                z_visual = self._encode_visual(frames)
                num_frames = frames.shape[1]

        if has_audio:
            z_audio = self._encode_audio(waveform, num_frames)

        if has_text:
            z_text = self._encode_text(input_ids, attention_mask)

        # === Step 2: Pairwise Cross-Modal Attention ===
        cross_attended = {}
        cross_attention_weights = {}

        if has_visual and has_audio:
            cross_attended["v_to_a"], w = self.cross_attn_v_to_a(
                z_visual, z_audio, return_attention_weights
            )
            if w is not None:
                cross_attention_weights["v_to_a"] = w

            cross_attended["a_to_v"], w = self.cross_attn_a_to_v(
                z_audio, z_visual, return_attention_weights
            )
            if w is not None:
                cross_attention_weights["a_to_v"] = w

        if has_visual and has_text:
            cross_attended["v_to_t"], w = self.cross_attn_v_to_t(
                z_visual, z_text, return_attention_weights
            )
            if w is not None:
                cross_attention_weights["v_to_t"] = w

            cross_attended["t_to_v"], w = self.cross_attn_t_to_v(
                z_text, z_visual, return_attention_weights
            )
            if w is not None:
                cross_attention_weights["t_to_v"] = w

        if has_audio and has_text:
            cross_attended["a_to_t"], w = self.cross_attn_a_to_t(
                z_audio, z_text, return_attention_weights
            )
            if w is not None:
                cross_attention_weights["a_to_t"] = w

            cross_attended["t_to_a"], w = self.cross_attn_t_to_a(
                z_text, z_audio, return_attention_weights
            )
            if w is not None:
                cross_attention_weights["t_to_a"] = w

        # === Step 3: Pool cross-attended features ===
        pooled_cross_features = []
        modality_scores = {}

        for key, features in cross_attended.items():
            pooled = self._pool_sequence(features)  # [B, d_fused]
            pooled_cross_features.append(pooled)

            # Per-pair contribution (magnitude-based heuristic before training)
            modality_scores[key] = pooled.norm(dim=-1, keepdim=True)

        # === Step 4: Temporal Self-Attention ===
        temporal_features = None
        temporal_attn_weights = None

        if cross_attended:
            # Concatenate all cross-attended sequences along temporal dim
            # and run temporal self-attention
            all_temporal = []
            for features in cross_attended.values():
                if features.ndim == 2:
                    all_temporal.append(features.unsqueeze(1))
                else:
                    all_temporal.append(features)

            temporal_input = torch.cat(all_temporal, dim=1)  # [B, T_total, d_fused]
            temporal_features, temporal_attn_weights = self.temporal_attention(
                temporal_input,
                return_attention=return_attention_weights,
            )
        elif has_visual:
            # Single modality fallback: temporal attention on visual
            temporal_features, temporal_attn_weights = self.temporal_attention(
                z_visual,
                return_attention=return_attention_weights,
            )
        else:
            # No temporal features available (single image, audio-only, or text-only)
            if has_audio:
                temporal_features = self._pool_sequence(z_audio)
            elif has_text:
                temporal_features = self._pool_sequence(z_text)
            else:
                temporal_features = torch.zeros(batch_size, self.d_fused, device=device)

        # === Step 5: Lip-Sync Consistency ===
        lip_sync_score = None
        lip_sync_per_frame = None

        if has_visual and has_audio and not is_image:
            lip_sync_score, lip_sync_per_frame = self.lip_sync_module(
                z_visual, z_audio, return_per_frame=True
            )
        else:
            lip_sync_score = torch.full((batch_size, 1), 0.5, device=device)

        # === Step 6: Assemble Classifier Input ===
        # Pad cross-attended features to always have 6 slots
        while len(pooled_cross_features) < 6:
            pooled_cross_features.append(
                torch.zeros(batch_size, self.d_fused, device=device)
            )

        classifier_input = torch.cat(
            pooled_cross_features + [temporal_features, lip_sync_score],
            dim=-1,
        )  # [B, 6*d_fused + d_fused + 1]

        # === Step 7: Classification ===
        fake_probability = self.classifier(classifier_input)

        # Also get raw logit for loss computation
        logit = self.classifier(classifier_input, return_logit=True)

        return UMFTOutput(
            fake_probability=fake_probability,
            modality_scores=modality_scores,
            lip_sync_score=lip_sync_score,
            lip_sync_per_frame=lip_sync_per_frame,
            cross_attention_weights=cross_attention_weights if return_attention_weights else None,
            temporal_attention_weights=temporal_attn_weights,
            logit=logit,
        )

    def _get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def _get_batch_size(
        self,
        frames: Optional[torch.Tensor],
        waveform: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
    ) -> int:
        """Get batch size from first available input."""
        if frames is not None:
            return frames.shape[0]
        if waveform is not None:
            return waveform.shape[0]
        if input_ids is not None:
            return input_ids.shape[0]
        raise ValueError("At least one modality input must be provided")

    @classmethod
    def from_config(cls, config: UMFTConfig) -> "CrossModalCrossAttentionFusion":
        """Create model from config."""
        return cls(config=config)

    def save_pretrained(self, save_path: str):
        """
        Save model weights and config.

        Args:
            save_path: Directory to save to
        """
        import os
        import json

        os.makedirs(save_path, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))

        # Save config
        from dataclasses import asdict
        config_dict = asdict(self.config)
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_path: str, map_location: str = "cpu") -> "CrossModalCrossAttentionFusion":
        """
        Load model from saved weights and config.

        Args:
            load_path: Directory containing model.pt and config.json
            map_location: Device to load weights to

        Returns:
            Loaded model
        """
        import os
        import json

        # Load config
        config_path = os.path.join(load_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = UMFTConfig(**config_dict)

        # Create model with non-pretrained encoders (weights loaded from checkpoint)
        config.pretrained_encoders = False
        model = cls(config=config)

        # Load weights
        weights_path = os.path.join(load_path, "model.pt")
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)

        return model

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count per component."""
        counts = {}
        for name, module in [
            ("visual_encoder", self.visual_encoder),
            ("audio_encoder", self.audio_encoder),
            ("text_encoder", self.text_encoder),
            ("cross_attention", nn.ModuleList([
                self.cross_attn_v_to_a, self.cross_attn_v_to_t,
                self.cross_attn_a_to_v, self.cross_attn_a_to_t,
                self.cross_attn_t_to_v, self.cross_attn_t_to_a,
            ])),
            ("temporal_attention", self.temporal_attention),
            ("lip_sync_module", self.lip_sync_module),
            ("classifier", self.classifier),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}

        counts["total"] = {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        return counts
