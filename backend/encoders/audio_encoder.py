"""
Argus Core v2 - Audio Feature Encoder
=======================================
Mel-Spectrogram + SpecRNet + Wav2Vec2 audio feature extractor.

UMFT Architecture:
    Spectral branch: Mel-spectrogram → SpecRNet-style residual CNN.
    Captures spectral artifacts, vocoder harmonics, frequency phase
    shifts characteristic of synthetic speech.

    SSL branch: Wav2Vec2-base-960h pretrained encoder extracts
    self-supervised speech representations capturing phonetic content,
    speaker characteristics, and prosodic patterns.

Output modes:
    - Frame-aligned: [B, T, d_fused] synchronized to video frame rate
    - Collapsed:     [B, d_fused] for backward compatibility

Both branches project to d_fused=512 for cross-attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from encoders.projection_heads import ModalityProjection, SharedProjectionConfig


class MelSpectrogramExtractor(nn.Module):
    """
    Learnable mel-spectrogram extraction layer.

    Converts raw waveform to log-mel spectrogram representation
    suitable for CNN-based spectral analysis.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        # Mel filterbank (fixed, not learnable)
        mel_basis = self._create_mel_filterbank(sample_rate, n_fft, n_mels)
        self.register_buffer("mel_basis", mel_basis)

        # Hann window
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def _create_mel_filterbank(
        self, sample_rate: int, n_fft: int, n_mels: int
    ) -> torch.Tensor:
        """Create mel-scale filterbank matrix."""
        f_min = 0.0
        f_max = sample_rate / 2.0

        mel_min = 2595.0 * torch.log10(torch.tensor(1.0 + f_min / 700.0))
        mel_max = 2595.0 * torch.log10(torch.tensor(1.0 + f_max / 700.0))

        mels = torch.linspace(mel_min.item(), mel_max.item(), n_mels + 2)
        hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        bins = torch.floor((n_fft + 1) * hz / sample_rate).long()

        fbank = torch.zeros(n_mels, n_fft // 2 + 1)
        for m in range(1, n_mels + 1):
            f_left = bins[m - 1].item()
            f_center = bins[m].item()
            f_right = bins[m + 1].item()

            for k in range(f_left, f_center):
                if f_center != f_left:
                    fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right != f_center:
                    fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

        return fbank

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram.

        Args:
            waveform: Raw audio [B, num_samples]

        Returns:
            Log-mel spectrogram [B, n_mels, time_frames]
        """
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )

        power_spec = stft.abs() ** 2
        mel_spec = torch.matmul(self.mel_basis, power_spec)
        log_mel = torch.log1p(mel_spec)

        return log_mel


class SpectralResidualBlock(nn.Module):
    """
    SpecRNet-style residual block for spectral feature extraction.

    Applies two Conv2D layers with BatchNorm and GELU activation,
    with a residual connection. Designed for efficient processing
    of mel-spectrogram representations.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
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


class AttentionPooling2D(nn.Module):
    """
    Attention-based pooling for spectral feature maps.

    Learns to weight spatial locations based on relevance
    to deepfake detection instead of uniform pooling.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps [B, C, H, W]

        Returns:
            Attention-weighted pooled features [B, C]
        """
        attn_weights = self.attention(x)  # [B, 1, H, W]
        attn_weights = torch.softmax(attn_weights.flatten(2), dim=-1)  # [B, 1, H*W]
        attn_weights = attn_weights.view_as(attn_weights)

        weighted = x * attn_weights
        pooled = weighted.sum(dim=(2, 3))  # [B, C]
        return pooled


class SpectralBranch(nn.Module):
    """
    SpecRNet-style spectral analysis branch.

    Architecture:
        Input: [B, 1, n_mels, time_frames]
        → 4 residual blocks with progressive channel increase
        → Attention pooling
        → Output: [B, spectral_dim=512]
    """

    def __init__(self, n_mels: int = 80, spectral_dim: int = 512):
        super().__init__()

        self.mel_extractor = MelSpectrogramExtractor(n_mels=n_mels)

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.block1 = SpectralResidualBlock(32, 64, stride=2)
        self.block2 = SpectralResidualBlock(64, 128, stride=2)
        self.block3 = SpectralResidualBlock(128, 256, stride=2)
        self.block4 = SpectralResidualBlock(256, 512, stride=2)

        self.attention_pool = AttentionPooling2D(512)
        self.fc = nn.Linear(512, spectral_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract spectral features from raw waveform.

        Args:
            waveform: Raw audio [B, num_samples]

        Returns:
            Spectral feature vector [B, spectral_dim]
        """
        mel = self.mel_extractor(waveform)  # [B, n_mels, time_frames]
        mel = mel.unsqueeze(1)  # [B, 1, n_mels, time_frames]

        x = self.stem(mel)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.attention_pool(x)  # [B, 512]
        x = self.fc(x)
        return x


class AudioFeatureEncoder(nn.Module):
    """
    Audio feature encoder combining spectral (SpecRNet) and SSL (Wav2Vec2) branches.

    v2 Enhancement: Returns frame-aligned features [B, T, d_fused] synchronized
    to video frame rate for the UMFT temporal cross-attention pipeline.

    Spectral branch: Mel-spectrogram + SpecRNet-style residual CNN extracts
    spectral artifacts and vocoder harmonics.

    SSL branch: Wav2Vec2-base-960h extracts self-supervised speech
    representations capturing phonetic content and speaker identity.

    Output modes:
        align_to_frames(num_frames) → [B, T, d_fused]  (UMFT pipeline)
        forward()                   → [B, d_fused]      (backward compatible)
    """

    def __init__(
        self,
        config: Optional[SharedProjectionConfig] = None,
        pretrained_wav2vec2: bool = True,
        freeze_wav2vec2_feature_extractor: bool = True,
    ):
        """
        Initialize audio feature encoder.

        Args:
            config: Shared projection configuration
            pretrained_wav2vec2: Load pretrained Wav2Vec2 weights
            freeze_wav2vec2_feature_extractor: Freeze CNN feature extractor
        """
        super().__init__()

        self.config = config or SharedProjectionConfig()

        # Spectral branch
        self.spectral_branch = SpectralBranch(
            n_mels=80,
            spectral_dim=self.config.d_model_audio_spectral,
        )

        # SSL branch: Wav2Vec2
        self.wav2vec2 = self._build_wav2vec2(
            pretrained_wav2vec2, freeze_wav2vec2_feature_extractor
        )

        # Projection heads
        self.spectral_projection = ModalityProjection(
            d_input=self.config.d_model_audio_spectral,
            d_output=self.config.d_fused,
            dropout_rate=self.config.dropout_rate,
        )
        self.ssl_projection = ModalityProjection(
            d_input=self.config.d_model_audio_ssl,
            d_output=self.config.d_fused,
            dropout_rate=self.config.dropout_rate,
        )

        # Temporal projection: map Wav2Vec2 hidden states to per-frame features
        self.temporal_fc = nn.Linear(self.config.d_fused, self.config.d_fused)
        nn.init.xavier_uniform_(self.temporal_fc.weight)
        nn.init.zeros_(self.temporal_fc.bias)

        self.output_norm = nn.LayerNorm(self.config.d_fused)

    def _build_wav2vec2(self, pretrained: bool, freeze_feature_extractor: bool) -> nn.Module:
        """
        Build Wav2Vec2 model from HuggingFace transformers.

        Args:
            pretrained: Load pretrained weights
            freeze_feature_extractor: Freeze the CNN feature extractor

        Returns:
            Wav2Vec2 model
        """
        from transformers import Wav2Vec2Model

        model_name = "facebook/wav2vec2-base-960h"
        if pretrained:
            model = Wav2Vec2Model.from_pretrained(model_name)
        else:
            from transformers import Wav2Vec2Config
            wav2vec2_config = Wav2Vec2Config(
                hidden_size=self.config.d_model_audio_ssl,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            model = Wav2Vec2Model(wav2vec2_config)

        if freeze_feature_extractor and hasattr(model, "feature_extractor"):
            for param in model.feature_extractor.parameters():
                param.requires_grad = False

        return model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features from raw waveform (collapsed).

        Args:
            waveform: Raw audio tensor [B, num_samples] (16kHz)

        Returns:
            Audio feature vector [B, d_fused]
        """
        # Spectral branch
        spectral_features = self.spectral_branch(waveform)  # [B, 512]
        spectral_projected = self.spectral_projection(spectral_features)

        # SSL branch: mean-pooled
        wav2vec2_output = self.wav2vec2(waveform)
        ssl_features = wav2vec2_output.last_hidden_state.mean(dim=1)  # [B, 768]
        ssl_projected = self.ssl_projection(ssl_features)

        combined = (spectral_projected + ssl_projected) / 2.0
        return self.output_norm(combined)

    def forward_temporal(
        self,
        waveform: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Extract frame-aligned audio features for UMFT temporal pipeline.

        Uses Wav2Vec2 hidden states (which are temporal) and interpolates
        them to match the video frame count for synchronized cross-modal
        attention.

        Args:
            waveform: Raw audio tensor [B, num_samples] (16kHz)
            num_frames: Number of video frames to align to

        Returns:
            Frame-aligned audio features [B, T, d_fused]
        """
        batch_size = waveform.shape[0]

        # Get Wav2Vec2 temporal hidden states: [B, T_audio, 768]
        wav2vec2_output = self.wav2vec2(waveform)
        ssl_temporal = wav2vec2_output.last_hidden_state  # [B, T_audio, 768]

        # Project each time step: [B, T_audio, d_fused]
        ssl_projected = self.ssl_projection(
            ssl_temporal.reshape(-1, ssl_temporal.shape[-1])
        ).reshape(batch_size, ssl_temporal.shape[1], -1)

        # Interpolate to match video frame count: [B, T, d_fused]
        # Transpose to [B, d_fused, T_audio] for F.interpolate
        ssl_projected_t = ssl_projected.permute(0, 2, 1)
        aligned = F.interpolate(
            ssl_projected_t, size=num_frames, mode="linear", align_corners=False
        )
        aligned = aligned.permute(0, 2, 1)  # [B, T, d_fused]

        # Add spectral features as a global bias to each frame
        spectral_features = self.spectral_branch(waveform)  # [B, 512]
        spectral_projected = self.spectral_projection(spectral_features)  # [B, d_fused]
        spectral_bias = spectral_projected.unsqueeze(1)  # [B, 1, d_fused]

        # Gated combination: temporal SSL + spectral bias
        combined = aligned + 0.3 * spectral_bias  # Learnable scaling via temporal_fc
        combined = self.temporal_fc(combined)

        return self.output_norm(combined)

    def get_feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.config.d_fused
