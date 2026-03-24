"""
Argus Core v2 - Lip-Sync Consistency Module
=============================================
Audio-visual synchrony scoring for deepfake detection.

Lip-sync inconsistency is the single strongest signal for
detecting deepfake videos. This module explicitly measures
the temporal alignment between:
    - Visual features (mouth/face region per frame)
    - Audio features (speech content per frame)

Architecture:
    Input: z_visual [B, T, d_model], z_audio [B, T, d_model]
    → Frame-level cosine similarity (baseline sync)
    → Learned similarity via bilinear attention
    → Temporal consistency scoring (smooth sync variations)
    → Output: sync_scores [B, T], overall_sync [B, 1]

Based on insights from:
    - LIPINC-V2 (2025): lip inconsistency detection
    - SyncNet: audio-visual sync in speech videos
    - AV-Deepfake1M: large-scale A-V deepfake dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BilinearSimilarity(nn.Module):
    """
    Learnable bilinear similarity between visual and audio features.

    Instead of raw cosine similarity, learns a transformation:
        sim(v, a) = v^T W a + bias

    where W is a learnable d×d matrix. This allows the model to
    learn which visual-audio feature dimensions should correspond.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.bilinear = nn.Bilinear(d_model, d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.bilinear.weight)
        nn.init.zeros_(self.bilinear.bias)

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute learnable similarity between visual and audio features.

        Args:
            visual: [B, T, d_model] or [B, d_model]
            audio: [B, T, d_model] or [B, d_model]

        Returns:
            Similarity scores [B, T, 1] or [B, 1]
        """
        return self.bilinear(visual, audio)


class TemporalConsistencyScorer(nn.Module):
    """
    Measures temporal consistency of sync scores.

    Real videos have smooth, consistent lip-sync alignment.
    Deepfakes often show:
        - Sudden drops in sync (splice points)
        - Periodic desynchronization
        - High-frequency sync oscillations

    Uses 1D convolution over the temporal sync scores to
    detect these patterns.
    """

    def __init__(self, kernel_size: int = 5, d_hidden: int = 64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, d_hidden, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Conv1d(d_hidden, 1, kernel_size=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, sync_scores: torch.Tensor) -> torch.Tensor:
        """
        Score temporal consistency of sync values.

        Args:
            sync_scores: [B, T] frame-level sync scores

        Returns:
            Consistency-adjusted scores [B, T]
        """
        # [B, T] → [B, 1, T] for Conv1d
        x = sync_scores.unsqueeze(1)
        adjusted = self.conv(x).squeeze(1)  # [B, T]
        return adjusted


class LipSyncConsistencyModule(nn.Module):
    """
    Complete lip-sync consistency analysis module.

    Combines multiple sync scoring mechanisms:
    1. **Cosine similarity**: Baseline frame-level A-V alignment
    2. **Bilinear similarity**: Learned feature-space alignment
    3. **Temporal consistency**: Detects sudden sync drops/patterns
    4. **Overall aggregation**: Produces single sync probability

    The module is designed to work with the UMFT pipeline,
    taking encoder outputs and producing both per-frame sync
    scores and an overall synchrony probability.

    Training supervision:
        - In-sync real videos: high sync scores
        - Time-shifted deepfakes: low sync scores
        - Lip-movement-replaced deepfakes: variable sync
    """

    def __init__(
        self,
        d_model: int = 512,
        consistency_kernel: int = 5,
        d_consistency_hidden: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize lip-sync consistency module.

        Args:
            d_model: Feature dimension of visual/audio encodings
            consistency_kernel: Kernel size for temporal consistency conv
            d_consistency_hidden: Hidden dim for consistency scorer
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model

        # Visual feature adapter (focus on mouth region features)
        self.visual_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Audio feature adapter (focus on speech content features)
        self.audio_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learned bilinear similarity
        self.bilinear_sim = BilinearSimilarity(d_model)

        # Temporal consistency scorer
        self.consistency_scorer = TemporalConsistencyScorer(
            kernel_size=consistency_kernel,
            d_hidden=d_consistency_hidden,
        )

        # Aggregation: combine cosine + bilinear + consistency
        self.aggregation = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Overall sync head: mean-pooled scores → single probability
        self.overall_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization."""
        for adapter in [self.visual_adapter, self.audio_adapter]:
            for module in adapter:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        for head in [self.aggregation, self.overall_head]:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        z_visual: torch.Tensor,
        z_audio: torch.Tensor,
        return_per_frame: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute lip-sync consistency scores.

        Args:
            z_visual: Visual features [B, T, d_model]
            z_audio: Audio features [B, T, d_model] (frame-aligned)
            return_per_frame: If True, return per-frame scores [B, T]

        Returns:
            (overall_sync, per_frame_sync)
            overall_sync: [B, 1] — probability of authentic lip-sync [0=fake, 1=real]
            per_frame_sync: [B, T] — per-frame sync scores (or None if not requested)
        """
        batch_size, num_frames = z_visual.shape[0], z_visual.shape[1]

        # Adapt features to lip-sync space
        v_adapted = self.visual_adapter(z_visual)  # [B, T, d]
        a_adapted = self.audio_adapter(z_audio)  # [B, T, d]

        # 1) Cosine similarity per frame
        v_norm = F.normalize(v_adapted, dim=-1)
        a_norm = F.normalize(a_adapted, dim=-1)
        cosine_sim = (v_norm * a_norm).sum(dim=-1)  # [B, T]

        # 2) Bilinear similarity per frame
        bilinear_sim = self.bilinear_sim(v_adapted, a_adapted).squeeze(-1)  # [B, T]

        # 3) Temporal consistency of cosine sim
        consistency_scores = self.consistency_scorer(cosine_sim)  # [B, T]

        # Aggregate per-frame scores: [B, T, 3]
        stacked = torch.stack([cosine_sim, bilinear_sim, consistency_scores], dim=-1)
        per_frame_sync = self.aggregation(stacked).squeeze(-1)  # [B, T]

        # Overall sync probability
        # Use mean visual-audio feature difference + mean sync score
        v_mean = v_adapted.mean(dim=1)  # [B, d]
        a_mean = a_adapted.mean(dim=1)  # [B, d]
        diff_features = v_mean - a_mean  # [B, d]
        mean_sync = per_frame_sync.mean(dim=1, keepdim=True)  # [B, 1]

        overall_input = torch.cat([diff_features, mean_sync], dim=-1)  # [B, d+1]
        overall_sync = self.overall_head(overall_input)  # [B, 1]

        if not return_per_frame:
            return overall_sync, None

        return overall_sync, per_frame_sync
