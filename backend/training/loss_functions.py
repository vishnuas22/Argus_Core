"""
Argus Core v2 - Loss Functions
================================
Task-specific loss functions for UMFT training.

Multi-task training objectives:
    1. Binary Focal Loss:       Classification (handles class imbalance)
    2. AV Contrastive Loss:     Cross-modal alignment learning
    3. Lip-Sync Loss:           Audio-visual synchrony supervision
    4. Multi-Task Loss:         Weighted combination with learnable weights

All losses accept raw logits (pre-sigmoid) for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss for deepfake detection.

    Focal loss addresses class imbalance by down-weighting
    easy examples and focusing on hard ones:

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        - p_t = p if y=1, else (1-p)
        - α balances positive/negative classes
        - γ focuses on hard examples (γ=0 → standard BCE)

    Deepfake datasets are often imbalanced (more real than fake
    or vice versa), making focal loss essential.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weight for positive class (fake). Default 0.75.
            gamma: Focusing parameter. Higher = more focus on hard examples.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model output [B, 1] (pre-sigmoid)
            targets: Binary labels [B, 1] or [B] (0=real, 1=fake)

        Returns:
            Focal loss scalar
        """
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        # BCE with logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Compute probabilities
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AudioVisualContrastiveLoss(nn.Module):
    """
    Contrastive loss for audio-visual alignment learning.

    Encourages matched A-V pairs to be close in embedding space
    and mismatched pairs to be far apart. This teaches the model
    to detect A-V inconsistencies characteristic of deepfakes.

    Uses InfoNCE-style contrastive loss:
        L = -log(exp(sim(v_i, a_i)/τ) / Σ_j exp(sim(v_i, a_j)/τ))
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature scaling for similarity scores.
                         Lower = sharper distribution.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between visual and audio features.

        Args:
            visual_features: [B, d_model] mean-pooled visual features
            audio_features: [B, d_model] mean-pooled audio features

        Returns:
            Contrastive loss scalar
        """
        # L2 normalize
        v_norm = F.normalize(visual_features, dim=-1)
        a_norm = F.normalize(audio_features, dim=-1)

        # Compute similarity matrix: [B, B]
        sim_matrix = torch.matmul(v_norm, a_norm.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        batch_size = visual_features.shape[0]
        labels = torch.arange(batch_size, device=visual_features.device)

        # Cross-entropy loss (both directions)
        loss_v_to_a = F.cross_entropy(sim_matrix, labels)
        loss_a_to_v = F.cross_entropy(sim_matrix.T, labels)

        return (loss_v_to_a + loss_a_to_v) / 2.0


class LipSyncLoss(nn.Module):
    """
    Loss for supervising the lip-sync consistency module.

    Uses BCE loss on the overall sync score and per-frame scores:
    - Real videos → sync_score ≈ 1.0
    - Fake videos → sync_score ≈ 0.0
    - Time-shifted audio → sync_score ≈ 0.0
    """

    def __init__(self, per_frame_weight: float = 0.3):
        super().__init__()
        self.per_frame_weight = per_frame_weight

    def forward(
        self,
        overall_sync: torch.Tensor,
        per_frame_sync: Optional[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute lip-sync supervision loss.

        Args:
            overall_sync: [B, 1] overall sync probability
            per_frame_sync: [B, T] per-frame sync scores (optional)
            targets: [B] or [B, 1] — 0=fake (no sync), 1=real (sync)

        Returns:
            Lip-sync loss scalar
        """
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        # Invert labels: real=1 means good sync, fake=0 means bad sync
        sync_targets = 1.0 - targets  # Inverted: real videos SHOULD sync

        # Wait, actually: real videos should have HIGH sync (target=1)
        # Fake videos should have LOW sync (target=0)
        sync_targets = targets  # Real=1→high_sync, Fake=0→low_sync? No...
        # Labels: 0=real, 1=fake. For sync: real=high_sync, fake=low_sync
        sync_targets = 1.0 - targets  # real(0) → sync(1), fake(1) → sync(0)

        overall_loss = F.binary_cross_entropy(overall_sync, sync_targets)

        if per_frame_sync is not None:
            # Expand targets to match per-frame shape
            frame_targets = sync_targets.unsqueeze(-1).expand_as(
                per_frame_sync.unsqueeze(-1) if per_frame_sync.ndim == 1 else per_frame_sync
            ).squeeze(-1)

            # Clamp per_frame_sync to valid range
            per_frame_clamped = per_frame_sync.clamp(0.001, 0.999)
            frame_loss = F.binary_cross_entropy(per_frame_clamped, frame_targets)
            return overall_loss + self.per_frame_weight * frame_loss

        return overall_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable task weights.

    Combines classification, contrastive, and lip-sync losses
    using learned uncertainty-based weighting (Kendall et al., 2018):

        L = Σ_i (1 / (2σ_i²)) * L_i + log(σ_i)

    This automatically balances loss magnitudes across tasks.
    """

    def __init__(
        self,
        num_tasks: int = 3,
        task_names: Optional[list] = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or ["classification", "contrastive", "lip_sync"]

        # Learnable log-variance parameters (initialized to 0 → weight=1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted multi-task loss.

        Args:
            losses: Dict mapping task names to their individual losses

        Returns:
            Total weighted loss
        """
        total_loss = torch.tensor(0.0, device=self.log_vars.device)

        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue

            task_loss = losses[name]
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * task_loss + self.log_vars[i]

        return total_loss

    def get_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        weights = {}
        for i, name in enumerate(self.task_names):
            weights[name] = torch.exp(-self.log_vars[i]).item()
        return weights
