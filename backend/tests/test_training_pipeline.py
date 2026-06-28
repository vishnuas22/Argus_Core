"""
Argus Core v2 - Training Pipeline Tests
==========================================
Tests for the training infrastructure components.

Validates:
    - Loss function computation (focal, contrastive, lip-sync, multi-task)
    - Single training step (forward + backward + optimizer step)
    - Loss decrease on synthetic data over multiple steps
    - Metric computation (AUC, EER, ECE)
"""

import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.loss_functions import (
    BinaryFocalLoss,
    AudioVisualContrastiveLoss,
    LipSyncLoss,
    MultiTaskLoss,
)
from training.evaluation import compute_metrics


class TestBinaryFocalLoss:
    """Test focal loss for classification."""

    def test_output_shape(self):
        """Test loss output is scalar."""
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()

        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # Scalar

    def test_positive_loss(self):
        """Loss should always be positive."""
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(16, 1)
        targets = torch.randint(0, 2, (16, 1)).float()

        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        """Perfect predictions should give very low focal loss."""
        loss_fn = BinaryFocalLoss(gamma=2.0)

        # "Perfect" predictions: high logit for fake, low for real
        logits = torch.tensor([[10.0], [-10.0], [10.0], [-10.0]])
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01, f"Perfect predictions should give near-zero focal loss, got {loss.item()}"

    def test_gamma_reduces_easy_example_weight(self):
        """Higher gamma should reduce weight of easy examples."""
        logits = torch.tensor([[5.0], [-5.0], [0.1]])  # Two easy, one hard
        targets = torch.tensor([[1.0], [0.0], [1.0]])

        loss_gamma0 = BinaryFocalLoss(gamma=0.0)(logits, targets)
        loss_gamma2 = BinaryFocalLoss(gamma=2.0)(logits, targets)

        # Focal loss with gamma>0 should be lower (downweights easy examples)
        assert loss_gamma2.item() < loss_gamma0.item()

    def test_1d_targets(self):
        """Test with 1D target tensor."""
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(4, 1)
        targets = torch.randint(0, 2, (4,)).float()

        loss = loss_fn(logits, targets)
        assert loss.ndim == 0


class TestContrastiveLoss:
    """Test audio-visual contrastive loss."""

    def test_output_shape(self):
        """Loss should be scalar."""
        loss_fn = AudioVisualContrastiveLoss()
        v = torch.randn(8, 64)
        a = torch.randn(8, 64)

        loss = loss_fn(v, a)
        assert loss.ndim == 0

    def test_positive_loss(self):
        """Loss should be positive."""
        loss_fn = AudioVisualContrastiveLoss()
        v = torch.randn(8, 64)
        a = torch.randn(8, 64)

        loss = loss_fn(v, a)
        assert loss.item() > 0

    def test_identical_features_low_loss(self):
        """Identical features should have lower loss than random."""
        loss_fn = AudioVisualContrastiveLoss(temperature=0.07)

        features = torch.randn(8, 64)
        loss_identical = loss_fn(features, features)

        random_a = torch.randn(8, 64)
        loss_random = loss_fn(features, random_a)

        assert loss_identical.item() < loss_random.item()


class TestLipSyncLoss:
    """Test lip-sync supervision loss."""

    def test_output_shape(self):
        """Loss should be scalar."""
        loss_fn = LipSyncLoss()
        overall = torch.sigmoid(torch.randn(4, 1))
        per_frame = torch.sigmoid(torch.randn(4, 8))
        targets = torch.randint(0, 2, (4,)).float()

        loss = loss_fn(overall, per_frame, targets)
        assert loss.ndim == 0

    def test_without_per_frame(self):
        """Should work without per-frame scores."""
        loss_fn = LipSyncLoss()
        overall = torch.sigmoid(torch.randn(4, 1))
        targets = torch.randint(0, 2, (4,)).float()

        loss = loss_fn(overall, None, targets)
        assert loss.ndim == 0


class TestMultiTaskLoss:
    """Test multi-task loss with learnable weights."""

    def test_output_shape(self):
        """Loss should be scalar."""
        multi_loss = MultiTaskLoss(num_tasks=3)
        losses = {
            "classification": torch.tensor(0.5),
            "contrastive": torch.tensor(0.3),
            "lip_sync": torch.tensor(0.2),
        }

        total = multi_loss(losses)
        assert total.ndim == 0

    def test_weights_learnable(self):
        """Task weights should be learnable parameters."""
        multi_loss = MultiTaskLoss(num_tasks=3)
        assert multi_loss.log_vars.requires_grad

    def test_missing_task(self):
        """Should handle missing task losses gracefully."""
        multi_loss = MultiTaskLoss(num_tasks=3)
        losses = {"classification": torch.tensor(0.5)}

        total = multi_loss(losses)
        assert total.ndim == 0

    def test_get_weights(self):
        """Should return current task weights."""
        multi_loss = MultiTaskLoss(num_tasks=3)
        weights = multi_loss.get_weights()

        assert len(weights) == 3
        for name, weight in weights.items():
            assert weight > 0


class TestMetrics:
    """Test evaluation metric computation."""

    def test_perfect_predictions(self):
        """Perfect predictions should give AUC=1, accuracy=1."""
        probs = torch.tensor([0.9, 0.95, 0.1, 0.05])
        labels = torch.tensor([1, 1, 0, 0])

        metrics = compute_metrics(probs, labels)

        assert metrics["accuracy"] == 1.0
        assert metrics["auc_roc"] >= 0.99
        assert metrics["f1"] == 1.0
        assert metrics["ece"] < 0.2

    def test_random_predictions(self):
        """Random predictions should give AUC ≈ 0.5."""
        torch.manual_seed(42)
        probs = torch.rand(1000)
        labels = torch.randint(0, 2, (1000,))

        metrics = compute_metrics(probs, labels)

        # AUC should be near 0.5 for random
        assert 0.3 < metrics["auc_roc"] < 0.7

    def test_metric_keys(self):
        """All expected metrics should be present."""
        probs = torch.rand(100)
        labels = torch.randint(0, 2, (100,))

        metrics = compute_metrics(probs, labels)

        expected_keys = ["accuracy", "precision", "recall", "f1",
                         "specificity", "auc_roc", "eer", "ece"]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"


class TestTrainingStep:
    """Test that a single training step works end-to-end."""

    def test_single_training_step(self):
        """Forward + backward + optimizer step should work."""
        from core.cross_attention_fusion import CrossModalCrossAttentionFusion, UMFTConfig

        config = UMFTConfig(
            d_fused=32,
            pretrained_encoders=False,
            freeze_vit_layers=0,
            freeze_wav2vec2_feature_extractor=False,
            freeze_roberta_layers=0,
            num_cross_attn_heads=2,
            cross_attn_d_ff=64,
            num_temporal_layers=1,
            num_temporal_heads=2,
            temporal_d_ff=64,
            classifier_d_hidden_1=16,
            classifier_d_hidden_2=8,
        )
        model = CrossModalCrossAttentionFusion(config=config)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = BinaryFocalLoss()

        # Synthetic batch
        frames = torch.randn(2, 4, 3, 224, 224)
        labels = torch.tensor([[1.0], [0.0]])

        # Forward
        output = model(frames=frames)
        loss = loss_fn(output.logit, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_loss_decreases(self):
        """Loss should decrease over multiple training steps on synthetic data."""
        from core.cross_attention_fusion import CrossModalCrossAttentionFusion, UMFTConfig

        config = UMFTConfig(
            d_fused=32,
            pretrained_encoders=False,
            freeze_vit_layers=0,
            freeze_wav2vec2_feature_extractor=False,
            freeze_roberta_layers=0,
            num_cross_attn_heads=2,
            cross_attn_d_ff=64,
            num_temporal_layers=1,
            num_temporal_heads=2,
            temporal_d_ff=64,
            classifier_d_hidden_1=16,
            classifier_d_hidden_2=8,
        )
        model = CrossModalCrossAttentionFusion(config=config)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = BinaryFocalLoss(gamma=0.0)  # Standard BCE for clearer signal

        # Fixed synthetic data (overfit to it)
        torch.manual_seed(42)
        frames = torch.randn(4, 4, 3, 224, 224)
        labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        losses = []
        for step in range(10):
            output = model(frames=frames)
            loss = loss_fn(output.logit, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally trend downward
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
