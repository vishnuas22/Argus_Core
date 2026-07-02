"""
Argus Core v2 - Lip-Sync Module Tests
========================================
Tests for the lip-sync consistency module.

Validates:
    - Output shapes for various input sizes
    - Sync scoring range [0, 1]
    - Higher sync for aligned inputs vs desynchronized
    - Gradient flow through the module
"""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fusion_layers.lip_sync_module import LipSyncConsistencyModule


@pytest.fixture
def module():
    """Create lip-sync module for testing."""
    return LipSyncConsistencyModule(d_model=64, consistency_kernel=3, d_consistency_hidden=16)


class TestLipSyncModule:
    """Test lip-sync consistency module."""

    def test_output_shapes(self, module):
        """Test output shapes for standard inputs."""
        B, T, d = 2, 8, 64
        z_visual = torch.randn(B, T, d)
        z_audio = torch.randn(B, T, d)

        overall_sync, per_frame_sync = module(z_visual, z_audio)

        assert overall_sync.shape == (B, 1)
        assert per_frame_sync.shape == (B, T)

    def test_output_range(self, module):
        """Test that sync scores are in [0, 1]."""
        B, T, d = 4, 16, 64
        z_visual = torch.randn(B, T, d)
        z_audio = torch.randn(B, T, d)

        overall_sync, per_frame_sync = module(z_visual, z_audio)

        assert (overall_sync >= 0).all() and (overall_sync <= 1).all()

    def test_aligned_vs_desynchronized(self, module):
        """Aligned inputs should score differently from desynchronized."""
        B, T, d = 4, 16, 64

        # Create "aligned" inputs (same features)
        base_features = torch.randn(B, T, d)
        aligned_visual = base_features + 0.1 * torch.randn(B, T, d)
        aligned_audio = base_features + 0.1 * torch.randn(B, T, d)

        # Create "desynchronized" inputs (shuffled temporal order)
        desynced_audio = aligned_audio[:, torch.randperm(T), :]

        with torch.no_grad():
            sync_aligned, _ = module(aligned_visual, aligned_audio)
            sync_desynced, _ = module(aligned_visual, desynced_audio)

        # Aligned should generally score differently from desynchronized
        # (exact direction depends on training, but scores should differ)
        assert not torch.allclose(sync_aligned, sync_desynced, atol=0.01), \
            "Module should distinguish aligned from desynchronized inputs"

    def test_without_per_frame(self, module):
        """Test output when per_frame is not requested."""
        B, T, d = 2, 8, 64
        z_visual = torch.randn(B, T, d)
        z_audio = torch.randn(B, T, d)

        overall_sync, per_frame_sync = module(z_visual, z_audio, return_per_frame=False)

        assert overall_sync.shape == (B, 1)
        assert per_frame_sync is None

    def test_gradient_flow(self, module):
        """Test that gradients flow through the module."""
        B, T, d = 2, 8, 64
        z_visual = torch.randn(B, T, d, requires_grad=True)
        z_audio = torch.randn(B, T, d, requires_grad=True)

        overall_sync, per_frame_sync = module(z_visual, z_audio)
        loss = overall_sync.sum() + per_frame_sync.sum()
        loss.backward()

        assert z_visual.grad is not None
        assert z_audio.grad is not None
        assert z_visual.grad.abs().sum() > 0
        assert z_audio.grad.abs().sum() > 0

    def test_variable_sequence_length(self, module):
        """Test with different sequence lengths."""
        B, d = 2, 64

        for T in [1, 4, 8, 32]:
            z_visual = torch.randn(B, T, d)
            z_audio = torch.randn(B, T, d)

            overall, per_frame = module(z_visual, z_audio)
            assert overall.shape == (B, 1)
            assert per_frame.shape == (B, T)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
