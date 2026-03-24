"""
Argus Core v2 - UMFT Forward Pass Tests
==========================================
Validates the complete end-to-end forward pass of the
Unified Multimodal Forensic Transformer.

Tests all modality combinations:
    - Visual only (video)
    - Visual only (image)
    - Audio only
    - Text only
    - Visual + Audio
    - Visual + Text
    - Audio + Text
    - Full multimodal (Visual + Audio + Text)

Validates:
    - Output shapes
    - Probability range [0, 1]
    - Gradient flow through all components
    - Cross-attention weight dimensions
    - Lip-sync score availability for A-V inputs
"""

import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.cross_attention_fusion import CrossModalCrossAttentionFusion, UMFTConfig, UMFTOutput


@pytest.fixture
def config():
    """Lightweight config for fast testing (no pretrained weights)."""
    return UMFTConfig(
        d_fused=64,  # Small dims for CPU testing
        pretrained_encoders=False,
        freeze_vit_layers=0,
        freeze_wav2vec2_feature_extractor=False,
        freeze_roberta_layers=0,
        num_cross_attn_heads=4,
        cross_attn_d_ff=128,
        num_temporal_layers=1,
        num_temporal_heads=4,
        temporal_d_ff=128,
        classifier_d_hidden_1=32,
        classifier_d_hidden_2=16,
        max_frames=32,
    )


@pytest.fixture
def model(config):
    """Create a lightweight UMFT model for testing."""
    model = CrossModalCrossAttentionFusion(config=config)
    model.eval()
    return model


class TestUMFTForwardPass:
    """Test UMFT forward pass across all modality combinations."""

    def test_visual_only_video(self, model, config):
        """Test with video input only [B, T, C, H, W]."""
        B, T = 2, 4
        frames = torch.randn(B, T, 3, 224, 224)

        output = model(frames=frames)

        assert isinstance(output, UMFTOutput)
        assert output.fake_probability.shape == (B, 1)
        assert (output.fake_probability >= 0).all() and (output.fake_probability <= 1).all()
        assert output.logit is not None
        assert output.logit.shape == (B, 1)

    def test_visual_only_image(self, model, config):
        """Test with single image input [B, C, H, W]."""
        B = 2
        frames = torch.randn(B, 3, 224, 224)

        output = model(frames=frames)

        assert output.fake_probability.shape == (B, 1)
        assert (output.fake_probability >= 0).all() and (output.fake_probability <= 1).all()
        # No lip-sync for images (should be default 0.5)
        assert output.lip_sync_score is not None

    def test_audio_only(self, model, config):
        """Test with audio input only [B, num_samples]."""
        B = 2
        waveform = torch.randn(B, 16000 * 3)  # 3 seconds at 16kHz

        output = model(waveform=waveform)

        assert output.fake_probability.shape == (B, 1)
        assert (output.fake_probability >= 0).all() and (output.fake_probability <= 1).all()

    def test_text_only(self, model, config):
        """Test with text input only [B, seq_len]."""
        B = 2
        input_ids = torch.randint(0, 50265, (B, 64))
        attention_mask = torch.ones(B, 64, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert output.fake_probability.shape == (B, 1)
        assert (output.fake_probability >= 0).all() and (output.fake_probability <= 1).all()

    def test_visual_audio(self, model, config):
        """Test with video + audio (most common deepfake detection case)."""
        B, T = 2, 4
        frames = torch.randn(B, T, 3, 224, 224)
        waveform = torch.randn(B, 16000 * 3)

        output = model(frames=frames, waveform=waveform)

        assert output.fake_probability.shape == (B, 1)
        # Lip-sync should be computed for V+A video
        assert output.lip_sync_score is not None
        assert output.lip_sync_score.shape == (B, 1)
        assert output.lip_sync_per_frame is not None
        assert output.lip_sync_per_frame.shape[0] == B

        # Should have V→A and A→V cross-attention pairs
        assert "v_to_a" in output.modality_scores
        assert "a_to_v" in output.modality_scores

    def test_visual_text(self, model, config):
        """Test with video + text."""
        B, T = 2, 4
        frames = torch.randn(B, T, 3, 224, 224)
        input_ids = torch.randint(0, 50265, (B, 64))
        attention_mask = torch.ones(B, 64, dtype=torch.long)

        output = model(frames=frames, input_ids=input_ids, attention_mask=attention_mask)

        assert output.fake_probability.shape == (B, 1)
        assert "v_to_t" in output.modality_scores
        assert "t_to_v" in output.modality_scores

    def test_audio_text(self, model, config):
        """Test with audio + text."""
        B = 2
        waveform = torch.randn(B, 16000 * 3)
        input_ids = torch.randint(0, 50265, (B, 64))
        attention_mask = torch.ones(B, 64, dtype=torch.long)

        output = model(waveform=waveform, input_ids=input_ids, attention_mask=attention_mask)

        assert output.fake_probability.shape == (B, 1)
        assert "a_to_t" in output.modality_scores
        assert "t_to_a" in output.modality_scores

    def test_full_multimodal(self, model, config):
        """Test with all three modalities (video + audio + text)."""
        B, T = 2, 4
        frames = torch.randn(B, T, 3, 224, 224)
        waveform = torch.randn(B, 16000 * 3)
        input_ids = torch.randint(0, 50265, (B, 64))
        attention_mask = torch.ones(B, 64, dtype=torch.long)

        output = model(
            frames=frames,
            waveform=waveform,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert output.fake_probability.shape == (B, 1)

        # All 6 cross-attention pairs should be present
        expected_pairs = ["v_to_a", "v_to_t", "a_to_v", "a_to_t", "t_to_v", "t_to_a"]
        for pair in expected_pairs:
            assert pair in output.modality_scores, f"Missing cross-attention pair: {pair}"

        # Lip-sync should be computed
        assert output.lip_sync_score is not None

    def test_attention_weights_returned(self, model, config):
        """Test that attention weights are returned when requested."""
        B, T = 2, 4
        frames = torch.randn(B, T, 3, 224, 224)
        waveform = torch.randn(B, 16000 * 3)

        output = model(frames=frames, waveform=waveform, return_attention_weights=True)

        assert output.cross_attention_weights is not None
        assert len(output.cross_attention_weights) > 0


class TestGradientFlow:
    """Verify gradient flows through all model components."""

    def test_gradient_flow_visual(self, model, config):
        """Verify gradients flow back through visual encoder."""
        model.train()
        frames = torch.randn(2, 4, 3, 224, 224, requires_grad=True)

        output = model(frames=frames)
        loss = output.fake_probability.sum()
        loss.backward()

        # Check gradients exist for trainable encoder parameters
        has_grad = False
        for name, param in model.visual_encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grad = True
                    break

        assert has_grad, "No gradients flowing through visual encoder"

    def test_gradient_flow_full(self, model, config):
        """Verify gradients flow through full multimodal pipeline."""
        model.train()
        frames = torch.randn(2, 4, 3, 224, 224)
        waveform = torch.randn(2, 16000 * 3)
        input_ids = torch.randint(0, 50265, (2, 64))

        output = model(frames=frames, waveform=waveform, input_ids=input_ids)
        loss = output.fake_probability.sum()
        loss.backward()

        # Check classifier head has gradients
        for name, param in model.classifier.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for classifier param: {name}"

    def test_logit_used_for_loss(self, model, config):
        """Verify logit output can be used with BCE loss."""
        model.train()
        frames = torch.randn(2, 4, 3, 224, 224)
        labels = torch.tensor([[1.0], [0.0]])

        output = model(frames=frames)
        loss = nn.functional.binary_cross_entropy_with_logits(output.logit, labels)
        loss.backward()

        assert loss.item() > 0


class TestModelPersistence:
    """Test model saving and loading."""

    def test_save_and_load(self, model, config, tmp_path):
        """Test save_pretrained and from_pretrained."""
        save_dir = str(tmp_path / "test_model")
        model.save_pretrained(save_dir)

        # Verify files exist
        assert os.path.exists(os.path.join(save_dir, "model.pt"))
        assert os.path.exists(os.path.join(save_dir, "config.json"))

        # Load model
        loaded = CrossModalCrossAttentionFusion.from_pretrained(save_dir)

        # Verify outputs match
        frames = torch.randn(1, 4, 3, 224, 224)

        model.eval()
        loaded.eval()

        with torch.no_grad():
            orig_output = model(frames=frames)
            loaded_output = loaded(frames=frames)

        torch.testing.assert_close(
            orig_output.fake_probability,
            loaded_output.fake_probability,
            atol=1e-5, rtol=1e-5,
        )

    def test_parameter_count(self, model):
        """Test parameter count reporting."""
        counts = model.get_parameter_count()

        assert "total" in counts
        assert counts["total"]["total"] > 0
        assert counts["total"]["trainable"] > 0
        assert counts["total"]["trainable"] <= counts["total"]["total"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_input_raises_error(self, model):
        """Providing no inputs should raise ValueError."""
        with pytest.raises(ValueError, match="At least one modality"):
            model()

    def test_batch_size_one(self, model):
        """Test with batch size 1."""
        frames = torch.randn(1, 4, 3, 224, 224)
        output = model(frames=frames)
        assert output.fake_probability.shape == (1, 1)

    def test_single_frame_video(self, model):
        """Test with T=1 (single frame as video)."""
        frames = torch.randn(2, 1, 3, 224, 224)
        output = model(frames=frames)
        assert output.fake_probability.shape == (2, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
