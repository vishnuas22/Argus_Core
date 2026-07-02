"""
Argus Core v2 - Classification Head
=====================================
Final classification MLP for the UMFT pipeline.

v2 Architecture:
    Accepts concatenated features from:
    - 6 cross-modal attention pairs (each d_model)
    - Temporal self-attention output (d_model)
    - Lip-sync consistency score (1)
    
    Total input: 6*d_model + d_model + 1 = 3585 (with d_model=512)

    Network:
    z_concat [B, d_input]
    → Linear(d_input, d_hidden_1) → GELU → Dropout(0.3)
    → Linear(d_hidden_1, d_hidden_2) → GELU → Dropout(0.2)
    → Linear(d_hidden_2, 1) → Sigmoid (or raw logit)
    → fake_probability [B, 1]

Includes learnable temperature scaling for post-hoc calibration.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class TemperatureScaling(nn.Module):
    """
    Learnable temperature scaling for probability calibration.

    After training, the model's output probabilities may be
    poorly calibrated (over-confident or under-confident).
    Temperature scaling adjusts the logit by a learned scalar:

        P(fake) = σ(logit / T)

    where T is a learnable parameter initialized to 1.0.

    This is the simplest and most effective post-hoc calibration
    method (Guo et al., 2017).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature.clamp(min=0.01)


class DeepfakeClassificationHead(nn.Module):
    """
    MLP classification head for deepfake detection.

    v2 Enhancement: Flexible input dimension to accept concatenated
    features from multiple fusion sources. Includes learnable
    temperature scaling for calibrated probability output.

    The Sigmoid output produces a continuous probability [0, 1]:
    - Values near 1.0 indicate high fake probability
    - Values near 0.0 indicate high authentic probability
    - Values near 0.5 indicate uncertainty

    No threshold is applied — downstream components (TrustScorer)
    handle verdict determination through calibrated thresholds.
    """

    def __init__(
        self,
        d_input: int = 512,
        d_hidden_1: int = 256,
        d_hidden_2: int = 128,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2,
        use_temperature_scaling: bool = True,
    ):
        """
        Initialize classification head.

        Args:
            d_input: Input feature dimension from fusion layer.
                     For UMFT: 6 * d_model + d_model + 1 = 3585
                     For v1 compat: 512
            d_hidden_1: First hidden layer dimension
            d_hidden_2: Second hidden layer dimension
            dropout_1: Dropout rate after first hidden layer
            dropout_2: Dropout rate after second hidden layer
            use_temperature_scaling: Use learnable temperature scaling
        """
        super().__init__()

        self.d_input = d_input
        self.use_temperature_scaling = use_temperature_scaling

        # Input projection (adaptive to handle different input dims)
        self.input_norm = nn.LayerNorm(d_input)

        # MLP classification network
        self.network = nn.Sequential(
            nn.Linear(d_input, d_hidden_1),
            nn.GELU(),
            nn.Dropout(dropout_1),
            nn.Linear(d_hidden_1, d_hidden_2),
            nn.GELU(),
            nn.Dropout(dropout_2),
            nn.Linear(d_hidden_2, 1),
        )

        # Temperature scaling for calibration
        if use_temperature_scaling:
            self.temp_scaling = TemperatureScaling()
        else:
            self.temp_scaling = None

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier uniform initialization for all linear layers."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        fused_features: torch.Tensor,
        return_logit: bool = False,
    ) -> torch.Tensor:
        """
        Classify fused features as fake or authentic.

        Args:
            fused_features: Fused cross-modal representation [B, d_input]
            return_logit: If True, return raw logit instead of probability

        Returns:
            If return_logit: Raw logit [B, 1]
            Otherwise: Fake probability [B, 1] in range [0, 1]
        """
        x = self.input_norm(fused_features)
        logit = self.network(x)  # [B, 1]

        if self.temp_scaling is not None:
            logit = self.temp_scaling(logit)

        if return_logit:
            return logit

        return self.sigmoid(logit)

    def forward_with_features(
        self,
        fused_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify and return intermediate features for explainability.

        Args:
            fused_features: Fused cross-modal representation [B, d_input]

        Returns:
            (probability, penultimate_features)
            probability: [B, 1] fake probability
            penultimate_features: [B, d_hidden_2] for gradient-based XAI
        """
        x = self.input_norm(fused_features)

        # Run through network, extracting penultimate features
        for i, module in enumerate(self.network):
            x = module(x)
            if i == 4:  # After second GELU (index 4: Linear→GELU→Drop→Linear→GELU)
                penultimate = x.clone()

        logit = x  # After final Linear
        if self.temp_scaling is not None:
            logit = self.temp_scaling(logit)

        probability = self.sigmoid(logit)

        return probability, penultimate
