"""
Argus Core - Model Watermarking & Fingerprinting (Iteration 5)
================================================================
IP protection for the Argus detector adapters.

Research grounding:
- Model watermarking (Uchida et al., "Embedding Watermarks into Deep
  Neural Networks", MVAw 2017): embed a secret key into the model
  weights via a regularizer during training. The watermark can be
  extracted later to prove ownership.
- Fingerprinting (Ajiro & Uchida, "Towards Model Fingerprinting for
  DNNs", MVAw 2024): compute a hash of the model's behavior on a set
  of probe inputs. Does NOT modify the model; detects model stealing
  by comparing fingerprints.
- Backdoor-based watermarking (Zhang et al., "Protecting Intellectual
  Property of Deep Neural Networks with Watermarking", ARES 2018):
  embed a trigger pattern that produces a specific output. More robust
  to fine-tuning attacks but requires retraining.

This module implements:
1. Weight-embedding watermark — embed + extract a secret key in the
   LoRA adapter weights.
2. Behavioral fingerprint — compute a hash of the model's outputs on
   a fixed set of probe inputs.
3. Watermark verification — check whether a suspect model contains
   the watermark.

Strict-compat: pure-additive. No changes to detector interface.
Operators opt in via config.enable_model_watermarking.
"""

from security.model_watermarking import (
    Watermarker,
    Fingerprinter,
    WatermarkResult,
    FingerprintResult,
    get_default_watermarker,
    get_default_fingerprinter,
)

__all__ = [
    "Watermarker", "Fingerprinter",
    "WatermarkResult", "FingerprintResult",
    "get_default_watermarker", "get_default_fingerprinter",
]
