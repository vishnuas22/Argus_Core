"""
Argus Core - Adversarial Defense Stack (Iteration 2)
=====================================================
Training-free defenses for production deepfake detection on T4/A10.

Research grounding:
- Randomized Preprocessing Sanitizer (RPS): training-free defense that
  randomly selects one of K preprocessing transforms per inference.
  Defeats single-transform adaptive EOT attackers (Qiu et al., "Mitigating
  Adversarial Attacks on Deepfake Detection via Randomized Preprocessing",
  ACM Workshop 2025).
- XAI Adversarial Gate: flag-don't-classify paradigm — uses GradCAM/
  Eigen-CAM consistency to detect inputs whose explanation is unstable,
  a known signature of adversarial examples (Wang et al., "XAI-Based
  Adversarial Detection", arXiv 2024).
- Randomized Smoothing Lite (RS-lite): Cohen et al. 2019 lineage. We use
  n=64-128 forward passes (not the full n=10^5) as a soft signal, not a
  certified radius. Reduces to ~12% latency overhead at batch=1 on T4.

DUMB benchmark (arXiv 2601.05986, Jan 2026) shows PGD achieves 99.6%
white-box ASR on undefended deepfake detectors — this stack is the
minimum viable defense.

All defenses are strict-additive: setting config.enable_adversarial_defenses
to False restores pre-iteration behavior.
"""

from defenses.randomized_preprocessing import (
    RandomizedPreprocessingSanitizer,
    get_default_rps,
)
from defenses.adversarial_gate import AdversarialGate, get_default_gate
from defenses.randomized_smoothing_lite import (
    RandomizedSmoothingLite,
    get_default_rslite,
)
from defenses.certified_robustness import (
    BRONetWrapper,
    RandomizedSmoothingCertifier,
    CertificationResult,
    get_default_rs_certifier,
)

__all__ = [
    "RandomizedPreprocessingSanitizer", "get_default_rps",
    "AdversarialGate", "get_default_gate",
    "RandomizedSmoothingLite", "get_default_rslite",
    # Iteration 5: certified robustness
    "BRONetWrapper",
    "RandomizedSmoothingCertifier", "CertificationResult",
    "get_default_rs_certifier",
]
