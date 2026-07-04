"""
Argus Core - Core Business Logic Layer
======================================
Central processing components for deepfake detection.

Layer 4 of the Argus Core architecture.

Modules:
- engine: Model inference engine with VRAM management
- explain: GradCAM++ and textual explanation generation
- fusion: Multi-modal result aggregation with attention weighting
- scorer: Trust Score computation with Platt calibration

Usage:
    from core import get_inference_engine, get_multi_modal_fusion, get_trust_scorer

    engine = get_inference_engine()
    fusion = get_multi_modal_fusion()
    scorer = get_trust_scorer()

NOTE ON IMPORT ROBUSTNESS
-------------------------
Previous versions of this file imported every submodule eagerly. That made
``import core`` fail on CPU-only hosts where ``torch`` is intentionally not
installed, because ``core.fusion`` and ``core.cross_attention_fusion`` both
``import torch`` at module load time. As a result, even pure-Python modules
like ``core.engine``, ``core.scorer`` and ``core.explain`` became
un-importable, which broke unit tests, CLI tools, and the FastAPI app
bootstrap on CPU-only deployments.

The fix is to import each submodule defensively: if a submodule's optional
ML dependency is missing, we skip the import and continue. Callers that
need the missing functionality will get a clear ImportError at the point
of use, not at ``import core`` time.
"""

# ---- Always-available (pure Python) --------------------------------------
from core.engine import (
    InferenceEngine,
    InferenceResult,
    BatchInferenceResult,
    get_inference_engine,
    initialize_inference_engine,
)

from core.explain import (
    ExplainabilityEngine,
    Region,
    HeatmapResult,
    ManipulationType,
    get_explainability_engine,
)

from core.scorer import (
    TrustScorer,
    ScoringConfig,
    VerdictThresholds,
    PlattParams,
    get_trust_scorer,
)

# ---- Optional (require torch) -------------------------------------------
# These submodules import torch at module-load time. On CPU-only hosts
# without torch installed, importing them would raise ModuleNotFoundError
# and break the entire `core` package. Wrap each in try/except so the
# rest of the package remains usable.
try:
    from core.fusion import (
        MultiModalFusion,
        FusionConfig,
        UncertaintyEstimator,
        get_multi_modal_fusion,
    )
except ModuleNotFoundError as _e:
    import warnings as _warnings
    _warnings.warn(
        f"core.fusion unavailable (missing optional dependency): {_e}. "
        f"Multi-modal fusion features will be disabled.",
        ImportWarning,
        stacklevel=2,
    )

try:
    from core.cross_attention_fusion import (
        CrossModalCrossAttentionFusion,
        UMFTConfig as CrossAttentionConfig,
    )
except ModuleNotFoundError as _e:
    import warnings as _warnings
    _warnings.warn(
        f"core.cross_attention_fusion unavailable (missing optional "
        f"dependency): {_e}. Cross-attention neural fusion will be disabled.",
        ImportWarning,
        stacklevel=2,
    )


__all__ = [
    # Engine
    "InferenceEngine",
    "InferenceResult",
    "BatchInferenceResult",
    "get_inference_engine",
    "initialize_inference_engine",

    # Explain
    "ExplainabilityEngine",
    "Region",
    "HeatmapResult",
    "ManipulationType",
    "get_explainability_engine",

    # Fusion (may be missing on CPU-only hosts)
    "MultiModalFusion",
    "FusionConfig",
    "UncertaintyEstimator",
    "get_multi_modal_fusion",
    "CrossModalCrossAttentionFusion",
    "CrossAttentionConfig",

    # Scorer
    "TrustScorer",
    "ScoringConfig",
    "VerdictThresholds",
    "PlattParams",
    "get_trust_scorer",
]
