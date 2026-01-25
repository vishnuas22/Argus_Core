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
"""

from core.engine import (
    InferenceEngine,
    InferenceResult,
    BatchInferenceResult,
    get_inference_engine,
    initialize_inference_engine
)

from core.explain import (
    ExplainabilityEngine,
    Region,
    HeatmapResult,
    ManipulationType,
    get_explainability_engine
)

from core.fusion import (
    MultiModalFusion,
    FusionConfig,
    AttentionWeightComputer,
    UncertaintyEstimator,
    get_multi_modal_fusion
)

from core.scorer import (
    TrustScorer,
    ScoringConfig,
    VerdictThresholds,
    PlattParams,
    get_trust_scorer
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
    
    # Fusion
    "MultiModalFusion",
    "FusionConfig",
    "AttentionWeightComputer",
    "UncertaintyEstimator",
    "get_multi_modal_fusion",
    
    # Scorer
    "TrustScorer",
    "ScoringConfig",
    "VerdictThresholds",
    "PlattParams",
    "get_trust_scorer",
]
