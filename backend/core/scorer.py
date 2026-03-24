"""
Argus Core - Trust Score Computation
====================================
Computes calibrated Trust Score and determines verdict.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - core/scorer.py

SOTA Algorithms:
- Bayesian Score Calibration: Platt scaling for well-calibrated probabilities
- Dynamic Thresholding: Content-type aware thresholds from config

Score Ranges (configurable):
- 80-100: Authentic
- 60-79: Likely Authentic
- 40-59: Uncertain (flag for human review)
- 20-39: Likely Fake
- 0-19: Fake

Integration:
- Imports: schemas/internal.py, config.py
- Inputs: AggregatedResult
- Outputs: TrustScore, Verdict
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from config import config
from schemas.schemas import (
    Verdict, ContentType, TrustScore, AggregatedResult,
    ModalityResult, Modality
)
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PlattParams:
    """
    Platt scaling parameters for probability calibration.
    
    Platt scaling fits a logistic regression on top of model outputs
    to produce well-calibrated probabilities.
    
    P(y=1|f) = 1 / (1 + exp(A*f + B))
    
    Where f is the raw score, A and B are learned parameters.
    """
    a: float = -2.0  # Default: steeper sigmoid
    b: float = 0.0   # Default: centered at 0.5
    
    def transform(self, score: float) -> float:
        """Apply Platt transformation. Uses |a| to ensure proper sharpening."""
        a_abs = abs(self.a)
        return 1.0 / (1.0 + np.exp(-a_abs * (score - 0.5) + self.b))


# Default Platt parameters per content type (pre-calibrated)
# NOTE: Platt calibration with negative 'a' inverts the sigmoid and flattens
# the signal toward 0.5, making the system indecisive. Set a=1.0 to enable
# normal sharpening sigmoid, or set use_platt_calibration=False to disable.
DEFAULT_PLATT_PARAMS = {
    ContentType.VIDEO_WITH_SPEECH: PlattParams(a=1.0, b=0.0),
    ContentType.VIDEO_NO_SPEECH: PlattParams(a=1.0, b=0.0),
    ContentType.AUDIO_ONLY: PlattParams(a=1.0, b=0.0),
    ContentType.IMAGE_ONLY: PlattParams(a=1.0, b=0.0),
    ContentType.TEXT_ONLY: PlattParams(a=1.0, b=0.0),
}


@dataclass
class VerdictThresholds:
    """
    Verdict determination thresholds.
    
    Trust Score ranges for each verdict level.
    """
    authentic: int = 80
    likely_authentic: int = 60
    uncertain: int = 40
    likely_fake: int = 20
    # Below likely_fake = fake
    
    @classmethod
    def from_config(cls) -> "VerdictThresholds":
        """Create from global config."""
        return cls(
            authentic=config.verdict_threshold_authentic,
            likely_authentic=config.verdict_threshold_likely_authentic,
            uncertain=config.verdict_threshold_uncertain,
            likely_fake=config.verdict_threshold_likely_fake
        )
    
    def get_verdict(self, score: float) -> Verdict:
        """
        Get verdict from score.
        
        Args:
            score: Trust score (0-100)
            
        Returns:
            Verdict enum
        """
        if score >= self.authentic:
            return Verdict.AUTHENTIC
        elif score >= self.likely_authentic:
            return Verdict.LIKELY_AUTHENTIC
        elif score >= self.uncertain:
            return Verdict.UNCERTAIN
        elif score >= self.likely_fake:
            return Verdict.LIKELY_FAKE
        else:
            return Verdict.FAKE


@dataclass
class ScoringConfig:
    """Configuration for scoring system."""
    # Score transformation
    use_platt_calibration: bool = False  # Disabled: Platt with negative 'a' flattens signal
    score_power: float = 1.0  # Power transform for score distribution
    
    # Uncertainty handling
    uncertainty_penalty: float = 0.1  # Reduce confidence when uncertain
    max_uncertainty_for_verdict: float = 0.4  # Above this, force "uncertain"
    
    # Content-type adjustments
    use_content_type_thresholds: bool = True
    
    # Score bounds
    min_score: float = 0.0
    max_score: float = 100.0


class TrustScorer:
    """
    Computes calibrated Trust Score and verdict.
    
    The Trust Score is a 0-100 value representing the probability
    that content is authentic (not manipulated).
    
    Score Interpretation:
    - 80-100: High confidence authentic
    - 60-79: Likely authentic, minor concerns
    - 40-59: Uncertain, requires human review
    - 20-39: Likely fake, significant concerns
    - 0-19: High confidence fake
    
    The scorer applies:
    1. Platt calibration for well-calibrated probabilities
    2. Content-type specific threshold adjustments
    3. Uncertainty integration into final score
    4. Verdict determination from thresholds
    
    Usage:
        scorer = TrustScorer()
        trust_score, verdict = scorer.compute(aggregated_result)
    """
    
    def __init__(
        self,
        scoring_config: Optional[ScoringConfig] = None,
        thresholds: Optional[VerdictThresholds] = None,
        platt_params: Optional[Dict[ContentType, PlattParams]] = None
    ):
        """
        Initialize trust scorer.
        
        Args:
            scoring_config: Scoring configuration
            thresholds: Verdict thresholds
            platt_params: Platt calibration parameters per content type
        """
        self.config = scoring_config or ScoringConfig()
        self.thresholds = thresholds or VerdictThresholds.from_config()
        self.platt_params = platt_params or DEFAULT_PLATT_PARAMS.copy()
        
        logger.info(
            f"TrustScorer initialized: platt={self.config.use_platt_calibration}, "
            f"thresholds=({self.thresholds.authentic}/{self.thresholds.likely_authentic}/"
            f"{self.thresholds.uncertain}/{self.thresholds.likely_fake})"
        )
    
    def compute(
        self,
        aggregated: AggregatedResult,
        content_type: Optional[ContentType] = None
    ) -> Tuple[TrustScore, Verdict]:
        """
        Compute final Trust Score and verdict.
        
        Applies calibration and thresholding to produce final results.
        
        Args:
            aggregated: Multi-modal fusion result
            content_type: Content type for calibration selection
            
        Returns:
            Tuple of (TrustScore, Verdict)
        """
        # Start with fused score (0-1 range, higher = more likely fake)
        raw_score = aggregated.fused_score
        uncertainty = aggregated.uncertainty
        
        # Invert: Trust Score is authenticity (100 = authentic, 0 = fake)
        # Raw score is manipulation probability
        authenticity_prob = 1.0 - raw_score
        calibration_applied = False
        
        # Apply Platt calibration if enabled
        if self.config.use_platt_calibration and content_type:
            authenticity_prob = self.calibrate_probability(
                authenticity_prob, content_type
            )
            calibration_applied = True
        
        # Apply power transform for score distribution
        if self.config.score_power != 1.0:
            authenticity_prob = np.power(authenticity_prob, self.config.score_power)
        
        # Convert to 0-100 scale
        score_value = authenticity_prob * 100.0
        
        # NOTE: Uncertainty penalty removed to avoid double-counting.
        # The image analyzer already reduces confidence when signals disagree.
        # Applying penalty again in the scorer over-penalizes disagreement.
        
        # Clamp to bounds
        score_value = float(np.clip(
            score_value,
            self.config.min_score,
            self.config.max_score
        ))
        
        # Compute confidence
        confidence = self._compute_confidence(aggregated, uncertainty)
        
        # Create TrustScore
        trust_score = TrustScore(
            value=round(score_value, 1),
            confidence=round(confidence, 3),
            calibrated=calibration_applied
        )
        
        # Determine verdict
        if uncertainty > self.config.max_uncertainty_for_verdict:
            # High uncertainty forces "uncertain" verdict
            verdict = Verdict.UNCERTAIN
        else:
            verdict = self.thresholds.get_verdict(score_value)
        
        logger.debug(
            f"Score computed: raw={raw_score:.3f}, trust={score_value:.1f}, "
            f"verdict={verdict.value}, confidence={confidence:.3f}"
        )
        
        return trust_score, verdict
    
    def calibrate_probability(
        self,
        raw_prob: float,
        content_type: ContentType
    ) -> float:
        """
        Apply Platt scaling for probability calibration.
        
        Platt scaling ensures that predicted probabilities match
        empirical frequencies (e.g., 70% means 70 out of 100 correct).
        
        Args:
            raw_prob: Raw probability (0-1)
            content_type: Content type for parameter selection
            
        Returns:
            Calibrated probability (0-1)
        """
        params = self.platt_params.get(content_type, PlattParams())
        
        # Transform using sigmoid with learned parameters
        calibrated = params.transform(raw_prob)
        
        return float(np.clip(calibrated, 0.001, 0.999))
    
    def _compute_confidence(
        self,
        aggregated: AggregatedResult,
        uncertainty: float
    ) -> float:
        """
        Compute confidence in the score.
        
        Confidence represents how reliable the score is, based on:
        - Number of modalities analyzed
        - Individual modality confidences
        - Score extremity (confident predictions get higher confidence)
        
        Note: Uncertainty is already factored into modality confidences,
        so we don't double-penalize here.
        
        Args:
            aggregated: Aggregated result
            uncertainty: Fusion uncertainty (used for context, not penalty)
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from modality results
        if aggregated.modality_results:
            modality_confs = [r.confidence for r in aggregated.modality_results]
            base_confidence = np.mean(modality_confs)
        else:
            base_confidence = 0.5
        
        # Adjust for number of modalities
        # Single modality should still allow high confidence if the model is certain
        num_modalities = len(aggregated.modality_results)
        if num_modalities >= 3:
            modality_factor = 1.0
        elif num_modalities == 2:
            modality_factor = 0.95
        else:
            # Single modality - allow up to 0.9 confidence
            modality_factor = 0.9
        
        # Score extremity factor - predictions near 0 or 1 are more confident
        score_extremity = abs(aggregated.fused_score - 0.5) * 2
        extremity_factor = 0.85 + 0.15 * score_extremity
        
        # Combine factors - don't apply uncertainty penalty again
        # as it's already in the modality confidence
        confidence = base_confidence * modality_factor * extremity_factor
        
        return float(np.clip(confidence, 0.1, 0.95))
    
    def compute_from_scores(
        self,
        modality_scores: Dict[Modality, float],
        modality_confidences: Optional[Dict[Modality, float]] = None,
        content_type: Optional[ContentType] = None
    ) -> Tuple[TrustScore, Verdict]:
        """
        Convenience method to compute score from raw modality scores.
        
        Creates ModalityResult objects and aggregates before scoring.
        
        Args:
            modality_scores: Dict of modality to manipulation score (0-1)
            modality_confidences: Optional confidence per modality
            content_type: Content type
            
        Returns:
            Tuple of (TrustScore, Verdict)
        """
        modality_confidences = modality_confidences or {}
        
        results = []
        for modality, score in modality_scores.items():
            confidence = modality_confidences.get(modality, 0.7)
            results.append(ModalityResult(
                modality=modality,
                score=score,
                confidence=confidence,
                details={}
            ))
        
        # Simple weighted aggregation
        if results:
            total_weight = sum(r.confidence for r in results)
            fused = sum(r.score * r.confidence for r in results) / total_weight
            uncertainty = np.std([r.score for r in results]) if len(results) > 1 else 0.5
        else:
            fused = 0.5
            uncertainty = 1.0
        
        aggregated = AggregatedResult(
            modality_results=results,
            fused_score=fused,
            uncertainty=uncertainty,
            weights_used={m.value: c / total_weight for m, c in modality_confidences.items()}
            if modality_confidences else {}
        )
        
        return self.compute(aggregated, content_type)
    
    def get_score_interpretation(
        self,
        trust_score: TrustScore
    ) -> Dict[str, Any]:
        """
        Get human-readable interpretation of score.
        
        Args:
            trust_score: Trust score to interpret
            
        Returns:
            Dict with interpretation details
        """
        score = trust_score.value
        
        # Determine band
        if score >= 80:
            band = "high_authentic"
            color = "green"
            summary = "Content appears authentic with high confidence"
            recommendation = "No further verification needed"
        elif score >= 60:
            band = "likely_authentic"
            color = "light_green"
            summary = "Content likely authentic with minor concerns"
            recommendation = "Consider spot-checking key elements"
        elif score >= 40:
            band = "uncertain"
            color = "yellow"
            summary = "Analysis inconclusive - content requires review"
            recommendation = "Human expert review recommended"
        elif score >= 20:
            band = "likely_fake"
            color = "orange"
            summary = "Content shows signs of manipulation"
            recommendation = "Detailed forensic review recommended"
        else:
            band = "fake"
            color = "red"
            summary = "Content appears manipulated with high confidence"
            recommendation = "Do not trust - investigate source"
        
        return {
            "score": score,
            "band": band,
            "color": color,
            "summary": summary,
            "recommendation": recommendation,
            "confidence": trust_score.confidence,
            "calibrated": trust_score.calibrated,
            "thresholds": {
                "authentic": self.thresholds.authentic,
                "likely_authentic": self.thresholds.likely_authentic,
                "uncertain": self.thresholds.uncertain,
                "likely_fake": self.thresholds.likely_fake
            }
        }
    
    def get_verdict_explanation(
        self,
        verdict: Verdict,
        trust_score: TrustScore
    ) -> str:
        """
        Generate explanation for verdict.
        
        Args:
            verdict: Determined verdict
            trust_score: Associated trust score
            
        Returns:
            Human-readable explanation string
        """
        score = trust_score.value
        conf_pct = trust_score.confidence * 100
        
        explanations = {
            Verdict.AUTHENTIC: (
                f"Content verified as authentic (Trust Score: {score:.0f}/100, "
                f"Confidence: {conf_pct:.0f}%). No manipulation indicators detected."
            ),
            Verdict.LIKELY_AUTHENTIC: (
                f"Content appears authentic (Trust Score: {score:.0f}/100, "
                f"Confidence: {conf_pct:.0f}%). Minor anomalies detected but likely "
                "due to compression or processing, not manipulation."
            ),
            Verdict.UNCERTAIN: (
                f"Analysis inconclusive (Trust Score: {score:.0f}/100, "
                f"Confidence: {conf_pct:.0f}%). Mixed signals detected. "
                "Human expert review is recommended before making a determination."
            ),
            Verdict.LIKELY_FAKE: (
                f"Content shows manipulation indicators (Trust Score: {score:.0f}/100, "
                f"Confidence: {conf_pct:.0f}%). Multiple analysis modalities flagged "
                "concerns. Treat with skepticism pending forensic review."
            ),
            Verdict.FAKE: (
                f"Content detected as manipulated (Trust Score: {score:.0f}/100, "
                f"Confidence: {conf_pct:.0f}%). Strong manipulation artifacts detected. "
                "Do not trust this content without verification."
            )
        }
        
        return explanations.get(verdict, "Unable to determine verdict.")
    
    def update_thresholds(
        self,
        authentic: Optional[int] = None,
        likely_authentic: Optional[int] = None,
        uncertain: Optional[int] = None,
        likely_fake: Optional[int] = None
    ) -> None:
        """
        Update verdict thresholds.
        
        Args:
            authentic: New threshold for authentic verdict
            likely_authentic: New threshold for likely_authentic
            uncertain: New threshold for uncertain
            likely_fake: New threshold for likely_fake
        """
        if authentic is not None:
            self.thresholds.authentic = authentic
        if likely_authentic is not None:
            self.thresholds.likely_authentic = likely_authentic
        if uncertain is not None:
            self.thresholds.uncertain = uncertain
        if likely_fake is not None:
            self.thresholds.likely_fake = likely_fake
        
        logger.info(
            f"Updated thresholds: {self.thresholds.authentic}/"
            f"{self.thresholds.likely_authentic}/{self.thresholds.uncertain}/"
            f"{self.thresholds.likely_fake}"
        )
    
    def fit_platt_parameters(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        content_type: ContentType
    ) -> PlattParams:
        """
        Fit Platt scaling parameters from calibration data.
        
        Uses logistic regression to fit A and B parameters.
        
        Args:
            scores: Raw model scores (N,)
            labels: True labels (0=fake, 1=authentic)
            content_type: Content type to update
            
        Returns:
            Fitted PlattParams
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Fit logistic regression
            lr = LogisticRegression(solver='lbfgs')
            lr.fit(scores.reshape(-1, 1), labels)
            
            # Extract parameters
            # P(y=1|f) = 1 / (1 + exp(-(w*f + b)))
            # Platt form: 1 / (1 + exp(A*f + B))
            # So A = -w, B = -b
            a = -float(lr.coef_[0][0])
            b = -float(lr.intercept_[0])
            
            params = PlattParams(a=a, b=b)
            self.platt_params[content_type] = params
            
            logger.info(f"Fitted Platt params for {content_type}: A={a:.3f}, B={b:.3f}")
            return params
            
        except ImportError:
            logger.warning("sklearn not available, using default Platt params")
            return self.platt_params.get(content_type, PlattParams())
    
    def batch_compute(
        self,
        aggregated_results: List[AggregatedResult],
        content_types: Optional[List[ContentType]] = None
    ) -> List[Tuple[TrustScore, Verdict]]:
        """
        Compute scores for a batch of results.
        
        Args:
            aggregated_results: List of aggregated results
            content_types: Optional content types per result
            
        Returns:
            List of (TrustScore, Verdict) tuples
        """
        content_types = content_types or [None] * len(aggregated_results)
        
        return [
            self.compute(agg, ct)
            for agg, ct in zip(aggregated_results, content_types)
        ]


# Singleton instance
_scorer: Optional[TrustScorer] = None


def get_trust_scorer() -> TrustScorer:
    """Get singleton trust scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = TrustScorer()
    return _scorer
