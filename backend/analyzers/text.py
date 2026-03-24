"""
Argus Core - Text Analyzer
===========================
AI-generated text detection using perplexity/burstiness analysis and RADAR model.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/text.py

SOTA Algorithms:
- Primary: RADAR (IBM NeurIPS) - adversarially robust detector
- Secondary: GPT-2 perplexity scoring, burstiness analysis
- Ensemble: Weighted combination for robustness

Metrics:
- Perplexity: Low = likely AI (too predictable)
- Burstiness: Low variance = likely AI (uniform sentence structure)
- Vocabulary diversity: Low = likely AI
- RADAR score: Adversarially trained classifier

Integration:
- Imports: core/engine.py
- Inputs: text: str
- Outputs: TextResult

Target Hardware: RTX 3050 (4GB VRAM) with optimized inference
"""

import asyncio
import re
import math
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
import time
from collections import Counter

from analyzers.base import (
    BaseAnalyzer,
    compute_confidence,
    infer_fake_class_index,
    extract_fake_probabilities,
)
from schemas.schemas import (
    Modality, PreprocessedData, ModalityResult, ContentType, TextResult
)
from config import config
from utils.logging import get_logger
from utils.errors import ValidationError, InferenceError
from models.model_init import ensure_models_for_analyzer, is_model_ready

if TYPE_CHECKING:
    from core.engine import InferenceEngine

logger = get_logger(__name__)


# Minimum text length for reliable analysis
MIN_TEXT_LENGTH = 50
# Optimal text length for best accuracy
OPTIMAL_TEXT_LENGTH = 500


@dataclass
class PerplexityFeatures:
    """
    Perplexity-based features for AI text detection.
    
    Low perplexity indicates highly predictable text,
    which is characteristic of AI-generated content.
    """
    mean_perplexity: float = 0.0  # Average perplexity across sentences
    median_perplexity: float = 0.0  # Median perplexity
    perplexity_variance: float = 0.0  # Variance in perplexity
    low_perplexity_ratio: float = 0.0  # Ratio of very low perplexity sentences
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean_perplexity": round(self.mean_perplexity, 4),
            "median_perplexity": round(self.median_perplexity, 4),
            "perplexity_variance": round(self.perplexity_variance, 4),
            "low_perplexity_ratio": round(self.low_perplexity_ratio, 4)
        }


@dataclass
class BurstinessFeatures:
    """
    Burstiness features for AI text detection.
    
    AI-generated text tends to have uniform sentence structure
    with low variance in length and complexity.
    """
    sentence_length_variance: float = 0.0  # Variance in sentence lengths
    burstiness_score: float = 0.0  # Overall burstiness measure
    paragraph_variance: float = 0.0  # Variance in paragraph lengths
    complexity_variance: float = 0.0  # Variance in sentence complexity
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "sentence_length_variance": round(self.sentence_length_variance, 4),
            "burstiness_score": round(self.burstiness_score, 4),
            "paragraph_variance": round(self.paragraph_variance, 4),
            "complexity_variance": round(self.complexity_variance, 4)
        }


@dataclass
class VocabularyFeatures:
    """
    Vocabulary diversity features.
    
    AI-generated text may have repetitive vocabulary patterns
    or unusual word frequency distributions.
    """
    type_token_ratio: float = 0.0  # Unique words / total words
    hapax_legomena_ratio: float = 0.0  # Words appearing once / total words
    vocabulary_richness: float = 0.0  # Overall vocabulary richness score
    repetition_score: float = 0.0  # N-gram repetition score
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "type_token_ratio": round(self.type_token_ratio, 4),
            "hapax_legomena_ratio": round(self.hapax_legomena_ratio, 4),
            "vocabulary_richness": round(self.vocabulary_richness, 4),
            "repetition_score": round(self.repetition_score, 4)
        }


@dataclass
class StylisticFeatures:
    """
    Stylistic features for text analysis.
    
    Captures writing style patterns that may differ
    between human and AI-generated text.
    """
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    punctuation_density: float = 0.0
    question_ratio: float = 0.0
    exclamation_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "avg_word_length": round(self.avg_word_length, 2),
            "punctuation_density": round(self.punctuation_density, 4),
            "question_ratio": round(self.question_ratio, 4),
            "exclamation_ratio": round(self.exclamation_ratio, 4)
        }


@dataclass
class TextAnalysisDetails:
    """
    Detailed text analysis results.
    
    Contains all intermediate analysis features for transparency.
    """
    # Neural detector scores
    roberta_score: float = 0.0  # Primary: RoBERTa-base OpenAI Detector
    radar_score: Optional[float] = None  # Legacy: RADAR classifier score
    gpt2_detector_score: float = 0.0  # Secondary: GPT-2 based detector
    
    # Feature-based analysis
    perplexity_features: Optional[PerplexityFeatures] = None
    burstiness_features: Optional[BurstinessFeatures] = None
    vocabulary_features: Optional[VocabularyFeatures] = None
    stylistic_features: Optional[StylisticFeatures] = None
    
    # Metadata
    text_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    language_detected: str = "en"
    primary_detector: str = "roberta"  # Which detector was used as primary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ModalityResult details."""
        return {
            "roberta_score": round(self.roberta_score, 4),
            "radar_score": round(self.radar_score, 4) if self.radar_score else None,
            "gpt2_detector_score": round(self.gpt2_detector_score, 4),
            "perplexity_features": self.perplexity_features.to_dict() if self.perplexity_features else None,
            "burstiness_features": self.burstiness_features.to_dict() if self.burstiness_features else None,
            "vocabulary_features": self.vocabulary_features.to_dict() if self.vocabulary_features else None,
            "stylistic_features": self.stylistic_features.to_dict() if self.stylistic_features else None,
            "text_length": self.text_length,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "language_detected": self.language_detected,
            "primary_detector": self.primary_detector
        }


class TextAnalyzer(BaseAnalyzer):
    """
    AI-generated text detection.
    
    Multi-method detection pipeline:
    1. Text preprocessing (cleaning, tokenization)
    2. Perplexity analysis (GPT-2 based)
    3. Burstiness analysis (sentence structure variance)
    4. Vocabulary analysis (diversity metrics)
    5. RADAR neural classifier (adversarially trained)
    6. Ensemble scoring
    
    Supported Detection:
    - ChatGPT/GPT-4 generated text
    - Claude, Gemini, LLaMA outputs
    - Paraphrased AI content
    - Machine-translated text
    
    Usage:
        analyzer = TextAnalyzer()
        result = await analyzer.analyze(preprocessed_data, engine)
    
    Or direct analysis:
        text_result = await analyzer.analyze_text("Some text...", engine)
    """
    
    def __init__(
        self,
        min_text_length: int = MIN_TEXT_LENGTH,
        perplexity_threshold: float = 50.0,
        burstiness_threshold: float = 0.5
    ):
        """
        Initialize text analyzer.
        
        Args:
            min_text_length: Minimum characters for analysis
            perplexity_threshold: Threshold for low perplexity (AI-likely)
            burstiness_threshold: Threshold for low burstiness (AI-likely)
        """
        super().__init__(
            analyzer_name="TextAnalyzer",
            supported_modalities=[Modality.TEXT],
            version="1.0.0"
        )
        
        self.min_text_length = min_text_length
        self.perplexity_threshold = perplexity_threshold
        self.burstiness_threshold = burstiness_threshold
        
        # Weight configuration for ensemble
        self.weights = {
            "roberta": 0.35,  # Primary: RoBERTa-base OpenAI Detector
            "perplexity": 0.35,  # Perplexity analysis (increased weight)
            "burstiness": 0.18,  # Burstiness analysis
            "vocabulary": 0.12   # Vocabulary diversity
        }
        
        logger.info(
            f"TextAnalyzer initialized: min_len={min_text_length}, "
            f"perplexity_thresh={perplexity_threshold}"
        )
    
    def get_required_models(self) -> List[str]:
        """
        Return models required for text analysis.
        
        Returns:
            List of model registry keys
        """
        return [
            "roberta_ai_detector",  # Primary: RoBERTa-base OpenAI Detector
            "gpt2_perplexity"       # Secondary: GPT-2 for perplexity scoring
        ]
    
    def validate_input(self, data: PreprocessedData) -> None:
        """
        Validate input data for text analysis.
        
        Args:
            data: PreprocessedData to validate
            
        Raises:
            ValidationError: If data is invalid for text analysis
        """
        super().validate_input(data)
        
        # Verify this is text content
        if data.content_type != ContentType.TEXT_ONLY:
            raise ValidationError(
                f"TextAnalyzer requires text content, got {data.content_type}"
            )
        
        # Check for text content
        if not data.text_content:
            raise ValidationError(
                "TextAnalyzer requires text_content in PreprocessedData"
            )
        
        # Validate minimum length
        if len(data.text_content) < self.min_text_length:
            raise ValidationError(
                f"TextAnalyzer requires at least {self.min_text_length} characters, "
                f"got {len(data.text_content)}"
            )
    
    async def _analyze_impl(
        self,
        data: PreprocessedData,
        engine: "InferenceEngine"
    ) -> ModalityResult:
        """
        Core text analysis implementation.
        
        Args:
            data: PreprocessedData with text content
            engine: InferenceEngine for model inference
            
        Returns:
            ModalityResult with detection score and details
        """
        text_result, details = await self.analyze_text(data.text_content, engine)
        
        return ModalityResult(
            modality=Modality.TEXT,
            score=text_result.ai_probability,
            confidence=self._compute_confidence(details),
            details=details.to_dict()
        )
    
    async def analyze_text(
        self,
        text: str,
        engine: "InferenceEngine"
    ) -> Tuple[TextResult, TextAnalysisDetails]:
        """
        Analyze text for AI generation patterns.
        
        Main entry point for direct text analysis.
        
        Args:
            text: Input text (minimum 50 characters recommended)
            engine: InferenceEngine
            
        Returns:
            Tuple of (TextResult, TextAnalysisDetails)
        """
        start_time = time.time()
        details = TextAnalysisDetails()
        
        # Basic text statistics
        details.text_length = len(text)
        words = self._tokenize_words(text)
        sentences = self._tokenize_sentences(text)
        details.word_count = len(words)
        details.sentence_count = len(sentences)
        
        # Validate length
        if len(text) < self.min_text_length:
            logger.warning(f"Text too short: {len(text)} < {self.min_text_length}")
            return self._create_default_result(), details
        
        logger.debug(f"Analyzing text: {len(words)} words, {len(sentences)} sentences")
        
        # 1. Compute perplexity features
        perplexity_features = self.compute_perplexity_features(text, sentences)
        details.perplexity_features = perplexity_features
        
        # 2. Compute burstiness features
        burstiness_features = self.compute_burstiness(sentences)
        details.burstiness_features = burstiness_features
        
        # 3. Compute vocabulary features
        vocabulary_features = self.compute_vocabulary_features(words)
        details.vocabulary_features = vocabulary_features
        
        # 4. Compute stylistic features
        stylistic_features = self.compute_stylistic_features(text, words, sentences)
        details.stylistic_features = stylistic_features
        
        # 5. Primary: Run RoBERTa AI detector
        roberta_score = 0.5
        use_roberta = False
        try:
            roberta_score = await self._run_roberta_detector(text, engine)
            details.roberta_score = roberta_score
            details.primary_detector = "roberta"
            use_roberta = True
            logger.info(f"RoBERTa detection: ai_prob={roberta_score:.4f}")
        except Exception as e:
            logger.warning(f"RoBERTa detection failed, using perplexity fallback: {e}")
            details.primary_detector = "perplexity"
        
        # 6. Secondary: Run GPT-2 based perplexity detector
        gpt2_score = await self._run_gpt2_detector(text, engine)
        details.gpt2_detector_score = gpt2_score
        
        # 7. Compute ensemble score
        ai_probability = self._compute_ensemble_score(
            roberta_score,
            perplexity_features,
            burstiness_features,
            vocabulary_features,
            use_roberta=use_roberta
        )
        
        inference_time = (time.time() - start_time) * 1000
        confidence = self._compute_confidence(details)
        self._metrics.record_analysis(True, inference_time, confidence)
        
        logger.info(
            f"Text analysis complete: ai_prob={ai_probability:.3f}, "
            f"perplexity={perplexity_features.mean_perplexity:.2f}, "
            f"time={inference_time:.2f}ms"
        )
        
        text_result = TextResult(
            ai_probability=ai_probability,
            perplexity_score=perplexity_features.mean_perplexity,
            burstiness_score=burstiness_features.burstiness_score,
            radar_score=details.radar_score
        )
        
        return text_result, details
    
    def compute_perplexity_features(
        self,
        text: str,
        sentences: List[str]
    ) -> PerplexityFeatures:
        """
        Compute perplexity-based features.
        
        Uses simplified perplexity approximation based on
        n-gram frequency analysis. True perplexity would require
        a language model (handled by neural detector).
        
        Args:
            text: Full text
            sentences: Tokenized sentences
            
        Returns:
            PerplexityFeatures
        """
        if not sentences:
            return PerplexityFeatures(mean_perplexity=100.0)
        
        # Compute pseudo-perplexity per sentence
        sentence_perplexities = []
        
        for sentence in sentences:
            words = self._tokenize_words(sentence)
            if len(words) < 3:
                continue
            
            # Simple perplexity approximation based on word predictability
            # Using word frequency as proxy
            perplexity = self._estimate_sentence_perplexity(words)
            sentence_perplexities.append(perplexity)
        
        if not sentence_perplexities:
            return PerplexityFeatures(mean_perplexity=100.0)
        
        mean_perp = np.mean(sentence_perplexities)
        median_perp = np.median(sentence_perplexities)
        variance = np.var(sentence_perplexities)
        
        # Count low perplexity sentences
        low_perp_count = sum(1 for p in sentence_perplexities 
                           if p < self.perplexity_threshold)
        low_perp_ratio = low_perp_count / len(sentence_perplexities)
        
        return PerplexityFeatures(
            mean_perplexity=float(mean_perp),
            median_perplexity=float(median_perp),
            perplexity_variance=float(variance),
            low_perplexity_ratio=float(low_perp_ratio)
        )
    
    def _estimate_sentence_perplexity(self, words: List[str]) -> float:
        """
        Estimate sentence perplexity using word frequency.
        
        Lower perplexity = more predictable = more likely AI.
        
        Args:
            words: List of words in sentence
            
        Returns:
            Estimated perplexity
        """
        # Common English word frequencies (simplified)
        # In production, would use actual frequency data
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'having'
        }
        
        # Calculate "surprise" based on word commonality
        surprises = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in common_words:
                surprise = 1.0  # Very predictable
            elif len(word_lower) > 10:
                surprise = 5.0  # Long words are less predictable
            else:
                surprise = 3.0  # Average word
            surprises.append(surprise)
        
        if not surprises:
            return 100.0
        
        # Perplexity is geometric mean of surprises
        log_surprises = [math.log(s) for s in surprises]
        avg_log = np.mean(log_surprises)
        perplexity = math.exp(avg_log)
        
        # Scale to typical perplexity range (10-500)
        return perplexity * 20
    
    def compute_burstiness(self, sentences: List[str]) -> BurstinessFeatures:
        """
        Compute burstiness (sentence length variance) features.
        
        Human writing has "bursty" patterns with varied sentence lengths.
        AI-generated text tends to be more uniform.
        
        Args:
            sentences: List of sentences
            
        Returns:
            BurstinessFeatures
        """
        if len(sentences) < 2:
            return BurstinessFeatures(burstiness_score=0.5)
        
        # Sentence lengths in words
        lengths = [len(self._tokenize_words(s)) for s in sentences]
        
        # Remove very short sentences (e.g., single words)
        lengths = [l for l in lengths if l > 2]
        
        if len(lengths) < 2:
            return BurstinessFeatures(burstiness_score=0.5)
        
        # Compute variance
        mean_length = np.mean(lengths)
        variance = np.var(lengths)
        
        # Coefficient of variation (normalized variance)
        if mean_length > 0:
            cv = np.std(lengths) / mean_length
        else:
            cv = 0.0
        
        # Burstiness score: high CV = more human-like
        # Low CV = more AI-like
        burstiness_score = float(cv)
        
        # Complexity variance (based on punctuation density per sentence)
        complexities = []
        for s in sentences:
            punct_count = sum(1 for c in s if c in '.,;:!?-—')
            complexities.append(punct_count / (len(s) + 1))
        
        complexity_variance = float(np.var(complexities))
        
        return BurstinessFeatures(
            sentence_length_variance=float(variance),
            burstiness_score=burstiness_score,
            paragraph_variance=0.0,  # Would require paragraph detection
            complexity_variance=complexity_variance
        )
    
    def compute_vocabulary_features(self, words: List[str]) -> VocabularyFeatures:
        """
        Compute vocabulary diversity features.
        
        AI-generated text may have:
        - Lower type-token ratio (less diverse vocabulary)
        - More repetitive patterns
        - Unusual word frequency distributions
        
        Args:
            words: List of words
            
        Returns:
            VocabularyFeatures
        """
        if not words:
            return VocabularyFeatures(type_token_ratio=0.5)
        
        # Normalize words
        words_lower = [w.lower() for w in words if len(w) > 0]
        
        if not words_lower:
            return VocabularyFeatures(type_token_ratio=0.5)
        
        # Type-token ratio
        unique_words = set(words_lower)
        ttr = len(unique_words) / len(words_lower)
        
        # Hapax legomena ratio (words appearing exactly once)
        word_counts = Counter(words_lower)
        hapax = sum(1 for w, c in word_counts.items() if c == 1)
        hapax_ratio = hapax / len(words_lower)
        
        # N-gram repetition
        repetition_score = self._compute_ngram_repetition(words_lower)
        
        # Overall vocabulary richness
        vocabulary_richness = (ttr * 0.4 + hapax_ratio * 0.3 + 
                             (1 - repetition_score) * 0.3)
        
        return VocabularyFeatures(
            type_token_ratio=float(ttr),
            hapax_legomena_ratio=float(hapax_ratio),
            vocabulary_richness=float(vocabulary_richness),
            repetition_score=float(repetition_score)
        )
    
    def _compute_ngram_repetition(self, words: List[str], n: int = 3) -> float:
        """
        Compute n-gram repetition score.
        
        High repetition may indicate AI-generated content.
        
        Args:
            words: List of words
            n: N-gram size
            
        Returns:
            Repetition score [0, 1]
        """
        if len(words) < n:
            return 0.0
        
        # Generate n-grams
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        if not ngrams:
            return 0.0
        
        # Count repeated n-grams
        ngram_counts = Counter(ngrams)
        repeated = sum(1 for c in ngram_counts.values() if c > 1)
        
        # Repetition ratio
        repetition = repeated / len(ngram_counts) if ngram_counts else 0.0
        
        return float(repetition)
    
    def compute_stylistic_features(
        self,
        text: str,
        words: List[str],
        sentences: List[str]
    ) -> StylisticFeatures:
        """
        Compute stylistic features.
        
        Args:
            text: Full text
            words: Tokenized words
            sentences: Tokenized sentences
            
        Returns:
            StylisticFeatures
        """
        # Average sentence length
        if sentences:
            avg_sentence_length = np.mean([len(self._tokenize_words(s)) for s in sentences])
        else:
            avg_sentence_length = 0.0
        
        # Average word length
        if words:
            avg_word_length = np.mean([len(w) for w in words if len(w) > 0])
        else:
            avg_word_length = 0.0
        
        # Punctuation density
        punct_count = sum(1 for c in text if c in '.,;:!?-—\'\"')
        punct_density = punct_count / (len(text) + 1)
        
        # Question ratio
        question_count = text.count('?')
        question_ratio = question_count / (len(sentences) + 1)
        
        # Exclamation ratio
        exclamation_count = text.count('!')
        exclamation_ratio = exclamation_count / (len(sentences) + 1)
        
        return StylisticFeatures(
            avg_sentence_length=float(avg_sentence_length),
            avg_word_length=float(avg_word_length),
            punctuation_density=float(punct_density),
            question_ratio=float(question_ratio),
            exclamation_ratio=float(exclamation_ratio)
        )
    
    async def _run_radar_detector(
        self,
        text: str,
        engine: "InferenceEngine"
    ) -> Optional[float]:
        """
        Run RADAR neural detector (legacy fallback).
        
        RADAR is adversarially trained to be robust against
        paraphrasing and other evasion techniques.
        
        Args:
            text: Input text
            engine: InferenceEngine
            
        Returns:
            AI probability [0, 1] or None if model unavailable
        """
        try:
            # Tokenize text for model
            # RADAR expects tokenized input
            tokens = self._prepare_text_for_model(text)
            
            # Add batch dimension
            batch = np.expand_dims(tokens, 0).astype(np.int64)
            
            result = await engine.infer(
                "radar_text",
                batch,
                return_probabilities=True
            )
            
            return self._extract_ai_probability(
                result.class_probabilities
                if result.class_probabilities is not None
                else result.predictions
            )
            
        except Exception as e:
            logger.warning(f"RADAR detector failed: {e}")
            return None
    
    async def _run_roberta_detector(
        self,
        text: str,
        engine: "InferenceEngine"
    ) -> float:
        """
        Run ModernBERT AI Detector.
        
        ModernBERT-base fine-tuned for AI text detection.
        This is the primary detector with real model weights.
        
        Args:
            text: Input text
            engine: InferenceEngine
            
        Returns:
            AI probability [0, 1]
        """
        try:
            # Get proper tokenization with attention mask
            input_ids, attention_mask = self._prepare_text_with_attention(text, max_length=512)
            
            # Run inference with both inputs
            result = await engine.infer(
                "roberta_ai_detector",
                {"input_ids": input_ids, "attention_mask": attention_mask},
                return_probabilities=True
            )
            
            # Extract AI probability robustly across possible model output shapes.
            ai_prob = self._extract_ai_probability(
                result.class_probabilities
                if result.class_probabilities is not None
                else result.predictions
            )
            
            logger.debug(f"ModernBERT detector result: ai_prob={ai_prob:.4f}")
            return ai_prob
            
        except ImportError as e:
            logger.error(f"ModernBERT dependencies missing: {e}")
            raise InferenceError("roberta_ai_detector", f"Missing dependencies: {e}")
        except RuntimeError as e:
            logger.error(f"ModernBERT ONNX runtime error: {e}")
            raise InferenceError("roberta_ai_detector", f"Runtime error: {e}")
        except ValueError as e:
            logger.error(f"ModernBERT input validation error: {e}")
            raise InferenceError("roberta_ai_detector", f"Invalid input: {e}")
        except Exception as e:
            logger.error(f"ModernBERT detector failed: {type(e).__name__}: {e}")
            raise InferenceError("roberta_ai_detector", f"Unexpected error: {type(e).__name__}: {e}")

    def _extract_ai_probability(self, outputs: Any) -> float:
        """
        Extract AI-generated probability from model outputs with flexible shape handling.

        Supports outputs shaped as:
        - (B, C)
        - (B, 1, C)
        - (C,)
        - logits or probabilities
        """
        arr = np.asarray(outputs, dtype=np.float32)

        if arr.size == 0:
            return 0.5

        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
        else:
            # Preserve class axis (last axis), flatten remaining leading axes.
            num_classes = arr.shape[-1]
            arr = arr.reshape(-1, num_classes)

        if arr.shape[1] < 2:
            # Binary classifier expected; fallback to bounded mean if malformed.
            return float(np.clip(np.mean(arr), 0.0, 1.0))

        # Convert logits to probabilities if needed.
        row_sums = arr.sum(axis=-1)
        looks_like_probs = (
            arr.min() >= 0.0 and
            arr.max() <= 1.0 and
            np.allclose(row_sums, 1.0, atol=1e-2)
        )
        if looks_like_probs:
            probs = arr
        else:
            shifted = arr - np.max(arr, axis=-1, keepdims=True)
            exp_arr = np.exp(shifted)
            probs = exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)

        ai_idx = infer_fake_class_index(
            class_labels=["human", "ai_generated"],
            default_index=1
        )
        ai_probs = extract_fake_probabilities(
            probs,
            fake_class_index=ai_idx,
            apply_confidence_shrinkage=True
        )
        return float(np.mean(ai_probs))
    
    def _prepare_text_with_attention(self, text: str, max_length: int = 512) -> tuple:
        """Prepare text with both input_ids and attention_mask for transformer models."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            return (
                encoding["input_ids"].astype(np.int64),
                encoding["attention_mask"].astype(np.int64)
            )
        except Exception as e:
            logger.warning(f"Tokenizer unavailable, using fallback: {e}")
            tokens = self._fallback_tokenization(text, max_length)
            attention_mask = np.ones_like(tokens)
            attention_mask[tokens == 0] = 0
            return np.expand_dims(tokens, 0), np.expand_dims(attention_mask, 0)
    
    async def _run_gpt2_detector(
        self,
        text: str,
        engine: "InferenceEngine"
    ) -> float:
        """
        Run GPT-2 based detector.
        
        Uses GPT-2 perplexity as a detection signal.
        Low perplexity indicates likely AI-generated text.
        
        Args:
            text: Input text
            engine: InferenceEngine
            
        Returns:
            AI probability [0, 1]
        """
        try:
            tokens = self._prepare_text_for_model(text)
            batch = np.expand_dims(tokens, 0).astype(np.int64)
            
            result = await engine.infer(
                "gpt2_perplexity",
                batch,
                return_probabilities=True
            )
            
            return self._extract_ai_probability(
                result.class_probabilities
                if result.class_probabilities is not None
                else result.predictions
            )
            
        except Exception as e:
            logger.debug(f"GPT-2 detector skipped: {e}")
            # Fall back to perplexity-based estimate
            return self._estimate_ai_probability_from_perplexity()
    
    def _estimate_ai_probability_from_perplexity(self) -> float:
        """Estimate AI probability without neural model."""
        # Default neutral estimate
        return 0.5
    
    def _prepare_text_for_model(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Prepare text for neural model input using proper BPE tokenization.
        
        Uses transformers library tokenizer for ModernBERT/RoBERTa compatibility.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Token IDs array with shape [max_length]
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            return encoding["input_ids"][0].astype(np.int64)
        except Exception as e:
            logger.warning(f"Tokenizer not available, using fallback: {e}")
            return self._fallback_tokenization(text, max_length)
    
    def _fallback_tokenization(self, text: str, max_length: int = 512) -> np.ndarray:
        """Fallback tokenization when transformers tokenizer is unavailable."""
        text = text.strip()[:max_length * 4]
        tokens = [ord(c) % 30000 for c in text[:max_length]]
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
        return np.array(tokens, dtype=np.int64)
    
    def _compute_ensemble_score(
        self,
        roberta_score: float,
        perplexity: PerplexityFeatures,
        burstiness: BurstinessFeatures,
        vocabulary: VocabularyFeatures,
        use_roberta: bool = True
    ) -> float:
        """
        Compute ensemble AI probability score.
        
        Args:
            roberta_score: RoBERTa detector score (primary)
            perplexity: Perplexity features
            burstiness: Burstiness features
            vocabulary: Vocabulary features
            use_roberta: Whether RoBERTa was successfully used
            
        Returns:
            Ensemble AI probability [0, 1]
        """
        scores = {}
        weights = dict(self.weights)
        
        # RoBERTa score (primary detector)
        if use_roberta:
            scores["roberta"] = roberta_score
        else:
            # Redistribute weight to feature-based methods
            weights["perplexity"] += weights["roberta"] * 0.5
            weights["burstiness"] += weights["roberta"] * 0.3
            weights["vocabulary"] += weights["roberta"] * 0.2
            weights["roberta"] = 0
        
        # Perplexity score
        # Low perplexity = likely AI
        # Sigmoid calibration: maps perplexity to AI probability smoothly
        # Center at 80 (typical human text), steepness 0.03
        # Low perplexity (<50) -> high AI prob (>0.7)
        # Medium perplexity (80) -> moderate AI prob (~0.5)
        # High perplexity (>120) -> low AI prob (<0.3)
        perp_score = 1.0 / (1.0 + np.exp(0.03 * (perplexity.mean_perplexity - 80.0)))
        scores["perplexity"] = perp_score
        
        # Burstiness score
        # Low burstiness = likely AI
        # CV < 0.5 is suspicious
        burst_score = max(0, 1.0 - burstiness.burstiness_score * 2)
        scores["burstiness"] = burst_score
        
        # Vocabulary score
        # Low diversity = likely AI
        vocab_score = 1.0 - vocabulary.vocabulary_richness
        # High repetition = likely AI
        vocab_score = vocab_score * 0.7 + vocabulary.repetition_score * 0.3
        scores["vocabulary"] = vocab_score
        
        # Weighted combination
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.5
        
        ensemble_score = sum(
            scores.get(k, 0.5) * v 
            for k, v in weights.items()
        ) / total_weight
        
        return float(np.clip(ensemble_score, 0, 1))
    
    def _compute_confidence(self, details: TextAnalysisDetails) -> float:
        """
        Compute confidence based on analysis details.
        
        Args:
            details: Analysis details
            
        Returns:
            Confidence score [0, 1]
        """
        # Length factor
        length_factor = min(1.0, details.text_length / OPTIMAL_TEXT_LENGTH)
        
        # Primary detector factor (RoBERTa is more reliable)
        detector_factor = 0.9 if details.primary_detector == "roberta" else 0.6
        
        # Score extremity factor
        scores = []
        if details.roberta_score > 0:
            scores.append(details.roberta_score)
        if details.perplexity_features:
            scores.append(1.0 - min(1.0, details.perplexity_features.mean_perplexity / 200))
        
        if scores:
            mean_score = np.mean(scores)
            extremity = abs(mean_score - 0.5) * 2
        else:
            extremity = 0.0
        
        confidence = (
            0.4 * length_factor +
            0.3 * detector_factor +
            0.3 * extremity
        )
        
        return float(np.clip(confidence, 0.3, 0.95))
    
    def _tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence tokenization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _create_default_result(self) -> TextResult:
        """Create default result for insufficient data."""
        return TextResult(
            ai_probability=0.5,
            perplexity_score=100.0,
            burstiness_score=0.5,
            radar_score=None
        )


# Singleton instance
_text_analyzer: Optional[TextAnalyzer] = None


def get_text_analyzer() -> TextAnalyzer:
    """Get singleton text analyzer instance."""
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextAnalyzer()
    return _text_analyzer
