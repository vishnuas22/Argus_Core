"""
Argus Evaluation Package
========================
Benchmarking and evaluation tools for deepfake detection models.
"""

from evaluation.benchmark import (
    BenchmarkDataset,
    BenchmarkResult,
    BenchmarkReport,
    BenchmarkRunner,
    SOTATargets,
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkResult",
    "BenchmarkReport",
    "BenchmarkRunner",
    "SOTATargets",
]
