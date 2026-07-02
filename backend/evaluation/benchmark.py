"""
Argus Evaluation - Benchmark Framework
=======================================
Comprehensive benchmarking suite for evaluating all Argus detectors against
standard deepfake detection datasets and SOTA targets.

Supported datasets:
    - FaceForensics++ (FF++): 4 manipulation methods (Deepfakes, Face2Face,
      FaceSwap, NeuralTextures) at 3 compression levels (c0, c23, c40)
    - Celeb-DF v2: 5920 videos of celebrity deepfakes
    - ASVspoof 2019/2021: Audio anti-spoofing (LA, DF, PA tracks)
    - WildDeepfake: In-the-wild deepfake detection

SOTA benchmark targets:
    - FaceForensics++: AUC > 98% (c0), > 95% (c40)
    - Celeb-DF v2: AUC > 97%
    - ASVspoof 2021 LA: EER < 1.0%
    - Cross-generator: AUC > 95% on unseen methods

Usage:
    python -m evaluation.benchmark --dataset ffpp --split test
    python -m evaluation.benchmark --dataset celebdf
    python -m evaluation.benchmark --dataset asvspoof21
    python -m evaluation.benchmark --all
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class BenchmarkDataset(Enum):
    """Supported benchmark datasets."""
    FFPP_C0 = "ffpp_c0"           # FaceForensics++ raw
    FFPP_C23 = "ffpp_c23"         # FaceForensics++ low compression
    FFPP_C40 = "ffpp_c40"         # FaceForensics++ high compression
    CELEBDF = "celebdf"           # Celeb-DF v2
    ASVSPOOF19 = "asvspoof19"    # ASVspoof 2019
    ASVSPOOF21 = "asvspoof21"    # ASVspoof 2021
    WILDDEEPFAKE = "wilddeepfake" # WildDeepfake


@dataclass
class SOTATargets:
    """SOTA benchmark performance targets."""
    # Image/Video detectors
    ffpp_c0_auc: float = 0.98
    ffpp_c23_auc: float = 0.96
    ffpp_c40_auc: float = 0.95
    celebdf_auc: float = 0.97
    cross_generator_auc: float = 0.95

    # Audio detectors
    asvspoof19_eer: float = 0.012    # 1.2%
    asvspoof21_eer: float = 0.010    # 1.0%
    asvspoof21_auc: float = 0.98

    # Cross-modal
    multimodal_auc: float = 0.98


@dataclass
class BenchmarkResult:
    """Result of a single benchmark evaluation."""
    dataset: str
    detector_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    elapsed_seconds: float = 0.0
    passed_sota: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Full benchmark report across all detectors and datasets."""
    results: List[BenchmarkResult] = field(default_factory=list)
    sota_targets: SOTATargets = field(default_factory=SOTATargets)
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def compute_summary(self) -> None:
        """Compute aggregate summary statistics."""
        if not self.results:
            return

        # Group by detector
        by_detector: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            by_detector.setdefault(r.detector_name, []).append(r)

        detector_summaries = {}
        for det_name, det_results in by_detector.items():
            auc_values = [r.metrics.get("auc_roc", 0.5) for r in det_results]
            eer_values = [r.metrics.get("eer", 0.5) for r in det_results]
            detector_summaries[det_name] = {
                "mean_auc": float(np.mean(auc_values)),
                "min_auc": float(np.min(auc_values)),
                "mean_eer": float(np.mean(eer_values)),
                "num_benchmarks": len(det_results),
                "passed_sota": all(r.passed_sota for r in det_results),
            }

        self.summary = {
            "num_detectors": len(by_detector),
            "num_benchmarks": len(self.results),
            "detectors": detector_summaries,
            "overall_sota_compliance": sum(
                1 for d in detector_summaries.values() if d["passed_sota"]
            ) / max(len(detector_summaries), 1),
        }

    def to_dict(self) -> Dict:
        """Serialize report to dict."""
        return {
            "sota_targets": asdict(self.sota_targets),
            "results": [asdict(r) for r in self.results],
            "summary": self.summary,
            "generated_at": self.generated_at,
        }

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        self.compute_summary()
        report_dict = self.to_dict()
        with open(path, "w") as f:
            json.dump(report_dict, f, indent=2)
        logger.info("Benchmark report saved to %s", path)


class BenchmarkRunner:
    """
    Run detector benchmarks against standard datasets.

    Evaluates each registered detector on the specified dataset and computes
    comprehensive metrics. Compares against SOTA targets.
    """

    def __init__(
        self,
        data_dir: str = "/data/benchmarks",
        output_dir: str = "/data/benchmark_results",
        device: str = "cpu",
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        self.sota_targets = SOTATargets()
        os.makedirs(output_dir, exist_ok=True)

    async def run_benchmark(
        self,
        dataset: BenchmarkDataset,
        detectors: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> BenchmarkReport:
        """
        Run benchmark on specified dataset.

        Args:
            dataset: Dataset to evaluate on.
            detectors: List of detector names to evaluate. None = all.
            max_samples: Limit number of samples for quick testing.

        Returns:
            BenchmarkReport with all results.
        """
        from datetime import datetime, timezone

        report = BenchmarkReport(
            sota_targets=self.sota_targets,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Get dataset samples
        samples = self._load_dataset(dataset, max_samples)
        if not samples:
            logger.warning("No samples found for dataset %s", dataset.value)
            return report

        logger.info(
            "Running benchmark: dataset=%s, samples=%d",
            dataset.value, len(samples),
        )

        # Get detectors to evaluate
        detector_names = detectors or self._get_all_detectors(dataset)

        for det_name in detector_names:
            try:
                result = await self._evaluate_detector(
                    det_name, dataset, samples
                )
                result.passed_sota = self._check_sota_targets(
                    dataset, result.metrics
                )
                report.add_result(result)
                logger.info(
                    "  %s: AUC=%.4f, EER=%.4f, passed_sota=%s",
                    det_name,
                    result.metrics.get("auc_roc", 0.5),
                    result.metrics.get("eer", 0.5),
                    result.passed_sota,
                )
            except Exception as e:
                logger.error("Benchmark failed for %s: %s", det_name, e)

        return report

    def _load_dataset(
        self, dataset: BenchmarkDataset, max_samples: Optional[int]
    ) -> List[Dict]:
        """
        Load dataset samples.

        Returns list of dicts with keys: "path", "label" (0=real, 1=fake),
        "split" (train/val/test).
        """
        dataset_dir = os.path.join(self.data_dir, dataset.value)
        if not os.path.exists(dataset_dir):
            logger.warning(
                "Dataset directory not found: %s. "
                "Create symlinks to your dataset in %s.",
                dataset_dir, self.data_dir,
            )
            return []

        samples = []
        for split in ["real", "fake"]:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                continue
            label = 0 if split == "real" else 1
            for fname in os.listdir(split_dir):
                if fname.endswith((".mp4", ".wav", ".npy", ".jpg", ".png")):
                    samples.append({
                        "path": os.path.join(split_dir, fname),
                        "label": label,
                        "split": split,
                    })

        if max_samples and len(samples) > max_samples:
            # Stratified subsampling
            real = [s for s in samples if s["label"] == 0][:max_samples // 2]
            fake = [s for s in samples if s["label"] == 1][:max_samples // 2]
            samples = real + fake

        return samples

    def _get_all_detectors(self, dataset: BenchmarkDataset) -> List[str]:
        """Get list of applicable detectors for a dataset."""
        is_audio = dataset in (
            BenchmarkDataset.ASVSPOOF19,
            BenchmarkDataset.ASVSPOOF21,
        )

        if is_audio:
            return [
                "AASIST3AudioDetector",
                "Wav2Vec2XLSRMoELoRADetector",
                "ECAPATDNNAudioDetector",
                "CDPMambaDetector",
            ]
        else:
            return [
                "CLIPLoRAImageDetector",
                "DINOv2ImageDetector",
                "SigLIPImageDetector",
                "SBIDetector",
                "UCFCrossForgeryDetector",
            ]

    async def _evaluate_detector(
        self,
        detector_name: str,
        dataset: BenchmarkDataset,
        samples: List[Dict],
    ) -> BenchmarkResult:
        """Evaluate a single detector on dataset samples."""
        start_time = time.time()

        # Instantiate detector
        detector = self._create_detector(detector_name)
        if detector is None:
            raise ValueError(f"Unknown detector: {detector_name}")

        # Run inference on all samples
        predictions = []
        labels = []

        for sample in samples:
            try:
                result = await self._run_detection(detector, sample)
                predictions.append(result)
                labels.append(sample["label"])
            except Exception as e:
                logger.debug("Detection failed for %s: %s", sample["path"], e)

        elapsed = time.time() - start_time

        if not predictions:
            return BenchmarkResult(
                dataset=dataset.value,
                detector_name=detector_name,
                elapsed_seconds=elapsed,
            )

        # Compute metrics
        if not _TORCH_AVAILABLE:
            logger.warning("torch not available — computing basic metrics without compute_metrics")
            probs = np.array(predictions)
            labels_arr = np.array(labels)
            auc = float(np.trapz(labels_arr, probs)) if len(probs) > 1 else 0.5
            return BenchmarkResult(
                dataset=dataset.value,
                detector_name=detector_name,
                metrics={"auc_roc": auc},
                num_samples=len(predictions),
                elapsed_seconds=elapsed,
            )
        from training.evaluation import compute_metrics
        probs = np.array(predictions)
        probs_tensor = torch.from_numpy(probs)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        metrics = compute_metrics(probs_tensor, labels_tensor)

        return BenchmarkResult(
            dataset=dataset.value,
            detector_name=detector_name,
            metrics=metrics,
            num_samples=len(predictions),
            elapsed_seconds=elapsed,
        )

    def _create_detector(self, name: str):
        """Create a detector instance by name."""
        try:
            if name == "CLIPLoRAImageDetector":
                from detectors import CLIPLoRAImageDetector
                return CLIPLoRAImageDetector()
            elif name == "DINOv2ImageDetector":
                from detectors import DINOv2ImageDetector
                return DINOv2ImageDetector()
            elif name == "SigLIPImageDetector":
                from detectors import SigLIPImageDetector
                return SigLIPImageDetector()
            elif name == "SBIDetector":
                from detectors import SBIDetector
                return SBIDetector()
            elif name == "UCFCrossForgeryDetector":
                from detectors import UCFCrossForgeryDetector
                return UCFCrossForgeryDetector()
            elif name == "AASIST3AudioDetector":
                from detectors import AASIST3AudioDetector
                return AASIST3AudioDetector()
            elif name == "Wav2Vec2XLSRMoELoRADetector":
                from detectors import Wav2Vec2XLSRMoELoRADetector
                return Wav2Vec2XLSRMoELoRADetector()
            elif name == "ECAPATDNNAudioDetector":
                from detectors import ECAPATDNNAudioDetector
                return ECAPATDNNAudioDetector()
            elif name == "CDPMambaDetector":
                from detectors import CDPMambaDetector
                return CDPMambaDetector()
            else:
                logger.warning("Unknown detector: %s", name)
                return None
        except Exception as e:
            logger.error("Failed to create detector %s: %s", name, e)
            return None

    async def _run_detection(self, detector, sample: Dict) -> float:
        """Run a single detection and return spoof probability."""
        import cv2

        path = sample["path"]
        ext = os.path.splitext(path)[1].lower()

        if ext in (".jpg", ".png", ".jpeg"):
            # Image sample
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Cannot read image: {path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = await detector.detect(img_rgb)
            return result.score

        elif ext == ".wav":
            # Audio sample
            import scipy.io.wavfile as wavfile
            sr, audio = wavfile.read(path)
            audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            result = await detector.detect(audio, sample_rate=sr)
            return result.score

        elif ext == ".npy":
            # Pre-computed features
            data = np.load(path)
            if data.ndim == 3:
                # Image array
                result = await detector.detect(data)
                return result.score
            else:
                # Audio waveform
                result = await detector.detect(data)
                return result.score

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _check_sota_targets(
        self, dataset: BenchmarkDataset, metrics: Dict[str, float]
    ) -> bool:
        """Check if detector meets SOTA targets for the dataset."""
        targets = self.sota_targets

        if dataset == BenchmarkDataset.FFPP_C0:
            return metrics.get("auc_roc", 0.5) >= targets.ffpp_c0_auc
        elif dataset == BenchmarkDataset.FFPP_C23:
            return metrics.get("auc_roc", 0.5) >= targets.ffpp_c23_auc
        elif dataset == BenchmarkDataset.FFPP_C40:
            return metrics.get("auc_roc", 0.5) >= targets.ffpp_c40_auc
        elif dataset == BenchmarkDataset.CELEBDF:
            return metrics.get("auc_roc", 0.5) >= targets.celebdf_auc
        elif dataset == BenchmarkDataset.ASVSPOOF19:
            return metrics.get("eer", 0.5) <= targets.asvspoof19_eer
        elif dataset == BenchmarkDataset.ASVSPOOF21:
            return metrics.get("eer", 0.5) <= targets.asvspoof21_eer

        return False


# ─────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import asyncio

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Argus Benchmark Runner")
    parser.add_argument(
        "--dataset",
        choices=[d.value for d in BenchmarkDataset],
        help="Dataset to benchmark on",
    )
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--detectors", nargs="+", help="Specific detectors to evaluate")
    parser.add_argument("--max-samples", type=int, help="Max samples per dataset")
    parser.add_argument("--data-dir", default="/data/benchmarks", help="Dataset root")
    parser.add_argument("--output-dir", default="/data/benchmark_results", help="Output dir")
    args = parser.parse_args()

    runner = BenchmarkRunner(data_dir=args.data_dir, output_dir=args.output_dir)

    datasets = (
        list(BenchmarkDataset) if args.all
        else [BenchmarkDataset(args.dataset)] if args.dataset
        else []
    )

    if not datasets:
        parser.print_help()
        exit(1)

    for ds in datasets:
        report = asyncio.run(runner.run_benchmark(
            ds, detectors=args.detectors, max_samples=args.max_samples
        ))
        report.compute_summary()
        output_path = os.path.join(args.output_dir, f"benchmark_{ds.value}.json")
        report.save(output_path)
        print(f"\nBenchmark complete: {ds.value}")
        print(json.dumps(report.summary, indent=2))
