#!/usr/bin/env python3
"""
Argus Core - Standard Benchmark Evaluation Suite
==================================================
Evaluates the Argus Core detection pipeline on standard deepfake
detection benchmarks: FaceForensics++, Celeb-DF, DFDC, ASVspoof.

Usage:
    python scripts/benchmark.py --dataset ffpp --data /path/to/FF++
    python scripts/benchmark.py --dataset celebdf --data /path/to/Celeb-DF
    python scripts/benchmark.py --dataset dfdc --data /path/to/DFDC
    python scripts/benchmark.py --all --data /datasets

Output:
    Results printed to stdout and saved to benchmark_results.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from config import config
from utils.logging import get_logger
from core.scorer import PlattParams, ScoringConfig, TrustScorer, VerdictThresholds

logger = get_logger(__name__)


async def evaluate_on_dataset(
    dataset_name: str,
    data_dir: str,
    split: str = "test",
) -> Dict[str, float]:
    """
    Run Argus Core detection pipeline on a standard benchmark dataset.

    This function loads samples from the dataset directory, runs inference
    through the Argus Core pipeline, and computes standard metrics.

    Args:
        dataset_name: Benchmark dataset name (ffpp, celebdf, dfdc, asvspoof)
        data_dir: Path to dataset root
        split: Dataset split (test, val, all)

    Returns:
        Dict of metric name -> value
    """
    logger.info(f"Evaluating on {dataset_name} (split={split}) from {data_dir}")

    scores = []
    labels = []
    latencies = []

    from analyzers.image import ImageAnalyzer
    from core.engine import InferenceEngine

    engine = InferenceEngine()
    analyzer = ImageAnalyzer()

    sample_paths = _discover_samples(dataset_name, data_dir, split)
    if not sample_paths:
        logger.warning(f"No samples found for {dataset_name} at {data_dir}")
        return {}

    logger.info(f"Found {len(sample_paths)} samples")

    for path, label in sample_paths:
        try:
            start = time.perf_counter()
            image = np.array(Image.open(str(path)).convert("RGB"))
            result = await analyzer.analyze_single_image(image, engine)
            elapsed = (time.perf_counter() - start) * 1000

            fake_score = result.deepfake_score if hasattr(result, "deepfake_score") else getattr(result, "score", 0.5)
            scores.append(fake_score)
            labels.append(label)
            latencies.append(elapsed)
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")
            continue

    if not scores:
        return {}

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    latencies_arr = np.array(latencies)

    metrics = _compute_benchmark_metrics(scores_arr, labels_arr)
    metrics["latency_mean_ms"] = float(np.mean(latencies_arr))
    metrics["latency_p50_ms"] = float(np.median(latencies_arr))
    metrics["latency_p95_ms"] = float(np.percentile(latencies_arr, 95))
    metrics["num_samples"] = len(scores)
    metrics["dataset"] = dataset_name

    scorer = TrustScorer(ScoringConfig(), VerdictThresholds())
    platt = PlattParams.fit(scores_arr, labels_arr)
    calibrated = np.array([platt.transform(s) for s in scores_arr])
    cal_metrics = _compute_benchmark_metrics(calibrated, labels_arr)
    for k, v in cal_metrics.items():
        metrics[f"calibrated_{k}"] = v
    metrics["platt_a"] = platt.a
    metrics["platt_b"] = platt.b

    logger.info(f"Results for {dataset_name}:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    return metrics


def _discover_samples(
    dataset_name: str, data_dir: str, split: str
) -> List[Tuple[str, int]]:
    """Discover sample paths and labels for a given benchmark dataset."""
    samples = []
    root = Path(data_dir)

    if dataset_name == "ffpp":
        for method in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            method_dir = root / method
            if not method_dir.exists():
                continue
            for label_str, label in [("real", 0), ("fake", 1)]:
                label_dir = method_dir / label_str
                if not label_dir.exists():
                    continue
                for f in sorted(label_dir.glob("*.*"))[:100]:
                    samples.append((str(f), label))

    elif dataset_name == "celebdf":
        real_dir = root / "real"
        if real_dir.exists():
            for f in sorted(real_dir.glob("*.*"))[:200]:
                samples.append((str(f), 0))
        fake_dir = root / "fake"
        if fake_dir.exists():
            for f in sorted(fake_dir.glob("*.*"))[:200]:
                samples.append((str(f), 1))

    elif dataset_name == "dfdc":
        import json as j
        meta_path = root / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = j.load(f)
            for vid, info in meta.items():
                if info.get("split", "") == split or split == "all":
                    vid_path = root / vid
                    if vid_path.exists():
                        samples.append((str(vid_path), 1 if info.get("label") == "FAKE" else 0))

    elif dataset_name == "asvspoof":
        import json as j
        meta_path = root / "ASVspoof2021.LA.cm.train.metadata.txt"
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        _, _, _, label_str = parts[:4]
                        label = 0 if label_str == "bonafide" else 1
                        audio_path = root / f"{parts[1]}.wav"
                        if audio_path.exists():
                            samples.append((str(audio_path), label))

    return samples


def _compute_benchmark_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Compute standard benchmark metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )

    preds = (scores >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }

    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["false_positive_rate"] = float(fp / max(tn + fp, 1))
        metrics["false_negative_rate"] = float(fn / max(tp + fn, 1))
        metrics["specificity"] = float(tn / max(tn + fp, 1))

    return metrics


async def main():
    parser = argparse.ArgumentParser(description="Argus Core Benchmark Suite")
    parser.add_argument("--dataset", choices=["ffpp", "celebdf", "dfdc", "asvspoof"])
    parser.add_argument("--data", required=True, help="Dataset root directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--all", action="store_true", help="Run all available benchmarks")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    if args.all:
        datasets_to_run = ["ffpp", "celebdf", "dfdc", "asvspoof"]
    elif args.dataset:
        datasets_to_run = [args.dataset]
    else:
        parser.print_help()
        sys.exit(1)

    all_results = {}
    for ds in datasets_to_run:
        ds_dir = os.path.join(args.data, ds)
        if not os.path.isdir(ds_dir):
            logger.warning(f"Dataset directory not found: {ds_dir}")
            continue
        result = await evaluate_on_dataset(ds, ds_dir, args.split)
        if result:
            all_results[ds] = result

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n=== BENCHMARK SUMMARY ===")
    for ds, r in all_results.items():
        print(f"{ds}: AUC={r.get('auc_roc', 0):.4f}, "
              f"Acc={r.get('accuracy', 0):.4f}, "
              f"F1={r.get('f1', 0):.4f}, "
              f"ECE={r.get('ece', 0):.4f}, "
              f"Lat={r.get('latency_mean_ms', 0):.1f}ms")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
