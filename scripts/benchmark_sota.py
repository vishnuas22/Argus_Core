#!/usr/bin/env python3
"""
Argus Core - SOTA Benchmark Harness (Iteration 1.5)
====================================================
Runs the Argus SOTA detector ensemble against standard deepfake
benchmark test sets and reports:
  - Image: AUC, Accuracy, EER on Celeb-DF v2 / FaceForensics++ test
  - Audio: EER, Accuracy, t-DCF on ASVspoof 2019 LA eval
  - Video: AUC, Accuracy on DFDC / FF++ test

Usage:
  # Benchmark image detectors on Celeb-DF v2 test set
  python scripts/benchmark_sota.py \\
      --modality image \\
      --test-set celebdf_v2 \\
      --test-root /data/Celeb-DF_v2/Test \\
      --output /tmp/bench_image_celebdf.json

  # Benchmark audio detectors on ASVspoof 2019 LA eval
  python scripts/benchmark_sota.py \\
      --modality audio \\
      --test-set asvspoof2019_la \\
      --test-root /data/ASVspoof2019_LA_eval \\
      --output /tmp/bench_audio_asvspoof.json

  # Benchmark video detectors on FF++ test set
  python scripts/benchmark_sota.py \\
      --modality video \\
      --test-set faceforensics \\
      --test-root /data/FF++/test \\
      --output /tmp/bench_video_ffpp.json

Outputs a JSON file with per-detector and ensemble metrics.

References:
- AUC: standard ROC AUC.
- EER: equal error rate, intersection of FPR and FNR curves.
- t-DCF: ASVspoof 2019 minimum normalized detection cost function.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure backend is importable when run from the host
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


# =====================================================================
# Metrics
# =====================================================================

def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Standard ROC AUC."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probs))
    except Exception:
        # Manual computation fallback
        if len(np.unique(labels)) < 2:
            return 0.5
        order = np.argsort(-probs)
        sorted_labels = labels[order]
        tps = np.cumsum(sorted_labels == 1)
        fps = np.cumsum(sorted_labels == 0)
        tpr = tps / max(tps[-1], 1)
        fpr = fps / max(fps[-1], 1)
        # Trapezoidal AUC
        return float(np.trapz(tpr, fpr))


def compute_eer(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """Equal Error Rate and its threshold."""
    if len(np.unique(labels)) < 2:
        return 0.5, 0.5
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, probs)
        fnr = 1.0 - tpr
        # Find threshold where FPR ≈ FNR
        idx = int(np.argmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        return eer, float(thresholds[idx])
    except Exception:
        return 0.5, 0.5


def compute_accuracy(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(int)
    return float(np.mean(preds == labels))


def compute_min_tdcf(
    labels: np.ndarray,
    probs: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """ASVspoof 2019 minimum normalized t-DCF."""
    if len(np.unique(labels)) < 2:
        return 1.0
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, probs)
        fnr = 1.0 - tpr
        # ASVspoof 2019 t-DCF formula
        c_def = c_miss * p_target + c_fa * (1 - p_target)
        c_det = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        c_norm = c_det / c_def
        return float(np.min(c_norm))
    except Exception:
        return 1.0


# =====================================================================
# Test set loaders
# =====================================================================

def load_image_test_set(test_root: str, test_set: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load image test set. Returns (images, labels)."""
    from PIL import Image
    root = Path(test_root)
    real_dir = root / "real"
    fake_dir = root / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        # Try alternative layouts
        if test_set == "celebdf_v2":
            real_dir = root / "Celeb-real"
            fake_dir = root / "Celeb-synthesis"
        elif test_set == "faceforensics":
            real_dir = root / "original"
            fake_dir = root / "manipulated"

    images: List[np.ndarray] = []
    labels: List[int] = []

    if real_dir.exists():
        for p in sorted(real_dir.glob("*.jpg")) + sorted(real_dir.glob("*.png")):
            try:
                img = np.array(Image.open(p).convert("RGB"))
                images.append(img)
                labels.append(0)
            except Exception:
                continue
    if fake_dir.exists():
        for p in sorted(fake_dir.glob("*.jpg")) + sorted(fake_dir.glob("*.png")):
            try:
                img = np.array(Image.open(p).convert("RGB"))
                images.append(img)
                labels.append(1)
            except Exception:
                continue

    if not images:
        raise RuntimeError(
            f"Test set {test_set}: no images found under {test_root}. "
            "Expected layout: <root>/real/*.jpg, <root>/fake/*.jpg. "
            "See TRAINING.md for setup."
        )
    return images, np.array(labels)


def load_audio_test_set(test_root: str, test_set: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load audio test set. Returns (waveforms, labels)."""
    import librosa
    root = Path(test_root)

    if test_set == "asvspoof2019_la":
        # Expected: <root>/ASVspoof2019_LA_eval/flac/*.flac
        #           <root>/ASVspoof2019_LA_eval/ASVspoof2019.LA.evalcmc.txt
        flac_dir = root / "ASVspoof2019_LA_eval" / "flac"
        protocol = root / "ASVspoof2019_LA_eval" / "ASVspoof2019.LA.evalcm.txt"
        if not flac_dir.exists():
            flac_dir = root / "flac"
            protocol = root / "protocol.txt"
    else:
        flac_dir = root / "real_audio"
        protocol = None

    waveforms: List[np.ndarray] = []
    labels: List[int] = []

    if protocol and protocol.exists():
        with open(protocol, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                file_name = parts[1]
                label = 0 if parts[4] == "bonafide" else 1
                audio_path = flac_dir / f"{file_name}.flac"
                if audio_path.exists():
                    try:
                        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
                        waveforms.append(wav)
                        labels.append(label)
                    except Exception:
                        continue
    else:
        # Layout: <root>/real/*.wav, <root>/fake/*.wav
        for p in sorted((root / "real").glob("*.wav")) if (root / "real").exists() else []:
            try:
                wav, _ = librosa.load(p, sr=16000, mono=True)
                waveforms.append(wav)
                labels.append(0)
            except Exception:
                continue
        for p in sorted((root / "fake").glob("*.wav")) if (root / "fake").exists() else []:
            try:
                wav, _ = librosa.load(p, sr=16000, mono=True)
                waveforms.append(wav)
                labels.append(1)
            except Exception:
                continue

    if not waveforms:
        raise RuntimeError(
            f"Test set {test_set}: no audio found under {test_root}. "
            "See TRAINING.md for ASVspoof 2019 setup."
        )
    return waveforms, np.array(labels)


def load_video_test_set(test_root: str, test_set: str) -> Tuple[List[List[np.ndarray]], np.ndarray]:
    """Load video test set. Returns (frame_sequences, labels)."""
    import cv2
    root = Path(test_root)
    real_dir = root / "real"
    fake_dir = root / "fake"

    sequences: List[List[np.ndarray]] = []
    labels: List[int] = []

    def _extract_frames(video_path: Path, num_frames: int = 16) -> Optional[List[np.ndarray]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        idx = np.linspace(0, total - 1, num_frames).astype(int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        return frames

    if real_dir.exists():
        for p in sorted(real_dir.glob("*.mp4")) + sorted(real_dir.glob("*.avi")):
            frames = _extract_frames(p)
            if frames:
                sequences.append(frames)
                labels.append(0)
    if fake_dir.exists():
        for p in sorted(fake_dir.glob("*.mp4")) + sorted(fake_dir.glob("*.avi")):
            frames = _extract_frames(p)
            if frames:
                sequences.append(frames)
                labels.append(1)

    if not sequences:
        raise RuntimeError(
            f"Test set {test_set}: no videos found under {test_root}. "
            "Expected layout: <root>/real/*.mp4, <root>/fake/*.mp4."
        )
    return sequences, np.array(labels)


# =====================================================================
# Detector runners
# =====================================================================

async def run_image_detectors(images: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Run all image SOTA detectors. Returns dict[name] = probs array."""
    from detectors import CLIPLoRAImageDetector, DINOv2ImageDetector
    results: Dict[str, np.ndarray] = {}

    # CLIP + LoRA (with optional fine-tuned head)
    clip = CLIPLoRAImageDetector(
        fine_tuned_head_repo=os.environ.get("ARGUS_CLIP_FINE_TUNED_HEAD", ""),
    )
    scores = []
    for img in images:
        r = await clip.detect(img)
        scores.append(r.score)
    results["clip_lora"] = np.array(scores)

    # DINOv2
    dino = DINOv2ImageDetector()
    scores = []
    for img in images:
        r = await dino.detect(img)
        scores.append(r.score)
    results["dinov2"] = np.array(scores)

    return results


async def run_audio_detectors(waveforms: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Run all audio SOTA detectors. Returns dict[name] = probs array."""
    from detectors import AASIST3AudioDetector, Wav2Vec2XLSRMoELoRADetector
    results: Dict[str, np.ndarray] = {}

    aasist = AASIST3AudioDetector()
    scores = []
    for wav in waveforms:
        r = await aasist.detect(wav, sample_rate=16000)
        scores.append(r.score)
    results["aasist3"] = np.array(scores)

    wav2vec = Wav2Vec2XLSRMoELoRADetector()
    scores = []
    for wav in waveforms:
        r = await wav2vec.detect(wav, sample_rate=16000)
        scores.append(r.score)
    results["wav2vec2_xls_r"] = np.array(scores)

    return results


async def run_video_detectors(sequences: List[List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Run all video SOTA detectors. Returns dict[name] = probs array."""
    from detectors import VideoMAEDetector, AltFreeVideoDetector
    results: Dict[str, np.ndarray] = {}

    vmae = VideoMAEDetector()
    scores = []
    for frames in sequences:
        r = await vmae.detect(frames)
        scores.append(r.score)
    results["videomae"] = np.array(scores)

    alt = AltFreeVideoDetector()
    scores = []
    for frames in sequences:
        r = await alt.detect(frames)
        scores.append(r.score)
    results["altfree"] = np.array(scores)

    return results


# =====================================================================
# Main
# =====================================================================

async def main_async(args):
    print(f"=== Argus SOTA Benchmark ===")
    print(f"  Modality:  {args.modality}")
    print(f"  Test set:  {args.test_set}")
    print(f"  Test root: {args.test_root}")
    print(f"  Output:    {args.output}")

    # Load test set
    if args.modality == "image":
        samples, labels = load_image_test_set(args.test_root, args.test_set)
        detector_scores = await run_image_detectors(samples)
    elif args.modality == "audio":
        samples, labels = load_audio_test_set(args.test_root, args.test_set)
        detector_scores = await run_audio_detectors(samples)
    elif args.modality == "video":
        samples, labels = load_video_test_set(args.test_root, args.test_set)
        detector_scores = await run_video_detectors(samples)
    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    print(f"\nLoaded {len(labels)} samples ({int((labels == 0).sum())} real, "
          f"{int((labels == 1).sum())} fake)")

    # Compute per-detector metrics
    print("\n=== Per-Detector Metrics ===")
    per_detector: Dict[str, Dict[str, float]] = {}
    for name, probs in detector_scores.items():
        auc = compute_auc(labels, probs)
        eer, eer_thresh = compute_eer(labels, probs)
        acc = compute_accuracy(labels, probs, threshold=0.5)
        if args.modality == "audio":
            min_tdcf = compute_min_tdcf(labels, probs)
        else:
            min_tdcf = None
        print(f"  {name:25s} AUC={auc:.4f}  EER={eer:.4f}  Acc={acc:.4f}"
              + (f"  min-tDCF={min_tdcf:.4f}" if min_tdcf is not None else ""))
        per_detector[name] = {
            "auc": auc, "eer": eer, "eer_threshold": eer_thresh,
            "accuracy": acc,
            **({"min_tdcf": min_tdcf} if min_tdcf is not None else {}),
        }

    # Ensemble (simple mean + logit-space mean)
    print("\n=== Ensemble Metrics ===")
    all_probs = np.stack(list(detector_scores.values()), axis=0)  # (n_detectors, n_samples)
    mean_probs = all_probs.mean(axis=0)

    auc = compute_auc(labels, mean_probs)
    eer, eer_thresh = compute_eer(labels, mean_probs)
    acc = compute_accuracy(labels, mean_probs, threshold=0.5)
    print(f"  {'mean_ensemble':25s} AUC={auc:.4f}  EER={eer:.4f}  Acc={acc:.4f}")
    ensemble_metrics = {
        "mean_ensemble": {
            "auc": auc, "eer": eer, "eer_threshold": eer_thresh, "accuracy": acc,
        }
    }

    # DiversityEnsemble (per-sample combiner)
    try:
        from detectors import combine_detector_results
        from detectors.base import DetectionResult
        combined_probs = []
        for i in range(len(labels)):
            results = [
                DetectionResult(score=float(all_probs[d, i]), confidence=0.7,
                                model_name=name, backend="pytorch")
                for d, name in enumerate(detector_scores.keys())
            ]
            fused = combine_detector_results(results)
            combined_probs.append(fused.score)
        combined_probs = np.array(combined_probs)
        auc = compute_auc(labels, combined_probs)
        eer, eer_thresh = compute_eer(labels, combined_probs)
        acc = compute_accuracy(labels, combined_probs, threshold=0.5)
        print(f"  {'diversity_ensemble':25s} AUC={auc:.4f}  EER={eer:.4f}  Acc={acc:.4f}")
        ensemble_metrics["diversity_ensemble"] = {
            "auc": auc, "eer": eer, "eer_threshold": eer_thresh, "accuracy": acc,
        }
    except Exception as e:
        print(f"  diversity_ensemble: failed ({e})")

    # Write JSON output
    output = {
        "modality": args.modality,
        "test_set": args.test_set,
        "num_samples": int(len(labels)),
        "num_real": int((labels == 0).sum()),
        "num_fake": int((labels == 1).sum()),
        "per_detector": per_detector,
        "ensemble": ensemble_metrics,
    }
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote results to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Argus SOTA detectors")
    parser.add_argument("--modality", required=True, choices=["image", "audio", "video"])
    parser.add_argument("--test-set", required=True,
                        choices=["celebdf_v2", "faceforensics", "asvspoof2019_la", "dfdc"])
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
