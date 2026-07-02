#!/usr/bin/env python3
"""
Argus Core - Calibration Fitter (Iteration 2)
===============================================
Fits Temperature Scaling + Conformal RAPS + Drift Reference from a
held-out calibration set.

Usage:
  python scripts/fit_calibration.py \\
      --modality image \\
      --calibration-json /data/calibration_image.json \\
      --output-dir /models/calibration

  python scripts/fit_calibration.py \\
      --modality audio \\
      --calibration-json /data/calibration_audio.json \\
      --output-dir /models/calibration

Calibration JSON format:
  {
    "logits": [[2.1, -1.3], [1.5, 0.5], ...],   // (N, C) raw logits
    "labels": [0, 0, 1, ...],                    // (N,) true labels
    "probs": [[0.9, 0.1], ...],                  // (N, C) probabilities (optional)
    "embeddings": [[...], ...]                    // (N, D) embeddings for drift ref
  }

If "logits" is missing, the script derives pseudo-logits from "probs"
via logit(p) = log(p / (1-p)) for binary case.

Outputs:
  /models/calibration/temperature_scaler.json
  /models/calibration/conformal_raps.json
  /models/calibration/drift_reference.{json,npz}
  /models/calibration/audit_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


def load_calibration_data(path: str) -> dict:
    """Load calibration data from JSON."""
    with open(path, "r") as fh:
        data = json.load(fh)
    return data


def derive_logits_from_probs(probs: np.ndarray) -> np.ndarray:
    """Derive pseudo-logits from probabilities via logit transform."""
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    if probs.ndim == 1:
        # Binary: P(class=1) → logit
        return np.stack([np.log(1 - probs), np.log(probs)], axis=-1)
    else:
        # Multi-class: log(p)
        return np.log(probs)


async def main_async(args):
    print(f"=== Argus Calibration Fitter ===")
    print(f"  Modality:       {args.modality}")
    print(f"  Calibration set: {args.calibration_json}")
    print(f"  Output dir:     {args.output_dir}")

    # Load data
    data = load_calibration_data(args.calibration_json)
    labels = np.array(data["labels"], dtype=np.int64)
    print(f"\nLoaded {len(labels)} calibration samples")

    # Get logits
    if "logits" in data:
        logits = np.array(data["logits"], dtype=np.float64)
    elif "probs" in data:
        logits = derive_logits_from_probs(np.array(data["probs"], dtype=np.float64))
    else:
        print("ERROR: calibration JSON must contain 'logits' or 'probs'")
        sys.exit(1)

    # Get probs (for conformal)
    if "probs" in data:
        probs = np.array(data["probs"], dtype=np.float64)
    else:
        # Compute from logits
        scaled = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(scaled)
        probs = exp / exp.sum(axis=-1, keepdims=True)

    # Get embeddings (for drift reference) — optional
    embeddings = None
    if "embeddings" in data:
        embeddings = np.array(data["embeddings"], dtype=np.float64)
        print(f"  Embeddings: {embeddings.shape}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Fit Temperature Scaling
    print("\n=== Fitting Temperature Scaling ===")
    from calibration.temperature_scaling import fit_temperature_scaler, TemperatureScaler
    scaler = fit_temperature_scaler(logits, labels)
    print(f"  T = {scaler.temperature:.4f}")
    print(f"  N = {scaler.num_samples}")
    scaler_path = os.path.join(args.output_dir, "temperature_scaler.json")
    scaler.save(scaler_path)

    # 2. Run calibration audit (before + after TS)
    print("\n=== Calibration Audit ===")
    from calibration.calibration_audit import run_calibration_audit
    # Use binary P(class=1) for the audit
    if probs.shape[-1] == 2:
        probs_binary = probs[:, 1]
    else:
        probs_binary = probs.max(axis=-1)  # multi-class: use max prob

    audit_before = run_calibration_audit(probs_binary, labels)
    print(f"  BEFORE TS: ECE={audit_before.ece_15:.4f}, Brier={audit_before.brier_score:.4f}, NLL={audit_before.nll:.4f}")

    # Apply TS and re-audit
    calibrated_probs = scaler.calibrate_logits(logits)
    if calibrated_probs.shape[-1] == 2:
        cal_binary = calibrated_probs[:, 1]
    else:
        cal_binary = calibrated_probs.max(axis=-1)
    audit_after = run_calibration_audit(cal_binary, labels)
    print(f"  AFTER TS:  ECE={audit_after.ece_15:.4f}, Brier={audit_after.brier_score:.4f}, NLL={audit_after.nll:.4f}")

    audit_path = os.path.join(args.output_dir, "audit_report.json")
    with open(audit_path, "w") as fh:
        json.dump({
            "modality": args.modality,
            "num_samples": len(labels),
            "before_ts": audit_before.to_dict(),
            "after_ts": audit_after.to_dict(),
            "temperature": scaler.temperature,
        }, fh, indent=2)
    print(f"  Audit saved to {audit_path}")

    # 3. Fit Conformal RAPS
    print("\n=== Fitting Conformal RAPS ===")
    from calibration.conformal import fit_conformal_raps
    alpha = float(args.alpha)
    # Use the calibrated probs for conformal fitting
    conformal = fit_conformal_raps(
        calibrated_probs, labels, alpha=alpha,
        lambda_raps=0.0, k_raps=1,
    )
    print(f"  q_hat = {conformal.q_hat:.4f}")
    print(f"  alpha = {conformal.alpha}")
    conformal_path = os.path.join(args.output_dir, "conformal_raps.json")
    conformal.save(conformal_path)

    # 4. Build drift reference (if embeddings provided)
    if embeddings is not None:
        print("\n=== Building Drift Reference ===")
        from monitoring.reference_store import ReferenceStore
        store = ReferenceStore(max_reference_size=1000)
        store.build_from_embeddings(embeddings, modality=args.modality)
        ref_path = os.path.join(args.output_dir, "drift_reference")
        store.save(ref_path)
        print(f"  Reference: {store.num_samples} samples, modality={store.modality}")

    print(f"\n=== Done. Artifacts in {args.output_dir} ===")


def main():
    parser = argparse.ArgumentParser(description="Fit calibration artifacts for Argus")
    parser.add_argument("--modality", required=True, choices=["image", "audio", "video"])
    parser.add_argument("--calibration-json", required=True,
                        help="JSON with 'logits'/'probs', 'labels', optional 'embeddings'")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write calibration artifacts")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Conformal miscoverage rate (default 0.10 = 90% coverage)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
