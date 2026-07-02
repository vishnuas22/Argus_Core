"""
Argus Core v2 - Evaluation Metrics
=====================================
Comprehensive evaluation suite for deepfake detection models.

Metrics:
    - AUC-ROC: Area under ROC curve (primary metric)
    - EER: Equal Error Rate (FAR = FRR threshold)
    - Accuracy, Precision, Recall, F1
    - Calibration: Expected Calibration Error (ECE)
    - Cross-dataset generalization matrix
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


def compute_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute standard detection metrics.

    Args:
        probabilities: [N] predicted probabilities
        labels: [N] ground truth labels (0=real, 1=fake)
        threshold: Decision threshold

    Returns:
        Dict of metric name → value
    """
    probs = probabilities.cpu().numpy()
    targets = labels.cpu().numpy().astype(int)
    preds = (probs >= threshold).astype(int)

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = float(np.mean(preds == targets))

    # Per-class metrics
    tp = float(np.sum((preds == 1) & (targets == 1)))
    fp = float(np.sum((preds == 1) & (targets == 0)))
    tn = float(np.sum((preds == 0) & (targets == 0)))
    fn = float(np.sum((preds == 0) & (targets == 1)))

    metrics["precision"] = tp / max(tp + fp, 1)
    metrics["recall"] = tp / max(tp + fn, 1)
    metrics["f1"] = (
        2 * metrics["precision"] * metrics["recall"]
        / max(metrics["precision"] + metrics["recall"], 1e-8)
    )
    metrics["specificity"] = tn / max(tn + fp, 1)

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        metrics["auc_roc"] = float(roc_auc_score(targets, probs))
    except Exception:
        metrics["auc_roc"] = _compute_auc_manual(probs, targets)

    # EER
    metrics["eer"] = _compute_eer(probs, targets)

    # ECE (calibration)
    metrics["ece"] = _compute_ece(probs, targets, n_bins=10)

    return metrics


def _compute_auc_manual(probs: np.ndarray, targets: np.ndarray) -> float:
    """Manual AUC computation using trapezoidal rule."""
    sorted_indices = np.argsort(-probs)
    sorted_targets = targets[sorted_indices]

    num_pos = np.sum(targets == 1)
    num_neg = np.sum(targets == 0)

    if num_pos == 0 or num_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for t in sorted_targets:
        if t == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / num_pos)
        fpr_list.append(fp / num_neg)

    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return float(auc)


def _compute_eer(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute Equal Error Rate."""
    thresholds = np.linspace(0, 1, 1000)
    min_diff = float("inf")
    eer = 0.5

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)

        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))
        num_neg = np.sum(targets == 0)
        num_pos = np.sum(targets == 1)

        far = fp / max(num_neg, 1)
        frr = fn / max(num_pos, 1)

        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2

    return float(eer)


def _compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.

    Measures how well predicted probabilities match actual frequencies.
    Lower is better; ECE < 0.05 indicates good calibration.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if np.sum(mask) == 0:
            continue

        bin_confidence = np.mean(probs[mask])
        bin_accuracy = np.mean(targets[mask])
        bin_count = np.sum(mask)

        ece += (bin_count / len(probs)) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: UMFT model
        dataloader: Evaluation data loader
        device: Device to run on

    Returns:
        Complete metrics dictionary
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            output = model(
                frames=batch.get("frames"),
                waveform=batch.get("waveform"),
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
            )

            all_probs.append(output.fake_probability.cpu().squeeze())
            all_labels.append(batch["label"].cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    metrics = compute_metrics(probs, labels)

    logger.info(
        f"Evaluation — AUC={metrics['auc_roc']:.4f}, "
        f"EER={metrics['eer']:.4f}, "
        f"Acc={metrics['accuracy']:.4f}, "
        f"F1={metrics['f1']:.4f}, "
        f"ECE={metrics['ece']:.4f}"
    )

    return metrics


def cross_dataset_evaluation(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model across multiple datasets for generalization analysis.

    Args:
        model: UMFT model
        dataloaders: Dict mapping dataset name → DataLoader
        device: Device

    Returns:
        Nested dict: dataset_name → metrics
    """
    results = {}

    for name, loader in dataloaders.items():
        logger.info(f"Evaluating on {name}...")
        metrics = evaluate_model(model, loader, device)
        results[name] = metrics

    # Log cross-dataset summary
    logger.info("\n=== Cross-Dataset Generalization ===")
    logger.info(f"{'Dataset':<20} {'AUC':>8} {'EER':>8} {'Acc':>8} {'F1':>8}")
    logger.info("-" * 52)
    for name, metrics in results.items():
        logger.info(
            f"{name:<20} "
            f"{metrics['auc_roc']:>8.4f} "
            f"{metrics['eer']:>8.4f} "
            f"{metrics['accuracy']:>8.4f} "
            f"{metrics['f1']:>8.4f}"
        )

    return results
