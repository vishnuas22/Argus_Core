"""
Argus Core v2 - Training Package
==================================
Complete training infrastructure for the UMFT model.
"""

from training.trainer import UMFTTrainer
from training.dataset_loader import create_dataloader, MultiModalDeepfakeDataset
from training.augmentation import DegradationPipeline, AudioDegradationPipeline
from training.evaluation import compute_metrics, evaluate_model, cross_dataset_evaluation
from training.loss_functions import BinaryFocalLoss, AudioVisualContrastiveLoss, LipSyncLoss, MultiTaskLoss
from training.curriculum_controller import RLCurriculumController

__all__ = [
    "UMFTTrainer",
    "create_dataloader",
    "MultiModalDeepfakeDataset",
    "DegradationPipeline",
    "AudioDegradationPipeline",
    "RLCurriculumController",
    "compute_metrics",
    "evaluate_model",
    "cross_dataset_evaluation",
    "BinaryFocalLoss",
    "AudioVisualContrastiveLoss",
    "LipSyncLoss",
    "MultiTaskLoss",
]
