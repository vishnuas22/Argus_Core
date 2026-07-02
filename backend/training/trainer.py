"""
Argus Core v2 - UMFT Trainer
==============================
Training loop and management for the Unified Multimodal
Forensic Transformer.

Features:
    - Mixed precision training (AMP)
    - Gradient accumulation for effective larger batch sizes
    - Cosine annealing learning rate schedule with warmup
    - Best-model checkpointing based on validation AUC
    - Logging to console (MLflow/WandB hooks available)
    - Multi-task loss optimization
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            scale = self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            import math
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self.min_lr + (self.base_lrs[i] - self.min_lr) * scale

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class UMFTTrainer:
    """
    Trainer for the UMFT deepfake detection model.

    Handles the complete training lifecycle:
        1. Forward pass through UMFT model
        2. Multi-task loss computation
        3. Mixed-precision backward pass
        4. Gradient accumulation and optimizer step
        5. Validation loop with metric tracking
        6. Checkpointing and logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 30,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.05,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_every: int = 50,
        device: str = "auto",
    ):
        """
        Initialize trainer.

        Args:
            model: UMFT model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            epochs: Number of training epochs
            gradient_accumulation_steps: Steps to accumulate before optimizer step
            warmup_ratio: Fraction of total steps for warmup
            use_amp: Use automatic mixed precision
            checkpoint_dir: Directory for model checkpoints
            log_every: Log metrics every N steps
            device: Device to train on ('auto', 'cuda', 'cpu', 'mps')
        """
        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp and self.device.type == "cuda"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every

        # Optimizer: AdamW with separate LR for encoders
        encoder_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": learning_rate * 0.1},  # Lower LR for encoders
            {"params": other_params, "lr": learning_rate},
        ], weight_decay=weight_decay)

        # Learning rate schedule
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, total_steps
        )

        # Loss functions (imported lazily to avoid circular imports)
        from training.loss_functions import BinaryFocalLoss, AudioVisualContrastiveLoss, LipSyncLoss, MultiTaskLoss
        self.focal_loss = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        self.contrastive_loss = AudioVisualContrastiveLoss(temperature=0.07)
        self.lip_sync_loss = LipSyncLoss(per_frame_weight=0.3)
        self.multi_task_loss = MultiTaskLoss(num_tasks=3).to(self.device)

        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.global_step = 0
        self.best_val_auc = 0.0

        # RL curriculum controller (extracted from augmentation pipeline)
        self._curriculum_controller = self._extract_controller(train_loader)

        logger.info(
            f"Trainer initialized: device={self.device}, "
            f"epochs={epochs}, batch_size={train_loader.batch_size}, "
            f"grad_accum={gradient_accumulation_steps}, "
            f"effective_batch={train_loader.batch_size * gradient_accumulation_steps}, "
            f"total_steps={total_steps}, warmup={warmup_steps}"
        )

    def _extract_controller(self, loader):
        """Extract RL curriculum controller from augmentation pipeline, if any."""
        if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'transform'):
            pipeline = loader.dataset.transform
            if hasattr(pipeline, 'get_controller'):
                return pipeline.get_controller()
        return None

    def _step_controller(self, epoch: int, val_auc: float = None, train_loss: float = None):
        """Advance RL curriculum controller if present."""
        if self._curriculum_controller is not None:
            info = self._curriculum_controller.step(
                epoch=epoch,
                max_epochs=self.epochs,
                val_auc=val_auc,
                train_loss=train_loss,
            )
            if val_auc is not None:
                logger.info(
                    f"  RL-Controller: reward={info.get('reward', 0):.3f}, "
                    f"temp={info.get('temperature', 1.0):.2f}, "
                    f"severity={info.get('severity_mult', 1.0):.2f}"
                )

    def train(self) -> Dict[str, float]:
        """
        Run the full training loop.

        Returns:
            Final metrics dict
        """
        logger.info("Starting training...")

        for epoch in range(self.epochs):
            train_metrics = self._train_epoch(epoch)
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} — "
                f"loss={train_metrics['loss']:.4f}, "
                f"lr={self.scheduler.get_lr():.2e}"
            )

            if self.val_loader:
                val_metrics = self._validate(epoch)
                logger.info(
                    f"  Val — loss={val_metrics['loss']:.4f}, "
                    f"auc={val_metrics.get('auc', 0):.4f}, "
                    f"acc={val_metrics.get('accuracy', 0):.4f}"
                )

                # Step RL curriculum controller (reward = val AUC change)
                self._step_controller(epoch, val_auc=val_metrics.get('auc'))

                # Checkpoint best model
                if val_metrics.get("auc", 0) > self.best_val_auc:
                    self.best_val_auc = val_metrics["auc"]
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                    logger.info(f"  ★ New best AUC: {self.best_val_auc:.4f}")
            else:
                # No val loader: use training loss trend as reward
                self._step_controller(epoch, train_loss=train_metrics.get('loss'))

            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, train_metrics)

        # Save final model
        self._save_checkpoint(self.epochs - 1, train_metrics, is_final=True)

        return {"best_val_auc": self.best_val_auc}

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        # Propagate epoch to curriculum degradation pipeline if present
        if hasattr(self.train_loader.dataset, 'transform') and hasattr(self.train_loader.dataset.transform, 'set_epoch'):
            self.train_loader.dataset.transform.set_epoch(epoch)

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            with autocast(enabled=self.use_amp):
                output = self.model(
                    frames=batch.get("frames"),
                    waveform=batch.get("waveform"),
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                )

                # Compute losses
                losses = {}
                labels = batch["label"].unsqueeze(1)

                # Classification loss (focal)
                losses["classification"] = self.focal_loss(output.logit, labels)

                # Lip-sync loss (if available)
                if output.lip_sync_score is not None and batch.get("waveform") is not None:
                    losses["lip_sync"] = self.lip_sync_loss(
                        output.lip_sync_score, output.lip_sync_per_frame, batch["label"]
                    )

                # Multi-task weighted loss
                loss = self.multi_task_loss(losses) / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (every N accumulation steps)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % self.log_every == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"  Step {batch_idx + 1}/{len(self.train_loader)} — "
                    f"loss={avg_loss:.4f}, lr={self.scheduler.get_lr():.2e}"
                )

        return {"loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation loop."""
        self.model.eval()
        all_probs = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            output = self.model(
                frames=batch.get("frames"),
                waveform=batch.get("waveform"),
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
            )

            labels = batch["label"].unsqueeze(1)
            loss = self.focal_loss(output.logit, labels)
            total_loss += loss.item()
            num_batches += 1

            all_probs.append(output.fake_probability.cpu())
            all_labels.append(labels.cpu())

        probs = torch.cat(all_probs).squeeze()
        labels = torch.cat(all_labels).squeeze()

        # Compute metrics
        metrics = {"loss": total_loss / max(num_batches, 1)}

        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            preds = (probs > 0.5).long()
            metrics["auc"] = roc_auc_score(labels.numpy(), probs.numpy())
            metrics["accuracy"] = accuracy_score(labels.numpy(), preds.numpy())
        except Exception:
            metrics["auc"] = 0.0
            metrics["accuracy"] = (probs.round() == labels).float().mean().item()

        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_val_auc": self.best_val_auc,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        elif is_final:
            path = self.checkpoint_dir / "final_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"

        torch.save(checkpoint, path)
        logger.info(f"  Checkpoint saved: {path}")

        # Also save model in from_pretrained format
        if is_best:
            pretrained_dir = self.checkpoint_dir / "best_pretrained"
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(str(pretrained_dir))

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}, "
            f"best_auc={self.best_val_auc:.4f}"
        )
