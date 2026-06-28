"""
Argus Core - Training Entry Point
==================================
CLI for training the UMFT deepfake detection model.

Usage:
    python scripts/train.py --data-dir /path/to/datasets --epochs 30

Requires:
    - PyTorch with CUDA or MPS
    - A UMFT-compatible model class with:
        .logit, .fake_probability, .lip_sync_score, .lip_sync_per_frame attributes
    - Dataset organized as FaceForensics++ / DFDC / AV-Deepfake1M structure
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train UMFT deepfake detection model"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory containing training datasets"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-GPU batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="AdamW weight decay"
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--num-frames", type=int, default=16,
        help="Frames sampled per video"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Training device"
    )
    parser.add_argument(
        "--curriculum-epochs", type=int, default=30,
        help="Epochs for degradation curriculum ramp-up"
    )
    parser.add_argument(
        "--rl-curriculum", action="store_true",
        help="Enable RL-based adaptive curriculum controller"
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Optional dataset manifest file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from training import create_dataloader, UMFTTrainer

    curriculum_controller = None
    if args.rl_curriculum:
        from training import RLCurriculumController
        curriculum_controller = RLCurriculumController()

    train_loader = create_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="train",
        num_frames=args.num_frames,
        manifest_file=args.manifest,
        use_degradation_curriculum=True,
        curriculum_epochs=args.curriculum_epochs,
        curriculum_controller=curriculum_controller,
    )

    val_loader = create_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="val",
        num_frames=args.num_frames,
        manifest_file=args.manifest,
        use_degradation_curriculum=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    try:
        from core.cross_attention_fusion import UMFTConfig, CrossAttentionEngine
        model = CrossAttentionEngine(UMFTConfig())
    except ImportError:
        print("ERROR: No UMFT model class found.")
        print("The training pipeline expects a torch.nn.Module with:")
        print("  - .logit (raw logit)")
        print("  - .fake_probability (sigmoid output)")
        print("  - .lip_sync_score, .lip_sync_per_frame (optional)")
        print()
        print("Implement your model and instantiate it here.")
        sys.exit(1)

    trainer = UMFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    print(f"Training on {trainer.device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    metrics = trainer.train()
    print(f"Training complete. Best val AUC: {metrics.get('best_val_auc', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
