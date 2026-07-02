#!/usr/bin/env python3
"""
Argus Core - LoRA Adapter Training Pipeline (Iteration 1.5)
============================================================
Trains per-modality LoRA adapters for the SOTA detector backbones:
  - image  → CLIP ViT-B/16 + LoRA   (datasets: FF++, Celeb-DF v2)
  - image  → DINOv2 + MAC head      (datasets: FF++, Celeb-DF v2)
  - audio  → Wav2Vec2-XLS-R + LoRA  (dataset: ASVspoof 2019 LA)
  - video  → VideoMAE + LoRA        (datasets: FF++, DFDC)

Usage:
  # Image (CLIP + LoRA on FaceForensics++)
  python scripts/train_lora_adapters.py \\
      --modality image --backbone clip \\
      --dataset faceforensics \\
      --dataset-root /data/faceforensics \\
      --output-dir /models/clip_lora_image_adapter \\
      --epochs 10 --batch-size 32 --lr 1e-4

  # Audio (Wav2Vec2-XLS-R + LoRA on ASVspoof 2019 LA)
  python scripts/train_lora_adapters.py \\
      --modality audio --backbone wav2vec2_xls_r \\
      --dataset asvspoof2019 \\
      --dataset-root /data/asvspoof2019 \\
      --output-dir /models/wav2vec2_xls_r_moe_lora \\
      --epochs 20 --batch-size 8 --lr 5e-5

  # Video (VideoMAE + LoRA on FF++)
  python scripts/train_lora_adapters.py \\
      --modality video --backbone videomae \\
      --dataset faceforensics \\
      --dataset-root /data/faceforensics \\
      --output-dir /models/videomae_finetune \\
      --epochs 15 --batch-size 4 --lr 1e-4

Outputs (per modality):
  /models/<adapter_dir>/adapter_config.json   — LoRA hyperparams
  /models/<adapter_dir>/adapter_model.safetensors — LoRA weights
  /models/<adapter_dir>/classifier.pt         — trained linear head
  /models/<adapter_dir>/training_metrics.json — per-epoch metrics

References:
- LoRA: Hu et al., ICLR 2022. https://arxiv.org/abs/2106.09685
- ForAda (CLIP+LoRA for deepfake): CVPR 2025.
- Wav2Vec2-XLS-R: Babu et al., INTERSPEECH 2022.
- VideoMAE: Tong et al., NeurIPS 2022.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure backend is importable when run from the host
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class TrainConfig:
    # Modality + backbone
    modality: str = "image"           # image | audio | video
    backbone: str = "clip"            # clip | dinov2 | wav2vec2_xls_r | videomae
    # Dataset
    dataset: str = "faceforensics"    # faceforensics | celebdf | asvspoof2019 | dfdc
    dataset_root: str = "/data"
    # Output
    output_dir: str = "/models/adapter"
    # Training
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # Compute
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42
    # Logging
    log_every: int = 50
    save_every_n_epochs: int = 1


# =====================================================================
# Backbone loaders
# =====================================================================

def load_image_backbone(backbone: str, cache_dir: Optional[str] = None):
    """Load a frozen image backbone + its processor."""
    from transformers import AutoModel, AutoImageProcessor
    if backbone == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model_id = "openai/clip-vit-base-patch16"
        processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
        # We LoRA the vision model only
        backbone_model = model.vision_model
        hidden = model.config.projection_dim
    elif backbone == "dinov2":
        model_id = "facebook/dinov2-base"
        processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
        backbone_model = model
        hidden = model.config.hidden_size
    else:
        raise ValueError(f"Unknown image backbone: {backbone}")
    for p in backbone_model.parameters():
        p.requires_grad = False
    return backbone_model, processor, hidden


def load_audio_backbone(backbone: str, cache_dir: Optional[str] = None):
    """Load a frozen audio backbone + its processor."""
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    if backbone == "wav2vec2_xls_r":
        model_id = "facebook/wav2vec2-xls-r-300m"
    else:
        raise ValueError(f"Unknown audio backbone: {backbone}")
    processor = Wav2Vec2Processor.from_pretrained(model_id, cache_dir=cache_dir)
    model = Wav2Vec2Model.from_pretrained(model_id, cache_dir=cache_dir)
    for p in model.parameters():
        p.requires_grad = False
    return model, processor, model.config.hidden_size


def load_video_backbone(backbone: str, cache_dir: Optional[str] = None):
    """Load a frozen video backbone + its processor."""
    from transformers import VideoMAEModel, AutoImageProcessor
    if backbone == "videomae":
        model_id = "MCG-NJU/videomae-base"
    else:
        raise ValueError(f"Unknown video backbone: {backbone}")
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    except Exception:
        from transformers import VideoMAEImageProcessor
        processor = VideoMAEImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = VideoMAEModel.from_pretrained(model_id, cache_dir=cache_dir)
    for p in model.parameters():
        p.requires_grad = False
    return model, processor, model.config.hidden_size


# =====================================================================
# LoRA injection
# =====================================================================

def apply_lora(model: nn.Module, cfg: TrainConfig) -> nn.Module:
    """Apply LoRA adapters to the backbone."""
    from peft import LoraConfig, get_peft_model

    # LoRA target modules differ per backbone
    if cfg.backbone == "clip":
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    elif cfg.backbone == "dinov2":
        target_modules = ["query", "value", "key"]
    elif cfg.backbone == "wav2vec2_xls_r":
        target_modules = ["q_proj", "v_proj", "k_proj"]
    elif cfg.backbone == "videomae":
        target_modules = ["query", "value"]
    else:
        target_modules = ["query", "value"]

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
    )
    return get_peft_model(model, lora_config)


# =====================================================================
# Classifier head
# =====================================================================

class ClassifierHead(nn.Module):
    """Simple linear head on top of [CLS] / pooled features."""
    def __init__(self, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x)


# =====================================================================
# Dataset stubs (operators implement real loaders for their data)
# =====================================================================
#
# WHY STUBS: real deepfake datasets are 100s of GB and require
# accepted licenses (FF++ is commercial, Celeb-DF has a research
# license, ASVspoof has its own EULA). We cannot bundle them.
# Operators implement `_load_samples` to point at their local data.
# The structure below is the contract the trainer expects.

class DeepfakeDataset(Dataset):
    """
    Base class for deepfake training datasets.

    Subclasses MUST implement `_load_samples` returning a list of
    dicts with keys:
      - image:  file path        (modality=image)
      - audio:  file path        (modality=audio)
      - frames: list of paths    (modality=video)
      - label:  0 (real) | 1 (fake)
    """
    def __init__(self, cfg: TrainConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    def _load_samples(self):
        """Override in subclass."""
        raise NotImplementedError(
            f"Operators must implement _load_samples for dataset "
            f"{self.cfg.dataset}. See TRAINING.md."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        if self.cfg.modality == "image":
            return self._load_image(sample)
        elif self.cfg.modality == "audio":
            return self._load_audio(sample)
        elif self.cfg.modality == "video":
            return self._load_video(sample)
        raise ValueError(f"Unknown modality: {self.cfg.modality}")

    # ------------------------------------------------------------------
    def _load_image(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        from PIL import Image
        img = Image.open(sample["image"]).convert("RGB")
        return {"image": img, "label": sample["label"]}

    def _load_audio(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        import librosa
        wav, sr = librosa.load(sample["audio"], sr=16000, mono=True)
        return {"waveform": torch.from_numpy(wav).float(), "label": sample["label"]}

    def _load_video(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        from PIL import Image
        frames = []
        for p in sample["frames"][:16]:
            frames.append(Image.open(p).convert("RGB"))
        # Pad if fewer than 16
        while len(frames) < 16:
            frames.append(frames[-1])
        return {"frames": frames, "label": sample["label"]}


class FaceForensicsDataset(DeepfakeDataset):
    """FaceForensics++ dataset loader (operators supply paths)."""
    def _load_samples(self):
        root = Path(self.cfg.dataset_root)
        # Expected layout:
        #   <root>/real/<split>/*.png
        #   <root>/fake/<split>/*.png
        real_dir = root / "real" / self.split
        fake_dir = root / "fake" / self.split
        if not real_dir.exists() or not fake_dir.exists():
            logger.warning(
                "FaceForensics++ paths missing: %s, %s — "
                "creating empty dataset. See TRAINING.md for setup.",
                real_dir, fake_dir,
            )
            return
        for p in real_dir.glob("*.png"):
            self.samples.append({"image": str(p), "label": 0})
        for p in fake_dir.glob("*.png"):
            self.samples.append({"image": str(p), "label": 1})
        logger.info(
            "FaceForensics++ %s: %d real, %d fake",
            self.split,
            sum(1 for s in self.samples if s["label"] == 0),
            sum(1 for s in self.samples if s["label"] == 1),
        )


class CelebDFDataset(DeepfakeDataset):
    """Celeb-DF v2 dataset loader."""
    def _load_samples(self):
        root = Path(self.cfg.dataset_root)
        real_dir = root / "Celeb-real" / self.split
        fake_dir = root / "Celeb-synthesis" / self.split
        if not real_dir.exists() or not fake_dir.exists():
            logger.warning(
                "Celeb-DF paths missing: %s, %s — creating empty dataset. "
                "See TRAINING.md for setup.",
                real_dir, fake_dir,
            )
            return
        for p in real_dir.glob("*.jpg"):
            self.samples.append({"image": str(p), "label": 0})
        for p in fake_dir.glob("*.jpg"):
            self.samples.append({"image": str(p), "label": 1})


class ASVSpoof2019Dataset(DeepfakeDataset):
    """ASVspoof 2019 LA dataset loader."""
    def _load_samples(self):
        root = Path(self.cfg.dataset_root)
        # Expected layout (ASVspoof 2019 LA):
        #   <root>/LA/<split>/flac/*.flac
        #   <root>/LA/<split>/<split>.txt  (key, file, _, _, label)
        split_dir = root / "LA" / self.split
        if not split_dir.exists():
            logger.warning(
                "ASVspoof 2019 path missing: %s — creating empty dataset. "
                "See TRAINING.md for setup.",
                split_dir,
            )
            return
        protocol = split_dir / f"{self.split}.txt"
        if not protocol.exists():
            logger.warning("ASVspoof protocol file missing: %s", protocol)
            return
        audio_dir = split_dir / "flac"
        with open(protocol, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # ASVspoof protocol: speaker, file, _, _, label, _
                file_name = parts[1]
                label = 0 if parts[4] == "bonafide" else 1
                audio_path = audio_dir / f"{file_name}.flac"
                if audio_path.exists():
                    self.samples.append({"audio": str(audio_path), "label": label})
        logger.info(
            "ASVspoof 2019 LA %s: %d samples (%d bonafide, %d spoof)",
            self.split,
            len(self.samples),
            sum(1 for s in self.samples if s["label"] == 0),
            sum(1 for s in self.samples if s["label"] == 1),
        )


def get_dataset(cfg: TrainConfig, split: str) -> DeepfakeDataset:
    if cfg.dataset == "faceforensics":
        return FaceForensicsDataset(cfg, split=split)
    elif cfg.dataset == "celebdf":
        return CelebDFDataset(cfg, split=split)
    elif cfg.dataset == "asvspoof2019":
        return ASVSpoof2019Dataset(cfg, split=split)
    elif cfg.dataset == "dfdc":
        # DFDC has the same layout as FF++ in this loader
        return FaceForensicsDataset(cfg, split=split)
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


# =====================================================================
# Collators
# =====================================================================

class ImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs, labels


class AudioCollator:
    def __init__(self, processor, max_seconds: float = 5.0, sr: int = 16000):
        self.processor = processor
        self.max_samples = int(max_seconds * sr)

    def __call__(self, batch):
        wavs = []
        labels = []
        for b in batch:
            w = b["waveform"]
            if len(w) > self.max_samples:
                w = w[:self.max_samples]
            else:
                w = torch.nn.functional.pad(w, (0, self.max_samples - len(w)))
            wavs.append(w.numpy())
            labels.append(b["label"])
        labels = torch.tensor(labels, dtype=torch.long)
        inputs = self.processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs, labels


class VideoCollator:
    def __init__(self, processor, num_frames: int = 16):
        self.processor = processor
        self.num_frames = num_frames

    def __call__(self, batch):
        all_frames = []
        labels = []
        for b in batch:
            frames = b["frames"]
            # Sample 16 frames uniformly
            if len(frames) > self.num_frames:
                idx = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
                frames = [frames[i] for i in idx]
            all_frames.append(frames)
            labels.append(b["label"])
        labels = torch.tensor(labels, dtype=torch.long)
        inputs = self.processor(all_frames, return_tensors="pt")
        return inputs, labels


# =====================================================================
# Trainer
# =====================================================================

def train_one_epoch(
    model: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
) -> Dict[str, float]:
    """Train one epoch. Returns metrics dict."""
    model.train()
    head.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()

    for step, (inputs, labels) in enumerate(dataloader):
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
        labels = labels.to(cfg.device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(**inputs)
            # Extract [CLS] / pooled features
            if hasattr(outputs, "last_hidden_state"):
                feat = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, "pooler_output"):
                feat = outputs.pooler_output
            else:
                feat = outputs[0].mean(dim=1)
            logits = head(feat)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(filter(lambda p: p.requires_grad, model.parameters())) + list(head.parameters()),
                cfg.max_grad_norm,
            )
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if step > 0 and step % cfg.log_every == 0:
            logger.info(
                "epoch %d step %d/%d: loss=%.4f acc=%.4f",
                epoch, step, len(dataloader),
                total_loss / total_samples,
                total_correct / total_samples,
            )

    return {
        "epoch": epoch,
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    cfg: TrainConfig,
) -> Dict[str, float]:
    """Validate. Returns metrics dict including AUC."""
    model.eval()
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs = []
    all_labels = []
    loss_fn = nn.CrossEntropyLoss()

    for inputs, labels in dataloader:
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
        labels = labels.to(cfg.device)
        outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output"):
            feat = outputs.pooler_output
        else:
            feat = outputs[0].mean(dim=1)
        logits = head(feat)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    # Compute AUC if scikit-learn available
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        auc = 0.5

    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_accuracy": total_correct / max(total_samples, 1),
        "val_auc": auc,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for Argus SOTA detectors")
    parser.add_argument("--modality", required=True, choices=["image", "audio", "video"])
    parser.add_argument("--backbone", required=True,
                        choices=["clip", "dinov2", "wav2vec2_xls_r", "videomae"])
    parser.add_argument("--dataset", required=True,
                        choices=["faceforensics", "celebdf", "asvspoof2019", "dfdc"])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        modality=args.modality,
        backbone=args.backbone,
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Output dir
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    logger.info("=== Argus LoRA Training ===")
    logger.info("Config: %s", asdict(cfg))

    # ----- Load backbone -----
    if cfg.modality == "image":
        backbone_model, processor, hidden = load_image_backbone(cfg.backbone)
    elif cfg.modality == "audio":
        backbone_model, processor, hidden = load_audio_backbone(cfg.backbone)
    elif cfg.modality == "video":
        backbone_model, processor, hidden = load_video_backbone(cfg.backbone)
    else:
        raise ValueError(f"Unknown modality: {cfg.modality}")

    # ----- Apply LoRA -----
    backbone_model = apply_lora(backbone_model, cfg)
    backbone_model = backbone_model.to(cfg.device)
    trainable = sum(p.numel() for p in backbone_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone_model.parameters())
    logger.info(
        "LoRA applied: %d trainable / %d total params (%.2f%%)",
        trainable, total, 100.0 * trainable / max(total, 1),
    )

    # ----- Classifier head -----
    head = ClassifierHead(hidden_size=hidden, num_classes=2).to(cfg.device)

    # ----- Datasets -----
    train_ds = get_dataset(cfg, "train")
    val_ds = get_dataset(cfg, "val")
    if len(train_ds) == 0:
        logger.error(
            "Training dataset is empty. This means the dataset loader "
            "could not find files at %s. See TRAINING.md for dataset setup.",
            cfg.dataset_root,
        )
        sys.exit(1)

    if cfg.modality == "image":
        collator = ImageCollator(processor)
    elif cfg.modality == "audio":
        collator = AudioCollator(processor)
    else:
        collator = VideoCollator(processor)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collator, pin_memory=True,
    ) if len(val_ds) > 0 else None

    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, backbone_model.parameters()))
        + list(head.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ----- Training loop -----
    all_metrics: List[Dict[str, Any]] = []
    best_val_auc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(backbone_model, head, train_loader, optimizer, cfg, epoch)
        if val_loader is not None:
            val_metrics = validate(backbone_model, head, val_loader, cfg)
            metrics = {**train_metrics, **val_metrics}
            if val_metrics["val_auc"] > best_val_auc:
                best_val_auc = val_metrics["val_auc"]
                _save_checkpoint(backbone_model, head, cfg, suffix="_best")
                logger.info(
                    "New best val AUC: %.4f (epoch %d)",
                    best_val_auc, epoch,
                )
        else:
            metrics = train_metrics
        all_metrics.append(metrics)
        logger.info("Epoch %d metrics: %s", epoch, metrics)

        if epoch % cfg.save_every_n_epochs == 0:
            _save_checkpoint(backbone_model, head, cfg, suffix=f"_e{epoch}")

    # Final save
    _save_checkpoint(backbone_model, head, cfg, suffix="_final")
    with open(os.path.join(cfg.output_dir, "training_metrics.json"), "w") as fh:
        json.dump(all_metrics, fh, indent=2)

    logger.info("=== Training complete ===")
    logger.info("Best val AUC: %.4f", best_val_auc)
    logger.info("Adapters saved to: %s", cfg.output_dir)


def _save_checkpoint(model: nn.Module, head: nn.Module, cfg: TrainConfig, suffix: str = ""):
    """Save LoRA adapter + classifier head."""
    out = Path(cfg.output_dir)
    # Save LoRA adapter (PEFT format)
    try:
        model.save_pretrained(out)
        logger.info("Saved LoRA adapter to %s", out)
    except Exception as e:
        logger.warning("Could not save PEFT adapter: %s", e)
    # Save classifier head separately
    head_path = out / f"classifier{suffix}.pt"
    torch.save(head.state_dict(), head_path)
    logger.info("Saved classifier head to %s", head_path)


if __name__ == "__main__":
    main()
