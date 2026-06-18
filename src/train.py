"""
Training script for WLE binary classification with ResNet-50.

Features:
- Mixed precision (AMP)
- Gradient clipping
- LR scheduling (cosine, plateau, onecycle)
- Early stopping
- Best/last checkpoint saving
- Resume training
- Class imbalance handling (WeightedRandomSampler or class-weighted loss)
- Comprehensive metric logging
- tqdm progress bars

Usage:
    python -m src.train --epochs 50 --batch-size 32 --lr 1e-4
"""

import os
import sys
import time
import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import parse_args, TrainConfig
from src.utils import set_seed, get_device, compute_metrics, format_metrics, save_train_log
from src.augmentations import get_train_transforms, get_val_transforms
from src.dataset import build_dataloaders, WLEBinaryDataset
from src.model import build_model
from src.plotting import plot_loss_curve, plot_metrics_curve


# ---------------------------------------------------------------------------
# Optimizer & Scheduler builders
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, config: TrainConfig) -> optim.Optimizer:
    """Build optimizer based on config."""
    if config.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr, momentum=0.9,
                         weight_decay=config.weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> object:
    """Build LR scheduler based on config."""
    if config.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=config.patience // 2,
            verbose=True
        )
    elif config.scheduler == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr,
            epochs=config.epochs, steps_per_epoch=steps_per_epoch
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


# ---------------------------------------------------------------------------
# Epoch runners
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    scaler: Optional[GradScaler],
    is_train: bool = True,
) -> Dict:
    """Run one epoch (training or validation)."""
    if is_train:
        model.train()
    else:
        model.eval()

    all_preds = []
    all_probs = []
    all_targets = []
    running_loss = 0.0

    desc = "Training" if is_train else "Validation"
    pbar = tqdm(loader, desc=desc, leave=False, ncols=120)

    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)  # (B, 1)

        with (torch.enable_grad() if is_train else torch.no_grad()):
            if scaler is not None and is_train:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # Collect predictions
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(np.float32)
        targets = labels.detach().cpu().numpy().flatten()

        all_preds.append(preds)
        all_probs.append(probs)
        all_targets.append(targets)
        running_loss += loss.item() * images.size(0)

        # Update progress bar
        batch_preds = preds
        batch_acc = (batch_preds == targets).mean()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    avg_loss = running_loss / len(loader.dataset)

    metrics = compute_metrics(all_targets, all_probs, all_preds)

    return {
        "loss": avg_loss,
        "preds": all_preds,
        "probs": all_probs,
        "targets": all_targets,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: TrainConfig) -> None:
    """Run the full training pipeline."""

    # ---- Setup ----
    set_seed(config.seed)
    device = get_device()

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    plots_dir = os.path.join(config.output_dir, "plots")
    reports_dir = os.path.join(config.output_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Save config
    config.save(os.path.join(config.output_dir, "config.json"))
    print(f"Configuration:\n{config.to_dict()}")

    # ---- Data ----
    print("\n--- Loading Data ---")
    train_transforms = get_train_transforms(config.image_size)
    val_transforms = get_val_transforms(config.image_size)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        train_transform=train_transforms,
        val_transform=val_transforms,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_weighted_sampler=(
            config.use_imbalance_handler
            and config.imbalance_method == "weighted_sampler"
        ),
    )

    # Print dataset statistics
    print("\n--- Dataset Statistics ---")
    train_dataset = train_loader.dataset
    train_dataset.print_stats()

    val_dataset = val_loader.dataset
    val_dataset.print_stats()

    # ---- Model ----
    print("\n--- Building Model ---")
    model = build_model(pretrained=config.pretrained, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ---- Loss (BCEWithLogitsLoss) ----
    pos_weight = None
    if config.use_imbalance_handler and config.imbalance_method == "class_weight":
        weights = train_dataset.get_class_weights()
        # pos_weight = neg_count / pos_count (weight for the positive class)
        pos_weight = torch.tensor([weights[0] / weights[1]], device=device)
        print(f"  Using class-weighted loss, pos_weight = {pos_weight.item():.3f}")
    else:
        print("  Using unweighted BCEWithLogitsLoss")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- Optimizer & Scheduler ----
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    # ---- AMP ----
    scaler = GradScaler() if (config.use_amp and device.type == "cuda") else None
    print(f"  AMP: {'enabled' if scaler is not None else 'disabled'}")

    # ---- Training state ----
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [],
        "val_roc_auc": [], "lr": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    # CSV log header
    log_path = os.path.join(reports_dir, "train_log.csv")
    save_train_log(log_path, 0, 0, 0, {}, {}, 0, header=True)

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # ---- Train ----
        train_results = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, is_train=True
        )

        # Step LR scheduler (for non-plateau schedulers)
        if config.scheduler == "onecycle":
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        elif config.scheduler == "cosine":
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validate ----
        val_results = run_epoch(
            model, val_loader, criterion, None, device, None, is_train=False
        )

        # Step LR scheduler (for plateau scheduler)
        if config.scheduler == "plateau":
            scheduler.step(val_results["loss"])
            current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # ---- Record history ----
        history["train_loss"].append(train_results["loss"])
        history["val_loss"].append(val_results["loss"])
        history["train_acc"].append(train_results["accuracy"])
        history["train_precision"].append(train_results["precision"])
        history["train_recall"].append(train_results["recall"])
        history["train_f1"].append(train_results["f1"])
        history["val_acc"].append(val_results["accuracy"])
        history["val_precision"].append(val_results["precision"])
        history["val_recall"].append(val_results["recall"])
        history["val_f1"].append(val_results["f1"])
        history["val_roc_auc"].append(val_results["roc_auc"])
        history["lr"].append(current_lr)

        # ---- Log to CSV ----
        save_train_log(
            log_path, epoch,
            train_results["loss"], val_results["loss"],
            train_results, val_results, current_lr
        )

        # ---- Print progress ----
        print(f"\nEpoch {epoch}/{config.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_results['loss']:.4f}  "
              f"Acc: {train_results['accuracy']:.4f}  "
              f"F1: {train_results['f1']:.4f}")
        print(f"  Val   Loss: {val_results['loss']:.4f}  "
              f"Acc: {val_results['accuracy']:.4f}  "
              f"F1: {val_results['f1']:.4f}  "
              f"AUC: {val_results['roc_auc']:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # ---- Checkpointing ----
        # Save last model
        last_path = os.path.join(config.checkpoint_dir, "last_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "history": history,
            "config": config.to_dict(),
        }, last_path)

        # Save best model
        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": config.to_dict(),
                "val_loss": val_results["loss"],
                "val_acc": val_results["accuracy"],
                "val_f1": val_results["f1"],
                "val_auc": val_results["roc_auc"],
            }, best_path)
            print(f"  >>> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        # ---- Early stopping ----
        if epochs_no_improve >= config.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs!")
            break

    # ---- Training complete ----
    print("\n" + "=" * 60)
    print(f"Training complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    print("=" * 60)

    # ---- Plot training curves ----
    print("\n--- Generating Training Plots ---")
    plot_loss_curve(
        history["train_loss"], history["val_loss"],
        os.path.join(plots_dir, "loss_curve.png")
    )
    plot_metrics_curve(history, os.path.join(plots_dir, "metrics_curve.png"))
    print(f"  Saved to {plots_dir}/")


if __name__ == "__main__":
    config = parse_args()
    train(config)
