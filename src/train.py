"""
Training script for WLE binary classification with ResNet-18.

Features:
- Mixed precision (AMP)
- Gradient clipping
- Differential learning rates (head vs backbone layers)
- LR warmup + cosine/plateau/onecycle scheduling
- Stochastic depth (DropPath)
- Label smoothing
- Exponential Moving Average (EMA) of weights
- Early stopping on F1/AUC/loss
- Best/last checkpoint saving
- Class imbalance handling
- Comprehensive metric logging

Usage:
    python -m src.train --epochs 60 --batch-size 16 --lr 1e-3 --image-size 512
"""

import os
import sys
import time
import copy
import math
from typing import Dict, List, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import parse_args, TrainConfig
from src.utils import set_seed, get_device, compute_metrics, save_train_log, compute_optimal_threshold
from src.augmentations import get_train_transforms, get_val_transforms
from src.dataset import build_dataloaders
from src.model import build_model, ResNet18Binary
from src.plotting import plot_loss_curve, plot_metrics_curve


# ---------------------------------------------------------------------------
# Label Smoothing BCE Loss
# ---------------------------------------------------------------------------

class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing.

    Instead of targets 0/1, uses (smoothing/2) and (1 - smoothing/2).
    This prevents the model from becoming overconfident on hard labels.
    """

    def __init__(self, smoothing: float = 0.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0:
            return self.bce(logits, targets).mean()
        # Smooth targets: 0 → smoothing/2, 1 → 1 - smoothing/2
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, smooth_targets).mean()


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = OrderedDict()
        self.backup = OrderedDict()
        self._register()

    def _register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                    self.shadow[name] = new_average

    def apply_shadow(self) -> None:
        """Replace model params with EMA shadow (for validation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        """Restore original params (for continued training)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup.clear()


# ---------------------------------------------------------------------------
# Optimizer & Scheduler builders
# ---------------------------------------------------------------------------

def build_optimizer(
    model: ResNet18Binary,
    config: TrainConfig,
    use_differential_lr: bool = True,
) -> optim.Optimizer:
    """Build optimizer with optional differential learning rates."""
    if use_differential_lr:
        param_groups = model.get_param_groups(
            base_lr=config.lr,
            head_lr_mult=1.0,
            backbone_lr_mult=config.backbone_lr_mult,
        )
    else:
        param_groups = model.parameters()

    if config.optimizer == "adamw":
        return optim.AdamW(param_groups, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return optim.SGD(param_groups, momentum=0.9,
                         weight_decay=config.weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


class WarmupScheduler:
    """Linear warmup followed by the main scheduler."""

    def __init__(self, optimizer, warmup_epochs: int, steps_per_epoch: int, main_scheduler):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.main_scheduler = main_scheduler
        self.current_step = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup from 0 to base_lr
            progress = self.current_step / self.warmup_steps
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.base_lrs[i] * progress
        else:
            self.main_scheduler.step()

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "main_scheduler": self.main_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state["current_step"]
        self.main_scheduler.load_state_dict(state["main_scheduler"])


def build_scheduler(
    optimizer: optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> object:
    """Build LR scheduler based on config, wrapped in warmup."""
    if config.scheduler == "cosine":
        main = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs * steps_per_epoch
        )
    elif config.scheduler == "plateau":
        main = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=config.patience // 2, verbose=True
        )
    elif config.scheduler == "onecycle":
        main = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr,
            epochs=config.epochs, steps_per_epoch=steps_per_epoch
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")

    if config.warmup_epochs > 0:
        return WarmupScheduler(optimizer, config.warmup_epochs, steps_per_epoch, main)
    return main


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
    scheduler: Optional[object] = None,       # OneCycleLR (per-batch)
    scheduler_batch_step: bool = False,        # Whether to step scheduler per batch
) -> Dict:
    """Run one epoch (training or validation).

    Args:
        scheduler: If OneCycleLR, pass it here for per-batch stepping.
        scheduler_batch_step: If True, scheduler.step() is called after every batch.
    """
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
                with autocast("cuda"):
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

            # Per-batch LR stepping for OneCycleLR
            if scheduler_batch_step and scheduler is not None:
                scheduler.step()

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

def _get_best_metric_value(val_results: Dict, metric: str) -> float:
    """Extract the early-stopping metric value from validation results."""
    if metric == "f1":
        return val_results["f1"]  # Higher is better
    elif metric == "auc":
        return val_results["roc_auc"]  # Higher is better
    elif metric == "loss":
        return -val_results["loss"]  # Lower is better → negate for "higher is better"
    else:
        return val_results["f1"]


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

    # Proper 3-way split: train (80%) / val (20%) from train_dir, test from test_dir
    train_loader, val_loader, test_loader = build_dataloaders(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        train_transform=train_transforms,
        val_transform=val_transforms,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_ratio=config.val_ratio,
        seed=config.seed,
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
    test_dataset = test_loader.dataset
    print(f"  Test set (held-out):")
    test_dataset.print_stats()

    # ---- Model ----
    print("\n--- Building Model ---")
    model = build_model(
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        head_dropout=config.head_dropout,
        drop_path_rate=config.drop_path_rate,
        device=device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = total_params - frozen_params
    print(f"  Architecture: ResNet-18 (~11M params)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Head dropout: {config.head_dropout}")
    print(f"  DropPath rate: {config.drop_path_rate}")
    if config.freeze_backbone and config.unfreeze_epoch > 0:
        print(f"  Backbone frozen until epoch {config.unfreeze_epoch}")
    if config.unfreeze_epoch > 0:
        print(f"  Differential LR: head=1×, backbone={config.backbone_lr_mult}× "
              f"(layer4=1.0×, layer3=0.5×, layer2=0.2×, layer1=0.1×)")

    # ---- Loss (Label Smoothing BCEWithLogitsLoss) ----
    pos_weight = None
    if config.use_imbalance_handler and config.imbalance_method == "class_weight":
        dist = train_dataset.get_class_distribution()
        count_neg = dist[train_dataset.classes[0]]
        count_pos = dist[train_dataset.classes[1]]
        pos_weight = torch.tensor([count_neg / count_pos], device=device)
        print(f"  pos_weight = {pos_weight.item():.4f}")

    criterion = LabelSmoothingBCEWithLogitsLoss(
        smoothing=config.label_smoothing,
        pos_weight=pos_weight,
    )
    print(f"  Label smoothing: {config.label_smoothing}")

    # ---- Optimizer & Scheduler (with differential LR) ----
    optimizer = build_optimizer(model, config, use_differential_lr=True)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
    print(f"  Warmup: {config.warmup_epochs} epochs")
    print(f"  Steps/epoch: {steps_per_epoch}")

    # ---- AMP ----
    scaler = GradScaler("cuda") if (config.use_amp and device.type == "cuda") else None
    print(f"  AMP: {'enabled' if scaler is not None else 'disabled'}")

    # ---- EMA ----
    ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None
    print(f"  EMA: {'enabled (decay=' + str(config.ema_decay) + ')' if ema else 'disabled'}")

    # ---- Training state ----
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [],
        "val_roc_auc": [], "lr": [],
    }

    best_metric = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0

    log_path = os.path.join(reports_dir, "train_log.csv")
    save_train_log(log_path, 0, 0, 0, {}, {}, 0, header=True)

    print("\n" + "=" * 60)
    print(f"Starting Training (early stopping on val_{config.early_stop_metric})")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # ---- Phase transition: unfreeze backbone with differential LR ----
        if config.freeze_backbone and epoch == config.unfreeze_epoch:
            print(f"\n>>> Unfreezing backbone at epoch {epoch} "
                  f"(differential LR, backbone_mult={config.backbone_lr_mult})")
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, config, use_differential_lr=True)
            # Reset scheduler for new optimizer; no more warmup
            main_sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(config.epochs - config.unfreeze_epoch + 1) * steps_per_epoch,
            )
            scheduler = main_sched
            # Re-register EMA to capture newly trainable backbone params
            if ema is not None:
                ema._register()
            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_now = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            print(f"  Trainable: {trainable_now:,} ({frozen_now:,} frozen)")

        # ---- Train ----
        # Cosine + warmup: step per batch. Plateau + OneCycle: also per batch.
        use_batch_step = config.scheduler in ("cosine", "onecycle")
        train_results = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            is_train=True,
            scheduler=scheduler if use_batch_step else None,
            scheduler_batch_step=use_batch_step,
        )

        # Update EMA after each training epoch
        if ema is not None:
            ema.update()

        # Epoch-level LR step for plateau (not stepped per batch)
        if config.scheduler == "plateau":
            pass  # Stepped after validation below

        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validate (with EMA weights if enabled) ----
        if ema is not None:
            ema.apply_shadow()
        val_results = run_epoch(
            model, val_loader, criterion, None, device, None, is_train=False
        )
        if ema is not None:
            ema.restore()

        if config.scheduler == "plateau":
            if config.early_stop_metric == "loss":
                scheduler.step(val_results["loss"])
            else:
                scheduler.step(val_results[config.early_stop_metric])
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

        save_train_log(
            log_path, epoch,
            train_results["loss"], val_results["loss"],
            train_results, val_results, current_lr
        )

        # ---- Print ----
        print(f"\nEpoch {epoch}/{config.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_results['loss']:.4f}  "
              f"Acc: {train_results['accuracy']:.4f}  "
              f"F1: {train_results['f1']:.4f}")
        print(f"  Val   Loss: {val_results['loss']:.4f}  "
              f"Acc: {val_results['accuracy']:.4f}  "
              f"F1: {val_results['f1']:.4f}  "
              f"AUC: {val_results['roc_auc']:.4f}")
        if len(optimizer.param_groups) > 1:
            lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups[:3]]
            print(f"  LRs: {', '.join(lrs)}...")
        else:
            print(f"  LR: {current_lr:.2e}")

        # ---- Checkpointing ----
        last_path = os.path.join(config.checkpoint_dir, "last_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_shadow": ema.shadow if ema else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "history": history,
            "config": config.to_dict(),
        }, last_path)

        # Save best model (evaluated with EMA weights)
        current_metric = _get_best_metric_value(val_results, config.early_stop_metric)
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_no_improve = 0

            # Compute optimal threshold with EMA weights
            if ema is not None:
                ema.apply_shadow()
            opt_threshold = compute_optimal_threshold(
                val_results["targets"], val_results["probs"]
            )
            if ema is not None:
                ema.restore()

            best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_shadow": ema.shadow if ema else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": config.to_dict(),
                "val_loss": val_results["loss"],
                "val_acc": val_results["accuracy"],
                "val_f1": val_results["f1"],
                "val_auc": val_results["roc_auc"],
                "optimal_threshold": opt_threshold,
            }, best_path)
            display_val = (
                f"{val_results[config.early_stop_metric]:.4f}"
                if config.early_stop_metric != "loss"
                else f"{val_results['loss']:.4f}"
            )
            print(f"  >>> Saved best model (val_{config.early_stop_metric}={display_val})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= config.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs!")
            break

    # ---- Training complete ----
    print("\n" + "=" * 60)
    print(f"Training complete. Best epoch: {best_epoch}")
    print(f"Best val_{config.early_stop_metric}: "
          f"{best_metric if config.early_stop_metric != 'loss' else -best_metric:.4f}")
    print("=" * 60)

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
