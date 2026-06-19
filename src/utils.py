"""Utility functions: seeding, device detection, metrics computation."""

import random
import os
import numpy as np
import torch
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True is safe with fixed input sizes (all images resized to same dims)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return cuda if available, else cpu."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU")
    return device


def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        targets: Ground truth binary labels (0/1)
        probs: Predicted probabilities (continuous)
        preds: Predicted binary labels (0/1)
        threshold: Decision threshold

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(targets, preds)
    metrics["precision"] = precision_score(targets, preds, zero_division=0)
    metrics["recall"] = recall_score(targets, preds, zero_division=0)
    metrics["f1"] = f1_score(targets, preds, zero_division=0)

    # AUC metrics
    if len(np.unique(targets)) > 1:
        metrics["roc_auc"] = roc_auc_score(targets, probs)
        metrics["pr_auc"] = average_precision_score(targets, probs)
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    # Specificity
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dict as a pretty string."""
    lines = [f"--- {prefix} Metrics ---" if prefix else "--- Metrics ---"]
    for k, v in metrics.items():
        lines.append(f"  {k}: {v:.4f}")
    return "\n".join(lines)


def compute_optimal_threshold(
    targets: np.ndarray,
    probs: np.ndarray,
) -> float:
    """
    Compute optimal decision threshold via Youden's J statistic
    (maximising sensitivity + specificity - 1) on the ROC curve.

    Args:
        targets: Ground truth binary labels (0/1)
        probs: Predicted probabilities

    Returns:
        Optimal threshold value in [0, 1]
    """
    from sklearn.metrics import roc_curve
    if len(np.unique(targets)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(targets, probs)
    j_scores = tpr - fpr  # Youden's J = sensitivity + specificity - 1
    idx = np.argmax(j_scores)
    return float(thresholds[idx])


def save_train_log(
    path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: float,
    header: bool = False
) -> None:
    """Append a row to the training CSV log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "w" if header else "a"
    with open(path, mode) as f:
        if header:
            cols = [
                "epoch", "train_loss", "val_loss",
                "train_acc", "train_precision", "train_recall", "train_f1",
                "val_acc", "val_precision", "val_recall", "val_f1",
                "val_roc_auc", "lr"
            ]
            f.write(",".join(cols) + "\n")

        row = [
            str(epoch),
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{train_metrics.get('accuracy', 0):.4f}",
            f"{train_metrics.get('precision', 0):.4f}",
            f"{train_metrics.get('recall', 0):.4f}",
            f"{train_metrics.get('f1', 0):.4f}",
            f"{val_metrics.get('accuracy', 0):.4f}",
            f"{val_metrics.get('precision', 0):.4f}",
            f"{val_metrics.get('recall', 0):.4f}",
            f"{val_metrics.get('f1', 0):.4f}",
            f"{val_metrics.get('roc_auc', 0):.4f}",
            f"{lr:.8f}",
        ]
        f.write(",".join(row) + "\n")
