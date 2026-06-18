"""Plotting utilities for training curves and evaluation visualizations."""

import os
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)


# Consistent style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
) -> None:
    """Plot training and validation loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss")
    ax.plot(epochs, val_losses, "r-", linewidth=1.5, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_metrics_curve(
    history: Dict[str, List[float]],
    save_path: str,
) -> None:
    """
    Plot validation metrics across epochs.

    Args:
        history: Dict with keys like 'val_acc', 'val_precision', 'val_recall', 'val_f1'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Map internal keys to display names
    metric_pairs = [
        ("val_acc", "Accuracy"),
        ("val_precision", "Precision"),
        ("val_recall", "Recall"),
        ("val_f1", "F1 Score"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(next(iter(history.values()))) + 1)

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    for (key, name), color in zip(metric_pairs, colors):
        if key in history:
            ax.plot(epochs, history[key], linewidth=1.5, label=name, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics Over Training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_roc_curve(
    targets: np.ndarray,
    probs: np.ndarray,
    save_path: str,
) -> None:
    """Plot ROC curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    from sklearn.metrics import roc_auc_score
    fpr, tpr, _ = roc_curve(targets, probs)
    auc = roc_auc_score(targets, probs)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_pr_curve(
    targets: np.ndarray,
    probs: np.ndarray,
    save_path: str,
) -> None:
    """Plot Precision-Recall curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    from sklearn.metrics import average_precision_score
    precision, recall, _ = precision_recall_curve(targets, probs)
    ap = average_precision_score(targets, probs)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, "b-", linewidth=2, label=f"PR (AP = {ap:.4f})")
    baseline = targets.sum() / len(targets)
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1,
                alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(
    targets: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Plot and save confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(targets, preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
