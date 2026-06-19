"""
Evaluation script for the WLE binary classification pipeline.

Loads the best checkpoint (or a user-specified one) and evaluates on the test set.
Generates:
- Classification metrics (accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC)
- Confusion matrix plot
- ROC curve plot
- Precision-Recall curve plot
- Classification report (sklearn)

Usage:
    python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth
"""

import os
import sys
import argparse
import json
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, compute_metrics
from src.augmentations import get_val_transforms
from src.dataset import WLEBinaryDataset
from src.model import build_model
from src.plotting import (
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix
)
from sklearn.metrics import classification_report


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Run evaluation and collect predictions."""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", ncols=100):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= threshold).astype(np.float32)
            targets = labels.cpu().numpy().flatten()

            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(targets)
            all_paths.extend(batch["path"])

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    return {
        "preds": all_preds,
        "probs": all_probs,
        "targets": all_targets,
        "paths": all_paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate binary classifier on WLE test set"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the model checkpoint (.pth)"
    )
    parser.add_argument(
        "--test-dir", type=str, default="data/CAD-SEL-Dataset-New/test",
        help="Path to test directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Base output directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="Image size (auto-detected from checkpoint config if not set)"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # ---- Load checkpoint ----
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_config = checkpoint.get("config", {})

    # Determine image size
    image_size = args.image_size
    if image_size is None:
        image_size = ckpt_config.get("image_size", 320)
    print(f"Image size: {image_size}")

    # ---- Build model ----
    pretrained = ckpt_config.get("pretrained", True)
    model = build_model(pretrained=pretrained, device=device)
    # Prefer EMA weights (better generalization) over raw weights
    if "ema_shadow" in checkpoint and checkpoint["ema_shadow"] is not None:
        model.load_state_dict(checkpoint["ema_shadow"])
        print("Loaded EMA (exponential moving average) weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded raw model weights (no EMA found)")
    epoch = checkpoint.get("epoch", "unknown")
    optimal_threshold = checkpoint.get("optimal_threshold", 0.5)
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Optimal threshold (Youden's J, from val set): {optimal_threshold:.4f}")

    # ---- Data ----
    val_transforms = get_val_transforms(image_size)
    # The dataset auto-enforces Non-Nets=0, NETs=1
    test_dataset = WLEBinaryDataset(
        root_dir=args.test_dir,
        transform=val_transforms,
    )

    print(f"Class mapping: {test_dataset.class_to_idx}")
    test_dataset.print_stats()

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Evaluate (with optimal threshold from validation) ----
    print(f"\n--- Running Evaluation (threshold={optimal_threshold:.4f}) ---")
    results = evaluate(model, test_loader, device, threshold=optimal_threshold)

    metrics = compute_metrics(
        results["targets"], results["probs"], results["preds"],
        threshold=optimal_threshold,
    )

    # ---- Print metrics ----
    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"  Threshold:   {optimal_threshold:.4f} (Youden's J from val)")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")

    # ---- Generate outputs ----
    plots_dir = os.path.join(args.output_dir, "plots")
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Classification report (using optimal threshold)
    class_names = test_dataset.classes
    report = classification_report(
        results["targets"], results["preds"],
        target_names=class_names, digits=4
    )
    report_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Threshold: {optimal_threshold:.4f} (Youden's J from validation)\n\n")
        f.write(report)
    print(f"\nClassification Report saved to {report_path}")
    print(report)

    # Plots
    print("Generating evaluation plots...")
    plot_confusion_matrix(
        results["targets"], results["preds"], class_names,
        os.path.join(plots_dir, "confusion_matrix.png")
    )
    plot_roc_curve(
        results["targets"], results["probs"],
        os.path.join(plots_dir, "roc_curve.png")
    )
    plot_pr_curve(
        results["targets"], results["probs"],
        os.path.join(plots_dir, "pr_curve.png")
    )
    print(f"Plots saved to {plots_dir}/")

    # Save metrics as JSON
    metrics_path = os.path.join(reports_dir, "test_metrics.json")
    # Convert numpy types for JSON serialization
    serializable = {k: float(v) if hasattr(v, 'item') else v
                    for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
