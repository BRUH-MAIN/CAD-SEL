"""Configuration and argument parsing for the WLE binary classification pipeline."""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainConfig:
    """Training configuration with all hyperparameters exposed via CLI."""

    # ---- Data ----
    train_dir: str = "data/CAD-SEL-Dataset-New/train"
    test_dir: str = "data/CAD-SEL-Dataset-New/test"
    image_size: int = 512                # Up from 320 — preserves diagnostic texture
    num_workers: int = 4
    val_ratio: float = 0.2

    # ---- Model ----
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_epoch: int = 12             # Extended: let head converge before unfreezing
    head_dropout: float = 0.5            # Aggressive dropout in classification head
    drop_path_rate: float = 0.1          # Stochastic depth throughout backbone

    # ---- Training ----
    batch_size: int = 16                 # Smaller → more updates/epoch → better regularization
    epochs: int = 60                     # More epochs to compensate for frozen phase
    lr: float = 1e-3                     # Base LR (head); backbone gets backbone_lr_mult * this
    backbone_lr_mult: float = 0.1        # Backbone LR = lr * backbone_lr_mult
    weight_decay: float = 1e-3           # Up from 1e-4 — stronger L2 regularization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    patience: int = 12                   # Early stopping patience
    seed: int = 42
    early_stop_metric: str = "f1"
    warmup_epochs: int = 5               # Linear LR warmup at start
    label_smoothing: float = 0.05        # Soft targets to prevent overconfidence
    use_ema: bool = True                 # Exponential moving average of weights
    ema_decay: float = 0.999             # EMA decay rate

    # ---- Class imbalance ----
    use_imbalance_handler: bool = True
    imbalance_method: str = "weighted_sampler"  # weighted_sampler | class_weight

    # ---- Output ----
    checkpoint_dir: str = "outputs/checkpoints"
    output_dir: str = "outputs"

    # ---- Mixed precision ----
    use_amp: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def parse_args() -> TrainConfig:
    """Parse command-line arguments and return a TrainConfig."""
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 binary classifier on CAD-SEL WLE images"
    )

    # Data
    parser.add_argument("--train-dir", type=str, default="data/CAD-SEL-Dataset-New/train")
    parser.add_argument("--test-dir", type=str, default="data/CAD-SEL-Dataset-New/test")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of train to hold out as validation (patient-level)")

    # Model
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze backbone in Phase 1, train only the head")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--unfreeze-epoch", type=int, default=12,
                        help="Epoch at which to unfreeze backbone (0 = never)")
    parser.add_argument("--head-dropout", type=float, default=0.5,
                        help="Dropout probability in classification head")
    parser.add_argument("--drop-path-rate", type=float, default=0.1,
                        help="Stochastic depth rate for backbone blocks")

    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Base LR for head; backbone gets lr * backbone-lr-mult")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1,
                        help="Multiplier for backbone LR relative to head")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "onecycle"])
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-metric", type=str, default="f1",
                        choices=["f1", "auc", "loss"],
                        help="Metric to use for early stopping and best-model selection")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of linear LR warmup epochs")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Label smoothing factor for BCEWithLogitsLoss")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate (0.999 = slow, 0.99 = fast)")

    # Class imbalance
    parser.add_argument("--use-imbalance-handler", action="store_true", default=True)
    parser.add_argument("--no-imbalance-handler", dest="use_imbalance_handler",
                        action="store_false")
    parser.add_argument("--imbalance-method", type=str, default="weighted_sampler",
                        choices=["weighted_sampler", "class_weight"])

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs")

    # AMP
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")

    args = parser.parse_args()
    return TrainConfig(**vars(args))
