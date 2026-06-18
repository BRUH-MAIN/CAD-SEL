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
    image_size: int = 320
    num_workers: int = 4

    # ---- Model ----
    pretrained: bool = True
    # We use BCEWithLogitsLoss (single logit) — simpler, more memory-efficient,
    # and allows easy pos_weight tuning for class imbalance.

    # ---- Training ----
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"          # adamw | sgd
    scheduler: str = "cosine"         # cosine | plateau | onecycle
    patience: int = 10                # early stopping patience
    seed: int = 42

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
        description="Train ResNet-50 binary classifier on CAD-SEL WLE images"
    )

    # Data
    parser.add_argument("--train-dir", type=str, default="data/CAD-SEL-Dataset-New/train")
    parser.add_argument("--test-dir", type=str, default="data/CAD-SEL-Dataset-New/test")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "onecycle"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

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
