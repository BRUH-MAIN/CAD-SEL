"""Dataset loader for WLE (White Light Endoscopy) binary classification.

Directory structure expected:
    data/CAD-SEL-Dataset-New/train/WLE-Set/
        NETs/          -> Class 1 (positive: neuroendocrine tumors)
            patient_001/
                image1.tiff
                ...
        Non-Nets/      -> Class 0 (negative: benign findings)
            Leiomyoma/
                ...
            Lipoma/
                ...
            NonNeoplasm/
                ...

Classes are automatically inferred from the top-level subdirectories under WLE-Set/.
All .tiff, .tif, .jpg, .jpeg, .png images are loaded recursively.
"""

import os
import glob
from typing import Tuple, List, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class WLEBinaryDataset(Dataset):
    """Binary classification dataset for WLE images.

    Args:
        root_dir: Path to the train/ or test/ directory.
        transform: Albumentations Compose (or None).
        classes: Optional ordered list of class names. If None, inferred from dirs.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        classes: Optional[List[str]] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        # Locate the WLE-Set folder
        wle_dir = os.path.join(root_dir, "WLE-Set")
        if not os.path.isdir(wle_dir):
            raise FileNotFoundError(
                f"WLE-Set/ not found under {root_dir}. "
                f"Expected structure: {root_dir}/WLE-Set/{{class}}/{{patient}}/image.tiff"
            )

        # Auto-detect class names from subdirectories
        # IMPORTANT: We want NETs as class 1 (positive/tumor) and
        # Non-Nets as class 0 (negative/benign). We enforce this ordering
        # regardless of alphabetical sort.
        if classes is None:
            raw_classes = sorted([
                d for d in os.listdir(wle_dir)
                if os.path.isdir(os.path.join(wle_dir, d))
            ])
            # Force Non-Nets=0, NETs=1
            self.classes = []
            for desired in ["Non-Nets", "NETs"]:
                if desired in raw_classes:
                    self.classes.append(desired)
            # Fallback: use sorted order if expected names not found
            if len(self.classes) != 2:
                self.classes = raw_classes
        else:
            self.classes = list(classes)

        if len(self.classes) != 2:
            raise ValueError(
                f"Expected exactly 2 classes under WLE-Set/, got {len(self.classes)}: {self.classes}"
            )

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all image paths and their labels
        self.samples: List[Tuple[str, int]] = []
        self.class_counts: Dict[str, int] = {cls: 0 for cls in self.classes}

        img_extensions = ("*.tiff", "*.tif", "*.jpg", "*.jpeg", "*.png")

        for cls_name in self.classes:
            cls_dir = os.path.join(wle_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            # Walk recursively through patient folders
            for ext in img_extensions:
                pattern = os.path.join(cls_dir, "**", ext)
                for img_path in glob.glob(pattern, recursive=True):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))
                    self.class_counts[cls_name] += 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No WLE images found under {wle_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Load image as RGB
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transform
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]  # Already a torch Tensor (C, H, W) normalized
        else:
            # Fallback: manual conversion
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.float32),
            "path": img_path,
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Return the number of samples per class."""
        return dict(self.class_counts)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced loss."""
        total = sum(self.class_counts.values())
        weights = []
        for cls in self.classes:
            count = self.class_counts.get(cls, 1)
            weights.append(total / (len(self.classes) * count))
        return torch.tensor(weights, dtype=torch.float32)

    def print_stats(self) -> None:
        """Print dataset statistics."""
        total = sum(self.class_counts.values())
        print(f"  Dataset: {self.root_dir}")
        print(f"  Total WLE images: {total}")
        print(f"  Classes: {self.classes}")
        for cls_name, count in self.class_counts.items():
            pct = 100.0 * count / total if total > 0 else 0
            print(f"    {cls_name}: {count} ({pct:.1f}%)")


def build_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test dataloaders.

    Uses the test_dir as validation set (no separate val split — the dataset
    already provides train/test splits).  During training, the test set serves
    as the validation set; final evaluation also uses the same test set with
    the best checkpoint.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = WLEBinaryDataset(
        root_dir=train_dir,
        transform=train_transform,
    )
    val_dataset = WLEBinaryDataset(
        root_dir=test_dir,
        transform=val_transform,
    )

    # Enforce consistent class ordering across train and val
    if val_dataset.class_to_idx != train_dataset.class_to_idx:
        # Re-label val samples to match train class indices
        old_to_new = {
            val_dataset.class_to_idx[cls]: train_dataset.class_to_idx[cls]
            for cls in val_dataset.classes
        }
        val_dataset.samples = [
            (path, old_to_new[label]) for path, label in val_dataset.samples
        ]
        val_dataset.class_to_idx = train_dataset.class_to_idx
        val_dataset.classes = train_dataset.classes

    # Build sampler for class imbalance
    train_sampler = None
    if use_weighted_sampler:
        labels = [label for _, label in train_dataset.samples]
        class_sample_counts = train_dataset.get_class_distribution()
        counts_list = [class_sample_counts[c] for c in train_dataset.classes]
        class_weights = 1.0 / torch.tensor(counts_list, dtype=torch.float)
        sample_weights = [class_weights[label].item() for label in labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, val_loader  # test = val
