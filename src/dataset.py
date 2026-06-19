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
import random
from typing import Tuple, List, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image


class WLEBinaryDataset(Dataset):
    """Binary classification dataset for WLE images.

    Args:
        root_dir: Path to the train/ or test/ directory.
        transform: Albumentations Compose (or None).
        classes: Optional ordered list of class names. If None, inferred from dirs.
        patient_ids: Optional whitelist of patient folder names. If provided,
            only images from these patients are included (used for patient-level
            train/val splitting).
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        classes: Optional[List[str]] = None,
        patient_ids: Optional[Dict[str, List[str]]] = None,
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
        # Non-Nets as class 0 (negative/benign). We enforce this ordering.
        if classes is None:
            raw_classes = sorted([
                d for d in os.listdir(wle_dir)
                if os.path.isdir(os.path.join(wle_dir, d))
            ])
            self.classes = []
            for desired in ["Non-Nets", "NETs"]:
                if desired in raw_classes:
                    self.classes.append(desired)
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
        self.patient_ids_used: Dict[str, List[str]] = {cls: [] for cls in self.classes}

        img_extensions = ("*.tiff", "*.tif", "*.jpg", "*.jpeg", "*.png")
        failed_loads = 0

        for cls_name in self.classes:
            cls_dir = os.path.join(wle_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            # Walk recursively through patient folders
            for ext in img_extensions:
                pattern = os.path.join(cls_dir, "**", ext)
                for img_path in glob.glob(pattern, recursive=True):
                    # Extract patient ID (parent folder name of the image)
                    patient_id = os.path.basename(os.path.dirname(img_path))

                    # Filter by patient_ids whitelist if provided
                    if patient_ids is not None:
                        if cls_name not in patient_ids:
                            continue
                        if patient_id not in patient_ids[cls_name]:
                            continue

                    # Pre-flight: verify image can be opened
                    try:
                        # Use PIL for robust TIFF support across platforms
                        pil_img = Image.open(img_path)
                        pil_img.verify()  # Verify without fully decoding
                        pil_img = Image.open(img_path)  # Re-open after verify
                        if pil_img.mode not in ("RGB", "L"):
                            pil_img = pil_img.convert("RGB")
                    except Exception:
                        failed_loads += 1
                        continue

                    self.samples.append((img_path, self.class_to_idx[cls_name]))
                    self.class_counts[cls_name] += 1
                    if patient_id not in self.patient_ids_used[cls_name]:
                        self.patient_ids_used[cls_name].append(patient_id)

        if failed_loads > 0:
            print(f"  WARNING: {failed_loads} corrupt/unreadable images skipped")

        if len(self.samples) == 0:
            raise RuntimeError(f"No WLE images found under {wle_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Load image as RGB (use PIL for robust TIFF handling)
        try:
            pil_img = Image.open(img_path)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img = np.array(pil_img)
        except Exception:
            # Fallback to OpenCV
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
        """
        Compute inverse-frequency class weights.
        Returns tensor [weight_class_0, weight_class_1].
        For imbalanced BCEWithLogitsLoss, pos_weight = weight_neg / weight_pos.
        """
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


def create_patient_split(
    root_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Create patient-level train/val split from the train directory.

    Returns:
        train_patients: {class_name: [patient_id, ...]}
        val_patients:   {class_name: [patient_id, ...]}
    """
    rng = random.Random(seed)
    train_patients: Dict[str, List[str]] = {}
    val_patients: Dict[str, List[str]] = {}

    wle_dir = os.path.join(root_dir, "WLE-Set")
    for cls_name in sorted(os.listdir(wle_dir)):
        cls_path = os.path.join(wle_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        patients = sorted([
            d for d in os.listdir(cls_path)
            if os.path.isdir(os.path.join(cls_path, d))
        ])
        rng.shuffle(patients)
        n_val = max(1, int(len(patients) * val_ratio))
        val_patients[cls_name] = patients[:n_val]
        train_patients[cls_name] = patients[n_val:]

    return train_patients, val_patients


def build_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    seed: int = 42,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test dataloaders.

    The train_dir is split 80/20 at the patient level into train/val.
    The test_dir is kept completely untouched as the held-out test set.

    Returns:
        train_loader, val_loader, test_loader
    """
    # --- Patient-level train/val split ---
    train_patients, val_patients = create_patient_split(
        train_dir, val_ratio=val_ratio, seed=seed
    )

    # Build datasets with patient filtering
    train_dataset = WLEBinaryDataset(
        root_dir=train_dir,
        transform=train_transform,
        patient_ids=train_patients,
    )
    val_dataset = WLEBinaryDataset(
        root_dir=train_dir,
        transform=val_transform,
        patient_ids=val_patients,
    )
    test_dataset = WLEBinaryDataset(
        root_dir=test_dir,
        transform=val_transform,
    )

    # Enforce consistent class ordering across all splits
    for ds in [val_dataset, test_dataset]:
        if ds.class_to_idx != train_dataset.class_to_idx:
            old_to_new = {
                ds.class_to_idx[cls]: train_dataset.class_to_idx[cls]
                for cls in ds.classes
            }
            ds.samples = [
                (path, old_to_new[label]) for path, label in ds.samples
            ]
            ds.class_to_idx = train_dataset.class_to_idx
            ds.classes = train_dataset.classes

    # Build sampler for class imbalance (train only)
    train_sampler = None
    if use_weighted_sampler:
        labels = [label for _, label in train_dataset.samples]
        class_sample_counts = train_dataset.get_class_distribution()
        counts_list = [class_sample_counts[c] for c in train_dataset.classes]
        class_weights = 1.0 / torch.tensor(counts_list, dtype=torch.float)
        sample_weights = [class_weights[label].item() for label in labels]
        # Oversample minority just enough to match majority, not infinitely
        majority_count = max(counts_list)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=majority_count * 2,  # 2× majority = mild oversampling
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
