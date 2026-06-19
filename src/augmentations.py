"""Albumentations-based augmentation and preprocessing transforms.

Design notes:
- Medical images require conservative augmentations: no extreme warping,
  no aggressive color shifts, no rotations beyond ±15°.
- Training uses a rich but safe set of augmentations.
- Validation/test use only resize + normalize (no augmentation).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


# ImageNet mean/std (standard for pretrained ResNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Training augmentations — medical-image-safe but stronger for small datasets.

    Key changes vs the original:
    - Wider RandomResizedCrop (0.7-1.0) for more scale diversity
    - Stronger brightness/contrast (±0.2) for illumination invariance
    - Reduced CoarseDropout (max 2 holes, 24×24) to avoid occluding small lesions
    - Added Rotate (±15°) for rotation invariance
    """
    return A.Compose([
        A.RandomResizedCrop(
            height=image_size, width=image_size,
            scale=(0.7, 1.0), ratio=(0.85, 1.18),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05),
        A.Rotate(limit=15, border_mode=0, value=0, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(
            max_holes=2, max_height=24, max_width=24,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.2
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 320) -> A.Compose:
    """Validation/test transforms — no augmentation, only resize + normalize."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
