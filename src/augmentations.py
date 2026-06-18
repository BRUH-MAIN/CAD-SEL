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


def get_train_transforms(image_size: int = 320) -> A.Compose:
    """
    Training augmentations — medical-image-safe transforms.

    Includes:
    - RandomResizedCrop (scale 0.8-1.0, mild variation)
    - HorizontalFlip (p=0.5)
    - VerticalFlip (p=0.05, low probability — organs can be upside-down in endoscopy)
    - RandomBrightnessContrast (p=0.5, mild limits ±0.1)
    - CLAHE (p=0.3, contrast enhancement common in medical imaging)
    - HueSaturationValue (p=0.3, small limits ±5)
    - GaussianBlur (p=0.1, mild blur)
    - CoarseDropout (p=0.2, light masking)
    - Normalize (ImageNet stats)
    """
    return A.Compose([
        A.RandomResizedCrop(
            height=image_size, width=image_size,
            scale=(0.8, 1.0), ratio=(0.9, 1.1),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05),
        A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(
            max_holes=4, max_height=32, max_width=32,
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
