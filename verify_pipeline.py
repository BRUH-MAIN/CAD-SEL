"""Quick validation script for the WLE classification pipeline."""
import sys
sys.path.insert(0, '.')

from src.dataset import WLEBinaryDataset
from src.augmentations import get_train_transforms, get_val_transforms

# Test train dataset
print("=== Train Dataset ===")
ds = WLEBinaryDataset("data/CAD-SEL-Dataset-New/train", transform=None)
ds.print_stats()
print(f"Class mapping: {ds.class_to_idx}")
print(f"Sample 0: label={ds[0]['label'].item()}")

# Test test dataset
print("\n=== Test Dataset ===")
ds_test = WLEBinaryDataset("data/CAD-SEL-Dataset-New/test", transform=None)
ds_test.print_stats()
print(f"Class mapping: {ds_test.class_to_idx}")

# Test with transforms
print("\n=== With Albumentations transforms ===")
ds_t = WLEBinaryDataset("data/CAD-SEL-Dataset-New/train", transform=get_train_transforms(320))
sample = ds_t[0]
print(f"Image shape: {sample['image'].shape}, label: {sample['label'].item()}")

print("\nAll checks passed!")
