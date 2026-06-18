"""ResNet-50 binary classifier for WLE images.

Design choice: BCEWithLogitsLoss with a single output logit.

Rationale:
- Binary classification is a single decision boundary problem.
- BCEWithLogitsLoss combines sigmoid + BCE in a numerically stable way.
- A single logit allows direct use of `pos_weight` for class imbalance,
  which is cleaner than CrossEntropyLoss weight tensors.
- The probability is simply torch.sigmoid(logit).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional


class ResNet50Binary(nn.Module):
    """ResNet-50 backbone with a binary classification head.

    Args:
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_backbone: If True, freeze all backbone layers (only train head).
        dropout: Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Replace the final fully-connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),  # Single logit for BCEWithLogitsLoss
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the new head
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape (B, 1)."""
        return self.backbone(x)


def build_model(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
    device: Optional[torch.device] = None,
) -> ResNet50Binary:
    """Factory function to build and optionally move the model to device."""
    model = ResNet50Binary(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model
