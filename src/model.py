"""ResNet-18 binary classifier for WLE images.

Design choices:
- ResNet-18 (~11M params) instead of ResNet-50 (~23.5M) — better ratio for
  a ~1,800-image medical dataset. Less capacity = less memorization.
- Deeper classification head: Dropout → Linear → BN → ReLU → Dropout → Linear
  gives the model a non-linear feature transformation before the final decision.
- Stochastic depth (torchvision.ops.stochastic_depth) throughout the backbone.
- BCEWithLogitsLoss with a single output logit for numerically stable binary
  classification with pos_weight support.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops import stochastic_depth
from typing import Optional


class ResNet18Binary(nn.Module):
    """ResNet-18 backbone with a deep binary classification head.

    Args:
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_backbone: If True, freeze all backbone layers (train only head).
        head_dropout: Dropout probability in the classification head.
        drop_path_rate: Stochastic depth rate (applied per block, linearly
            increasing from 0 to drop_path_rate across layers).
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        head_dropout: float = 0.5,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features  # 512

        # Deeper classification head with BatchNorm for regularization
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout * 0.6),
            nn.Linear(256, 1),
        )

        self.drop_path_rate = drop_path_rate

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(
        self, base_lr: float,
        head_lr_mult: float = 1.0,
        backbone_lr_mult: float = 0.1,
    ) -> list:
        """
        Return parameter groups with differential learning rates.
        Head: base_lr * head_lr_mult. Layer4: base_lr * backbone_lr_mult.
        Layer3: *0.5, Layer2: *0.2, Layer1: *0.1 of that.
        """
        head_params = list(self.backbone.fc.parameters())
        layers = {}
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.backbone, name, None)
            if layer is not None:
                layers[name] = list(layer.parameters())

        other_params = []
        layer_ids = set()
        for lp in layers.values():
            layer_ids.update(id(p) for p in lp)
        head_ids = set(id(p) for p in head_params)
        for _n, p in self.backbone.named_parameters():
            if id(p) not in head_ids and id(p) not in layer_ids:
                other_params.append(p)

        lr_map = {"layer4": 1.0, "layer3": 0.5, "layer2": 0.2, "layer1": 0.1}
        groups = [{"params": head_params, "lr": base_lr * head_lr_mult, "name": "head"}]
        for lname in ["layer1", "layer2", "layer3", "layer4"]:
            if lname in layers and layers[lname]:
                groups.append({
                    "params": layers[lname],
                    "lr": base_lr * backbone_lr_mult * lr_map[lname],
                    "name": lname,
                })
        if other_params:
            groups.append({
                "params": other_params,
                "lr": base_lr * backbone_lr_mult * lr_map["layer1"],
                "name": "stem",
            })
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stochastic depth in residual blocks."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        for i, layer in enumerate([
            self.backbone.layer1, self.backbone.layer2,
            self.backbone.layer3, self.backbone.layer4,
        ]):
            layer_dp = self.drop_path_rate * (i / 3.0)
            for block in layer:
                identity = x
                out = block.conv1(x)
                out = block.bn1(out)
                out = block.relu(out)
                out = block.conv2(out)
                out = block.bn2(out)
                if block.downsample is not None:
                    identity = block.downsample(identity)
                if self.training and layer_dp > 0:
                    out = stochastic_depth(out, layer_dp, "row", self.training)
                x = block.relu(out + identity)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x


def build_model(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    head_dropout: float = 0.5,
    drop_path_rate: float = 0.1,
    device: Optional[torch.device] = None,
) -> ResNet18Binary:
    model = ResNet18Binary(
        pretrained=pretrained, freeze_backbone=freeze_backbone,
        head_dropout=head_dropout, drop_path_rate=drop_path_rate,
    )
    if device is not None:
        model = model.to(device)
    return model
