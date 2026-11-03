"""ConvNeXt multi-task CNN: one shared backbone, a crop head and a disease head (brief 3.1, Fig. 1 right)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name: str = "convnext_tiny", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return an ImageNet-pretrained feature extractor (B,3,H,W) -> (B,D) and D."""
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights="DEFAULT" if pretrained else None)
        dim = m.classifier[2].in_features          # 768
        m.classifier[2] = nn.Identity()            # keep LayerNorm2d + Flatten, drop the ImageNet Linear
        return m, dim
    raise ValueError(f"unknown backbone {name}")


class MultiTaskCNN(nn.Module):
    def __init__(self, num_crops: int, num_diseases: int, backbone: str = "convnext_tiny",
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone, dim = build_backbone(backbone, pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.crop_head = nn.Linear(dim, num_crops)
        self.disease_head = nn.Linear(dim, num_diseases)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.features(x)
        return {"crop": self.crop_head(f), "disease": self.disease_head(f)}
