from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


class EfficientNetFood(nn.Module):
    """EfficientNet-B2 with a custom classification head for food recognition."""

    def __init__(
        self,
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: list[float] | None = None,
        head_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        dropout = dropout or [0.3, 0.2]

        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        base = efficientnet_b2(weights=weights)

        # Separate backbone (features) from classifier
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Infer backbone output channels
        in_features = base.classifier[1].in_features  # 1408 for B2

        self.head = nn.Sequential(
            nn.Dropout(p=dropout[0]),
            nn.Linear(in_features, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout[1]),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return self.head(pooled)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (for Phase 1 feature extraction)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_top_blocks(self, n: int = 3) -> None:
        """Unfreeze the top `n` blocks of the backbone (for Phase 2 fine-tuning)."""
        blocks = list(self.backbone.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
