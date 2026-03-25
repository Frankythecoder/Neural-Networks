from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.efficientnet import EfficientNetFood


_REGISTRY: dict[str, type[nn.Module]] = {
    "efficientnet_b2": EfficientNetFood,
}


def create_model(config: dict[str, Any]) -> nn.Module:
    """Create a model from config dict."""
    model_config = config["model"]
    name = model_config["name"]

    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_REGISTRY.keys())}")

    cls = _REGISTRY[name]
    return cls(
        num_classes=model_config.get("num_classes", 101),
        pretrained=model_config.get("pretrained", True),
        dropout=model_config.get("dropout", [0.3, 0.2]),
        head_hidden_dim=model_config.get("head_hidden_dim", 512),
    )
