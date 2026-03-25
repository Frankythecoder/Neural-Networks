from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def create_optimizer(
    model: nn.Module,
    config: dict[str, Any],
    phase: int = 1,
) -> AdamW:
    """Create AdamW optimizer with per-param-group LRs.

    Phase 1: single LR for head only (backbone frozen).
    Phase 2: discriminative LR (lower for backbone, higher for head).
    """
    training_cfg = config["training"]
    weight_decay = training_cfg["weight_decay"]

    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        lr = training_cfg["phase1"]["lr"]
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    # Phase 2: discriminative learning rates
    backbone_lr = training_cfg["phase2"]["backbone_lr"]
    head_lr = training_cfg["phase2"]["head_lr"]

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
    return AdamW(param_groups, weight_decay=weight_decay)


def create_scheduler(
    optimizer: AdamW,
    config: dict[str, Any],
    phase: int = 1,
) -> SequentialLR:
    """Create LR scheduler: linear warmup (3 epochs) then cosine decay."""
    training_cfg = config["training"]
    total_epochs = training_cfg[f"phase{phase}"]["epochs"]
    warmup_epochs = min(3, total_epochs)
    decay_epochs = max(total_epochs - warmup_epochs, 1)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)

    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
