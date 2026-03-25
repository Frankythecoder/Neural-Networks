from __future__ import annotations

import torch
from sklearn.metrics import precision_recall_fscore_support


def compute_topk_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, k: int = 1
) -> float:
    """Compute top-k accuracy from logits and labels."""
    _, topk_preds = logits.topk(k, dim=1)
    correct = topk_preds.eq(labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def compute_per_class_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, list[float]]:
    """Compute per-class precision, recall, and F1 score."""
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, labels=list(range(num_classes)), zero_division=0.0
    )
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }
