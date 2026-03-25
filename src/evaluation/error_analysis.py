"""Failure mode categorization and confusion pair analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def get_top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Find the most confused class pairs from predictions.

    Returns a list of dicts with keys: class_a, class_b, count.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Zero out diagonal (correct predictions)
    np.fill_diagonal(cm, 0)

    # Get top confused pairs (symmetric: merge (a,b) and (b,a))
    pairs: dict[tuple[int, int], int] = {}
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i == j or cm[i, j] == 0:
                continue
            key = (min(i, j), max(i, j))
            pairs[key] = pairs.get(key, 0) + cm[i, j]

    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        {"class_a": class_names[a], "class_b": class_names[b], "count": count}
        for (a, b), count in sorted_pairs
    ]


def get_confidence_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Compute confidence statistics for correct vs incorrect predictions."""
    probs = torch.softmax(logits, dim=1)
    max_probs, preds = probs.max(dim=1)

    correct_mask = preds == labels
    correct_conf = max_probs[correct_mask]
    incorrect_conf = max_probs[~correct_mask]

    return {
        "correct": {
            "mean": correct_conf.mean().item() if len(correct_conf) > 0 else 0.0,
            "std": correct_conf.std().item() if len(correct_conf) > 1 else 0.0,
            "count": int(correct_mask.sum().item()),
        },
        "incorrect": {
            "mean": incorrect_conf.mean().item() if len(incorrect_conf) > 0 else 0.0,
            "std": incorrect_conf.std().item() if len(incorrect_conf) > 1 else 0.0,
            "count": int((~correct_mask).sum().item()),
        },
    }
