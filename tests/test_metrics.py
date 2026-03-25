import pytest
import torch

from src.evaluation.metrics import compute_topk_accuracy, compute_per_class_metrics


class TestMetrics:
    def test_top1_accuracy_perfect(self):
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        labels = torch.tensor([1, 0])
        acc = compute_topk_accuracy(preds, labels, k=1)
        assert acc == 1.0

    def test_top1_accuracy_wrong(self):
        preds = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        labels = torch.tensor([0, 0])
        acc = compute_topk_accuracy(preds, labels, k=1)
        assert acc == 0.0

    def test_top5_accuracy(self):
        preds = torch.randn(10, 10)
        labels = torch.arange(10)
        for i in range(10):
            preds[i, labels[i]] = 100.0
        acc = compute_topk_accuracy(preds, labels, k=5)
        assert acc == 1.0

    def test_per_class_metrics_shape(self):
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        labels = torch.tensor([0, 1, 1, 0, 2, 2])
        metrics = compute_per_class_metrics(preds, labels, num_classes=3)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert len(metrics["precision"]) == 3
