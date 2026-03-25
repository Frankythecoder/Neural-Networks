import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture
def config_path(project_root):
    return project_root / "configs" / "default.yaml"


@pytest.fixture
def sample_config():
    return {
        "data": {
            "dataset": "food101",
            "image_size": 224,
            "batch_size": 4,
            "num_workers": 0,
            "val_split": 0.2,
            "augmentation": {
                "randaugment_num_ops": 2,
                "randaugment_magnitude": 9,
                "mixup_alpha": 0.2,
                "cutmix_alpha": 1.0,
            },
        },
        "model": {
            "name": "efficientnet_b2",
            "pretrained": False,
            "num_classes": 101,
            "dropout": [0.3, 0.2],
            "head_hidden_dim": 512,
        },
        "training": {
            "phase1": {"epochs": 1, "lr": 1e-3, "freeze_backbone": True},
            "phase2": {
                "epochs": 1,
                "backbone_lr": 1e-4,
                "head_lr": 5e-4,
                "unfreeze_blocks": 3,
            },
            "optimizer": "adamw",
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "gradient_clip": 1.0,
            "mixed_precision": False,
            "early_stopping": {"patience": 5, "monitor": "val_accuracy"},
        },
        "evaluation": {
            "metrics": ["top1_accuracy", "top5_accuracy", "per_class_f1"],
            "gradcam": True,
            "num_gradcam_samples": 5,
        },
        "serving": {"format": "onnx", "max_batch_size": 16, "max_image_size_mb": 10},
        "seed": 42,
    }
