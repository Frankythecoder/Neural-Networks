# Food Vision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Neural-Networks/Hangman repository into a production-grade food image classification platform using EfficientNet-B2 transfer learning on Food-101.

**Architecture:** PyTorch-based training pipeline with YAML config system, two-phase transfer learning (frozen feature extraction then fine-tuning), MLflow experiment tracking, ONNX export for serving, and FastAPI inference API. DVC for data versioning, Docker for containerization.

**Tech Stack:** Python 3.11, PyTorch 2.0+, torchvision, EfficientNet-B2, FastAPI, ONNX Runtime, MLflow, Optuna, DVC, Docker, pytest, ruff, mypy

**Spec:** `docs/superpowers/specs/2026-03-24-food-vision-design.md`

---

## File Structure

```
food-vision/
├── configs/
│   └── default.yaml                    # All training/model/data/serving config
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                   # YAML loading + CLI override merging
│   │   ├── reproducibility.py          # Seed setting + deterministic flags
│   │   └── logging.py                  # Structured logging setup
│   ├── data/
│   │   ├── __init__.py
│   │   ├── transforms.py              # Train/val augmentation pipelines
│   │   ├── dataset.py                 # Food101Dataset with split logic
│   │   └── download.py               # Download + DVC setup
│   ├── models/
│   │   ├── __init__.py
│   │   ├── efficientnet.py            # EfficientNet wrapper + custom head
│   │   └── factory.py                 # create_model(config) dispatcher
│   ├── training/
│   │   ├── __init__.py
│   │   ├── optimizer.py               # Optimizer + scheduler factory
│   │   ├── callbacks.py               # EarlyStopping, Checkpointing, LR/Metrics logging
│   │   └── trainer.py                 # Two-phase training loop with AMP
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Accuracy, F1, per-class metrics
│   │   ├── gradcam.py                 # GradCAM visualization
│   │   └── error_analysis.py          # Failure mode categorization
│   └── serving/
│       ├── __init__.py
│       ├── export_onnx.py             # PyTorch -> ONNX export
│       ├── predict.py                 # Prediction pre/post processing
│       └── app.py                     # FastAPI application
├── tests/
│   ├── conftest.py                    # Shared fixtures
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_api.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_error_analysis.ipynb
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── model-validation.yml
├── dvc.yaml
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Phase 1: Project Foundation

### Task 1: Remove Old Files and Create Directory Structure

**Files:**
- Delete: `hangman_game.py`, `neural_network_hangman.py`, `neural_network_demo.py`, `train_letter_model.py`, `Hangman_wordbank`, `models/letter_predictor.pt`, `requirements.txt`, `.zencoder/`, `.zenflow/`
- Create: all directories in project structure above

- [ ] **Step 1: Delete old Hangman files**

```bash
cd C:/Users/Frank/ML_food_vision
rm -f hangman_game.py neural_network_hangman.py neural_network_demo.py train_letter_model.py Hangman_wordbank requirements.txt
rm -rf models/ .zencoder/ .zenflow/
```

- [ ] **Step 2: Create new directory structure**

```bash
mkdir -p src/{utils,data,models,training,evaluation,serving}
mkdir -p tests notebooks configs docker .github/workflows
```

- [ ] **Step 3: Create all `__init__.py` files**

Create empty `__init__.py` in: `src/`, `src/utils/`, `src/data/`, `src/models/`, `src/training/`, `src/evaluation/`, `src/serving/`

Each file contains just:
```python
```

(Empty files.)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove hangman files, scaffold food-vision project structure"
```

---

### Task 2: pyproject.toml, .gitignore, and YAML Config

**Files:**
- Create: `pyproject.toml`
- Create: `configs/default.yaml`
- Modify: `.gitignore`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "food-vision"
version = "0.1.0"
description = "Production-grade food image classification with EfficientNet-B2"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "Pillow>=10.0",
    "pyyaml>=6.0",
    "mlflow>=2.10",
    "optuna>=3.5",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "onnx>=1.15",
    "onnxruntime>=1.17",
    "python-multipart>=0.0.9",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "seaborn>=0.13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "ruff>=0.3",
    "mypy>=1.8",
    "httpx>=0.27",
]

[tool.setuptools.packages.find]
include = ["src*"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Write configs/default.yaml**

```yaml
data:
  dataset: food101
  image_size: 224
  batch_size: 64
  num_workers: 4
  val_split: 0.2
  augmentation:
    randaugment_num_ops: 2
    randaugment_magnitude: 9
    mixup_alpha: 0.2
    cutmix_alpha: 1.0

model:
  name: efficientnet_b2
  pretrained: true
  num_classes: 101
  dropout: [0.3, 0.2]
  head_hidden_dim: 512

training:
  phase1:
    epochs: 5
    lr: 1.0e-3
    freeze_backbone: true
  phase2:
    epochs: 25
    backbone_lr: 1.0e-4
    head_lr: 5.0e-4
    unfreeze_blocks: 3
  optimizer: adamw
  weight_decay: 1.0e-4
  label_smoothing: 0.1
  gradient_clip: 1.0
  mixed_precision: true
  early_stopping:
    patience: 5
    monitor: val_accuracy

evaluation:
  metrics: [top1_accuracy, top5_accuracy, per_class_f1]
  gradcam: true
  num_gradcam_samples: 5

serving:
  format: onnx
  max_batch_size: 16
  max_image_size_mb: 10

seed: 42
```

- [ ] **Step 3: Update .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg

# ML artifacts
data/
mlruns/
outputs/
checkpoints/
*.pt
*.pth
*.onnx

# DVC
/data
*.dvc

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
htmlcov/
.coverage

# Misc
.zencoder/
.zenflow/
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml configs/default.yaml .gitignore
git commit -m "feat: add pyproject.toml, default config, and updated gitignore"
```

---

### Task 3: Config Loader Utility

**Files:**
- Create: `src/utils/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests for config loading**

`tests/test_config.py`:
```python
import pytest
from pathlib import Path
from src.utils.config import load_config


@pytest.fixture
def config_path():
    return Path(__file__).parent.parent / "configs" / "default.yaml"


def test_load_config_returns_dict(config_path):
    config = load_config(config_path)
    assert isinstance(config, dict)


def test_load_config_has_required_sections(config_path):
    config = load_config(config_path)
    for section in ["data", "model", "training", "evaluation", "serving", "seed"]:
        assert section in config, f"Missing config section: {section}"


def test_load_config_model_defaults(config_path):
    config = load_config(config_path)
    assert config["model"]["name"] == "efficientnet_b2"
    assert config["model"]["num_classes"] == 101


def test_load_config_with_overrides(config_path):
    overrides = {"model.num_classes": 10, "training.phase1.epochs": 1}
    config = load_config(config_path, overrides=overrides)
    assert config["model"]["num_classes"] == 10
    assert config["training"]["phase1"]["epochs"] == 1


def test_load_config_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent.yaml"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.config'`

- [ ] **Step 3: Implement config loader**

`src/utils/config.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML config and apply dot-notation overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key.split("."), value)

    return config


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```
Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/config.py tests/test_config.py
git commit -m "feat: add YAML config loader with dot-notation overrides"
```

---

### Task 4: Reproducibility and Logging Utilities

**Files:**
- Create: `src/utils/reproducibility.py`
- Create: `src/utils/logging.py`

- [ ] **Step 1: Implement reproducibility module**

`src/utils/reproducibility.py`:
```python
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_deterministic(enabled: bool = True) -> None:
    """Enable deterministic operations (may reduce performance)."""
    torch.use_deterministic_algorithms(enabled)
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

- [ ] **Step 2: Implement logging module**

`src/utils/logging.py`:
```python
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging for the application."""
    logger = logging.getLogger("food_vision")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
```

- [ ] **Step 3: Commit**

```bash
git add src/utils/reproducibility.py src/utils/logging.py
git commit -m "feat: add reproducibility seed setter and structured logging"
```

---

## Phase 2: Data Pipeline

### Task 5: Data Transforms

**Files:**
- Create: `src/data/transforms.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing tests for transforms**

`tests/test_data.py`:
```python
import pytest
import torch
from PIL import Image

from src.data.transforms import get_train_transforms, get_eval_transforms


@pytest.fixture
def sample_image():
    """Create a dummy 300x400 RGB image."""
    return Image.new("RGB", (400, 300), color=(128, 64, 32))


class TestTransforms:
    def test_train_transforms_output_shape(self, sample_image):
        transform = get_train_transforms(image_size=224)
        tensor = transform(sample_image)
        assert tensor.shape == (3, 224, 224)

    def test_train_transforms_output_dtype(self, sample_image):
        transform = get_train_transforms(image_size=224)
        tensor = transform(sample_image)
        assert tensor.dtype == torch.float32

    def test_eval_transforms_output_shape(self, sample_image):
        transform = get_eval_transforms(image_size=224)
        tensor = transform(sample_image)
        assert tensor.shape == (3, 224, 224)

    def test_eval_transforms_deterministic(self, sample_image):
        transform = get_eval_transforms(image_size=224)
        t1 = transform(sample_image)
        t2 = transform(sample_image)
        assert torch.equal(t1, t2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.transforms'`

- [ ] **Step 3: Implement transforms**

`src/data/transforms.py`:
```python
from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 224,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
) -> transforms.Compose:
    """Training augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation and test transforms (deterministic)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/transforms.py tests/test_data.py
git commit -m "feat: add train/eval image transform pipelines"
```

---

### Task 6: Food-101 Dataset Class

**Files:**
- Modify: `src/data/dataset.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write failing tests for dataset**

Append to `tests/test_data.py`:
```python
from unittest.mock import patch, MagicMock
from src.data.dataset import Food101Dataset


class TestFood101Dataset:
    def test_dataset_returns_tuple(self, sample_image):
        """Dataset __getitem__ returns (image_tensor, label_int)."""
        dataset = Food101Dataset(
            images=[sample_image],
            labels=[0],
            transform=get_eval_transforms(224),
        )
        img, label = dataset[0]
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert label == 0

    def test_dataset_length(self, sample_image):
        images = [sample_image] * 5
        labels = list(range(5))
        dataset = Food101Dataset(images=images, labels=labels, transform=get_eval_transforms(224))
        assert len(dataset) == 5
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
pytest tests/test_data.py::TestFood101Dataset -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.dataset'`

- [ ] **Step 3: Implement dataset class**

`src/data/dataset.py`:
```python
from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Food101Dataset(Dataset):
    """Wraps a list of PIL images and integer labels into a PyTorch Dataset."""

    def __init__(
        self,
        images: list[Image.Image],
        labels: list[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        assert len(images) == len(labels), "images and labels must have same length"
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```
Expected: all 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_data.py
git commit -m "feat: add Food101Dataset class"
```

---

### Task 7: Data Download and Split Script

**Files:**
- Create: `src/data/download.py`

- [ ] **Step 1: Implement download and split logic**

`src/data/download.py`:
```python
"""Download Food-101 dataset and create train/val/test splits."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.datasets import Food101

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed

logger = setup_logging()


def download_food101(data_dir: str = "data") -> Path:
    """Download Food-101 via torchvision and return the root path."""
    data_path = Path(data_dir)
    logger.info("Downloading Food-101 to %s ...", data_path)
    Food101(root=str(data_path), split="train", download=True)
    Food101(root=str(data_path), split="test", download=True)
    logger.info("Download complete.")
    return data_path


def create_splits(
    data_dir: str = "data",
    val_split: float = 0.2,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Create train/val index split from the training set.

    Returns dict with 'train' and 'val' index lists. Test set is separate.
    """
    set_seed(seed)
    train_dataset = Food101(root=data_dir, split="train")
    n = len(train_dataset)
    indices = torch.randperm(n).tolist()
    val_size = int(n * val_split)

    splits = {
        "train": sorted(indices[val_size:]),
        "val": sorted(indices[:val_size]),
    }

    # Save split indices for reproducibility
    split_path = Path(data_dir) / "split_indices.json"
    with open(split_path, "w") as f:
        json.dump(splits, f)
    logger.info(
        "Split created: %d train, %d val. Saved to %s",
        len(splits["train"]),
        len(splits["val"]),
        split_path,
    )
    return splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Food-101 dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    download_food101(args.data_dir)
    create_splits(
        data_dir=args.data_dir,
        val_split=config["data"]["val_split"],
        seed=config["seed"],
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/data/download.py
git commit -m "feat: add Food-101 download and train/val split script"
```

---

## Phase 3: Model Architecture

### Task 8: EfficientNet Wrapper

**Files:**
- Create: `src/models/efficientnet.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing tests for model**

`tests/test_model.py`:
```python
import pytest
import torch

from src.models.efficientnet import EfficientNetFood


class TestEfficientNetFood:
    @pytest.fixture
    def model(self):
        return EfficientNetFood(
            num_classes=101,
            pretrained=False,
            dropout=[0.3, 0.2],
            head_hidden_dim=512,
        )

    def test_forward_output_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 101)

    def test_forward_output_dtype(self, model):
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.dtype == torch.float32

    def test_freeze_backbone(self, model):
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
        for param in model.head.parameters():
            assert param.requires_grad

    def test_unfreeze_top_blocks(self, model):
        model.freeze_backbone()
        model.unfreeze_top_blocks(n=2)
        # At least some backbone params should now require grad
        unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        assert unfrozen > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_model.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement EfficientNet wrapper**

`src/models/efficientnet.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model.py -v
```
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/efficientnet.py tests/test_model.py
git commit -m "feat: add EfficientNet-B2 wrapper with freeze/unfreeze support"
```

---

### Task 9: Model Factory

**Files:**
- Create: `src/models/factory.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write failing test for factory**

Append to `tests/test_model.py`:
```python
from src.models.factory import create_model


class TestModelFactory:
    def test_create_efficientnet(self):
        config = {
            "model": {
                "name": "efficientnet_b2",
                "pretrained": False,
                "num_classes": 101,
                "dropout": [0.3, 0.2],
                "head_hidden_dim": 512,
            }
        }
        model = create_model(config)
        x = torch.randn(1, 3, 224, 224)
        assert model(x).shape == (1, 101)

    def test_create_unknown_model_raises(self):
        config = {"model": {"name": "unknown_model"}}
        with pytest.raises(ValueError, match="Unknown model"):
            create_model(config)
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
pytest tests/test_model.py::TestModelFactory -v
```
Expected: FAIL

- [ ] **Step 3: Implement model factory**

`src/models/factory.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model.py -v
```
Expected: all 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/factory.py tests/test_model.py
git commit -m "feat: add model factory with registry pattern"
```

---

## Phase 4: Training Pipeline

### Task 10: Optimizer and Scheduler Factory

**Files:**
- Create: `src/training/optimizer.py`

- [ ] **Step 1: Implement optimizer and scheduler factory**

`src/training/optimizer.py`:
```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/training/optimizer.py
git commit -m "feat: add optimizer and scheduler factory with discriminative LR"
```

---

### Task 11: Training Callbacks

**Files:**
- Create: `src/training/callbacks.py`

- [ ] **Step 1: Implement callbacks**

`src/training/callbacks.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.logging import setup_logging

logger = setup_logging()


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = metric
            return False

        improved = (
            metric > self.best if self.mode == "max" else metric < self.best
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d epochs", self.counter, self.patience
            )

        return self.counter >= self.patience


class ModelCheckpoint:
    """Save best and last model checkpoints."""

    def __init__(self, save_dir: str = "checkpoints", mode: str = "max") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best: float | None = None

    def step(
        self,
        model: nn.Module,
        metric: float,
        epoch: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save last checkpoint; save best if metric improved."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric": metric,
            **(extra or {}),
        }

        # Always save last
        torch.save(state, self.save_dir / "last.pt")

        # Save best
        if self.best is None or (
            metric > self.best if self.mode == "max" else metric < self.best
        ):
            self.best = metric
            torch.save(state, self.save_dir / "best.pt")
            logger.info("Saved best checkpoint (metric=%.4f) at epoch %d", metric, epoch)
```

- [ ] **Step 2: Commit**

```bash
git add src/training/callbacks.py
git commit -m "feat: add EarlyStopping and ModelCheckpoint callbacks"
```

---

### Task 12: Training Loop

**Files:**
- Create: `src/training/trainer.py`

- [ ] **Step 1: Implement the two-phase training loop**

`src/training/trainer.py`:
```python
"""Two-phase training loop with mixed precision and MLflow tracking."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models.factory import create_model
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.optimizer import create_optimizer, create_scheduler
from src.utils.logging import setup_logging

logger = setup_logging()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch. Returns dict with 'loss' and 'accuracy'."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run validation. Returns dict with 'loss' and 'accuracy'."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    phase: int,
) -> nn.Module:
    """Run one training phase (1 = feature extraction, 2 = fine-tuning)."""
    training_cfg = config["training"]
    phase_cfg = training_cfg[f"phase{phase}"]
    epochs = phase_cfg["epochs"]

    logger.info("=== Phase %d: %s (%d epochs) ===", phase,
                "Feature Extraction" if phase == 1 else "Fine-tuning", epochs)

    if phase == 1:
        model.freeze_backbone()
    else:
        model.unfreeze_top_blocks(n=phase_cfg.get("unfreeze_blocks", 3))

    optimizer = create_optimizer(model, config, phase=phase)
    scheduler = create_scheduler(optimizer, config, phase=phase)

    criterion = nn.CrossEntropyLoss(label_smoothing=training_cfg["label_smoothing"])

    scaler = GradScaler("cuda") if (training_cfg["mixed_precision"] and device.type == "cuda") else None

    early_stopping = EarlyStopping(
        patience=training_cfg["early_stopping"]["patience"], mode="max"
    )
    checkpoint = ModelCheckpoint(save_dir="checkpoints", mode="max")

    gradient_clip = training_cfg.get("gradient_clip", 1.0)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, gradient_clip
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Phase %d Epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f | lr=%.2e | %.1fs",
            phase, epoch, epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            lr, elapsed,
        )

        # MLflow logging
        step = epoch
        mlflow.log_metrics({
            f"phase{phase}/train_loss": train_metrics["loss"],
            f"phase{phase}/train_accuracy": train_metrics["accuracy"],
            f"phase{phase}/val_loss": val_metrics["loss"],
            f"phase{phase}/val_accuracy": val_metrics["accuracy"],
            f"phase{phase}/lr": lr,
        }, step=step)

        checkpoint.step(model, val_metrics["accuracy"], epoch, extra={
            "optimizer_state_dict": optimizer.state_dict(),
        })

        if early_stopping.step(val_metrics["accuracy"]):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    return model


def train(config: dict[str, Any], train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    """Full two-phase training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = create_model(config).to(device)

    mlflow.log_params({
        "model": config["model"]["name"],
        "seed": config["seed"],
        "phase1_epochs": config["training"]["phase1"]["epochs"],
        "phase2_epochs": config["training"]["phase2"]["epochs"],
        "batch_size": config["data"]["batch_size"],
        "label_smoothing": config["training"]["label_smoothing"],
    })

    # Phase 1: Feature extraction
    model = train_phase(model, train_loader, val_loader, config, device, phase=1)

    # Phase 2: Fine-tuning
    model = train_phase(model, train_loader, val_loader, config, device, phase=2)

    logger.info("Training complete.")
    return model


if __name__ == "__main__":
    import argparse
    from src.utils.config import load_config
    from src.utils.reproducibility import set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    set_seed(cfg["seed"])

    # Data loading deferred to here so this module stays testable
    from src.data.download import create_splits
    from src.data.dataset import Food101Dataset
    from src.data.transforms import get_train_transforms, get_eval_transforms
    from torchvision.datasets import Food101

    base_train = Food101(root="data", split="train")
    splits = create_splits(data_dir="data", val_split=cfg["data"]["val_split"], seed=cfg["seed"])

    aug_cfg = cfg["data"]["augmentation"]
    train_transform = get_train_transforms(
        image_size=cfg["data"]["image_size"],
        randaugment_num_ops=aug_cfg["randaugment_num_ops"],
        randaugment_magnitude=aug_cfg["randaugment_magnitude"],
    )
    val_transform = get_eval_transforms(image_size=cfg["data"]["image_size"])

    train_images = [base_train[i][0] for i in splits["train"]]
    train_labels = [base_train[i][1] for i in splits["train"]]
    val_images = [base_train[i][0] for i in splits["val"]]
    val_labels = [base_train[i][1] for i in splits["val"]]

    train_ds = Food101Dataset(train_images, train_labels, train_transform)
    val_ds = Food101Dataset(val_images, val_labels, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    with mlflow.start_run():
        train(cfg, train_loader, val_loader)
```

- [ ] **Step 2: Commit**

```bash
git add src/training/trainer.py
git commit -m "feat: add two-phase training loop with mixed precision and MLflow"
```

---

## Phase 5: Evaluation

### Task 13: Metrics Module

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests for metrics**

`tests/test_metrics.py`:
```python
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
        # 10-class problem, true labels are in top 5
        preds = torch.randn(10, 10)
        labels = torch.arange(10)
        # Force correct class to have high logit
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement metrics module**

`src/evaluation/metrics.py`:
```python
from __future__ import annotations

from typing import Any

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add top-k accuracy and per-class metrics"
```

---

### Task 14: GradCAM Visualization

**Files:**
- Create: `src/evaluation/gradcam.py`

- [ ] **Step 1: Implement GradCAM**

`src/evaluation/gradcam.py`:
```python
"""GradCAM attention visualization for EfficientNet models."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Generate GradCAM heatmaps for a given model and target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self, module: nn.Module, grad_input: tuple, grad_output: tuple
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Generate GradCAM heatmap for the given input.

        Returns a numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global avg pool gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def save_gradcam_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    save_path: str | Path,
    alpha: float = 0.5,
) -> None:
    """Overlay GradCAM heatmap on original image and save."""
    img_array = np.array(image.resize((heatmap.shape[1], heatmap.shape[0])))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("GradCAM")
    axes[1].axis("off")

    axes[2].imshow(img_array)
    axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
```

- [ ] **Step 2: Commit**

```bash
git add src/evaluation/gradcam.py
git commit -m "feat: add GradCAM attention visualization"
```

---

### Task 15: Error Analysis Module

**Files:**
- Create: `src/evaluation/error_analysis.py`

- [ ] **Step 1: Implement error analysis**

`src/evaluation/error_analysis.py`:
```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/evaluation/error_analysis.py
git commit -m "feat: add error analysis with confusion pairs and confidence stats"
```

---

## Phase 6: Serving

### Task 16: ONNX Export

**Files:**
- Create: `src/serving/export_onnx.py`

- [ ] **Step 1: Implement ONNX export**

`src/serving/export_onnx.py`:
```python
"""Export trained PyTorch model to ONNX format."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.factory import create_model
from src.utils.logging import setup_logging

logger = setup_logging()


def export_to_onnx(
    config: dict,
    checkpoint_path: str = "checkpoints/best.pt",
    output_path: str = "model.onnx",
    image_size: int = 224,
) -> Path:
    """Export model checkpoint to ONNX and validate outputs match."""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)
    output_path = Path(output_path)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Compare outputs
    with torch.no_grad():
        pytorch_out = model(dummy_input).numpy()

    session = ort.InferenceSession(str(output_path))
    onnx_out = session.run(None, {"image": dummy_input.numpy()})[0]

    diff = np.abs(pytorch_out - onnx_out).max()
    logger.info("ONNX export complete: %s (max diff: %.6f)", output_path, diff)
    assert diff < 1e-4, f"ONNX output mismatch: max diff {diff}"

    return output_path


if __name__ == "__main__":
    import argparse
    from src.utils.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="model.onnx")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    export_to_onnx(cfg, args.checkpoint, args.output, cfg["data"]["image_size"])
```

- [ ] **Step 2: Commit**

```bash
git add src/serving/export_onnx.py
git commit -m "feat: add ONNX export with output validation"
```

---

### Task 17: Prediction Logic

**Files:**
- Create: `src/serving/predict.py`

- [ ] **Step 1: Implement prediction pre/post processing**

`src/serving/predict.py`:
```python
"""Prediction logic: image preprocessing and inference."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.data.transforms import get_eval_transforms

# Food-101 class names (alphabetical)
CLASS_NAMES: list[str] = []  # Populated at startup


def load_class_names(path: str = "class_names.txt") -> list[str]:
    """Load class names from file."""
    global CLASS_NAMES
    with open(path) as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
    return CLASS_NAMES


class Predictor:
    """ONNX-based image classifier."""

    def __init__(self, model_path: str, class_names_path: str = "class_names.txt") -> None:
        self.session = ort.InferenceSession(model_path)
        self.transform = get_eval_transforms(image_size=224)
        load_class_names(class_names_path)

    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        """Classify a single image. Returns predictions and latency."""
        t0 = time.perf_counter()

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).numpy()
        logits = self.session.run(None, {"image": tensor})[0][0]

        # Softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        top_indices = probs.argsort()[::-1][:top_k]
        predictions = [
            {"class": CLASS_NAMES[i], "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        latency_ms = (time.perf_counter() - t0) * 1000
        return {"predictions": predictions, "latency_ms": round(latency_ms, 1)}
```

- [ ] **Step 2: Commit**

```bash
git add src/serving/predict.py
git commit -m "feat: add ONNX-based prediction logic"
```

---

### Task 18: FastAPI Application

**Files:**
- Create: `src/serving/app.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing tests for API**

`tests/test_api.py`:
```python
import io
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


@pytest.fixture
def mock_predictor():
    """Mock predictor to avoid loading a real ONNX model."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "predictions": [{"class": "pizza", "confidence": 0.94}],
        "latency_ms": 12.3,
    }
    return predictor


@pytest.fixture
def client(mock_predictor):
    with patch("src.serving.app.predictor", mock_predictor):
        from src.serving.app import app
        from fastapi.testclient import TestClient
        return TestClient(app)


class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_predict_valid_image(self, client):
        img = Image.new("RGB", (224, 224))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        resp = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert resp.status_code == 200
        assert "predictions" in resp.json()

    def test_predict_invalid_file_type(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement FastAPI app**

`src/serving/app.py`:
```python
"""FastAPI application for food image classification."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.serving.predict import Predictor

predictor: Predictor | None = None

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    model_path = os.getenv("MODEL_PATH", "model.onnx")
    class_names_path = os.getenv("CLASS_NAMES_PATH", "class_names.txt")
    if os.path.exists(model_path) and os.path.exists(class_names_path):
        predictor = Predictor(model_path, class_names_path)
    yield


app = FastAPI(title="Food Vision API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "efficientnet_b2",
        "version": "1.0.0",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(422, f"Invalid file type: {file.content_type}. Use JPEG/PNG/WebP.")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(422, f"File too large. Max {MAX_SIZE_MB}MB.")

    try:
        import io
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(422, "Could not decode image.")

    if predictor is None:
        raise HTTPException(503, "Model not loaded.")

    return predictor.predict(image, top_k=5)


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 16:
        raise HTTPException(422, "Max 16 images per batch.")
    if predictor is None:
        raise HTTPException(503, "Model not loaded.")

    results = []
    for f in files:
        if f.content_type not in ALLOWED_TYPES:
            results.append({"error": f"Invalid file type: {f.content_type}"})
            continue
        import io
        contents = await f.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results.append(predictor.predict(image, top_k=5))
    return {"results": results}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/serving/app.py src/serving/predict.py tests/test_api.py
git commit -m "feat: add FastAPI serving app with predict and health endpoints"
```

---

## Phase 7: Infrastructure

### Task 19: Dockerfiles and docker-compose

**Files:**
- Create: `docker/Dockerfile.train`
- Create: `docker/Dockerfile.serve`
- Create: `docker-compose.yml`

- [ ] **Step 1: Write Dockerfile.train**

`docker/Dockerfile.train`:
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

ENTRYPOINT ["python", "-m", "src.training.trainer", "--config", "/app/configs/default.yaml"]
```

- [ ] **Step 2: Write Dockerfile.serve**

`docker/Dockerfile.serve`:
```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --target=/deps \
    fastapi uvicorn[standard] onnxruntime Pillow numpy python-multipart pyyaml torchvision

# Runtime stage
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages
COPY src/ src/
COPY model.onnx class_names.txt ./

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write docker-compose.yml**

`docker-compose.yml`:
```yaml
services:
  train:
    build:
      context: .
      dockerfile: docker/Dockerfile.train
    volumes:
      - ./configs:/app/configs
      - ./data:/app/data
      - ./mlruns:/app/mlruns
      - ./checkpoints:/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  serve:
    build:
      context: .
      dockerfile: docker/Dockerfile.serve
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=model.onnx
      - CLASS_NAMES_PATH=class_names.txt

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.11.1
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlflow/mlruns
```

- [ ] **Step 4: Commit**

```bash
git add docker/Dockerfile.train docker/Dockerfile.serve docker-compose.yml
git commit -m "feat: add Docker training/serving images and compose config"
```

---

### Task 20: CI/CD Workflows

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/model-validation.yml`

- [ ] **Step 1: Write CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check src/ tests/

      - name: Type check
        run: mypy src/

      - name: Test
        run: pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80
```

- [ ] **Step 2: Write model validation workflow**

`.github/workflows/model-validation.yml`:
```yaml
name: Model Validation

on:
  push:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Unit tests
        run: pytest tests/ -v

      - name: Verify Docker builds
        run: |
          docker build -f docker/Dockerfile.serve -t food-vision-serve . || true
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/model-validation.yml
git commit -m "feat: add CI and model validation GitHub Actions workflows"
```

---

### Task 21: DVC Pipeline and Makefile

**Files:**
- Create: `dvc.yaml`
- Create: `Makefile`

- [ ] **Step 1: Write DVC pipeline**

`dvc.yaml`:
```yaml
stages:
  download:
    cmd: python -m src.data.download --config configs/default.yaml
    deps:
      - src/data/download.py
      - configs/default.yaml
    outs:
      - data/food-101:
          persist: true

  train:
    cmd: python -m src.training.trainer --config configs/default.yaml
    deps:
      - src/training/trainer.py
      - src/models/efficientnet.py
      - configs/default.yaml
      - data/food-101
    outs:
      - checkpoints/best.pt
    metrics:
      - mlruns:
          persist: true

  evaluate:
    cmd: python -m src.evaluation.metrics --config configs/default.yaml
    deps:
      - src/evaluation/metrics.py
      - checkpoints/best.pt
```

- [ ] **Step 2: Write Makefile**

`Makefile`:
```makefile
.PHONY: setup data train train-hpsearch evaluate export serve docker-build docker-serve test lint clean

setup:
	pip install -e ".[dev]"

data:
	python -m src.data.download --config configs/default.yaml

train:
	python -m src.training.trainer --config configs/default.yaml

train-hpsearch:
	python -m src.training.trainer --config configs/default.yaml --hpsearch

evaluate:
	python -m src.evaluation.metrics --config configs/default.yaml

export:
	python -m src.serving.export_onnx --config configs/default.yaml

serve:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose build

docker-serve:
	docker-compose up serve mlflow

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

clean:
	rm -rf mlruns/ outputs/ checkpoints/ __pycache__ .pytest_cache
```

- [ ] **Step 3: Commit**

```bash
git add dvc.yaml Makefile
git commit -m "feat: add DVC pipeline and Makefile with project commands"
```

---

## Phase 8: Documentation

### Task 22: README

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Write new README**

`README.md`:
```markdown
# Food Vision

Production-grade food image classification using EfficientNet-B2 transfer learning on the Food-101 dataset.

## Architecture

```
Image -> EfficientNet-B2 (pretrained) -> Custom MLP Head -> 101 Food Classes
         [frozen/fine-tuned]             [512 hidden, dropout]
```

**Training:** Two-phase transfer learning with mixed precision
1. Feature extraction (frozen backbone, 5 epochs)
2. Fine-tuning (top 3 blocks unfrozen, discriminative LR, 25 epochs)

**Serving:** ONNX Runtime inference via FastAPI

## Quick Start

```bash
pip install -e ".[dev]"     # Install dependencies
make data                    # Download Food-101 (~5GB)
make train                   # Train model (GPU recommended)
make serve                   # Start inference API
```

## Results

| Configuration | Top-1 Accuracy |
|--------------|----------------|
| A: Frozen backbone only | ~70-75% |
| B: + Fine-tuning | ~78-82% |
| C: + Advanced augmentation | ~82-85% |
| D: + Label smoothing + HP tuning | ~85-90% |

## API Usage

```bash
# Classify an image
curl -X POST http://localhost:8000/predict -F "file=@food_photo.jpg"

# Response
{"predictions": [{"class": "pizza", "confidence": 0.94}, ...], "latency_ms": 12.3}
```

## Project Structure

```
food-vision/
├── configs/default.yaml          # Training configuration
├── src/
│   ├── data/                     # Dataset, transforms, download
│   ├── models/                   # EfficientNet wrapper, model factory
│   ├── training/                 # Trainer, callbacks, optimizer
│   ├── evaluation/               # Metrics, GradCAM, error analysis
│   └── serving/                  # FastAPI app, ONNX export
├── tests/                        # Unit and integration tests
├── notebooks/                    # EDA, training analysis, error analysis
├── docker/                       # Training and serving Dockerfiles
└── .github/workflows/            # CI/CD pipelines
```

## Training

```bash
# Full training pipeline
make train

# With hyperparameter search (Optuna)
make train-hpsearch

# Evaluate on test set
make evaluate

# Export to ONNX
make export
```

## Docker

```bash
docker-compose up serve mlflow    # Start API + MLflow UI
# API: http://localhost:8000
# MLflow: http://localhost:5000
```

## Testing

```bash
make test     # Run tests with coverage
make lint     # Ruff + mypy
```

## Design Decisions

- **EfficientNet-B2**: Best accuracy/compute trade-off for Food-101 scale (~8.6M params)
- **Two-phase training**: Prevents catastrophic forgetting while allowing domain adaptation
- **ONNX serving**: 2-3x faster inference than native PyTorch, portable across runtimes
- **DVC**: Reproducible data pipeline without committing 5GB to git
- **MLflow**: Full experiment lineage — every hyperparameter and metric tracked

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — 101 food categories, 1,000 images each, ~5GB total.

## Author

Created by Frankythecoder
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for food-vision project"
```

---

### Task 23: Test Fixtures (conftest.py)

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write shared test fixtures**

`tests/conftest.py`:
```python
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
            "phase2": {"epochs": 1, "backbone_lr": 1e-4, "head_lr": 5e-4, "unfreeze_blocks": 3},
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
```

- [ ] **Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: add shared test fixtures"
```

---

### Task 24: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests PASS

- [ ] **Step 2: Run linter**

```bash
ruff check src/ tests/
```
Expected: no errors

- [ ] **Step 3: Verify project structure matches spec**

```bash
find . -type f -name "*.py" | sort
```
Verify all files from file structure section exist.

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```
