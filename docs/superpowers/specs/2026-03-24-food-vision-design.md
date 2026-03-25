# Food Vision — End-to-End ML Platform Design Spec

## Overview

Transform the Neural-Networks repository from a Hangman game into **food-vision**: a production-grade image classification system that recognizes food dishes from photos. The project targets ML Engineer job applications and demonstrates the full stack of ML engineering — from data versioning to deployed inference API.

**Problem:** Given a photo of a food dish, predict what it is (101 categories).
**Dataset:** Food-101 — 101 food categories, 1,000 images each, ~5 GB total.
**Why food recognition:** Publicly benchmarked dataset with published baselines, visually intuitive for demos, real-world applicable (nutrition tracking, restaurant apps), and challenging enough (visually similar dishes, varied lighting) to justify serious modeling.

**Repository:** The existing `Neural-Networks` GitHub repo will be renamed to `food-vision`. All Hangman files will be removed. Git history is preserved.

---

## Prerequisites

- Python >= 3.11
- NVIDIA GPU with CUDA 12.1+ toolkit
- Docker and Docker Compose
- DVC (`pip install dvc`)
- Make

---

## Project Structure

```
food-vision/
├── configs/                      # YAML experiment configs
│   ├── default.yaml              # Base config with all defaults
│   └── efficientnet_b2.yaml      # Model-specific overrides
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Food101Dataset class, transforms
│   │   ├── transforms.py         # Train/val/test augmentation pipelines
│   │   └── download.py           # Dataset download + DVC setup
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py            # Model factory (create_model from config)
│   │   └── efficientnet.py       # EfficientNet wrapper with custom head
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop with mixed precision, callbacks
│   │   ├── callbacks.py          # Early stopping, checkpointing, LR logging
│   │   └── optimizer.py          # Optimizer + scheduler factory
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Accuracy, F1, per-class metrics
│   │   ├── gradcam.py            # GradCAM attention visualization
│   │   └── error_analysis.py     # Failure mode categorization
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                # FastAPI application
│   │   ├── predict.py            # Prediction logic, pre/post processing
│   │   └── export_onnx.py        # PyTorch -> ONNX export script
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # YAML config loading + CLI overrides
│       ├── reproducibility.py    # Seed setting, deterministic flags
│       └── logging.py            # Structured logging setup
├── notebooks/
│   ├── 01_eda.ipynb              # Dataset exploration
│   ├── 02_training_analysis.ipynb # Ablation results, training curves
│   └── 03_error_analysis.ipynb   # GradCAM, confusion pairs, failure modes
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_data.py              # Data transforms, dataset class
│   ├── test_model.py             # Model output shapes, forward pass
│   ├── test_metrics.py           # Metric calculations
│   ├── test_config.py            # Config loading
│   └── test_api.py               # FastAPI endpoint integration tests
├── docker/
│   ├── Dockerfile.train          # Full training environment with CUDA
│   └── Dockerfile.serve          # Inference image (CPU ONNX ~500MB, GPU ONNX ~1.5GB)
├── .github/
│   └── workflows/
│       ├── ci.yml                # PR: lint + type check + unit tests
│       └── model-validation.yml  # Merge: integration tests + model gate
├── dvc.yaml                      # Data pipeline stages
├── .dvc/                         # DVC config
├── mlflow/                       # MLflow tracking config
├── docker-compose.yml            # MLflow server + inference API
├── Makefile                      # One-command workflows
├── pyproject.toml                # Dependencies, ruff, mypy config
├── README.md                     # Full documentation
└── .gitignore                    # Updated for ML project
```

---

## Data Pipeline

### Dataset: Food-101
- **Source:** torchvision.datasets.Food101 (auto-download) or manual download
- **Size:** 101 categories, 1,000 images each (750 train / 250 test per class)
- **Total:** ~101,000 images, ~5 GB

### Splits
- **Train:** 75% of original train set (56,812 images)
- **Validation:** 25% of original train set (18,938 images)
- **Test:** Original test set (25,250 images) — untouched, used only for final evaluation

### Data Versioning (DVC)
- `dvc.yaml` pipeline stages:
  1. `download` — fetch Food-101 dataset
  2. `preprocess` — validate images, create train/val split, save split indices
  3. `train` — run training pipeline
  4. `evaluate` — run evaluation suite
- Dataset tracked with DVC (not committed to git)
- Reproducible with `dvc repro`

### Augmentation Pipelines

**Training transforms:**
- RandomResizedCrop(224)
- RandomHorizontalFlip
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- RandAugment(num_ops=2, magnitude=9)
- Normalize(ImageNet mean/std)
- Mixup (alpha=0.2) and CutMix (alpha=1.0) applied at batch level

**Validation/Test transforms:**
- Resize(256) + CenterCrop(224)
- Normalize(ImageNet mean/std)

---

## Model Architecture

### EfficientNet-B2 with Transfer Learning

**Architecture:**
- Backbone: EfficientNet-B2 pretrained on ImageNet (torchvision)
- Custom MLP head: AdaptiveAvgPool -> Dropout(0.3) -> Linear(1408, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 101)
- Total params: ~7.8M (backbone) + ~0.77M (head) = ~8.6M

**Two-phase training:**

*Phase 1 — Feature extraction (5 epochs):*
- Freeze entire backbone
- Train only the classification head
- Higher learning rate (1e-3)
- Purpose: get the head to a reasonable starting point fast

*Phase 2 — Fine-tuning (25 epochs):*
- Unfreeze top 3 blocks of EfficientNet backbone
- Train with lower learning rate (1e-4) for backbone, (5e-4) for head
- Discriminative learning rates prevent catastrophic forgetting
- Early stopping with patience=5 based on validation accuracy

### Model Factory
- `create_model(config)` returns any supported architecture
- Start with EfficientNet-B2, easily extensible to B0/B4/ResNet50 for ablation

---

## Training Pipeline

### Optimizer & Scheduler
- **Optimizer:** AdamW (weight_decay=1e-4)
- **Scheduler:** Composed via `SequentialLR`:
  1. **Warmup phase (3 epochs):** `LinearLR` ramping from 0.01x to 1.0x base LR
  2. **Cosine decay phase (remaining epochs):** `CosineAnnealingLR` decaying to 1e-6
- **Gradient clipping:** max_norm=1.0

### Regularization
- Label smoothing: 0.1
- Dropout: 0.3 (head layer 1), 0.2 (head layer 2)
- Weight decay: 1e-4
- Mixup (alpha=0.2) + CutMix (alpha=1.0)

### Mixed Precision
- `torch.amp.GradScaler('cuda')` + `torch.amp.autocast('cuda')` for FP16 training (PyTorch 2.0+ API)
- ~2x training speedup, ~40% memory reduction

### Callbacks
- `EarlyStopping(patience=5, monitor='val_accuracy')`
- `ModelCheckpoint(save_best=True, save_last=True)`
- `LRLogger` — log learning rate to MLflow each epoch
- `MetricsLogger` — log train/val loss and accuracy to MLflow

### Experiment Tracking (MLflow)
- Every run logs:
  - Hyperparameters (from config YAML)
  - Per-epoch train/val loss and accuracy
  - Learning rate schedule
  - Best validation accuracy
  - Training time
  - Model artifacts (best checkpoint, ONNX export)
  - Augmentation configuration
  - GPU utilization stats

### Hyperparameter Tuning (Optuna)
- Optuna tunes **Phase 2 only** (Phase 1 uses fixed defaults as a warm-start)
- Search space:
  - `phase2.backbone_lr`: [1e-5, 1e-3] (log uniform) — maps to `training.phase2.backbone_lr`
  - `phase2.head_lr`: [1e-4, 1e-2] (log uniform) — maps to `training.phase2.head_lr`
  - `weight_decay`: [1e-5, 1e-2] (log uniform) — maps to `training.weight_decay`
  - `dropout`: [0.1, 0.5] (uniform) — maps to `model.dropout` (both layers use same value)
  - `label_smoothing`: [0.0, 0.2] (uniform) — maps to `training.label_smoothing`
  - `unfreeze_blocks`: [1, 2, 3, 4] (categorical) — maps to `training.phase2.unfreeze_blocks`
- 20-30 trials with TPE sampler
- Pruning underperforming trials early (MedianPruner)
- All trials logged to MLflow

---

## Ablation Studies

Run and report these 4 configurations to demonstrate incremental understanding:

| Ablation | Description | Expected Impact |
|----------|-------------|-----------------|
| A: Baseline | Frozen EfficientNet-B2 + MLP head, no augmentation beyond basic | ~70-75% top-1 |
| B: + Fine-tuning | Unfreeze top 3 blocks, discriminative LR | +5-8% |
| C: + Advanced augmentation | Add RandAugment, Mixup, CutMix | +2-3% |
| D: + Label smoothing + tuned hyperparams | Full pipeline with Optuna-tuned params | +1-2% |

**Target:** 85-90% top-1 accuracy (published SOTA on Food-101 is ~93% with larger models)

Each ablation is a separate MLflow experiment for easy comparison.

---

## Evaluation & Analysis

### Metrics
- Top-1 accuracy
- Top-5 accuracy
- Per-class precision, recall, F1-score
- Macro-averaged and weighted-averaged F1
- Inference latency (ms per image, PyTorch vs ONNX)

### Visualizations (generated as artifacts + in notebooks)
- Confusion matrix heatmap (full 101x101 + zoomed top-10 most confused pairs)
- Training/validation loss curves
- Training/validation accuracy curves
- Learning rate schedule plot
- Per-class accuracy bar chart (sorted, highlight best/worst 10)
- GradCAM attention maps — 3-5 examples per category for correct and incorrect predictions
- Failure analysis grid — misclassified images with predicted vs. true labels and confidence scores

### Error Analysis Notebook (03_error_analysis.ipynb)
- Top-20 most confused class pairs with example images
- Categorize failures: visually similar dishes, poor image quality, ambiguous labels, portion of dish shown
- Confidence distribution for correct vs. incorrect predictions
- Analysis of where GradCAM attention falls on misclassified images

---

## Model Serving

### ONNX Export
- Export best PyTorch checkpoint to ONNX format
- Validate ONNX model output matches PyTorch output (within tolerance)
- Optional: TensorRT optimization for maximum NVIDIA GPU throughput

### FastAPI Application

**Endpoints:**
- `POST /predict` — single image classification
  - Input: image file (JPEG/PNG, max 10MB)
  - Output: `{ "predictions": [{"class": "pizza", "confidence": 0.94}, ...], "latency_ms": 12.3 }`
  - Returns top-5 predictions with confidence scores
- `POST /predict/batch` — batch classification
  - Input: multiple image files (max 16)
  - Output: list of prediction results
- `GET /health` — health check
  - Output: `{ "status": "healthy", "model": "efficientnet_b2", "version": "1.0.0" }`
- `GET /metrics` — Prometheus-compatible metrics
  - Request count, latency histogram, error rate

**Implementation details:**
- Model loaded once at startup into GPU memory
- ONNX Runtime InferenceSession for inference
- Input validation: file type, file size, image dimensions
- Structured JSON logging for all requests (latency, prediction, confidence)
- Async request handling

---

## Containerization

### Dockerfile.train
- Base: `nvidia/cuda:12.1-devel-ubuntu22.04`
- Installs: Python 3.11, PyTorch (CUDA), all training dependencies
- Entrypoint: `python -m src.training.trainer --config /configs/default.yaml`
- Mounts: configs/, data/, mlruns/ as volumes

### Dockerfile.serve
- Base: `python:3.11-slim`
- Installs: FastAPI, ONNX Runtime (GPU), uvicorn, minimal deps only
- Multi-stage build to keep image small (~500MB)
- Copies: exported ONNX model + class labels
- Entrypoint: `uvicorn src.serving.app:app --host 0.0.0.0 --port 8000`

### docker-compose.yml
- `train` service: training container with GPU access
- `serve` service: inference API on port 8000
- `mlflow` service: MLflow tracking server on port 5000
- Shared volume for model artifacts

---

## CI/CD (GitHub Actions)

### ci.yml (on every PR)
- Lint with ruff
- Type check with mypy
- Run unit tests with pytest
- Check code coverage (>80%)

### model-validation.yml (on merge to main)
- Run integration tests (train on 10 images, predict via API)
- Validate exported ONNX model loads and produces valid output shapes
- Validate model accuracy on test set beats stored baseline threshold
- Build Docker images (no push — just verify they build)

---

## Testing

### Unit Tests
- `test_data.py`: transforms produce correct output shapes, dataset returns (image, label) tuples, augmentation pipeline runs without error
- `test_model.py`: model forward pass produces (batch, 101) output, create_model factory returns correct architecture, custom head has expected parameter count
- `test_metrics.py`: accuracy computation on known inputs, F1 score edge cases (zero predictions), confusion matrix shape
- `test_config.py`: YAML loading, CLI override merging, missing key defaults

### Integration Tests
- `test_api.py`: POST /predict returns 200 with valid image, returns 422 with invalid input, /health returns model info, /predict/batch handles multiple images
- Training integration: train 1 epoch on 10 images completes without error, checkpoint saves and loads correctly, MLflow run is created with expected metrics

---

## Configuration (YAML)

```yaml
# configs/default.yaml
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
    lr: 1e-3
    freeze_backbone: true
  phase2:
    epochs: 25
    backbone_lr: 1e-4
    head_lr: 5e-4
    unfreeze_blocks: 3
  optimizer: adamw
  weight_decay: 1e-4
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

---

## Makefile

```makefile
.PHONY: setup data train evaluate serve test lint clean

setup:
	pip install -e ".[dev]"
	pre-commit install

data:
	dvc pull
	python -m src.data.download

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
	rm -rf mlruns/ outputs/ __pycache__ .pytest_cache
```

---

## Migration Plan

### Files to remove from current repo:
- `hangman_game.py`
- `neural_network_hangman.py`
- `neural_network_demo.py`
- `train_letter_model.py`
- `Hangman_wordbank`
- `models/` directory
- `requirements.txt` (replaced by pyproject.toml)
- Current `README.md` (replaced with new one)

### Files to keep:
- `.git/` (preserve history)
- `.gitignore` (will be updated)

---

## Documentation

### README.md structure:
1. Project title + one-line description
2. Architecture diagram (text-based flowchart)
3. Results table (accuracy, comparison to baselines)
4. GradCAM example images (2-3)
5. Quick start (3 commands)
6. Detailed usage (training, evaluation, serving, Docker)
7. Project structure
8. Design decisions (why EfficientNet, why ONNX, why DVC)
9. Ablation results table

### Notebooks:
1. `01_eda.ipynb` — class distribution, sample images, image size stats, data quality
2. `02_training_analysis.ipynb` — ablation table, training curves, HP search results, LR schedule
3. `03_error_analysis.ipynb` — confusion matrix, GradCAM maps, failure categorization, confidence analysis
