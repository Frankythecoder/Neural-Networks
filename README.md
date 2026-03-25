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
