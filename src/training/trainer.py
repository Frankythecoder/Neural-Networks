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

    logger.info(
        "=== Phase %d: %s (%d epochs) ===",
        phase,
        "Feature Extraction" if phase == 1 else "Fine-tuning",
        epochs,
    )

    if phase == 1:
        model.freeze_backbone()
    else:
        model.unfreeze_top_blocks(n=phase_cfg.get("unfreeze_blocks", 3))

    optimizer = create_optimizer(model, config, phase=phase)
    scheduler = create_scheduler(optimizer, config, phase=phase)

    criterion = nn.CrossEntropyLoss(label_smoothing=training_cfg["label_smoothing"])

    scaler = (
        GradScaler("cuda")
        if (training_cfg["mixed_precision"] and device.type == "cuda")
        else None
    )

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
            phase,
            epoch,
            epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            lr,
            elapsed,
        )

        # MLflow logging
        mlflow.log_metrics(
            {
                f"phase{phase}/train_loss": train_metrics["loss"],
                f"phase{phase}/train_accuracy": train_metrics["accuracy"],
                f"phase{phase}/val_loss": val_metrics["loss"],
                f"phase{phase}/val_accuracy": val_metrics["accuracy"],
                f"phase{phase}/lr": lr,
            },
            step=epoch,
        )

        checkpoint.step(
            model,
            val_metrics["accuracy"],
            epoch,
            extra={"optimizer_state_dict": optimizer.state_dict()},
        )

        if early_stopping.step(val_metrics["accuracy"]):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    return model


def train(
    config: dict[str, Any], train_loader: DataLoader, val_loader: DataLoader
) -> nn.Module:
    """Full two-phase training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = create_model(config).to(device)

    mlflow.log_params(
        {
            "model": config["model"]["name"],
            "seed": config["seed"],
            "phase1_epochs": config["training"]["phase1"]["epochs"],
            "phase2_epochs": config["training"]["phase2"]["epochs"],
            "batch_size": config["data"]["batch_size"],
            "label_smoothing": config["training"]["label_smoothing"],
        }
    )

    # Phase 1: Feature extraction
    model = train_phase(model, train_loader, val_loader, config, device, phase=1)

    # Phase 2: Fine-tuning
    model = train_phase(model, train_loader, val_loader, config, device, phase=2)

    logger.info("Training complete.")
    return model


if __name__ == "__main__":
    import argparse

    from torchvision.datasets import Food101

    from src.data.dataset import Food101Dataset
    from src.data.download import create_splits
    from src.data.transforms import get_eval_transforms, get_train_transforms
    from src.utils.config import load_config
    from src.utils.reproducibility import set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    set_seed(cfg["seed"])

    base_train = Food101(root="data", split="train")
    splits = create_splits(
        data_dir="data", val_split=cfg["data"]["val_split"], seed=cfg["seed"]
    )

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
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    with mlflow.start_run():
        train(cfg, train_loader, val_loader)
