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
