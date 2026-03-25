from __future__ import annotations

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
