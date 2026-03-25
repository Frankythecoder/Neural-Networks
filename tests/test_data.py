import pytest
import torch
from PIL import Image

from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.dataset import Food101Dataset


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


class TestFood101Dataset:
    def test_dataset_returns_tuple(self, sample_image):
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
