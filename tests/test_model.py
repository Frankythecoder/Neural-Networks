import pytest
import torch

from src.models.efficientnet import EfficientNetFood
from src.models.factory import create_model


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
        unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        assert unfrozen > 0


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
