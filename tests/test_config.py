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
