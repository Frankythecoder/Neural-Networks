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
